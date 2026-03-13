"""
ChurnMaster Pro — prediction.py
================================
Taken directly from ChurnIQ Pro and merged here.

Provides:
  load_bundle()       — thread-safe model loading (bundle or legacy PKLs)
  make_prediction()   — full rich ML prediction with reasons, actions, timeline
  generate_reasons()  — domain-aware churn explanation
  generate_actions()  — smart personalised retention playbook
  generate_timeline() — estimated time-to-churn window
  _classify_risk()    — LOW / MEDIUM / HIGH threshold classifier
  _loyalty_grade()    — A–F loyalty score grader

Why keep this separate from routes.py?
  Prediction logic is heavy and has its own threading concerns.
  Keeping it in its own module makes testing, swapping models, and
  hot-reloading safe without touching any route code.
"""

import os
import pickle
import logging
import threading

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from config import (
    BASE_DIR, MODELS_DIR,
    RISK_LOW_THRESHOLD, RISK_HIGH_THRESHOLD,
    GRADE_A_MIN, GRADE_B_MIN, GRADE_C_MIN, GRADE_D_MIN,
    DOMAIN_REASONS,
)

logger = logging.getLogger(__name__)

# ── Thread-safe bundle cache ──────────────────────────────────────────────
_lock         = threading.Lock()
_bundle_mtime = 0.0
BUNDLE        = None


# ══════════════════════════════════════════════════════════════════════════
#  MODEL LOADING — bundle-first, legacy fallback
# ══════════════════════════════════════════════════════════════════════════

def _get_bundle_path() -> str | None:
    candidates = [
        os.path.join(MODELS_DIR, "model_bundle.pkl"),
        os.path.join(BASE_DIR,   "model_bundle.pkl"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def load_bundle() -> dict | None:
    """
    Load and cache the model bundle with thread-safe hot-reload.

    Priority:
      1. models/model_bundle.pkl  (ChurnIQ-style bundle — preferred)
      2. churn_model.pkl + scaler.pkl + feature_names.pkl  (customer_churn legacy)
      3. None — no model available

    The returned dict always has keys:
      model, scaler, imputer, feature_names, domain, importance_map,
      metrics, n_samples, n_features, target_col, legacy (bool)
    """
    global BUNDLE, _bundle_mtime

    with _lock:
        # ── Path 1: model_bundle.pkl (ChurnIQ Pro trained bundle) ─────────
        path = _get_bundle_path()
        if path:
            mtime = os.path.getmtime(path)
            if BUNDLE is None or mtime != _bundle_mtime:
                logger.info("Loading model bundle from: %s", path)
                try:
                    with open(path, "rb") as f:
                        BUNDLE = pickle.load(f)
                    _bundle_mtime = mtime
                    logger.info(
                        "Bundle loaded: domain=%s  samples=%d  features=%d",
                        BUNDLE.get("domain", "?"),
                        BUNDLE.get("n_samples", 0),
                        BUNDLE.get("n_features", 0),
                    )
                except Exception as exc:
                    logger.error("Failed to load bundle: %s", exc)
                    return None
            return BUNDLE

        # ── Path 2: legacy separate PKL files (customer_churn style) ──────
        model_p   = os.path.join(BASE_DIR, "churn_model.pkl")
        scaler_p  = os.path.join(BASE_DIR, "scaler.pkl")
        feature_p = os.path.join(BASE_DIR, "feature_names.pkl")

        if not os.path.exists(model_p):
            return None

        mtime = os.path.getmtime(model_p)
        if BUNDLE is not None and mtime == _bundle_mtime:
            return BUNDLE

        logger.info("Loading legacy PKL files from: %s", BASE_DIR)
        try:
            with open(model_p,   "rb") as f: model         = pickle.load(f)
            with open(scaler_p,  "rb") as f: scaler        = pickle.load(f)
            with open(feature_p, "rb") as f: feature_names = pickle.load(f)
        except Exception as exc:
            logger.error("Failed to load legacy PKL files: %s", exc)
            return None

        # Build importance map from legacy model (if it supports feature_importances_)
        imp_map: dict = {}
        if hasattr(model, "estimators_"):
            for _, est in model.estimators_:
                if hasattr(est, "feature_importances_"):
                    for fn, imp in zip(feature_names, est.feature_importances_):
                        imp_map[fn] = imp_map.get(fn, 0.0) + imp
            total = sum(imp_map.values()) or 1.0
            imp_map = {k: v / total for k, v in imp_map.items()}
        elif hasattr(model, "feature_importances_"):
            imp_map = dict(zip(feature_names, model.feature_importances_))

        imputer = SimpleImputer(strategy="median")

        BUNDLE = {
            "model":          model,
            "scaler":         scaler,
            "imputer":        imputer,
            "feature_names":  feature_names,
            "domain":         "Telecom",
            "target_col":     "Churn",
            "importance_map": imp_map,
            "metrics":        {"accuracy": 0.0, "auc": 0.0},
            "cv_accuracy":    0.0,
            "n_samples":      0,
            "n_features":     len(feature_names),
            "legacy":         True,
        }
        _bundle_mtime = mtime
        logger.info("Legacy PKL model loaded: %d features", len(feature_names))
        return BUNDLE


# ══════════════════════════════════════════════════════════════════════════
#  CORE PREDICTION ENGINE
# ══════════════════════════════════════════════════════════════════════════

def make_prediction(input_dict: dict, bundle: dict) -> dict:
    """
    Transform a raw feature dict into a full rich prediction result.

    Returns dict with keys:
      prediction    — 0 or 1
      probability   — float 0–100
      risk          — 'LOW' | 'MEDIUM' | 'HIGH'
      label         — 'Will Churn' | 'Will Stay'
      breakdown     — top-8 feature impact list
      reasons       — up to 6 domain-aware churn reason cards
      actions       — up to 5 personalised retention actions
      timeline      — time-to-churn estimate
      loyalty_score — int 0–100
      loyalty_grade — 'A'–'F'
    """
    feature_names = bundle["feature_names"]

    # Build feature row aligned to trained feature list
    row = {f: 0.0 for f in feature_names}
    for k, v in input_dict.items():
        if k in feature_names:
            try:
                row[k] = float(v)
            except (TypeError, ValueError):
                row[k] = 0.0

    X = pd.DataFrame([row])[feature_names]

    try:
        X_imp = bundle["imputer"].transform(X)
    except Exception:
        X_imp = X.fillna(0.0).values

    X_sc  = bundle["scaler"].transform(X_imp)
    pred  = int(bundle["model"].predict(X_sc)[0])
    proba = float(bundle["model"].predict_proba(X_sc)[0][1])
    risk  = _classify_risk(proba)

    imp_map   = bundle.get("importance_map", {})
    breakdown = sorted(
        [
            {"feature": f, "impact": round(imp_map.get(f, 0) * proba * 100, 2)}
            for f in feature_names
            if imp_map.get(f, 0) > 0.005
        ],
        key=lambda x: x["impact"],
        reverse=True,
    )[:8]

    reasons  = _generate_reasons(input_dict, breakdown, bundle, proba)
    actions  = _generate_actions(pred, proba, risk, breakdown, bundle)
    timeline = _generate_timeline(proba, input_dict)
    score    = max(0, min(100, round((1 - proba) * 100)))
    grade    = _loyalty_grade(score)

    return {
        "prediction":    pred,
        "probability":   round(proba * 100, 2),
        "risk":          risk,
        "label":         "Will Churn" if pred == 1 else "Will Stay",
        "breakdown":     breakdown,
        "reasons":       reasons,
        "actions":       actions,
        "timeline":      timeline,
        "loyalty_score": score,
        "loyalty_grade": grade,
    }


# ══════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════

def _classify_risk(proba: float) -> str:
    if proba < RISK_LOW_THRESHOLD:  return "LOW"
    if proba < RISK_HIGH_THRESHOLD: return "MEDIUM"
    return "HIGH"


def _loyalty_grade(score: int) -> str:
    if score >= GRADE_A_MIN: return "A"
    if score >= GRADE_B_MIN: return "B"
    if score >= GRADE_C_MIN: return "C"
    if score >= GRADE_D_MIN: return "D"
    return "F"


# ══════════════════════════════════════════════════════════════════════════
#  REASON ENGINE
# ══════════════════════════════════════════════════════════════════════════

def _generate_reasons(input_dict: dict, breakdown: list, bundle: dict, proba: float) -> list:
    domain     = bundle.get("domain", "General")
    domain_map = DOMAIN_REASONS.get(domain, DOMAIN_REASONS["General"])
    reasons    = []

    for item in breakdown[:6]:
        feat   = item["feature"]
        impact = item["impact"]
        val    = input_dict.get(feat, 0)
        sev    = "HIGH" if impact > 8 else ("MEDIUM" if impact > 3 else "LOW")
        matched = False

        for keyword, (icon, title, detail) in domain_map.items():
            if keyword.lower() in feat.lower():
                reasons.append({
                    "icon": icon, "title": title, "detail": detail,
                    "impact": impact, "severity": sev, "feature": feat, "value": val,
                })
                matched = True
                break

        if not matched:
            pretty = feat.replace("_", " ").title()
            reasons.append({
                "icon": "🔸",
                "title": f"{pretty} Risk Factor",
                "detail": (
                    f"This feature shows a pattern associated with churn in your dataset. "
                    f"Value of {val} contributes {impact:.1f}% to the churn probability."
                ),
                "impact": impact, "severity": sev, "feature": feat, "value": val,
            })

    if proba >= 0.75:
        reasons.insert(0, {
            "icon": "🚨", "title": "Critical Churn Risk",
            "detail": (
                f"With {round(proba * 100)}% churn probability, this customer is in the danger "
                f"zone. Multiple risk factors are compounding. Immediate action required."
            ),
            "impact": proba * 100, "severity": "HIGH", "feature": "overall", "value": proba,
        })

    return reasons[:6]


# ══════════════════════════════════════════════════════════════════════════
#  SMART RETENTION ACTIONS
# ══════════════════════════════════════════════════════════════════════════

def _generate_actions(pred: int, proba: float, risk: str, breakdown: list, bundle: dict) -> list:
    features = [b["feature"].lower() for b in breakdown[:5]]
    actions  = []

    if risk == "HIGH":
        actions.append({
            "priority": "TODAY", "color": "#ef4444", "icon": "📞",
            "action":   "Personal Retention Call",
            "detail":   "Call this customer directly. Human touch has 40% success rate for high-risk customers. Acknowledge their concerns before offering solutions.",
            "effort": "High", "impact": "Very High", "success_rate": "40%",
        })

    if any(w in f for f in features for w in ["contract", "plan", "subscription"]):
        actions.append({
            "priority": "URGENT" if risk != "LOW" else "HELPFUL",
            "color": "#f97316", "icon": "📋",
            "action": "Offer Long-Term Contract Incentive",
            "detail": "A 10–15% discount to switch to annual/2-year plan reduces churn by 60%. Present as exclusive loyalty offer.",
            "effort": "Low", "impact": "Very High", "success_rate": "60%",
        })

    if any(w in f for f in features for w in ["payment", "check", "billing"]):
        actions.append({
            "priority": "IMPORTANT", "color": "#f97316", "icon": "💳",
            "action": "Switch to Auto-Pay with Reward",
            "detail": "Auto-pay customers churn 30% less. Offer 1 free month to switch — removes monthly billing friction points.",
            "effort": "Low", "impact": "High", "success_rate": "55%",
        })

    if any(w in f for f in features for w in ["security", "support", "protection", "tech"]):
        actions.append({
            "priority": "IMPORTANT", "color": "#f97316", "icon": "🎁",
            "action": "Free Add-On Trial (3 Months)",
            "detail": "Add Online Security or Tech Support free for 3 months. Customers with 3+ services have 50% lower churn. 70% of trial users convert.",
            "effort": "Low", "impact": "High", "success_rate": "70%",
        })

    actions.append({
        "priority": "LONG TERM", "color": "#10b981", "icon": "🏆",
        "action": "Enrol in Loyalty Rewards Program",
        "detail": "Loyalty program members churn 35% less than non-members. Offer points per billing cycle redeemable for upgrades or discounts.",
        "effort": "Low", "impact": "High", "success_rate": "35%",
    })

    return actions[:5]


# ══════════════════════════════════════════════════════════════════════════
#  CHURN TIMELINE ESTIMATE
# ══════════════════════════════════════════════════════════════════════════

def _generate_timeline(proba: float, input_dict: dict) -> dict:
    if proba < RISK_LOW_THRESHOLD:
        return {
            "likely_churn_months": None,
            "label":       "Low Risk — No Imminent Churn",
            "description": "Strong retention signals. No churn expected in next 12 months based on current patterns.",
            "urgency":     "low",
        }
    if proba < RISK_HIGH_THRESHOLD:
        months = max(1, round(6 * (1 - proba) + 1))
        return {
            "likely_churn_months": months,
            "label":       f"May Churn in ~{months} Months",
            "description": f"Based on current risk profile, this customer may disengage within {months} months if no retention action is taken.",
            "urgency":     "medium",
        }
    months = max(1, round(3 * (1 - proba)))
    plural = "s" if months > 1 else ""
    return {
        "likely_churn_months": months,
        "label":       f"⚠️ Likely to Churn Within {months} Month{plural}",
        "description": f"High churn probability indicates this customer may leave within {months} month(s). Immediate retention action is critical.",
        "urgency":     "high",
    }

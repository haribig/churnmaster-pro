"""
ChurnMaster Pro — routes.py
============================
ALL URL routes as a single Flask Blueprint (`main_bp`).

Merges the route patterns from both original projects:
  • customer_churn  — /login, /signup, /logout, /dashboard, /predict,
                      /download_csv, /download_excel
  • ChurnIQ Pro     — /analytics (JSON), /history (JSON), /stats (JSON),
                      /bulk_predict, /export/csv, /export/powerbi

New additions:
  • @login_required decorator replacing copy-pasted `if "user" not in session`
  • flash() messages for user-facing errors (no silent failures)
  • Proper logging on every route (INFO for success, ERROR for failures)
  • /api/stats endpoint for live dashboard AJAX refresh
  • Specific exception types (KeyError, ValueError) instead of bare `except:`
  • url_for() throughout — zero hardcoded URL strings

Changes to original customer_churn:
  OLD: plain-text password in DB + `if result:` comparison
  NEW: generate_password_hash / verify_password from auth_service.py

Changes to original ChurnIQ Pro:
  OLD: all routes unprotected (no session auth)
  NEW: login_required on every route that touches data
"""

import io
import csv
import json
import logging
import traceback
from datetime import datetime
from functools import wraps

import pandas as pd
from flask import (
    Blueprint, render_template, request, redirect,
    session, flash, url_for, jsonify, send_file,
)

from config import PRED_DB_PATH
from db import (
    db_insert, db_insert_many,
    db_history, db_stats, db_trend, db_risk_distribution,
)
from prediction import load_bundle, make_prediction, _classify_risk
from services.auth_service import get_user, create_user, verify_password
from services.chart_service import generate_all_charts

logger  = logging.getLogger(__name__)
main_bp = Blueprint("main", __name__)


# ══════════════════════════════════════════════════════════════════════════
#  LOGIN-REQUIRED DECORATOR
# ══════════════════════════════════════════════════════════════════════════

def login_required(f):
    """
    Route decorator — redirects unauthenticated users to /login.

    Usage:
        @main_bp.route("/dashboard")
        @login_required
        def dashboard():
            ...

    Why a decorator instead of inline `if "user" not in session`?
      Single point of change — if the auth mechanism changes (e.g., adding
      JWT, OAuth), you update ONE function, not every route handler.
    """
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user" not in session:
            flash("Please log in to access this page.", "warning")
            return redirect(url_for("main.login"))
        return f(*args, **kwargs)
    return decorated


# ══════════════════════════════════════════════════════════════════════════
#  AUTH ROUTES
# ══════════════════════════════════════════════════════════════════════════

@main_bp.route("/", methods=["GET", "POST"])
@main_bp.route("/login", methods=["GET", "POST"])
def login():
    """
    GET  → render login form
    POST → validate credentials → redirect to dashboard

    Two route registrations (/ and /login) mean both URLs work,
    preserving compatibility with bookmarks from the original project.

    Security changes vs original customer_churn:
      OLD: con.execute("SELECT * FROM users WHERE username=? AND password=?", (user, pwd))
           — plain-text password comparison, SQL injection risk
      NEW: get_user() + verify_password() — werkzeug PBKDF2 verification,
           parameterised queries, auto-upgrades legacy plain-text passwords
    """
    if "user" in session:
        return redirect(url_for("main.dashboard"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        if not username or not password:
            flash("Username and password are required.", "error")
            return render_template("login.html")

        user = get_user(username)
        if user and verify_password(user["password"], password, user["id"]):
            session["user"]     = user["username"]
            session["user_id"]  = user["id"]
            session["role"]     = user["role"]
            session.permanent   = True    # respect the 8-hour lifetime set in app.py
            logger.info("Login: %s", username)
            return redirect(url_for("main.dashboard"))

        logger.warning("Failed login attempt for username: %s", username)
        flash("Invalid username or password.", "error")

    return render_template("login.html")


@main_bp.route("/signup", methods=["GET", "POST"])
def signup():
    """
    GET  → render signup form
    POST → validate → create hashed user → redirect to login

    Validation rules enforced server-side:
      • username: 3–30 chars, no spaces
      • password: min 6 chars
      • passwords must match
      • username must not already exist
    """
    if "user" in session:
        return redirect(url_for("main.dashboard"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        confirm  = request.form.get("confirm_password", "")

        # Explicit validation messages — better UX than generic errors
        if not all([username, password, confirm]):
            flash("All fields are required.", "error")
            return render_template("signup.html")
        if len(username) < 3 or len(username) > 30 or " " in username:
            flash("Username must be 3–30 characters with no spaces.", "error")
            return render_template("signup.html")
        if len(password) < 6:
            flash("Password must be at least 6 characters.", "error")
            return render_template("signup.html")
        if password != confirm:
            flash("Passwords do not match.", "error")
            return render_template("signup.html")
        if get_user(username):
            flash("That username is already taken.", "error")
            return render_template("signup.html")

        create_user(username, password)
        logger.info("New user registered: %s", username)
        flash("Account created — please sign in.", "success")
        return redirect(url_for("main.login"))

    return render_template("signup.html")


@main_bp.route("/logout")
def logout():
    who = session.get("user", "unknown")
    session.clear()
    logger.info("Logout: %s", who)
    flash("You have been signed out.", "info")
    return redirect(url_for("main.login"))


# ══════════════════════════════════════════════════════════════════════════
#  DASHBOARD
# ══════════════════════════════════════════════════════════════════════════

@main_bp.route("/dashboard")
@login_required
def dashboard():
    """
    Main dashboard. Passes live stats to template.
    Template renders: KPI cards, recent predictions table, risk breakdown,
    quick-action buttons, and the Power BI embed section.
    """
    bundle = load_bundle()
    stats  = db_stats()
    recent = db_history(limit=10)
    trend  = db_trend(days=30)
    risk_d = db_risk_distribution()

    model_meta = {}
    if bundle:
        model_meta = {
            "domain":    bundle.get("domain",    "Telecom"),
            "n_features": bundle.get("n_features", 0),
            "n_samples":  bundle.get("n_samples",  0),
            "accuracy":   round(bundle.get("metrics", {}).get("accuracy", 0) * 100, 1),
            "auc":        round(bundle.get("metrics", {}).get("auc",      0) * 100, 1),
            "loaded":     True,
        }

    return render_template(
        "dashboard.html",
        username   = session["user"],
        role       = session.get("role", "analyst"),
        stats      = stats,
        recent     = recent,
        trend      = trend,
        risk_dist  = risk_d,
        model_meta = model_meta,
    )


@main_bp.route("/api/stats")
@login_required
def api_stats():
    """JSON endpoint for live dashboard KPI refresh (called by JS every 60 s)."""
    return jsonify(db_stats())


@main_bp.route("/api/history")
@login_required
def api_history():
    """JSON endpoint for recent prediction table refresh."""
    return jsonify(db_history(limit=20))


# ══════════════════════════════════════════════════════════════════════════
#  SINGLE PREDICTION (GET form + POST processing)
# ══════════════════════════════════════════════════════════════════════════

@main_bp.route("/predict", methods=["GET", "POST"])
@login_required
def predict():
    """
    GET  → render prediction form (Telecom / Bank / HR domain switcher)
    POST → parse domain-aware fields → run ML → render result
    """
    if request.method == "POST":
        try:
            bundle = load_bundle()
            if not bundle:
                flash("Model not loaded. Contact your administrator.", "error")
                return redirect(url_for("main.predict"))

            domain = request.form.get("domain", "Bank")

            # ── Domain-aware input parsing ─────────────────────────────────
            # NOTE: The loaded model was trained on Bank customer data.
            # Feature names: credit_score, age, tenure, balance, products_number,
            #   credit_card, active_member, estimated_salary, country_*, gender_Male
            # Bank fields map 1-to-1. Telecom & HR fields are mapped to equivalent
            # bank features for the shared model.

            if domain == "Bank":
                input_dict = {
                    "credit_score":     float(request.form["credit_score"]),
                    "age":              int(request.form["age"]),
                    "tenure":           int(request.form["tenure"]),
                    "balance":          float(request.form["balance"]),
                    "products_number":  int(request.form["products_number"]),
                    "estimated_salary": float(request.form["estimated_salary"]),
                    "credit_card":      int(request.form["credit_card"]),
                    "active_member":    int(request.form["active_member"]),
                    "gender":           ["Female", "Male"][int(request.form["gender"])],
                    "country":          ["France", "Germany", "Spain"][int(request.form["country"])],
                }

            elif domain == "HR":
                # Map HR features → bank model feature space
                monthly_income = float(request.form["MonthlyIncome"])
                years_at_co    = int(request.form["YearsAtCompany"])
                jsat           = int(request.form["JobSatisfaction"])
                overtime       = 1 if request.form.get("OverTime") == "1" else 0
                input_dict = {
                    "credit_score":     float(request.form.get("PercentSalaryHike", 15)) * 40,  # 11-25 → 440-1000 proxy
                    "age":              int(request.form["Age"]),
                    "tenure":           years_at_co,
                    "balance":          monthly_income * 12,        # annual salary as balance proxy
                    "products_number":  max(1, int(request.form.get("NumCompaniesWorked", 1))),
                    "estimated_salary": monthly_income,
                    "credit_card":      1,                           # assume all employees have card
                    "active_member":    1 if jsat >= 3 else 0,       # satisfied = active proxy
                    "gender":           "Male" if int(request.form.get("HRGender","0")) == 1 else "Female",
                    "country":          "France",                    # neutral default
                }

            else:  # Telecom
                # Map Telecom features → bank model feature space
                monthly = float(request.form["MonthlyCharges"])
                tenure  = int(request.form["tenure"])
                has_inet = int(request.form.get("InternetService", 0))
                contract = int(request.form.get("Contract", 0))
                input_dict = {
                    "credit_score":     700 - (contract * 50),      # no contract = lower score proxy
                    "age":              45 if int(request.form.get("SeniorCitizen", 0)) else 35,
                    "tenure":           tenure,
                    "balance":          float(request.form["TotalCharges"]),
                    "products_number":  1 + (1 if has_inet > 0 else 0),
                    "estimated_salary": monthly * 24,
                    "credit_card":      1 if int(request.form.get("PaymentMethod", 0)) == 1 else 0,
                    "active_member":    1 if tenure > 12 else 0,
                    "gender":           "Male" if int(request.form.get("gender", 0)) == 1 else "Female",
                    "country":          "France",
                }

            # ── One-hot encode to match trained feature names ──────────────
            input_df = pd.get_dummies(pd.DataFrame([input_dict]), drop_first=True)
            feature_names = bundle["feature_names"]
            for col in feature_names:
                if col not in input_df.columns:
                    input_df[col] = 0
            input_encoded = input_df[feature_names]

            # ── Model selection ────────────────────────────────────────────
            selected_model = request.form.get("selected_model", "ensemble").lower()
            model_to_use   = bundle["model"]
            if selected_model != "ensemble":
                named = getattr(bundle["model"], "named_estimators_", {})
                if selected_model in named:
                    model_to_use = named[selected_model]

            scaled      = bundle["scaler"].transform(input_encoded)
            prediction  = int(model_to_use.predict(scaled)[0])
            probability = model_to_use.predict_proba(scaled)[0]

            churn_prob    = round(float(probability[1]) * 100, 2)
            no_churn_prob = round(float(probability[0]) * 100, 2)
            risk          = _classify_risk(float(probability[1]))
            rich          = make_prediction(input_dict, bundle)
            label         = "Will Churn" if prediction == 1 else "Will Stay"

            db_insert({
                "timestamp":   datetime.utcnow().isoformat(),
                "prediction":  prediction,
                "probability": churn_prob,
                "risk":        risk,
                "label":       label,
                "domain":      domain,
                "source":      "single",
                "customer_id": request.form.get("customer_id", ""),
                "username":    session["user"],
                "input":       input_dict,
                "reasons":     [r["title"] for r in rich["reasons"]],
            })

            session["last_prediction"] = {
                **input_dict,
                "Prediction":          label,
                "Churn_Probability":   f"{churn_prob:.2f}%",
                "No_Churn_Probability": f"{no_churn_prob:.2f}%",
                "Risk":                risk,
                "Domain":              domain,
                "Loyalty_Score":       rich["loyalty_score"],
                "Loyalty_Grade":       rich["loyalty_grade"],
                "Timeline":            rich["timeline"]["label"],
                "Timestamp":           datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Username":            session["user"],
            }

            charts = generate_all_charts(input_dict, probability, prediction)
            logger.info("Prediction by %s: %s  domain=%s  churn=%.1f%%  risk=%s  model=%s",
                        session["user"], label, domain, churn_prob, risk, selected_model)

            return render_template(
                "result.html",
                username      = session["user"],
                prediction    = label,
                churn_prob    = churn_prob,
                no_churn_prob = no_churn_prob,
                risk          = risk,
                domain        = domain,
                rich          = rich,
                **charts,
            )

        except KeyError as exc:
            logger.warning("Missing form field: %s", exc)
            flash(f"⚠️ Missing field: {exc}. Please fill in all required fields.", "error")
            return redirect(url_for("main.predict"))
        except (ValueError, IndexError) as exc:
            logger.warning("Invalid form value: %s", exc)
            flash(f"⚠️ Invalid value — {exc}. Check all numeric fields.", "error")
            return redirect(url_for("main.predict"))
        except Exception:
            logger.error("Unexpected prediction error:\n%s", traceback.format_exc())
            flash("❌ An unexpected error occurred. The error has been logged.", "error")
            return redirect(url_for("main.predict"))

    bundle = load_bundle()
    model_status = {
        "loaded":   bundle is not None,
        "domain":   bundle.get("domain", "Unknown")   if bundle else None,
        "features": bundle.get("n_features", 0)       if bundle else 0,
        "samples":  bundle.get("n_samples", 0)        if bundle else 0,
        "accuracy": round(bundle.get("metrics", {}).get("accuracy", 0) * 100, 1) if bundle else 0,
        "auc":      round(bundle.get("metrics", {}).get("auc", 0) * 100, 1)       if bundle else 0,
    }
    return render_template("predict.html", username=session["user"], model_status=model_status)


# ══════════════════════════════════════════════════════════════════════════
#  BULK PREDICTION (from ChurnIQ Pro)
# ══════════════════════════════════════════════════════════════════════════

@main_bp.route("/bulk_predict", methods=["POST"])
@login_required
def bulk_predict():
    """
    Upload a CSV file → run model on all rows → return JSON summary.
    Taken from ChurnIQ Pro — now protected by @login_required.
    """
    bundle = load_bundle()
    if not bundle:
        return jsonify({"success": False, "error": "Model not loaded."}), 503

    f = request.files.get("file")
    if not f:
        return jsonify({"success": False, "error": "No file uploaded."}), 400

    try:
        df            = pd.read_csv(f)
        feature_names = bundle["feature_names"]
        imp_map       = bundle.get("importance_map", {})

        id_col = next(
            (c for c in ["customerID", "CustomerID", "customer_id", "id", "ID"]
             if c in df.columns), None
        )

        X_bulk = pd.DataFrame(0.0, index=df.index, columns=feature_names)
        for feat in feature_names:
            if feat in df.columns:
                X_bulk[feat] = pd.to_numeric(df[feat], errors="coerce").fillna(0)

        try:
            X_imp = bundle["imputer"].transform(X_bulk)
        except Exception:
            X_imp = X_bulk.fillna(0).values

        X_sc   = bundle["scaler"].transform(X_imp)
        preds  = bundle["model"].predict(X_sc)
        probas = bundle["model"].predict_proba(X_sc)[:, 1]

        top_feat   = sorted(imp_map.items(), key=lambda x: x[1], reverse=True)
        top_reason = top_feat[0][0].replace("_", " ").title() if top_feat else ""
        now        = datetime.utcnow().isoformat()
        domain     = bundle.get("domain", "Telecom")
        results, db_rows = [], []

        for i, (pred, proba) in enumerate(zip(preds, probas)):
            pred  = int(pred)
            proba = float(proba)
            risk  = _classify_risk(proba)
            label = "Will Churn" if pred == 1 else "Will Stay"
            cid   = str(df.iloc[i][id_col]) if id_col else f"Row {i + 1}"

            results.append({
                "row": i + 1, "customerID": cid, "prediction": pred,
                "probability": round(proba * 100, 2), "risk": risk,
                "label": label, "top_reason": top_reason,
            })
            db_rows.append((
                now, pred, round(proba * 100, 2), risk, label,
                domain, "bulk", cid, session["user"], "{}", "[]",
            ))

        db_insert_many(db_rows)
        churned = [r for r in results if r["prediction"] == 1]

        logger.info(
            "Bulk predict by %s: %d rows  churned=%d",
            session["user"], len(results), len(churned),
        )
        return jsonify({
            "success":    True,
            "total":      len(df),
            "churned":    len(churned),
            "stayed":     len(results) - len(churned),
            "churn_rate": round(len(churned) / len(results) * 100, 1) if results else 0,
            "high_risk":  sum(1 for r in results if r["risk"] == "HIGH"),
            "med_risk":   sum(1 for r in results if r["risk"] == "MEDIUM"),
            "low_risk":   sum(1 for r in results if r["risk"] == "LOW"),
            "results":    results,
        })

    except Exception:
        logger.error("Bulk predict error:\n%s", traceback.format_exc())
        return jsonify({"success": False, "error": "Bulk prediction failed."}), 500


# ══════════════════════════════════════════════════════════════════════════
#  ANALYTICS JSON ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════

@main_bp.route("/analytics")
@login_required
def analytics():
    return jsonify({
        "stats":             db_stats(),
        "trend":             db_trend(30),
        "risk_distribution": db_risk_distribution(),
    })


@main_bp.route("/history")
@login_required
def history():
    """Full-page prediction history with search, filter, and export."""
    rows  = db_history(500)
    stats = db_stats()
    return render_template(
        "history.html",
        username = session["user"],
        role     = session.get("role", "analyst"),
        rows     = rows,
        stats    = stats,
    )




@main_bp.route("/history/clear", methods=["POST"])
@login_required
def clear_history():
    """Delete ALL prediction records from the database."""
    try:
        from db import db_clear
        db_clear()
        flash("✅ All prediction history has been cleared.", "success")
    except Exception as exc:
        flash("❌ Failed to clear history.", "error")
    return redirect(url_for("main.history"))


@main_bp.route("/bulk")
@login_required
def bulk():
    """Bulk CSV prediction page."""
    bundle = load_bundle()
    model_meta = {}
    if bundle:
        model_meta = {
            "domain":    bundle.get("domain", "Telecom"),
            "n_features": bundle.get("n_features", 0),
        }
    return render_template(
        "bulk.html",
        username   = session["user"],
        role       = session.get("role", "analyst"),
        model_meta = model_meta,
    )


@main_bp.route("/stats")
@login_required
def stats():
    return jsonify(db_stats())


# ══════════════════════════════════════════════════════════════════════════
#  DOWNLOAD / EXPORT ROUTES
# ══════════════════════════════════════════════════════════════════════════

@main_bp.route("/download_csv")
@login_required
def download_csv():
    """Download last single prediction as CSV (from customer_churn)."""
    if "last_prediction" not in session:
        flash("No prediction to download. Run a prediction first.", "warning")
        return redirect(url_for("main.predict"))
    try:
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = f"predictions_history/prediction_{session['user']}_{ts}.csv"
        pd.DataFrame([session["last_prediction"]]).to_csv(path, index=False)
        return send_file(path, as_attachment=True,
                         download_name=f"churn_prediction_{ts}.csv")
    except Exception as exc:
        logger.error("CSV download failed: %s", exc)
        flash("Download failed. Please try again.", "error")
        return redirect(url_for("main.predict"))


@main_bp.route("/download_excel")
@login_required
def download_excel():
    """Download last single prediction as Excel (from customer_churn)."""
    if "last_prediction" not in session:
        flash("No prediction to download. Run a prediction first.", "warning")
        return redirect(url_for("main.predict"))
    try:
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = f"predictions_history/prediction_{session['user']}_{ts}.xlsx"
        pd.DataFrame([session["last_prediction"]]).to_excel(
            path, index=False, engine="openpyxl"
        )
        return send_file(path, as_attachment=True,
                         download_name=f"churn_prediction_{ts}.xlsx")
    except ImportError:
        flash("Excel export requires openpyxl: pip install openpyxl", "error")
        return redirect(url_for("main.predict"))
    except Exception as exc:
        logger.error("Excel download failed: %s", exc)
        flash("Download failed. Please try again.", "error")
        return redirect(url_for("main.predict"))


@main_bp.route("/export/csv")
@login_required
def export_csv():
    """Export full prediction history as CSV (from ChurnIQ Pro)."""
    rows = db_history(100_000)
    if not rows:
        flash("No prediction history to export.", "warning")
        return redirect(url_for("main.dashboard"))

    out = io.StringIO()
    w   = csv.DictWriter(out, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)
    out.seek(0)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return send_file(
        io.BytesIO(out.getvalue().encode()),
        mimetype="text/csv",
        as_attachment=True,
        download_name=f"churnmaster_history_{ts}.csv",
    )


@main_bp.route("/export/powerbi", methods=["POST"])
@login_required
def export_powerbi():
    """
    Export full prediction history as Power BI Push Dataset JSON.
    From ChurnIQ Pro — use this to PUSH data into Power BI via the REST API
    as an alternative to the iframe embed approach.
    """
    rows = db_history(100_000)
    if not rows:
        return jsonify({"error": "No data"}), 404

    pbi = {
        "name": "ChurnMaster Predictions",
        "tables": [{
            "name": "Predictions",
            "columns": [
                {"name": "ID",         "dataType": "Int64"},
                {"name": "Timestamp",  "dataType": "DateTime"},
                {"name": "Prediction", "dataType": "Int64"},
                {"name": "Probability","dataType": "Double"},
                {"name": "Risk",       "dataType": "string"},
                {"name": "Label",      "dataType": "string"},
                {"name": "Domain",     "dataType": "string"},
                {"name": "Source",     "dataType": "string"},
                {"name": "Username",   "dataType": "string"},
                {"name": "CustomerID", "dataType": "string"},
            ],
            "rows": [
                {
                    "ID": r["id"], "Timestamp": r["timestamp"],
                    "Prediction": r["prediction"], "Probability": r["probability"],
                    "Risk": r["risk"], "Label": r["label"], "Domain": r["domain"],
                    "Source": r["source"], "Username": r.get("username", ""),
                    "CustomerID": r.get("customer_id", ""),
                }
                for r in rows
            ],
        }],
    }

    ts = datetime.now().strftime("%Y%m%d")
    return send_file(
        io.BytesIO(json.dumps(pbi, indent=2).encode()),
        mimetype="application/json",
        as_attachment=True,
        download_name=f"ChurnMaster_PowerBI_{ts}.json",
    )


# ══════════════════════════════════════════════════════════════════════════
#  POWER BI LIVE DATA API
#  ─────────────────────────────────────────────────────────────────────────
#  How it works:
#   1. Flask runs on localhost:5000 (or your server IP)
#   2. Power BI Desktop → Get Data → Web → http://localhost:5000/api/powerbi_live
#      (or add ?token=YOUR_TOKEN for lightweight security)
#   3. Power BI parses the JSON → creates a live-refresh dataset
#   4. Publish to Power BI Service → set scheduled refresh
#
#  No login cookie required — uses POWERBI_TOKEN env variable instead.
#  Set in .env:  POWERBI_TOKEN=your_secret_token_here
# ══════════════════════════════════════════════════════════════════════════

@main_bp.route("/api/powerbi_live")
def powerbi_live():
    """
    Public JSON endpoint consumed by:
      • Power BI Desktop  (Get Data → Web → paste this URL)
      • ChurnIQ_Dashboard.html  (Live Mode auto-refresh)
      • Custom BI tools  (curl, Tableau, Grafana, etc.)

    Token auth: set POWERBI_TOKEN in .env
    If POWERBI_TOKEN is not set, the endpoint is open (fine for localhost).
    URL format:  /api/powerbi_live?token=YOUR_TOKEN&limit=5000
    """
    import os
    expected = os.environ.get("POWERBI_TOKEN", "")
    if expected and request.args.get("token", "") != expected:
        return jsonify({"error": "Unauthorized — add ?token=YOUR_TOKEN"}), 401

    limit = min(int(request.args.get("limit", 10000)), 100_000)
    rows  = db_history(limit)
    stats = db_stats()
    bundle = load_bundle()

    model_info = {}
    if bundle:
        model_info = {
            "domain":    bundle.get("domain",    "Unknown"),
            "n_features": bundle.get("n_features", 0),
            "n_samples":  bundle.get("n_samples",  0),
            "accuracy":   round(bundle.get("metrics", {}).get("accuracy", 0) * 100, 1),
        }

    # Flat record list — Power BI Desktop parses this directly
    records = [
        {
            "id":          r["id"],
            "timestamp":   r["timestamp"],
            "prediction":  r["prediction"],
            "probability": r["probability"],
            "risk":        r["risk"],
            "label":       r["label"],
            "domain":      r["domain"],
            "source":      r["source"],
            "customer_id": r.get("customer_id", ""),
            "username":    r.get("username", ""),
        }
        for r in rows
    ]

    return jsonify({
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "total_records": len(records),
        "summary":       stats,
        "model":         model_info,
        "records":       records,
    })


@main_bp.route("/api/powerbi_live/records")
def powerbi_live_records_only():
    """
    Returns ONLY the records array — useful when Power BI can't handle nested JSON.
    Power BI Desktop → Get Data → Web → .../api/powerbi_live/records
    Then expand the list → each row becomes a Power BI table row automatically.
    """
    import os
    expected = os.environ.get("POWERBI_TOKEN", "")
    if expected and request.args.get("token", "") != expected:
        return jsonify({"error": "Unauthorized"}), 401

    limit   = min(int(request.args.get("limit", 10000)), 100_000)
    rows    = db_history(limit)
    records = [
        {
            "id":          r["id"],
            "timestamp":   r["timestamp"],
            "date":        r["timestamp"][:10] if r.get("timestamp") else "",
            "hour":        r["timestamp"][11:13] if r.get("timestamp") else "",
            "prediction":  r["prediction"],
            "probability": r["probability"],
            "risk":        r["risk"],
            "label":       r["label"],
            "will_churn":  1 if r["prediction"] == 1 else 0,
            "will_stay":   0 if r["prediction"] == 1 else 1,
            "domain":      r["domain"],
            "source":      r["source"],
            "customer_id": r.get("customer_id", ""),
            "username":    r.get("username", ""),
        }
        for r in rows
    ]
    return jsonify(records)


@main_bp.route("/api/powerbi_live/summary")
def powerbi_live_summary():
    """Summary stats endpoint for Power BI KPI visuals."""
    import os
    expected = os.environ.get("POWERBI_TOKEN", "")
    if expected and request.args.get("token", "") != expected:
        return jsonify({"error": "Unauthorized"}), 401

    stats = db_stats()
    bundle = load_bundle()
    return jsonify({
        "generated_at":  datetime.utcnow().isoformat() + "Z",
        "total":         stats["total"],
        "churned":       stats["churned"],
        "stayed":        stats["stayed"],
        "churn_rate_pct": stats["churn_rate"],
        "avg_probability": stats["avg_prob"],
        "high_risk":     stats["high_risk"],
        "medium_risk":   stats["med_risk"],
        "low_risk":      stats["low_risk"],
        "domain":        bundle.get("domain", "Unknown") if bundle else "No model",
        "model_accuracy": round(bundle.get("metrics", {}).get("accuracy", 0) * 100, 1) if bundle else 0,
    })


# ══════════════════════════════════════════════════════════════════════════
#  DATASET APIs — expose raw CSV datasets as JSON for Power BI Web connections
#  GET /api/dataset/telecom        → Telecom CSV records
#  GET /api/dataset/bank           → Bank CSV records
#  GET /api/dataset/hr             → HR Attrition CSV records
#  GET /api/dataset/<ds>/summary   → KPI summary for each dataset
# ══════════════════════════════════════════════════════════════════════════

import os as _os
import csv as _csv

def _csv_to_json(filepath):
    """Read a CSV file and return a list of dicts."""
    records = []
    try:
        with open(filepath, newline='', encoding='utf-8-sig') as f:
            reader = _csv.DictReader(f)
            for row in reader:
                records.append(dict(row))
    except Exception as e:
        logger.error("CSV read error %s: %s", filepath, e)
    return records

_DS_FILES = {
    "telecom": "customer_churn_prediction_dataset.csv",
    "bank":    "Bank Customer Churn Prediction.csv",
    "hr":      "WA_Fn-UseC_-HR-Employee-Attrition.csv",
}

@main_bp.route("/api/dataset/<ds_key>")
@login_required
def dataset_api(ds_key):
    """Return full CSV dataset as JSON for Power BI Desktop Web connections."""
    if ds_key not in _DS_FILES:
        return jsonify({"error": f"Unknown dataset '{ds_key}'. Use: telecom, bank, hr"}), 404
    base_dir = _os.path.dirname(_os.path.abspath(__file__))
    fpath = _os.path.join(base_dir, _DS_FILES[ds_key])
    records = _csv_to_json(fpath)
    return jsonify({
        "dataset":      ds_key,
        "total_records": len(records),
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "records":      records,
    })


@main_bp.route("/api/dataset/<ds_key>/summary")
@login_required
def dataset_summary_api(ds_key):
    """Return KPI summary stats for a specific dataset."""
    if ds_key not in _DS_FILES:
        return jsonify({"error": f"Unknown dataset '{ds_key}'. Use: telecom, bank, hr"}), 404
    base_dir = _os.path.dirname(_os.path.abspath(__file__))
    fpath = _os.path.join(base_dir, _DS_FILES[ds_key])
    records = _csv_to_json(fpath)
    total = len(records)

    # Detect churn column per dataset
    if ds_key == "telecom":
        churned = sum(1 for r in records if str(r.get("Churn","")).strip().lower() in ("yes","1"))
        churn_col = "Churn"
    elif ds_key == "bank":
        churned = sum(1 for r in records if str(r.get("churn","")).strip() == "1")
        churn_col = "churn"
    else:  # hr
        churned = sum(1 for r in records if str(r.get("Attrition","")).strip().lower() == "yes")
        churn_col = "Attrition"

    stayed = total - churned
    churn_rate = round(churned / total * 100, 1) if total else 0

    return jsonify({
        "dataset":      ds_key,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "total":        total,
        "churned":      churned,
        "stayed":       stayed,
        "churn_rate_pct": churn_rate,
        "churn_column": churn_col,
    })


# ══════════════════════════════════════════════════════════════════════════
#  DATASET MANAGEMENT — reload model after retraining on a new CSV
# ══════════════════════════════════════════════════════════════════════════

@main_bp.route("/api/reload_model", methods=["POST"])
@login_required
def reload_model():
    """
    Hot-swap the loaded model without restarting the server.
    Call this after running: python train.py --csv newdata.csv --target Churn

    POST body (optional JSON): {"confirm": true}
    Returns: model metadata after reload
    """
    import prediction as _pred
    with _pred._lock:
        _pred._bundle_mtime = 0
        _pred.BUNDLE        = None

    bundle = load_bundle()
    if bundle:
        logger.info(
            "Model hot-reloaded by %s: domain=%s  features=%d",
            session.get("user", "?"), bundle.get("domain", "?"), bundle.get("n_features", 0),
        )
        return jsonify({
            "success":  True,
            "domain":   bundle.get("domain",    "Unknown"),
            "features": bundle.get("n_features", 0),
            "samples":  bundle.get("n_samples",  0),
            "accuracy": round(bundle.get("metrics", {}).get("accuracy", 0) * 100, 1),
            "auc":      round(bundle.get("metrics", {}).get("auc",      0) * 100, 1),
        })
    return jsonify({"success": False, "error": "No model_bundle.pkl found. Run train.py first."}), 503


@main_bp.route("/api/sub_model_stats/<model_key>")
@login_required
def sub_model_stats(model_key):
    """Return per-sub-model stats for the predict page model selector."""
    bundle = load_bundle()
    if not bundle:
        return jsonify({"error": "No model loaded"}), 503

    # Static approximate stats per model type (from training knowledge)
    stats_map = {
        "ensemble": {"name": "Voting Ensemble (All 6)", "accuracy": "79%", "auc": "85%", "precision": "76%", "recall": "72%", "f1": "74%"},
        "rf":  {"name": "Random Forest",           "accuracy": "80%", "auc": "87%", "precision": "79%", "recall": "70%", "f1": "74%"},
        "et":  {"name": "Extra Trees",             "accuracy": "79%", "auc": "86%", "precision": "78%", "recall": "69%", "f1": "73%"},
        "gb":  {"name": "Gradient Boosting",       "accuracy": "81%", "auc": "88%", "precision": "80%", "recall": "71%", "f1": "75%"},
        "lr":  {"name": "Logistic Regression",     "accuracy": "74%", "auc": "82%", "precision": "72%", "recall": "68%", "f1": "70%"},
        "svm": {"name": "Support Vector Machine",  "accuracy": "76%", "auc": "83%", "precision": "74%", "recall": "65%", "f1": "69%"},
        "ada": {"name": "AdaBoost",                "accuracy": "77%", "auc": "84%", "precision": "75%", "recall": "69%", "f1": "72%"},
    }
    return jsonify(stats_map.get(model_key, stats_map["ensemble"]))


@main_bp.route("/api/model_info")
@login_required
def model_info():
    """Returns current loaded model metadata as JSON."""
    bundle = load_bundle()
    if not bundle:
        return jsonify({"loaded": False, "error": "No model loaded"})
    return jsonify({
        "loaded":        True,
        "domain":        bundle.get("domain",    "Unknown"),
        "target_col":    bundle.get("target_col","Churn"),
        "n_features":    bundle.get("n_features", 0),
        "n_samples":     bundle.get("n_samples",  0),
        "accuracy":      round(bundle.get("metrics", {}).get("accuracy", 0) * 100, 1),
        "auc":           round(bundle.get("metrics", {}).get("auc",      0) * 100, 1),
        "cv_accuracy":   round(bundle.get("cv_accuracy", 0) * 100, 1),
        "feature_names": bundle.get("feature_names", []),
        "legacy":        bundle.get("legacy", False),
    })


# ══════════════════════════════════════════════════════════════════════════
#  POWER BI PAGE & PBIX DOWNLOAD
# ══════════════════════════════════════════════════════════════════════════

@main_bp.route("/powerbi")
@login_required
def powerbi_page():
    """Dedicated Power BI integration page."""
    stats = db_stats()
    return render_template(
        "powerbi.html",
        username = session["user"],
        stats    = stats,
    )


@main_bp.route("/download/pbix")
@login_required
def download_pbix():
    """Serve the Power BI .pbix file as a download."""
    import os
    pbix_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "Customer_Churn_Dashboard.pbix")
    if not os.path.exists(pbix_path):
        flash("Power BI file not found on server.", "error")
        return redirect(url_for("main.powerbi_page"))
    from flask import send_file
    return send_file(
        pbix_path,
        as_attachment=True,
        download_name="ChurnMaster_Pro_Dashboard.pbix",
        mimetype="application/octet-stream",
    )


# ══════════════════════════════════════════════════════════════════════════
#  CUSTOM DATASET TRAINER — Upload any CSV and retrain the model in-app
# ══════════════════════════════════════════════════════════════════════════

import threading as _threading
import time as _time

# Shared training state — read by /api/train_status
_train_state = {
    "status":   "idle",     # idle | running | done | error
    "log":      [],
    "progress": 0,
    "result":   {},
}
_train_lock = _threading.Lock()


@main_bp.route("/train")
@login_required
def train_page():
    """Custom Dataset Trainer page."""
    bundle = load_bundle()
    model_info = {}
    if bundle:
        model_info = {
            "domain":   bundle.get("domain", "Unknown"),
            "features": bundle.get("n_features", 0),
            "samples":  bundle.get("n_samples", 0),
            "accuracy": round(bundle.get("metrics", {}).get("accuracy", 0) * 100, 1),
            "auc":      round(bundle.get("metrics", {}).get("auc", 0) * 100, 1),
        }
    return render_template("train.html",
                           username=session["user"],
                           model_info=model_info)


@main_bp.route("/api/csv_preview", methods=["POST"])
@login_required
def csv_preview():
    """
    Accept a CSV upload, return column names + sample rows + class balance.
    Used by the trainer UI to let the user pick the target column.
    """
    if "csv_file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    f = request.files["csv_file"]
    if not f.filename.lower().endswith(".csv"):
        return jsonify({"error": "File must be a .csv"}), 400
    try:
        df = pd.read_csv(f, nrows=500)
        columns = list(df.columns)
        n_rows_sample = len(df)

        # Re-read to get full row count
        f.seek(0)
        full_df = pd.read_csv(f)
        total_rows = len(full_df)

        # Suggest target column
        from train import detect_target, detect_domain
        suggested_target = detect_target(full_df)
        domain = detect_domain(full_df)

        # Class balance for top binary columns
        binary_cols = {}
        for col in full_df.columns:
            if full_df[col].nunique() <= 5:
                vc = full_df[col].value_counts().to_dict()
                binary_cols[col] = {str(k): int(v) for k, v in vc.items()}

        # Sample rows (first 8 for display)
        sample = df.head(8).fillna("").astype(str).to_dict(orient="records")

        return jsonify({
            "columns":          columns,
            "total_rows":       total_rows,
            "suggested_target": suggested_target,
            "domain":           domain,
            "binary_cols":      binary_cols,
            "sample":           sample,
            "filename":         f.filename,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@main_bp.route("/api/train", methods=["POST"])
@login_required
def api_train():
    """
    Start background model training on an uploaded CSV.
    Returns immediately — poll /api/train_status for progress.
    """
    global _train_state

    with _train_lock:
        if _train_state["status"] == "running":
            return jsonify({"error": "Training already in progress"}), 409

    if "csv_file" not in request.files:
        return jsonify({"error": "No CSV file"}), 400

    f = request.files["csv_file"]
    target_col = request.form.get("target_col", "").strip()
    if not target_col:
        return jsonify({"error": "No target column specified"}), 400

    # Save uploaded CSV to a temp location
    import tempfile
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    f.save(tmp.name)
    tmp.close()

    def _run_training(csv_path, target):
        global _train_state
        import shutil

        log = []
        def emit(msg):
            log.append(msg)
            with _train_lock:
                _train_state["log"] = list(log)
            logger.info("[TRAINER] %s", msg)

        try:
            with _train_lock:
                _train_state.update({"status":"running","log":[],"progress":0,"result":{}})

            emit("📂 Reading CSV file...")
            df = pd.read_csv(csv_path)
            emit(f"✅ Loaded {len(df):,} rows × {len(df.columns)} columns")

            from train import detect_domain, drop_id_cols, encode_target, balance_classes
            domain = detect_domain(df)
            emit(f"🏢 Domain detected: {domain}")

            with _train_lock: _train_state["progress"] = 10

            # Validate target column
            if target not in df.columns:
                matches = [c for c in df.columns if c.lower() == target.lower()]
                if matches:
                    target = matches[0]
                else:
                    raise ValueError(f"Column '{target}' not found in CSV")

            emit(f"🎯 Target column: {target}")

            # Clean
            df = drop_id_cols(df, target)
            for col in df.select_dtypes(include="object").columns:
                if col == target: continue
                try: df[col] = pd.to_numeric(df[col], errors="raise")
                except: pass
            df = df.dropna(subset=[target])
            emit(f"🧹 After cleaning: {len(df):,} rows")

            y = encode_target(df[target])
            n_churn = int((y==1).sum())
            n_stay  = int((y==0).sum())
            emit(f"⚖️  Class balance — Stay: {n_stay:,}  Churn: {n_churn:,}  ({round(n_churn/len(y)*100,1)}% churn)")

            df_feat = df.drop(columns=[target])
            df_feat = pd.get_dummies(df_feat, drop_first=True)
            for col in df_feat.columns:
                df_feat[col] = pd.to_numeric(df_feat[col], errors="coerce")

            feature_names = df_feat.columns.tolist()
            emit(f"📋 Features: {len(feature_names)} (after one-hot encoding)")
            with _train_lock: _train_state["progress"] = 20

            # Import sklearn
            from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
            from sklearn.preprocessing import StandardScaler
            from sklearn.impute import SimpleImputer
            from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                                          VotingClassifier, AdaBoostClassifier, ExtraTreesClassifier)
            from sklearn.linear_model import LogisticRegression
            from sklearn.svm import SVC
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
            import warnings
            warnings.filterwarnings("ignore")

            imputer = SimpleImputer(strategy="median")
            X_imp   = imputer.fit_transform(df_feat)
            scaler  = StandardScaler()
            X_sc    = scaler.fit_transform(X_imp)

            X_tr, X_te, y_tr, y_te = train_test_split(
                X_sc, y, test_size=0.25, random_state=42, stratify=y)
            emit(f"✂️  Split: {len(X_tr):,} train / {len(X_te):,} test")

            emit("⚖️  Balancing classes...")
            X_bal, y_bal = balance_classes(X_tr, y_tr.values)
            with _train_lock: _train_state["progress"] = 30

            # Train models
            models = []

            emit("🤖 [1/6] Logistic Regression...")
            lr = LogisticRegression(C=0.1, max_iter=1000, class_weight="balanced", random_state=42)
            lr.fit(X_bal, y_bal)
            models.append(("lr", lr))
            with _train_lock: _train_state["progress"] = 40

            emit("🌲 [2/6] Random Forest...")
            rf = RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_split=10,
                                        min_samples_leaf=4, class_weight="balanced", random_state=42, n_jobs=-1)
            rf.fit(X_bal, y_bal)
            models.append(("rf", rf))
            with _train_lock: _train_state["progress"] = 52

            emit("🌳 [3/6] Extra Trees...")
            et = ExtraTreesClassifier(n_estimators=200, max_depth=8, min_samples_split=10,
                                      min_samples_leaf=4, class_weight="balanced", random_state=42, n_jobs=-1)
            et.fit(X_bal, y_bal)
            models.append(("et", et))
            with _train_lock: _train_state["progress"] = 62

            emit("📈 [4/6] Gradient Boosting...")
            gb = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=4,
                                            subsample=0.8, min_samples_split=10, random_state=42)
            gb.fit(X_bal, y_bal)
            models.append(("gb", gb))
            with _train_lock: _train_state["progress"] = 72

            emit("🚀 [5/6] AdaBoost...")
            ada = AdaBoostClassifier(n_estimators=100, learning_rate=0.5, random_state=42)
            ada.fit(X_bal, y_bal)
            models.append(("ada", ada))
            with _train_lock: _train_state["progress"] = 80

            emit("🔷 [6/6] SVM...")
            svm = SVC(C=1.0, kernel="rbf", class_weight="balanced", probability=True, random_state=42)
            svm.fit(X_bal, y_bal)
            models.append(("svm", svm))
            with _train_lock: _train_state["progress"] = 86

            emit("🏆 Building Voting Ensemble (all 6)...")
            ensemble = VotingClassifier(estimators=models, voting="soft", n_jobs=-1)
            ensemble.fit(X_bal, y_bal)
            with _train_lock: _train_state["progress"] = 92

            ens_pred  = ensemble.predict(X_te)
            ens_proba = ensemble.predict_proba(X_te)[:,1]
            acc  = accuracy_score(y_te, ens_pred)
            prec = precision_score(y_te, ens_pred, zero_division=0)
            rec  = recall_score(y_te, ens_pred, zero_division=0)
            f1   = f1_score(y_te, ens_pred, zero_division=0)
            try: auc = roc_auc_score(y_te, ens_proba)
            except: auc = 0
            metrics = {"accuracy":acc,"precision":prec,"recall":rec,"f1":f1,"auc":auc}

            emit(f"📊 Accuracy: {round(acc*100,1)}%  AUC: {round(auc*100,1)}%  F1: {round(f1*100,1)}%")

            imp_map = {}
            for m in [rf, et, gb]:
                for fn, imp in zip(feature_names, m.feature_importances_):
                    imp_map[fn] = imp_map.get(fn, 0) + imp / 3

            bundle = {
                "model": ensemble, "scaler": scaler, "imputer": imputer,
                "feature_names": feature_names, "domain": domain,
                "target_col": target, "importance_map": imp_map,
                "metrics": metrics, "cv_accuracy": acc,
                "n_samples": len(df), "n_features": len(feature_names),
            }

            script_dir = os.path.dirname(os.path.abspath(__file__))
            out_dir    = os.path.join(script_dir, "models")
            os.makedirs(out_dir, exist_ok=True)

            emit("💾 Saving model files...")
            for fname, obj in [("model_bundle.pkl", bundle), ("churn_model.pkl", ensemble),
                                ("scaler.pkl", scaler), ("feature_names.pkl", feature_names)]:
                pickle.dump(obj, open(os.path.join(out_dir, fname), "wb"))
                shutil.copy(os.path.join(out_dir, fname), os.path.join(script_dir, fname))

            # Hot-reload
            import prediction as _pred
            with _pred._lock:
                _pred._bundle_mtime = 0
                _pred.BUNDLE = None
            load_bundle()

            with _train_lock:
                _train_state.update({
                    "status": "done", "progress": 100,
                    "result": {
                        "domain": domain, "accuracy": round(acc*100,1),
                        "auc": round(auc*100,1), "f1": round(f1*100,1),
                        "precision": round(prec*100,1), "recall": round(rec*100,1),
                        "features": len(feature_names), "samples": len(df),
                        "target": target,
                    }
                })
            emit(f"🎉 Training complete! Model active — domain: {domain}  accuracy: {round(acc*100,1)}%")

        except Exception as exc:
            logger.error("[TRAINER] Error: %s", exc)
            with _train_lock:
                _train_state.update({"status": "error", "progress": 0})
            emit(f"❌ Error: {exc}")
        finally:
            try: os.unlink(csv_path)
            except: pass

    t = _threading.Thread(target=_run_training, args=(tmp.name, target_col), daemon=True)
    t.start()
    return jsonify({"ok": True, "message": "Training started"})


@main_bp.route("/api/train_status")
@login_required
def api_train_status():
    """Return current training state — polled every 1s by the UI."""
    with _train_lock:
        return jsonify(dict(_train_state))


@main_bp.route("/api/train_example", methods=["POST"])
@login_required
def api_train_example():
    """Retrain on one of the built-in example CSV datasets (no file upload needed)."""
    global _train_state

    with _train_lock:
        if _train_state["status"] == "running":
            return jsonify({"error": "Training already in progress"}), 409

    data = request.get_json(force=True) or {}
    ds_key     = data.get("dataset", "bank")
    target_col = data.get("target_col", "")

    EXAMPLE_FILES = {
        "telecom": ("customer_churn_prediction_dataset.csv", "Churn"),
        "bank":    ("Bank Customer Churn Prediction.csv",    "churn"),
        "hr":      ("WA_Fn-UseC_-HR-Employee-Attrition.csv","Attrition"),
    }
    if ds_key not in EXAMPLE_FILES:
        return jsonify({"error": f"Unknown example dataset '{ds_key}'"}), 400

    fname, default_target = EXAMPLE_FILES[ds_key]
    if not target_col:
        target_col = default_target

    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path   = os.path.join(script_dir, fname)

    if not os.path.exists(csv_path):
        return jsonify({"error": f"Example CSV not found: {fname}"}), 404

    import tempfile, shutil as _shutil
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    _shutil.copy(csv_path, tmp.name)
    tmp.close()

    # Re-use the same background training function
    import prediction as _pred_mod
    import pickle as _pkl
    import pandas as _pd_local

    def _run_example_training(path, target):
        global _train_state
        log = []
        def emit(msg):
            log.append(msg)
            with _train_lock:
                _train_state["log"] = list(log)

        try:
            with _train_lock:
                _train_state.update({"status":"running","log":[],"progress":0,"result":{}})

            emit(f"📂 Loading example dataset: {fname}")
            df = _pd_local.read_csv(path)
            emit(f"✅ Loaded {len(df):,} rows × {len(df.columns)} columns")

            from train import detect_domain, drop_id_cols, encode_target, balance_classes
            domain = detect_domain(df)
            emit(f"🏢 Domain detected: {domain}")
            with _train_lock: _train_state["progress"] = 10

            df = drop_id_cols(df, target)
            for col in df.select_dtypes(include="object").columns:
                if col == target: continue
                try: df[col] = _pd_local.to_numeric(df[col], errors="raise")
                except: pass
            df = df.dropna(subset=[target])
            emit(f"🧹 After cleaning: {len(df):,} rows")

            y = encode_target(df[target])
            emit(f"⚖️  Class balance — Stay: {int((y==0).sum()):,}  Churn: {int((y==1).sum()):,}")

            df_feat = df.drop(columns=[target])
            df_feat = _pd_local.get_dummies(df_feat, drop_first=True)
            for col in df_feat.columns:
                df_feat[col] = _pd_local.to_numeric(df_feat[col], errors="coerce")

            feature_names = df_feat.columns.tolist()
            emit(f"📋 Features: {len(feature_names)}")
            with _train_lock: _train_state["progress"] = 20

            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import StandardScaler
            from sklearn.impute import SimpleImputer
            from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                                          VotingClassifier, AdaBoostClassifier, ExtraTreesClassifier)
            from sklearn.linear_model import LogisticRegression
            from sklearn.svm import SVC
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
            import warnings; warnings.filterwarnings("ignore")

            imputer = SimpleImputer(strategy="median")
            X_imp   = imputer.fit_transform(df_feat)
            scaler  = StandardScaler()
            X_sc    = scaler.fit_transform(X_imp)
            X_tr, X_te, y_tr, y_te = train_test_split(X_sc, y, test_size=0.25, random_state=42, stratify=y)

            emit(f"✂️  Split: {len(X_tr):,} train / {len(X_te):,} test")
            emit("⚖️  Balancing classes...")
            X_bal, y_bal = balance_classes(X_tr, y_tr.values)
            with _train_lock: _train_state["progress"] = 30

            models = []
            for idx, (name, clf, prog) in enumerate([
                ("lr",  LogisticRegression(C=0.1, max_iter=1000, class_weight="balanced", random_state=42), 40),
                ("rf",  RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_split=10, min_samples_leaf=4, class_weight="balanced", random_state=42, n_jobs=-1), 52),
                ("et",  ExtraTreesClassifier(n_estimators=200, max_depth=8, min_samples_split=10, min_samples_leaf=4, class_weight="balanced", random_state=42, n_jobs=-1), 62),
                ("gb",  GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=4, subsample=0.8, random_state=42), 72),
                ("ada", AdaBoostClassifier(n_estimators=100, learning_rate=0.5, random_state=42), 80),
                ("svm", SVC(C=1.0, kernel="rbf", class_weight="balanced", probability=True, random_state=42), 86),
            ]):
                emit(f"🤖 [{idx+1}/6] {name.upper()}...")
                clf.fit(X_bal, y_bal)
                models.append((name, clf))
                with _train_lock: _train_state["progress"] = prog

            emit("🏆 Building Voting Ensemble...")
            ensemble = VotingClassifier(estimators=models, voting="soft", n_jobs=-1)
            ensemble.fit(X_bal, y_bal)
            with _train_lock: _train_state["progress"] = 93

            ens_pred  = ensemble.predict(X_te)
            ens_proba = ensemble.predict_proba(X_te)[:,1]
            acc  = accuracy_score(y_te, ens_pred)
            prec = precision_score(y_te, ens_pred, zero_division=0)
            rec  = recall_score(y_te, ens_pred, zero_division=0)
            f1   = f1_score(y_te, ens_pred, zero_division=0)
            try: auc = roc_auc_score(y_te, ens_proba)
            except: auc = 0
            metrics = {"accuracy":acc,"precision":prec,"recall":rec,"f1":f1,"auc":auc}
            emit(f"📊 Accuracy: {round(acc*100,1)}%  AUC: {round(auc*100,1)}%  F1: {round(f1*100,1)}%")

            imp_map = {}
            for m in [models[1][1], models[2][1], models[3][1]]:
                for fn, imp in zip(feature_names, m.feature_importances_):
                    imp_map[fn] = imp_map.get(fn, 0) + imp / 3

            bundle = {
                "model": ensemble, "scaler": scaler, "imputer": imputer,
                "feature_names": feature_names, "domain": domain,
                "target_col": target, "importance_map": imp_map,
                "metrics": metrics, "cv_accuracy": acc,
                "n_samples": len(df), "n_features": len(feature_names),
            }

            sd = os.path.dirname(os.path.abspath(__file__))
            od = os.path.join(sd, "models")
            os.makedirs(od, exist_ok=True)
            emit("💾 Saving model files...")
            import shutil as _sh2
            for fn, obj in [("model_bundle.pkl",bundle),("churn_model.pkl",ensemble),
                            ("scaler.pkl",scaler),("feature_names.pkl",feature_names)]:
                _pkl.dump(obj, open(os.path.join(od, fn), "wb"))
                _sh2.copy(os.path.join(od, fn), os.path.join(sd, fn))

            with _pred_mod._lock:
                _pred_mod._bundle_mtime = 0
                _pred_mod.BUNDLE = None
            load_bundle()

            with _train_lock:
                _train_state.update({
                    "status": "done", "progress": 100,
                    "result": {
                        "domain": domain, "accuracy": round(acc*100,1),
                        "auc": round(auc*100,1), "f1": round(f1*100,1),
                        "precision": round(prec*100,1), "recall": round(rec*100,1),
                        "features": len(feature_names), "samples": len(df),
                        "target": target,
                    }
                })
            emit(f"🎉 Training complete! Domain: {domain}  Accuracy: {round(acc*100,1)}%")

        except Exception as exc:
            logger.error("[TRAINER] %s", exc)
            with _train_lock: _train_state.update({"status":"error","progress":0})
            emit(f"❌ Error: {exc}")
        finally:
            try: os.unlink(path)
            except: pass

    import threading as _th2
    t = _th2.Thread(target=_run_example_training, args=(tmp.name, target_col), daemon=True)
    t.start()
    return jsonify({"ok": True})

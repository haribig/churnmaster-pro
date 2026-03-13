"""
ChurnMaster Pro — config.py
============================
Single source of truth for all app-wide constants, paths, thresholds,
and dataset metadata. Combines config from both source projects.

Both databases and all paths are defined here so nothing is ever
hardcoded inside route handlers.
"""

import os
import secrets
import logging

# ── Load .env (optional — falls back to OS environment variables) ──────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── Paths ─────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR   = os.path.join(BASE_DIR, "models")

# On Vercel the project root is read-only; databases must live in /tmp.
# Locally (or any writable environment) they stay next to the project files.
_IS_VERCEL = os.environ.get("VERCEL") == "1"
_DB_DIR    = "/tmp" if _IS_VERCEL else BASE_DIR

HISTORY_DIR   = os.path.join(_DB_DIR, "predictions_history")

# Two separate databases (preserves both original project files)
USERS_DB_PATH = os.path.join(_DB_DIR, "users.db")       # user accounts
PRED_DB_PATH  = os.path.join(_DB_DIR, "churnmaster.db") # prediction history

os.makedirs(HISTORY_DIR, exist_ok=True)
os.makedirs(MODELS_DIR,  exist_ok=True)

# ── Security ──────────────────────────────────────────────────────────────
_env_key = os.environ.get("SECRET_KEY", "") or os.environ.get("CHURNIQ_SECRET_KEY", "")
if not _env_key:
    logging.getLogger(__name__).warning(
        "SECRET_KEY not set in environment — using a random key per session. "
        "Set SECRET_KEY in your .env file to persist sessions across restarts."
    )
    _env_key = secrets.token_hex(32)

SECRET_KEY = _env_key

# ── ML risk thresholds (from ChurnIQ Pro) ────────────────────────────────
RISK_LOW_THRESHOLD  = 0.30   # below this  → LOW risk
RISK_HIGH_THRESHOLD = 0.60   # above this  → HIGH risk
# between the two → MEDIUM risk

# ── Loyalty grade thresholds ──────────────────────────────────────────────
GRADE_A_MIN = 80
GRADE_B_MIN = 65
GRADE_C_MIN = 50
GRADE_D_MIN = 35

# ── Slider max values by feature keyword ─────────────────────────────────
SLIDER_MAX = {
    "month":  72,
    "year":   50,
    "tenure": 72,
    "day":    365,
}

# ── Domain-specific churn reason library (from ChurnIQ Pro) ──────────────
# Used by the reason engine to explain WHY a customer is likely to churn.
# Each key maps a feature name fragment → (icon, short_title, detailed_explanation)
DOMAIN_REASONS = {
    "Telecom": {
        "tenure":          ("📅", "Short Customer Tenure",
                            "Customers in their first year are 3× more likely to churn. "
                            "No deep habit or loyalty formed yet. Critical window for retention."),
        "MonthlyCharges":  ("💸", "High Monthly Bill",
                            "Monthly charge exceeds comfort zone. Customer is likely comparing "
                            "competitors and calculating if they're getting value for money."),
        "TotalCharges":    ("💰", "Low Lifetime Value",
                            "Customer hasn't accumulated significant spend — low switching cost "
                            "means less hesitation to leave."),
        "Contract":        ("📋", "No Long-Term Contract",
                            "Month-to-month customers have zero financial penalty to cancel. "
                            "Any friction or better offer from a competitor will trigger churn."),
        "InternetService": ("⚡", "High-Speed Internet User",
                            "Fiber optic customers receive aggressive competitor targeting. "
                            "They are price-sensitive and have the most options available."),
        "OnlineSecurity":  ("🔓", "No Security Add-on",
                            "Without security features, customers feel less invested in the "
                            "service ecosystem. Less stickiness means easier to leave."),
        "TechSupport":     ("🛠️", "No Tech Support",
                            "Unresolved technical issues are the #1 cause of sudden churn. "
                            "Without support, frustration silently builds until they leave."),
        "PaymentMethod":   ("💳", "Manual Payment Method",
                            "Electronic check users churn 30% more than auto-pay customers. "
                            "Manual payment creates monthly friction and churn opportunity."),
        "SeniorCitizen":   ("👴", "Senior Customer Segment",
                            "May face challenges with self-service digital platforms. "
                            "Needs extra support touchpoints or simplified service options."),
        "Partner":         ("👤", "No Partner / Single Account",
                            "Single-user accounts have lower switching cost — no family plan "
                            "or shared account to complicate cancellation."),
        "MultipleLines":   ("📱", "Limited Service Bundle",
                            "Customers with fewer service lines are less embedded in the "
                            "ecosystem and easier to poach by competitors."),
        "StreamingTV":     ("📺", "Streaming Service User",
                            "Streaming customers actively compare alternatives and churn if "
                            "pricing or quality falls short of expectations."),
    },
    "General": {
        "tenure":         ("📅", "Short Relationship Duration",
                           "Short tenure customers haven't built enough loyalty or dependency "
                           "to resist competitor offers."),
        "MonthlyCharges": ("💸", "High Cost Sensitivity",
                           "Price is a primary driver in this segment. Customer is likely "
                           "calculating cost-vs-value regularly."),
    },
}

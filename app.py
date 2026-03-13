"""
ChurnMaster Pro — app.py
=========================
Combined professional upgrade of:
  • customer_churn   (original project with telecom ML model + basic auth)
  • ChurnIQ Pro      (advanced ML engine, analytics DB, bulk predict, Power BI export)

What this file does (and ONLY this):
  • Creates the Flask app
  • Loads config / secret key
  • Initialises both databases (users + predictions)
  • Pre-loads the ML model bundle into memory
  • Registers the routes Blueprint

All URL routes live in routes.py — nothing routes-related here.
All ML logic lives in prediction.py.
All auth logic lives in services/auth_service.py.
All chart logic lives in services/chart_service.py.
All DB helpers live in db.py.

Run locally:
  python app.py

Production (gunicorn / waitress):
  gunicorn -w 4 "app:create_app()"          # Linux / Mac
  waitress-serve --port=5000 app:app         # Windows
"""

import os
import logging
from datetime import timedelta

from flask import Flask

# ── Logging ───────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("churnmaster")


# ══════════════════════════════════════════════════════════════════════════
#  APPLICATION FACTORY
# ══════════════════════════════════════════════════════════════════════════

def create_app() -> Flask:
    """
    Factory function — creates, configures, and returns the Flask app.
    Using a factory makes the app independently testable (no global state).
    """
    app = Flask(__name__)

    # ── Import config AFTER the module path is set ────────────────────────
    from config import SECRET_KEY
    app.secret_key = SECRET_KEY

    # Sessions are permanent (see login route) and expire after 8 hours
    app.permanent_session_lifetime = timedelta(hours=8)

    # ── Initialise databases ──────────────────────────────────────────────
    from services.auth_service import init_users_db
    from db import init_pred_db

    init_users_db()   # creates users.db → users table
    init_pred_db()    # creates churnmaster.db → predictions table

    # ── Pre-load ML model into memory ─────────────────────────────────────
    # Warm the cache so the first prediction request is instant
    from prediction import load_bundle
    bundle = load_bundle()
    if bundle:
        logger.info(
            "Model ready: domain=%s  features=%d  samples=%d",
            bundle.get("domain", "?"),
            bundle.get("n_features", 0),
            bundle.get("n_samples",  0),
        )
    else:
        logger.warning(
            "No model loaded — predictions will fail. "
            "Copy churn_model.pkl + scaler.pkl + feature_names.pkl into the project root, "
            "or run train.py to generate them."
        )

    # ── Register all routes as a Blueprint ───────────────────────────────
    # Blueprint is defined in routes.py — import here (after app is created)
    from routes import main_bp
    app.register_blueprint(main_bp)

    return app


# ══════════════════════════════════════════════════════════════════════════
#  MODULE-LEVEL APP INSTANCE (for gunicorn: "gunicorn app:app")
# ══════════════════════════════════════════════════════════════════════════

app = create_app()


# ══════════════════════════════════════════════════════════════════════════
#  LOCAL DEVELOPMENT ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    port  = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "true").lower() == "true"

    banner = "═" * 58
    logger.info(banner)
    logger.info("  🎯  ChurnMaster Pro — Combined Professional Edition")
    logger.info(banner)
    logger.info("  🌐  http://localhost:%d", port)
    logger.info("  🔒  Debug mode: %s", debug)
    logger.info("  📁  Predictions DB: churnmaster.db")
    logger.info("  👤  Users DB:       users.db")
    logger.info(banner)

    app.run(host="0.0.0.0", port=port, debug=debug)

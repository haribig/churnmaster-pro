"""
ChurnMaster Pro — db.py
========================
All SQLite prediction-history database helpers.
Taken from ChurnIQ Pro and adapted to use PRED_DB_PATH from config.

Separate from users.db (auth database managed by services/auth_service.py).
This database stores every single prediction made, enabling the analytics
dashboard, trend charts, and Power BI export.

Schema:
  predictions (
      id           PK AUTOINCREMENT,
      timestamp    ISO-8601 string,
      prediction   0 = stay, 1 = churn,
      probability  float 0–100,
      risk         'LOW' | 'MEDIUM' | 'HIGH',
      label        human-readable label,
      domain       dataset domain name,
      source       'single' | 'bulk',
      customer_id  optional customer identifier,
      username     who made the prediction,
      input_json   full feature dict as JSON,
      reasons_json list of reason titles as JSON
  )
"""

import sqlite3
import json
import logging
from datetime import datetime

from config import PRED_DB_PATH

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════
#  INITIALISATION
# ══════════════════════════════════════════════════════════════════════════

def init_pred_db() -> None:
    """Create the predictions table and indexes. Safe to call on every startup."""
    with sqlite3.connect(PRED_DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp    TEXT    NOT NULL,
                prediction   INTEGER NOT NULL,
                probability  REAL    NOT NULL,
                risk         TEXT    NOT NULL DEFAULT 'UNKNOWN',
                label        TEXT    NOT NULL DEFAULT '',
                domain       TEXT             DEFAULT '',
                source       TEXT             DEFAULT 'single',
                customer_id  TEXT             DEFAULT '',
                username     TEXT             DEFAULT '',
                input_json   TEXT             DEFAULT '{}',
                reasons_json TEXT             DEFAULT '[]'
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_ts   ON predictions (timestamp)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_pred ON predictions (prediction)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_risk ON predictions (risk)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_user ON predictions (username)")
        conn.commit()
    logger.info("Prediction database ready: %s", PRED_DB_PATH)


# ══════════════════════════════════════════════════════════════════════════
#  WRITE
# ══════════════════════════════════════════════════════════════════════════

def db_insert(p: dict) -> int:
    """Insert one prediction row. Returns the new row id."""
    with sqlite3.connect(PRED_DB_PATH) as conn:
        cur = conn.execute("""
            INSERT INTO predictions
              (timestamp, prediction, probability, risk, label,
               domain, source, customer_id, username, input_json, reasons_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            p.get("timestamp", datetime.utcnow().isoformat()),
            p["prediction"],
            round(float(p["probability"]), 2),
            p.get("risk",   "UNKNOWN"),
            p.get("label",  ""),
            p.get("domain", "Telecom"),
            p.get("source", "single"),
            p.get("customer_id", ""),
            p.get("username",    ""),
            json.dumps(p.get("input",   {})),
            json.dumps(p.get("reasons", [])),
        ))
        conn.commit()
        return cur.lastrowid


def db_insert_many(rows: list) -> None:
    """Batch-insert multiple rows in one transaction (used by bulk_predict)."""
    with sqlite3.connect(PRED_DB_PATH) as conn:
        conn.executemany("""
            INSERT INTO predictions
              (timestamp, prediction, probability, risk, label,
               domain, source, customer_id, username, input_json, reasons_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, rows)
        conn.commit()
    logger.debug("Batch-inserted %d prediction rows", len(rows))


def db_clear() -> None:
    """Delete ALL prediction records. Requires explicit call — not called automatically."""
    with sqlite3.connect(PRED_DB_PATH) as conn:
        conn.execute("DELETE FROM predictions")
        conn.commit()
    logger.warning("All prediction records deleted")


# ══════════════════════════════════════════════════════════════════════════
#  READ — single / list
# ══════════════════════════════════════════════════════════════════════════

def db_history(limit: int = 200, username: str = None) -> list:
    """Return the most recent *limit* predictions as list-of-dicts.
    Optionally filter by username (for per-user view)."""
    with sqlite3.connect(PRED_DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        if username:
            rows = conn.execute(
                "SELECT * FROM predictions WHERE username=? ORDER BY id DESC LIMIT ?",
                (username, limit)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
    return [dict(r) for r in rows]


# ══════════════════════════════════════════════════════════════════════════
#  READ — aggregate analytics
# ══════════════════════════════════════════════════════════════════════════

def db_stats() -> dict:
    """Return global aggregate statistics for the analytics dashboard."""
    with sqlite3.connect(PRED_DB_PATH) as conn:
        row = conn.execute("""
            SELECT
                COUNT(*),
                SUM(CASE WHEN prediction = 1 THEN 1 ELSE 0 END),
                AVG(probability),
                SUM(CASE WHEN risk = 'HIGH'   THEN 1 ELSE 0 END),
                SUM(CASE WHEN risk = 'MEDIUM' THEN 1 ELSE 0 END),
                SUM(CASE WHEN risk = 'LOW'    THEN 1 ELSE 0 END),
                SUM(CASE WHEN source = 'bulk' THEN 1 ELSE 0 END)
            FROM predictions
        """).fetchone()

    total = row[0] or 0
    return {
        "total":      total,
        "churned":    int(row[1] or 0),
        "stayed":     total - int(row[1] or 0),
        "churn_rate": round((row[1] or 0) / total * 100, 1) if total else 0.0,
        "avg_prob":   round(row[2] or 0.0, 1),
        "high_risk":  int(row[3] or 0),
        "med_risk":   int(row[4] or 0),
        "low_risk":   int(row[5] or 0),
        "bulk_total": int(row[6] or 0),
    }


def db_trend(days: int = 30) -> list:
    """Return daily churn-rate trend for the last *days* days."""
    with sqlite3.connect(PRED_DB_PATH) as conn:
        rows = conn.execute("""
            SELECT
                DATE(timestamp)                                      AS day,
                COUNT(*)                                             AS total,
                SUM(CASE WHEN prediction = 1 THEN 1 ELSE 0 END)     AS churned
            FROM predictions
            WHERE timestamp >= datetime('now', ?)
            GROUP BY day
            ORDER BY day
        """, (f"-{days} days",)).fetchall()

    return [
        {
            "day":     r[0],
            "total":   r[1],
            "churned": r[2],
            "rate":    round(r[2] / r[1] * 100, 1) if r[1] else 0,
        }
        for r in rows
    ]


def db_risk_distribution() -> dict:
    """Return {risk_level: count} breakdown."""
    with sqlite3.connect(PRED_DB_PATH) as conn:
        rows = conn.execute(
            "SELECT risk, COUNT(*) FROM predictions GROUP BY risk"
        ).fetchall()
    return {r[0]: r[1] for r in rows}

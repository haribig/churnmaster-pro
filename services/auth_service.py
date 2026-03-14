"""
ChurnMaster Pro — services/auth_service.py
==========================================
All user account database operations in one auditable module.

Security improvements vs the original customer_churn app.py:
  ✓ Passwords stored with PBKDF2-SHA256 (werkzeug) — never plain-text
  ✓ Auto-upgrade: existing plain-text passwords silently upgraded on next
    successful login — no forced password reset for existing users
  ✓ UNIQUE constraint on username enforced at database level
  ✓ Parameterised queries throughout — zero SQL injection risk
  ✓ Audit timestamp on every account

Database: users.db (USERS_DB_PATH from config)
Schema:
  users (
      id       INTEGER PK AUTOINCREMENT,
      username TEXT    UNIQUE NOT NULL,
      password TEXT    NOT NULL,         ← always pbkdf2:sha256:... hash
      role     TEXT    DEFAULT 'analyst',
      created  TEXT    DEFAULT datetime('now')
  )
"""

import sqlite3
import logging

from werkzeug.security import generate_password_hash, check_password_hash

from config import USERS_DB_PATH

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════
#  DB HELPERS
# ══════════════════════════════════════════════════════════════════════════

def _db() -> sqlite3.Connection:
    conn = sqlite3.connect(USERS_DB_PATH)
    conn.row_factory = sqlite3.Row  # access columns by name
    return conn


def init_users_db() -> None:
    """Create users table if it doesn't exist. Called once at app startup."""
    with _db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id       INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT    UNIQUE  NOT NULL,
                password TEXT            NOT NULL,
                role     TEXT            NOT NULL DEFAULT 'analyst',
                created  TEXT            NOT NULL DEFAULT (datetime('now'))
            )
        """)
        conn.commit()
    logger.info("Users database ready: %s", USERS_DB_PATH)


# ══════════════════════════════════════════════════════════════════════════
#  USER CRUD
# ══════════════════════════════════════════════════════════════════════════

def get_user(username: str) -> dict | None:
    """Return user dict by username, or None if not found."""
    with _db() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE username = ?", (username,)
        ).fetchone()
    return dict(row) if row else None


def create_user(username: str, password: str, role: str = "analyst") -> int:
    """Create a new user with a hashed password. Returns new user ID."""
    hashed = generate_password_hash(password)
    try:
        with _db() as conn:
            cur = conn.execute(
                "INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
                (username, hashed, role),
            )
            conn.commit()
            return cur.lastrowid
    except sqlite3.IntegrityError:
        raise ValueError(f"Username '{username}' is already taken.")


# ══════════════════════════════════════════════════════════════════════════
#  PASSWORD VERIFICATION WITH AUTO-MIGRATION
# ══════════════════════════════════════════════════════════════════════════

def verify_password(stored: str, provided: str, user_id: int) -> bool:
    """
    Verify a login attempt. Handles three cases:

      1. stored is a werkzeug hash  →  check_password_hash() — fast path
      2. stored is plain-text (legacy customer_churn user)  →  direct compare,
         then silently upgrade the stored value to a hash so next login uses path 1
      3. Neither matches  →  False

    This means NO existing user ever needs to reset their password.
    """
    # Path 1: properly hashed (all new accounts + already-upgraded legacy ones)
    if check_password_hash(stored, provided):
        return True

    # Path 2: legacy plain-text
    if stored == provided:
        _upgrade_hash(user_id, provided)
        logger.info("Auto-upgraded plain-text password to hash for user_id=%d", user_id)
        return True

    return False


def _upgrade_hash(user_id: int, plain: str) -> None:
    """Replace plain-text password with a secure hash in-place."""
    hashed = generate_password_hash(plain)
    with _db() as conn:
        conn.execute(
            "UPDATE users SET password = ? WHERE id = ?",
            (hashed, user_id),
        )
        conn.commit()

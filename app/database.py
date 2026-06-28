import json
from pathlib import Path
import sqlite3
from typing import Optional

from app.security import hash_password

DATABASE = Path(__file__).resolve().parents[1] / "aquasens.db"

VALID_ROLES = ("administrator", "operator")


def get_connection():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn


def initialize_database():
    conn = get_connection()
    cursor = conn.cursor()

    # ── Table utilisateurs ───────────────────────────────────────────
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'operator'
        )
    """)

    # ── Table historique diagnostics ──────────────────────────────────
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS history_diagnostic(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            ph REAL,
            solids REAL,
            conductivity REAL,
            turbidity REAL,
            temperature REAL,
            potability INTEGER,
            potability_label TEXT,
            confidence REAL,
            threshold REAL,
            inference_ms REAL
        )
    """)

    # ── Table historique filtrations ──────────────────────────────────
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS history_filtration(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            filter_id INTEGER,
            filter_name TEXT,
            reason TEXT,
            duration_s REAL,
            activated INTEGER,
            mock INTEGER
        )
    """)

    # ── Table historique prévisions ───────────────────────────────────
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS history_forecast(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            inference_ms REAL,
            n_alerts_critical INTEGER DEFAULT 0,
            n_alerts_warning INTEGER DEFAULT 0,
            n_alerts_info INTEGER DEFAULT 0,
            predictions_json TEXT,
            alerts_json TEXT
        )
    """)

    # Admin par défaut
    cursor.execute("SELECT id FROM users WHERE username=?", ("admin",))
    if cursor.fetchone() is None:
        cursor.execute(
            "INSERT INTO users(username, password_hash, role) VALUES(?,?,?)",
            ("admin", hash_password("Admin@2026"), "administrator"),
        )
        print("Administrateur créé (admin / Admin@2026)")

    # Opérateur par défaut
    cursor.execute("SELECT id FROM users WHERE username=?", ("operateur",))
    if cursor.fetchone() is None:
        cursor.execute(
            "INSERT INTO users(username, password_hash, role) VALUES(?,?,?)",
            ("operateur", hash_password("Oper@2026"), "operator"),
        )
        print("Opérateur créé (operateur / Oper@2026)")

    conn.commit()
    conn.close()


# ══════════════════════════════════════════════════════════════════════
# UTILISATEURS
# ══════════════════════════════════════════════════════════════════════

def get_user(username: str) -> Optional[dict]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username=?", (username,))
    user = cursor.fetchone()
    conn.close()
    return user


def list_users() -> list[dict]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, username, role FROM users ORDER BY id")
    users = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return users


def create_user(username: str, password: str, role: str = "operator") -> dict:
    if role not in VALID_ROLES:
        raise ValueError(f"Rôle invalide : {role}. Valides : {VALID_ROLES}")
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO users(username, password_hash, role) VALUES(?,?,?)",
        (username, hash_password(password), role),
    )
    conn.commit()
    user_id = cursor.lastrowid
    conn.close()
    return {"id": user_id, "username": username, "role": role}


def delete_user(username: str) -> bool:
    if username == "admin":
        raise ValueError("Impossible de supprimer le compte admin.")
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM users WHERE username=?", (username,))
    deleted = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return deleted


# ══════════════════════════════════════════════════════════════════════
# HISTORIQUES — ÉCRITURE
# ══════════════════════════════════════════════════════════════════════

def save_diagnostic(result: dict) -> None:
    raw = result.get("raw_values", {})
    conn = get_connection()
    conn.execute(
        """INSERT INTO history_diagnostic
           (timestamp, ph, solids, conductivity, turbidity, temperature,
            potability, potability_label, confidence, threshold, inference_ms)
           VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
        (
            result.get("timestamp"),
            raw.get("ph"),
            raw.get("Solids"),
            raw.get("Conductivity"),
            raw.get("Turbidity"),
            raw.get("Temperature"),
            result.get("potability_now"),
            result.get("potability_label"),
            result.get("confidence_proba"),
            result.get("threshold"),
            result.get("inference_time_ms"),
        ),
    )
    conn.commit()
    conn.close()


def save_filtration(actions: list[dict]) -> None:
    if not actions:
        return
    conn = get_connection()
    for a in actions:
        conn.execute(
            """INSERT INTO history_filtration
               (timestamp, filter_id, filter_name, reason, duration_s, activated, mock)
               VALUES (?,?,?,?,?,?,?)""",
            (
                a.get("timestamp"),
                a.get("filter_id"),
                a.get("filter_name"),
                a.get("reason"),
                a.get("duration_s"),
                int(a.get("activated", False)),
                int(a.get("mock", True)),
            ),
        )
    conn.commit()
    conn.close()


def save_forecast(forecast_result: dict) -> None:
    n = forecast_result.get("n_alerts", {})
    conn = get_connection()
    conn.execute(
        """INSERT INTO history_forecast
           (timestamp, inference_ms, n_alerts_critical, n_alerts_warning,
            n_alerts_info, predictions_json, alerts_json)
           VALUES (?,?,?,?,?,?,?)""",
        (
            forecast_result.get("timestamp"),
            forecast_result.get("inference_ms"),
            n.get("CRITICAL", 0),
            n.get("WARNING", 0),
            n.get("INFO", 0),
            json.dumps(forecast_result.get("predictions", {})),
            json.dumps(forecast_result.get("alerts", [])),
        ),
    )
    conn.commit()
    conn.close()


# ══════════════════════════════════════════════════════════════════════
# HISTORIQUES — LECTURE
# ══════════════════════════════════════════════════════════════════════

def get_history_diagnostic(limit: int = 100) -> list[dict]:
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM history_diagnostic ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_history_filtration(limit: int = 100) -> list[dict]:
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM history_filtration ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_history_forecast(limit: int = 50) -> list[dict]:
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM history_forecast ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]

from pathlib import Path
import sqlite3

from security import hash_password

DATABASE = Path(__file__).resolve().parents[1] / "users.db"


def get_connection():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn


def initialize_database():

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users(

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            username TEXT UNIQUE NOT NULL,

            password_hash TEXT NOT NULL,

            role TEXT NOT NULL
        )
    """)

    cursor.execute(
        "SELECT id FROM users WHERE username=?",
        ("admin",)
    )

    if cursor.fetchone() is None:

        cursor.execute(
            """
            INSERT INTO users(username,password_hash,role)

            VALUES(?,?,?)
            """,
            (
                "admin",
                hash_password("Admin@2026"),
                "administrator"
            )
        )

        print("✓ Administrateur créé")
        print("Utilisateur : admin")
        print("Mot de passe : Admin@2026")

    conn.commit()
    conn.close()


def get_user(username: str):

    conn = get_connection()

    cursor = conn.cursor()

    cursor.execute(
        "SELECT * FROM users WHERE username=?",
        (username,)
    )

    user = cursor.fetchone()

    conn.close()

    return user
"""
db.py  —  SQLite database layer for Banking-Specialist-AI-Agent
Place this file at the project root (same level as config.py).

Provides:
  - Automatic schema creation on first run
  - Seed data that mirrors the previous hard-coded mock values
  - A single `get_db()` helper used by every tool

Usage:
    from db import get_db
    with get_db() as conn:
        row = conn.execute("SELECT * FROM accounts WHERE id = ?", (account_id,)).fetchone()
"""

import sqlite3
import bcrypt
from pathlib import Path

# Database file lives at the project root
DB_PATH = Path(__file__).parent / "banking.db"


def get_db() -> sqlite3.Connection:
    """Return a connection with row_factory set so columns are accessible by name."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row          # access columns as row["column_name"]
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db() -> None:
    """Create tables and insert seed data (idempotent — safe to call on every startup)."""
    with get_db() as conn:
        # ── Schema ────────────────────────────────────────────────────────────

        conn.executescript("""
            CREATE TABLE IF NOT EXISTS accounts (
                id          TEXT PRIMARY KEY,
                owner       TEXT NOT NULL,
                balance     REAL NOT NULL DEFAULT 0.0,
                currency    TEXT NOT NULL DEFAULT 'USD',
                status      TEXT NOT NULL DEFAULT 'active',   -- active | frozen | closed
                type        TEXT NOT NULL DEFAULT 'checking'  -- checking | savings | credit
            );

            CREATE TABLE IF NOT EXISTS transactions (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                account_id  TEXT NOT NULL REFERENCES accounts(id),
                date        TEXT NOT NULL,               -- ISO-8601 date string
                description TEXT NOT NULL,
                amount      REAL NOT NULL,               -- negative = debit, positive = credit
                balance     REAL NOT NULL,               -- running balance after this txn
                location    TEXT,
                merchant    TEXT
            );

            CREATE TABLE IF NOT EXISTS credentials (
                account_id      TEXT PRIMARY KEY REFERENCES accounts(id),
                password_hash   BLOB NOT NULL
            );

            CREATE TABLE IF NOT EXISTS login_attempts (
                account_id  TEXT PRIMARY KEY,
                count       INTEGER NOT NULL DEFAULT 0,
                lock_until  REAL    NOT NULL DEFAULT 0.0  -- unix timestamp
            );

            CREATE TABLE IF NOT EXISTS loans (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                account_id      TEXT NOT NULL REFERENCES accounts(id),
                loan_type       TEXT NOT NULL,    -- mortgage | personal | auto
                amount          REAL NOT NULL,
                rate            REAL NOT NULL,
                term_months     INTEGER NOT NULL,
                status          TEXT NOT NULL DEFAULT 'active'
            );

            CREATE TABLE IF NOT EXISTS customers (
                id              TEXT PRIMARY KEY,   -- same as account_id
                credit_score    INTEGER NOT NULL DEFAULT 700,
                dti_ratio       REAL    NOT NULL DEFAULT 0.35,  -- debt-to-income
                kyc_status      TEXT    NOT NULL DEFAULT 'verified'
            );
        """)

        # ── Seed data (only insert if tables are empty) ───────────────────────

        if conn.execute("SELECT COUNT(*) FROM accounts").fetchone()[0] == 0:
            conn.executemany(
                "INSERT INTO accounts (id, owner, balance, currency, status, type) VALUES (?,?,?,?,?,?)",
                [
                    ("alice",   "Alice Johnson",  12_450.75, "USD", "active",  "checking"),
                    ("bob",     "Bob Smith",       4_820.00, "USD", "active",  "savings"),
                    ("charlie", "Charlie Wang",   31_200.50, "USD", "active",  "checking"),
                    ("diana",   "Diana Prince",    2_100.00, "USD", "frozen",  "checking"),
                ]
            )

        if conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0] == 0:
            conn.executemany(
                """INSERT INTO transactions
                   (account_id, date, description, amount, balance, location, merchant)
                   VALUES (?,?,?,?,?,?,?)""",
                [
                    # alice
                    ("alice", "2026-02-28", "Amazon Purchase",    -89.99,   12_540.74, "Online",        "amazon"),
                    ("alice", "2026-02-27", "Salary Deposit",   5_000.00,   12_630.73, "Direct Deposit", "employer"),
                    ("alice", "2026-02-25", "Netflix",            -15.99,    7_630.73, "Online",         "streaming"),
                    ("alice", "2026-02-24", "ATM Withdrawal",    -200.00,    7_646.72, "New York, NY",   "atm"),
                    ("alice", "2026-02-22", "Utility Bill",      -120.50,    7_846.72, "Online",         "utilities"),
                    # bob
                    ("bob",   "2026-02-28", "Grocery Store",      -54.30,    4_874.30, "Brooklyn, NY",   "grocery"),
                    ("bob",   "2026-02-26", "Paycheck",         2_500.00,    4_928.60, "Direct Deposit", "employer"),
                    # charlie — includes a suspicious large transaction for fraud demo
                    ("charlie", "2026-02-28", "Wire Transfer",  18_500.00,  31_200.50, "Cayman Islands", "wire"),
                    ("charlie", "2026-02-25", "Salary Deposit", 6_000.00,   12_700.50, "Direct Deposit", "employer"),
                ]
            )

        if conn.execute("SELECT COUNT(*) FROM credentials").fetchone()[0] == 0:
            conn.executemany(
                "INSERT INTO credentials (account_id, password_hash) VALUES (?,?)",
                [
                    ("alice",   bcrypt.hashpw(b"alice_password",   bcrypt.gensalt())),
                    ("bob",     bcrypt.hashpw(b"bob_password",     bcrypt.gensalt())),
                    ("charlie", bcrypt.hashpw(b"charlie_password", bcrypt.gensalt())),
                ]
            )

        if conn.execute("SELECT COUNT(*) FROM customers").fetchone()[0] == 0:
            conn.executemany(
                "INSERT INTO customers (id, credit_score, dti_ratio, kyc_status) VALUES (?,?,?,?)",
                [
                    ("alice",   760, 0.28, "verified"),
                    ("bob",     620, 0.45, "verified"),
                    ("charlie", 810, 0.18, "verified"),
                    ("diana",   550, 0.60, "pending"),
                ]
            )

        conn.commit()
        print(f"[db] Database ready at {DB_PATH}")


# Auto-initialise when the module is first imported
init_db()

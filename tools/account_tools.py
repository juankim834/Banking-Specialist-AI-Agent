"""
tools/account_tools.py  —  Account tool functions for the Account Specialist agent.

All mock hard-coded data has been replaced with SQLite queries via db.get_db().
Tool signatures are identical to the original, so no agent code needs to change.
"""

import time
import bcrypt

from agents import function_tool
from utils.audit_logger import log_event
from db import get_db


# ── Balance ───────────────────────────────────────────────────────────────────

@function_tool
def get_account_balance(account_id: str) -> dict:
    """Retrieve the balance and basic info for a given account ID."""
    log_event("AccountAgent", "get_account_balance", {"account_id": account_id})

    with get_db() as conn:
        row = conn.execute(
            "SELECT id, owner, balance, currency, status, type FROM accounts WHERE id = ?",
            (account_id,)
        ).fetchone()

    if row is None:
        return {"error": f"Account '{account_id}' not found."}

    return {
        "account_id": row["id"],
        "owner":      row["owner"],
        "balance":    row["balance"],
        "currency":   row["currency"],
        "status":     row["status"],
        "type":       row["type"],
    }


# ── Transaction history ───────────────────────────────────────────────────────

@function_tool
def get_transaction_history(account_id: str, limit: int = 10) -> list[dict]:
    """Fetch recent transaction history for an account, ordered newest first."""
    log_event("AccountAgent", "get_transaction_history", {
        "account_id": account_id, "limit": limit
    })

    with get_db() as conn:
        rows = conn.execute(
            """SELECT date, description, amount, balance, location, merchant
               FROM transactions
               WHERE account_id = ?
               ORDER BY date DESC, id DESC
               LIMIT ?""",
            (account_id, limit)
        ).fetchall()

    if not rows:
        return []

    return [
        {
            "date":        row["date"],
            "description": row["description"],
            "amount":      row["amount"],
            "balance":     row["balance"],
            "location":    row["location"],
            "merchant":    row["merchant"],
        }
        for row in rows
    ]


# ── Fund transfer ─────────────────────────────────────────────────────────────

@function_tool
def transfer_funds(from_account: str, to_account: str, amount: float, currency: str = "USD") -> dict:
    """
    Transfer funds between two accounts.
    Enforces a $5,000 single-transfer regulatory cap.
    Both accounts must exist and the source must have sufficient funds.
    """
    log_event("AccountAgent", "transfer_funds", {
        "from": from_account, "to": to_account, "amount": amount, "currency": currency
    })

    if amount <= 0:
        return {"status": "blocked", "reason": "Transfer amount must be positive."}

    if amount > 5_000:
        return {
            "status": "blocked",
            "reason": "Exceeds single-transfer regulatory limit of $5,000. Requires manual approval."
        }

    with get_db() as conn:
        src = conn.execute(
            "SELECT balance, status FROM accounts WHERE id = ?", (from_account,)
        ).fetchone()
        dst = conn.execute(
            "SELECT id FROM accounts WHERE id = ?", (to_account,)
        ).fetchone()

        if src is None:
            return {"status": "error", "reason": f"Source account '{from_account}' not found."}
        if dst is None:
            return {"status": "error", "reason": f"Destination account '{to_account}' not found."}
        if src["status"] != "active":
            return {"status": "blocked", "reason": f"Source account is '{src['status']}' and cannot send funds."}
        if src["balance"] < amount:
            return {"status": "blocked", "reason": "Insufficient funds."}

        import datetime, uuid
        txn_id = f"TXN-{datetime.date.today().year}-{uuid.uuid4().hex[:8].upper()}"
        today  = datetime.date.today().isoformat()

        new_src_balance = src["balance"] - amount
        conn.execute(
            "UPDATE accounts SET balance = ? WHERE id = ?",
            (new_src_balance, from_account)
        )
        conn.execute(
            "UPDATE accounts SET balance = balance + ? WHERE id = ?",
            (amount, to_account)
        )
        conn.execute(
            """INSERT INTO transactions (account_id, date, description, amount, balance)
               VALUES (?, ?, ?, ?, ?)""",
            (from_account, today, f"Transfer to {to_account}", -amount, new_src_balance)
        )
        conn.execute(
            """INSERT INTO transactions (account_id, date, description, amount, balance)
               VALUES (?, ?, ?, ?, ?)""",
            (to_account, today, f"Transfer from {from_account}", amount,
             conn.execute("SELECT balance FROM accounts WHERE id = ?", (to_account,)).fetchone()["balance"])
        )
        conn.commit()

    return {
        "status":         "success",
        "transaction_id": txn_id,
        "amount":         amount,
        "currency":       currency,
    }


# ── Credential verification (not a tool — called internally by verify_login) ──

MAX_ATTEMPTS = 5
LOCK_TIME    = 60  # seconds


def _verify_credentials(account_id: str, password: str) -> dict:
    """
    Verify account_id / password against the credentials table.
    Rate-limiting state is persisted in the login_attempts table.
    """
    now = time.time()

    with get_db() as conn:
        # Ensure a tracking row exists for this account_id
        conn.execute(
            "INSERT OR IGNORE INTO login_attempts (account_id, count, lock_until) VALUES (?, 0, 0.0)",
            (account_id,)
        )
        conn.commit()

        attempt = conn.execute(
            "SELECT count, lock_until FROM login_attempts WHERE account_id = ?",
            (account_id,)
        ).fetchone()

        # Reject if currently locked out
        if now < attempt["lock_until"]:
            return {"success": False, "reason": "Too many login attempts. Try again later."}

        cred = conn.execute(
            "SELECT password_hash FROM credentials WHERE account_id = ?",
            (account_id,)
        ).fetchone()

        # Unknown account — add delay to prevent username enumeration
        if cred is None:
            time.sleep(0.5)
            return {"success": False, "reason": "Invalid username or password."}

        password_ok = bcrypt.checkpw(password.encode(), cred["password_hash"])

        if not password_ok:
            new_count = attempt["count"] + 1
            if new_count >= MAX_ATTEMPTS:
                conn.execute(
                    "UPDATE login_attempts SET count = 0, lock_until = ? WHERE account_id = ?",
                    (now + LOCK_TIME, account_id)
                )
            else:
                conn.execute(
                    "UPDATE login_attempts SET count = ? WHERE account_id = ?",
                    (new_count, account_id)
                )
            conn.commit()
            time.sleep(0.5)
            return {"success": False, "reason": "Invalid username or password."}

        # Success — reset attempt counter
        conn.execute(
            "UPDATE login_attempts SET count = 0, lock_until = 0.0 WHERE account_id = ?",
            (account_id,)
        )
        conn.commit()

        owner = conn.execute(
            "SELECT owner FROM accounts WHERE id = ?", (account_id,)
        ).fetchone()

    return {
        "success":    True,
        "account_id": account_id,
        "owner":      owner["owner"] if owner else account_id,
    }


@function_tool
def verify_login(account_id: str, password: str) -> dict:
    """Verify account credentials during login. Returns success status and the account owner name."""
    log_event("AccountAgent", "verify_login", {"account_id": account_id})
    return _verify_credentials(account_id, password)

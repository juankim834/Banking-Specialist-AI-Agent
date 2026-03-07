from agents import function_tool
from utils.audit_logger import log_event

@function_tool
def get_account_balance(account_id: str) -> dict:
    """Retrieve the balance and basic info for a given account ID."""
    log_event("AccountAgent", "get_account_balance", {"account_id": account_id})
    # TODO: Replace with real DB/API call
    return {
        "account_id": account_id,
        "owner": "Jane Doe",
        "balance": 12_450.75,
        "currency": "USD",
        "status": "active",
        "type": "checking",
    }

@function_tool
def get_transaction_history(account_id: str, limit: int = 10) -> list[dict]:
    """Fetch recent transaction history for an account."""
    log_event("AccountAgent", "get_transaction_history", {"account_id": account_id, "limit": limit})
    return [
        {"date": "2026-02-28", "description": "Amazon Purchase",  "amount": -89.99,  "balance": 12_540.74},
        {"date": "2026-02-27", "description": "Salary Deposit",   "amount": 5_000.00, "balance": 12_630.73},
        {"date": "2026-02-25", "description": "Netflix",          "amount": -15.99,  "balance": 7_630.73},
        {"date": "2026-02-24", "description": "ATM Withdrawal",   "amount": -200.00, "balance": 7_646.72},
        {"date": "2026-02-22", "description": "Utility Bill",     "amount": -120.50, "balance": 7_846.72},
    ][:limit]

@function_tool
def transfer_funds(from_account: str, to_account: str, amount: float, currency: str = "USD") -> dict:
    """Transfer funds between accounts with basic threshold checks."""
    log_event("AccountAgent", "transfer_funds", {
        "from": from_account, "to": to_account, "amount": amount, "currency": currency
    })
    if amount > 50_000:
        return {"status": "blocked", "reason": "Exceeds single-transfer regulatory limit of $50,000. Requires manual approval."}
    return {"status": "success", "transaction_id": "TXN-2026-98765", "amount": amount, "currency": currency}


import bcrypt
import time

# simulated credentials store (in production, use a secure database)
_CREDENTIALS = {
    "alice": {
        "owner": "Alice",
        "password_hash": bcrypt.hashpw(b"alice_password", bcrypt.gensalt())
    }
}

_LOGIN_ATTEMPTS = {}
MAX_ATTEMPTS = 5
LOCK_TIME = 60


def _verify_credentials(account_id: str, password: str) -> dict:
    """Verify account_id / password against credentials store."""

    now = time.time()

    # Initiate tracking for this account if not present
    if account_id not in _LOGIN_ATTEMPTS:
        _LOGIN_ATTEMPTS[account_id] = {"count": 0, "lock_until": 0}

    attempt = _LOGIN_ATTEMPTS[account_id]

    # If the account is currently locked, reject immediately
    if now < attempt["lock_until"]:
        return {"success": False, "reason": "Too many login attempts. Try later."}

    user = _CREDENTIALS.get(account_id)

    # Standardize failure response to prevent username enumeration
    if not user:
        time.sleep(0.5)
        return {"success": False, "reason": "Invalid username or password"}

    password_ok = bcrypt.checkpw(password.encode(), user["password_hash"])

    if not password_ok:
        attempt["count"] += 1

        # Exceeding max attempts triggers a lockout
        if attempt["count"] >= MAX_ATTEMPTS:
            attempt["lock_until"] = now + LOCK_TIME
            attempt["count"] = 0

        time.sleep(0.5)
        return {"success": False, "reason": "Invalid username or password"}

    # Login successful, reset attempt tracking
    attempt["count"] = 0

    return {
        "success": True,
        "account_id": account_id,
        "owner": user["owner"]
    }


@function_tool
def verify_login(account_id: str, password: str) -> dict:
    """Verify account credentials during login. Returns success status and the account owner name."""
    log_event("AccountAgent", "verify_login", {"account_id": account_id})
    return _verify_credentials(account_id, password)
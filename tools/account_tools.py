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
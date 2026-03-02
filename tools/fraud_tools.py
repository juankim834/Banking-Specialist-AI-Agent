from agents import function_tool
from utils.audit_logger import log_event

@function_tool
def analyze_transaction_for_fraud(transaction_id: str, amount: float, location: str, merchant: str) -> dict:
    """Run fraud risk analysis on a specific transaction."""
    log_event("FraudAgent", "analyze_transaction_for_fraud", {
        "transaction_id": transaction_id, "amount": amount, "location": location
    })
    # Simulated heuristic fraud scoring
    risk_score = 0
    flags = []
    if amount > 10_000:
        risk_score += 40
        flags.append("High-value transaction")
    if location in ["Unknown", "High-risk country"]:
        risk_score += 50
        flags.append("Suspicious location")
    if merchant.lower() in ["crypto exchange", "gambling site"]:
        risk_score += 30
        flags.append("High-risk merchant category")

    risk_level = "LOW" if risk_score < 30 else "MEDIUM" if risk_score < 60 else "HIGH"
    return {
        "transaction_id": transaction_id,
        "risk_score": risk_score,
        "risk_level": risk_level,
        "flags": flags,
        "recommendation": "Block and escalate" if risk_level == "HIGH" else "Monitor" if risk_level == "MEDIUM" else "Approve",
    }

@function_tool
def freeze_account(account_id: str, reason: str) -> dict:
    """Freeze an account immediately due to suspected fraud."""
    log_event("FraudAgent", "freeze_account", {"account_id": account_id, "reason": reason})
    return {"status": "frozen", "account_id": account_id, "reason": reason, "timestamp": "2026-03-01T10:00:00Z"}
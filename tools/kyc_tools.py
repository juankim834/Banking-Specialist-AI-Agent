from agents import function_tool
from utils.audit_logger import log_event

@function_tool
def verify_customer_identity(customer_id: str, document_type: str) -> dict:
    """Perform KYC identity verification for a customer."""
    log_event("KYCAgent", "verify_customer_identity", {
        "customer_id": customer_id, "document_type": document_type
    })
    return {
        "customer_id": customer_id,
        "document_type": document_type,
        "verification_status": "VERIFIED",
        "risk_rating": "LOW",
        "aml_flags": [],
        "last_verified": "2025-12-01",
    }

@function_tool
def run_aml_check(customer_id: str, transaction_amount: float) -> dict:
    """Run Anti-Money Laundering (AML) check for a customer and transaction."""
    log_event("KYCAgent", "run_aml_check", {
        "customer_id": customer_id, "transaction_amount": transaction_amount
    })
    flagged = transaction_amount > 9_999  # CTR threshold
    return {
        "customer_id": customer_id,
        "transaction_amount": transaction_amount,
        "ctr_required": flagged,
        "suspicious_activity": False,
        "status": "FLAGGED_FOR_CTR" if flagged else "CLEAR",
        "note": "Currency Transaction Report required for transactions over $10,000" if flagged else "No AML concerns",
    }
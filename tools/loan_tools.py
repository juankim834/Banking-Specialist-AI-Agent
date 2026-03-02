from agents import function_tool
from utils.audit_logger import log_event

@function_tool
def check_loan_eligibility(customer_id: str, loan_amount: float, loan_type: str) -> dict:
    """Assess a customer's loan eligibility based on their profile."""
    log_event("LoanAgent", "check_loan_eligibility", {
        "customer_id": customer_id, "loan_amount": loan_amount, "loan_type": loan_type
    })
    # Simulated credit scoring
    credit_score = 720  # TODO: fetch from real credit bureau API
    dti_ratio = 0.32    # Debt-to-income ratio
    eligible = credit_score >= 680 and dti_ratio <= 0.43 and loan_amount <= 500_000

    return {
        "customer_id": customer_id,
        "credit_score": credit_score,
        "dti_ratio": dti_ratio,
        "requested_amount": loan_amount,
        "eligible": eligible,
        "max_eligible_amount": 250_000 if eligible else 0,
        "interest_rate_estimate": "5.75% APR" if eligible else "N/A",
        "reason": "Approved based on credit profile" if eligible else "Does not meet minimum eligibility criteria",
    }

@function_tool
def get_loan_products(loan_type: str) -> list[dict]:
    """Return available loan products for a given type."""
    log_event("LoanAgent", "get_loan_products", {"loan_type": loan_type})
    products = {
        "mortgage": [
            {"name": "30-Year Fixed", "rate": "6.25%", "min_amount": 50_000, "max_amount": 1_000_000},
            {"name": "15-Year Fixed", "rate": "5.75%", "min_amount": 50_000, "max_amount": 750_000},
        ],
        "personal": [
            {"name": "Standard Personal Loan", "rate": "8.99%", "min_amount": 1_000, "max_amount": 50_000},
        ],
        "auto": [
            {"name": "New Car Loan",  "rate": "4.99%", "min_amount": 5_000, "max_amount": 100_000},
            {"name": "Used Car Loan", "rate": "6.49%", "min_amount": 3_000, "max_amount": 50_000},
        ],
    }
    return products.get(loan_type.lower(), [{"error": f"Unknown loan type: {loan_type}"}])
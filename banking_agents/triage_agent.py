from agents import Agent
from config import TRIAGE_MODEL
from banking_agents.account_agent import account_agent
from banking_agents.fraud_agent   import fraud_agent
from banking_agents.loan_agent    import loan_agent
from banking_agents.kyc_agent     import kyc_agent
from banking_agents.support_agent import support_agent
from banking_agents.data_agent    import data_agent

triage_agent = Agent(
    name="Banking AI Assistant",
    model=TRIAGE_MODEL,
    instructions="""
You are the Banking AI Assistant Orchestrator. You handle every customer request by calling the right specialist tool(s) directly — you never hand off or transfer the customer.

RULES:
- NEVER say "I'm transferring you to...", "Let me connect you with...", or any similar phrase. Call the tool immediately and return the result to the user yourself.
- For compound requests (multiple topics in one message), call EACH relevant specialist tool in sequence, then combine all results into one clear reply.
- NEVER ask the customer for information that a tool can fetch (e.g. transaction details, account ID).
- The customer is already authenticated; their account ID is provided in the system context — pass it to tools as needed.

Specialist tools — call whichever match the request (can call multiple):
- account_specialist  → account balance, transaction history, fund transfers
- fraud_specialist    → suspicious / unrecognized transactions, fraud risk analysis, account freeze
- loan_specialist     → loan eligibility, mortgage / auto / personal loan products
- kyc_specialist      → KYC identity verification, AML compliance checks
- data_specialist     → stock prices, FRED economic data, PDF document analysis
- support_specialist  → general FAQs, branch hours, product info
""",
    tools=[
        account_agent.as_tool(
            tool_name="account_specialist",
            tool_description="Get account balance, transaction history, or process fund transfers.",
        ),
        fraud_agent.as_tool(
            tool_name="fraud_specialist",
            tool_description="Investigate suspicious or unrecognized transactions, run fraud risk analysis, and freeze accounts if needed.",
        ),
        loan_agent.as_tool(
            tool_name="loan_specialist",
            tool_description="Check loan eligibility and retrieve available loan products (mortgage, auto, personal).",
        ),
        kyc_agent.as_tool(
            tool_name="kyc_specialist",
            tool_description="Perform KYC identity verification and run AML compliance checks.",
        ),
        support_agent.as_tool(
            tool_name="support_specialist",
            tool_description="Answer general banking FAQs, branch hours, and product questions.",
        ),
        data_agent.as_tool(
            tool_name="data_specialist",
            tool_description="Fetch stock prices, FRED economic data, or analyze uploaded PDF financial documents.",
        ),
    ],
)

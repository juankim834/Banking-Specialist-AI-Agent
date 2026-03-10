import os
from dotenv import load_dotenv
from agents.extensions.models.litellm_model import LitellmModel

load_dotenv()

# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# TRIAGE_MODEL   = "gpt-4o-mini"   # Fast/cheap for routing
# SPECIALIST_MODEL = "gpt-4o"      # Powerful for complex reasoning

# Option A: Groq (free tier, extremely fast)
# TRIAGE_MODEL    = LitellmModel(model="groq/llama-3.1-8b-instant",   api_key=os.getenv("GROQ_API_KEY"))
# SPECIALIST_MODEL = LitellmModel(model="groq/llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY"))

# Option B: Anthropic Claude (powerful, great reasoning)
# TRIAGE_MODEL    = LitellmModel(model="anthropic/claude-haiku-3-5",         api_key=os.getenv("ANTHROPIC_API_KEY"))
# SPECIALIST_MODEL = LitellmModel(model="anthropic/claude-sonnet-4-5",       api_key=os.getenv("ANTHROPIC_API_KEY"))

# Option C: Google Gemini (generous free tier)
TRIAGE_MODEL = LitellmModel(model="groq/openai/gpt-oss-120b", api_key = os.getenv("GROQ_API_KEY"))

SPECIALIST_MODEL = LitellmModel(model="groq/llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY"))

# Option D: Mistral AI (EU-based, open-weight)
# TRIAGE_MODEL    = LitellmModel(model="mistral/mistral-small-latest",       api_key=os.getenv("MISTRAL_API_KEY"))
# SPECIALIST_MODEL = LitellmModel(model="mistral/mistral-large-latest",      api_key=os.getenv("MISTRAL_API_KEY"))

# ── Base compliance / reference documents ─────────────────────────────────────
# PDFs listed here are automatically indexed at server startup from the
# uploads/ directory.  Add any regulatory or policy PDFs you want the agent
# to always have available for hybrid RAG search.
#
# p560b.pdf → IRS Publication 560 "Retirement Plans for Small Business"
#             (SEP, SIMPLE, and Qualified Plans) — excellent compliance base.
BASE_DOCUMENTS: list[str] = [
    "p560b.pdf",
]
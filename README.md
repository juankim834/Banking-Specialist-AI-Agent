# 🏦 Banking AI Multi-Agent System

A production-ready AI banking assistant powered by the **OpenAI Agents SDK** and **FastAPI**. A central Triage Orchestrator routes every customer request to the right specialist agent, all served through a secure REST API and a real-time web chat interface.

---

## ✨ Features

- **7 AI Agents** — Triage Orchestrator + 6 domain specialists
- **Real-time streaming** — Server-Sent Events (SSE) for instant responses
- **Secure authentication** — HMAC-signed session tokens + bcrypt password hashing
- **PII Guardrail** — Blocks credit card numbers, SSNs, and account numbers before they reach any agent
- **Hybrid RAG search** — BM25 + vector retrieval with Reciprocal Rank Fusion over uploaded PDFs
- **External data** — Live stock prices (Yahoo Finance) and macroeconomic series (FRED)
- **Audit logging** — Structured JSON log of every agent action and security event
- **Multi-provider LLM** — Swap between Gemini, Groq, Claude, or Mistral in one config line

---

## 🤖 Agents

| Agent | Responsibility | Tools |
|---|---|---|
| **Triage Orchestrator** | Routes requests; calls specialist(s) as tools | 6 sub-agents |
| **Account Specialist** | Balances, transaction history, fund transfers | `get_account_balance`, `get_transaction_history`, `transfer_funds` |
| **Fraud Detection Specialist** | Risk scoring, suspicious transaction investigation, account freeze | `analyze_transaction_for_fraud`, `freeze_account` |
| **Loan & Credit Specialist** | Loan eligibility, mortgage/auto/personal product recommendations | `check_loan_eligibility`, `get_loan_products` |
| **KYC & Compliance Specialist** | Identity verification, AML checks, BSA compliance | `verify_customer_identity`, `run_aml_check` |
| **General Banking Support** | FAQs, branch info, product questions | — |
| **Data Synthesis Specialist** | Stock data, FRED economic series, PDF document analysis | `fetch_stock_data`, `fetch_fred_series`, `extract_pdf_text`, RAG tools |

---

## 🗂️ Project Structure

```
Banking-Specialist-AI-Agent/
├── config.py                  # LLM model selection & base document list
├── main.py                    # CLI entry point (Rich terminal chat)
├── server.py                  # FastAPI server + all REST endpoints
├── banking_agents/
│   ├── triage_agent.py        # Orchestrator
│   ├── account_agent.py
│   ├── fraud_agent.py
│   ├── loan_agent.py
│   ├── kyc_agent.py
│   ├── support_agent.py
│   └── data_agent.py
├── tools/
│   ├── account_tools.py
│   ├── fraud_tools.py
│   ├── loan_tools.py
│   ├── kyc_tools.py
│   ├── data_synthesis_tools.py
│   ├── rag_tools.py
│   ├── hybrid_search.py
│   └── document_processor.py
├── guardrails/
│   └── pii_guardrail.py       # Regex PII detection
├── utils/
│   └── audit_logger.py        # Structured JSON audit log
├── frontend/
│   ├── index.html             # Web chat SPA
│   └── walkthrough.html       # Full project documentation page
└── uploads/                   # Uploaded PDFs for RAG indexing
```

---

## 🚀 Getting Started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment variables

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_key_here
SESSION_SECRET=change-this-in-production
```

> To use a different LLM provider, open `config.py` and uncomment the desired block (Groq, Claude, Mistral, or Gemini).

### 3. Run the web server

```bash
python server.py
```

Then open [http://localhost:8000](http://localhost:8000) in your browser.

**Demo credentials:** `alice` / `alice_password`

### 4. (Optional) Run the CLI

```bash
python main.py
```

---

## 🌐 API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serves the web chat UI |
| `POST` | `/login` | Authenticate; returns a session token |
| `POST` | `/logout` | Invalidate session |
| `GET` | `/me` | Return authenticated account ID |
| `POST` | `/chat` | Stream agent response (SSE) |
| `POST` | `/upload-pdf` | Upload a PDF for RAG analysis |
| `GET` | `/documents` | List uploaded PDFs and index status |
| `POST` | `/index-document` | Index an uploaded PDF for hybrid search |
| `POST` | `/shutdown` | Gracefully stop the server |

---

## 🔒 Security

- **Session tokens** — HMAC-SHA256 signed; verified on every protected request
- **Bcrypt** — Passwords hashed with random salts; constant-time comparison
- **Rate limiting** — 5 failed login attempts triggers a 60-second lockout
- **PII guardrail** — Regex blocks card numbers, SSNs, passport IDs, and long account numbers before any agent processes them
- **Path traversal protection** — Uploaded filenames are sanitised with `pathlib.Path(name).name`
- **Audit log** — Every agent action, login, guardrail trigger, and error written to `audit.log`

---

## 📚 Documentation

For a full visual walkthrough of the architecture, agents, tools, API, and request flow, open the in-app docs page:

```
http://localhost:8000/static/walkthrough.html
```

Or click the **Docs** button in the top-right corner of the chat UI.

---

## 🛠️ Tech Stack

- [OpenAI Agents SDK](https://github.com/openai/openai-agents-python) — agent orchestration
- [FastAPI](https://fastapi.tiangolo.com/) + [Uvicorn](https://www.uvicorn.org/) — async web server
- [LiteLLM](https://github.com/BerriAI/litellm) — multi-provider LLM wrapper
- [bcrypt](https://pypi.org/project/bcrypt/) — password hashing
- [yfinance](https://github.com/ranaroussi/yfinance) — stock market data
- [fredapi](https://github.com/mortada/fredapi) — Federal Reserve economic data
- [Rich](https://github.com/Textualize/rich) — terminal UI for CLI mode

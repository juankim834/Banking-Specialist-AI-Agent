import asyncio
import hashlib
import hmac
import json
import os
import pathlib
import secrets
import shutil
import signal
from base64 import urlsafe_b64decode, urlsafe_b64encode
from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from agents import Runner, InputGuardrailTripwireTriggered, RunConfig
from agents.extensions.models.litellm_model import LitellmModel
from banking_agents.triage_agent import triage_agent
from guardrails.pii_guardrail import pii_guardrail
from tools.account_tools import _verify_credentials
from tools.rag_tools import _do_index as _rag_index, _indexed_docs as _rag_indexed_docs
from config import BASE_DOCUMENTS
from utils.audit_logger import log_event

# Attach PII guardrail
triage_agent.input_guardrails = [pii_guardrail]

# Upload directory for PDF files
UPLOAD_DIR = pathlib.Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Signing secret — set SESSION_SECRET env var in production
_SESSION_SECRET = os.getenv("SESSION_SECRET", "dev-banking-secret-key-change-in-production").encode()


def _create_token(account_id: str) -> str:
    """Return an HMAC-SHA256-signed token encoding the account_id."""
    payload = urlsafe_b64encode(account_id.encode()).decode()
    sig = hmac.new(_SESSION_SECRET, payload.encode(), hashlib.sha256).hexdigest()
    return f"{payload}.{sig}"


def _decode_token(token: str) -> str | None:
    """Verify token signature and return account_id, or None if invalid."""
    try:
        payload, sig = token.rsplit(".", 1)
        expected = hmac.new(_SESSION_SECRET, payload.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(sig, expected):
            return None
        return urlsafe_b64decode(payload.encode()).decode()
    except Exception:
        return None


app = FastAPI(title="Banking AI Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the frontend
app.mount("/static", StaticFiles(directory="frontend"), name="static")


class LoginRequest(BaseModel):
    account_id: str
    password: str


class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []
    api_key: str | None = None   # user-supplied API key (optional)
    model: str | None = None     # LiteLLM model string, e.g. "gemini/gemini-2.5-flash"


class IndexDocumentRequest(BaseModel):
    filename: str


@app.on_event("startup")
async def auto_index_base_documents():
    """Auto-index base compliance documents at server startup."""
    for filename in BASE_DOCUMENTS:
        safe_name = pathlib.Path(filename).name
        pdf_path = UPLOAD_DIR / safe_name
        if pdf_path.exists() and safe_name not in _rag_indexed_docs:
            try:
                result = _rag_index(safe_name)
                if result.get("status") == "indexed":
                    log_event("System", "base_doc_indexed", {
                        "filename": safe_name,
                        "chunks": result.get("total_chunks"),
                    })
                else:
                    log_event("System", "base_doc_index_skip", {
                        "filename": safe_name, "result": result,
                    })
            except Exception as exc:
                log_event("System", "base_doc_index_error", {
                    "filename": safe_name, "error": str(exc),
                })


@app.get("/")
async def root():
    return FileResponse("frontend/index.html")


@app.post("/login")
async def login(req: LoginRequest):
    """Authenticate with account_id + password; returns a session token on success."""
    result = _verify_credentials(req.account_id, req.password)
    if not result["success"]:
        log_event("Auth", "login_failed", {"account_id": req.account_id, "reason": result["reason"]})
        raise HTTPException(status_code=401, detail=result["reason"])
    token = _create_token(req.account_id)
    log_event("Auth", "login_success", {"account_id": req.account_id})
    return {"session_token": token, "account_id": req.account_id, "owner": result["owner"]}


@app.post("/logout")
async def logout(x_session_token: str = Header(None)):
    """Invalidate the current session token (client must discard it)."""
    account_id = _decode_token(x_session_token) if x_session_token else None
    if account_id:
        log_event("Auth", "logout", {"account_id": account_id})
    return {"status": "logged out"}


@app.get("/me")
async def me(x_session_token: str = Header(None)):
    """Return the authenticated account_id, or 401 if the token is invalid."""
    account_id = _decode_token(x_session_token) if x_session_token else None
    if not account_id:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return {"account_id": account_id}


@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...), x_session_token: str = Header(None)):
    """Upload a PDF document for analysis by the Data Synthesis Specialist."""
    if not x_session_token or not _decode_token(x_session_token):
        raise HTTPException(status_code=401, detail="Not authenticated")
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files (.pdf) are accepted")
    # Security: strip any path components from the filename to prevent directory traversal
    safe_name = pathlib.Path(file.filename).name
    dest = UPLOAD_DIR / safe_name
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    log_event("DataSynthesis", "pdf_uploaded", {"filename": safe_name})
    return {"filename": safe_name, "message": f"PDF '{safe_name}' uploaded successfully."}


@app.get("/documents")
async def list_documents(x_session_token: str = Header(None)):
    """List all PDF files in uploads/ with their RAG index status."""
    if not x_session_token or not _decode_token(x_session_token):
        raise HTTPException(status_code=401, detail="Not authenticated")
    docs = []
    for p in sorted(UPLOAD_DIR.glob("*.pdf")):
        docs.append({
            "filename": p.name,
            "indexed": p.name in _rag_indexed_docs,
            "is_base": p.name in BASE_DOCUMENTS,
            "size_kb": round(p.stat().st_size / 1024, 1),
        })
    return {"documents": docs}

@app.post("/index-document")
async def index_document(
    req: IndexDocumentRequest,
    x_session_token: str = Header(None),
):
    """Index an already-uploaded PDF for hybrid BM25 + vector RAG search."""
    if not x_session_token or not _decode_token(x_session_token):
        raise HTTPException(status_code=401, detail="Not authenticated")
    safe_name = pathlib.Path(req.filename).name
    if not (UPLOAD_DIR / safe_name).exists():
        raise HTTPException(
            status_code=404,
            detail=f"'{safe_name}' not found. Please upload it first.",
        )
    result = _rag_index(safe_name)
    log_event(
        "DataSynthesis", "document_indexed",
        {"filename": safe_name, "status": result.get("status")},
    )
    return result


@app.post("/chat")
async def chat(req: ChatRequest, x_session_token: str = Header(None)):
    """Stream the agent response as Server-Sent Events. Requires a valid session token."""
    account_id = _decode_token(x_session_token) if x_session_token else None
    if not account_id:
        raise HTTPException(status_code=401, detail="Not authenticated")

    async def event_stream():
        # Inject the authenticated account_id so agents never need to ask for it
        system_msg = {
            "role": "system",
            "content": (
                f"The authenticated customer's account ID is '{account_id}'. "
                "For every tool that requires an account_id or customer_id, use this value directly. "
                "Never ask the user for their account ID or customer ID."
            ),
        }
        conversation = [system_msg] + list(req.history)
        conversation.append({"role": "user", "content": req.message})
        log_event("TriageOrchestrator", "user_message", {"message": req.message})

        # Build optional RunConfig if the user supplied their own API key
        run_config = None
        if req.api_key and req.model:
            custom_model = LitellmModel(model=req.model, api_key=req.api_key)
            run_config = RunConfig(model=custom_model)

        try:
            result = await Runner.run(
                starting_agent=triage_agent,
                input=conversation,
                run_config=run_config,
                max_turns=10,
            )

            response_text = result.final_output
            last_agent = result.last_agent.name if result.last_agent else "Banking Agent"
            log_event(last_agent, "agent_response", {"response": response_text})

            payload = json.dumps({
                "type": "message",
                "content": response_text,
                "agent": last_agent,
            })
            yield f"data: {payload}\n\n"

        except InputGuardrailTripwireTriggered as e:
            log_event("Guardrail", "pii_blocked", {"reason": str(e)})
            payload = json.dumps({
                "type": "error",
                "content": "⚠️ Security Alert: Your message appears to contain sensitive personal information (PII). Please remove card numbers, SSNs, or account numbers and try again.",
                "agent": "Security",
            })
            yield f"data: {payload}\n\n"

        except Exception as e:
            log_event("System", "error", {"error": str(e)})
            payload = json.dumps({
                "type": "error",
                "content": f"❌ An error occurred: {str(e)}",
                "agent": "System",
            })
            yield f"data: {payload}\n\n"

        # Signal stream end
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")

@app.post("/shutdown")
async def shutdown():
    """Endpoint to gracefully shut down the server."""
    log_event("System", "shutdown_initiated", {})

    async def _kill():
        await asyncio.sleep(0.5)          # give the response time to reach the client
        os.kill(os.getpid(), signal.SIGINT)  # graceful Ctrl-C, uvicorn handles it cleanly

    asyncio.ensure_future(_kill())
    return {"message": "Server is shutting down..."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)

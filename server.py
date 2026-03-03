import asyncio
import json
import os
import signal
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from agents import Runner, InputGuardrailTripwireTriggered
from banking_agents.triage_agent import triage_agent
from guardrails.pii_guardrail import pii_guardrail
from utils.audit_logger import log_event

# Attach PII guardrail
triage_agent.input_guardrails = [pii_guardrail]

app = FastAPI(title="Banking AI Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the frontend
app.mount("/static", StaticFiles(directory="frontend"), name="static")


class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []


@app.get("/")
async def root():
    return FileResponse("frontend/index.html")


@app.post("/chat")
async def chat(req: ChatRequest):
    """Stream the agent response as Server-Sent Events."""

    async def event_stream():
        conversation = list(req.history)
        conversation.append({"role": "user", "content": req.message})
        log_event("TriageOrchestrator", "user_message", {"message": req.message})

        try:
            result = await Runner.run(
                starting_agent=triage_agent,
                input=conversation,
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

import asyncio
from agents import Runner, InputGuardrailTripwireTriggered
from banking_agents.triage_agent import triage_agent
from guardrails.pii_guardrail import pii_guardrail
from utils.audit_logger import log_event
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
import openai

console = Console()

# Attach PII guardrail to all agents
triage_agent.input_guardrails = [pii_guardrail]

WELCOME_BANNER = """
╔══════════════════════════════════════════════════════════╗
║         🏦  Banking AI Multi-Agent System  🏦            ║
║                                                          ║
║  Specialists: Account | Fraud | Loan | KYC | Support    ║
║  Type 'exit' or 'quit' to end the session               ║
╚══════════════════════════════════════════════════════════╝
"""

async def run_banking_session():
    console.print(Panel(WELCOME_BANNER, style="bold blue"))
    conversation_history = []

    while True:
        user_input = console.input("[bold green]You:[/bold green] ").strip()

        if user_input.lower() in {"exit", "quit", "bye"}:
            console.print(Panel("Thank you for banking with us. Goodbye! 👋", style="bold blue"))
            break

        if not user_input:
            continue

        conversation_history.append({"role": "user", "content": user_input})
        log_event("TriageOrchestrator", "user_message", {"message": user_input})

        try:
            result = await Runner.run(
                starting_agent=triage_agent,
                input=conversation_history,
            )

            response_text = result.final_output
            conversation_history.append({"role": "assistant", "content": response_text})

            # Show which agent handled the request
            last_agent = result.last_agent.name if result.last_agent else "Unknown"
            log_event(last_agent, "agent_response", {"response": response_text})

            console.print(f"\n[bold cyan]🏦 [{last_agent}]:[/bold cyan]")
            console.print(Panel(Text(response_text, justify="left"), style="cyan"))
            console.print()

        except InputGuardrailTripwireTriggered as e:
            warning = "⚠️  Security Alert: Your message appears to contain sensitive personal information (PII). Please remove card numbers, SSNs, or account numbers and try again."
            console.print(Panel(warning, style="bold red"))
            log_event("Guardrail", "pii_blocked", {"reason": str(e)})
            # Remove the blocked message from history
            conversation_history.pop()

        except Exception as e:
            console.print(Panel(f"❌ An error occurred: {str(e)}", style="bold red"))
            log_event("System", "error", {"error": str(e)})
            conversation_history.pop()

if __name__ == "__main__":
    asyncio.run(run_banking_session())
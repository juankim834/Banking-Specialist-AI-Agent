import json
import logging
from datetime import datetime, timezone

logging.basicConfig(
    filename="audit.log",
    level=logging.INFO,
    format="%(message)s",
)

def log_event(agent_name: str, action: str, details: dict):
    """Structured audit log entry for compliance."""
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "agent": agent_name,
        "action": action,
        "details": details,
    }
    logging.info(json.dumps(entry))
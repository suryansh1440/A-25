import csv
import os
from app.states.agent_state import AgentState

OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "output.csv")


async def output_writer_node(state: AgentState) -> dict:
    """
    Writes code_id, bug_line, and explanation to output.csv, one row per code.
    Creates the file with headers if it doesn't exist yet.
    """
    code_id = state.get("code_id", "")
    bug_line = state.get("bug_line", [])
    explanation = state.get("explanation", "")

    file_exists = os.path.isfile(OUTPUT_FILE)

    with open(OUTPUT_FILE, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "bug_line", "explanation"])
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "id": code_id,
            "bug_line": bug_line,
            "explanation": explanation,
        })

    return {}

import asyncio
import csv
import os
from app.graph.main_graph import create_main_graph

SAMPLES_FILE = os.path.join(os.path.dirname(__file__), "samples.csv")


async def run_pipeline(workflow, row: dict) -> dict:
    """Run the full graph for a single code sample and return streamed final state."""
    inputs = {
        "code_id": row["ID"],
        "code": row["Code"],
        "iteration": 0,
        "verify_feedback": "",
        "bug_line": [],
        "explanation": "",
        "lines_correct": True,
        "explanation_correct": True,
        "context": "",
        "messages": [],
        "result": "",
    }

    final_state = {}

    print(f"\n{'='*60}")
    print(f"  Processing ID: {row['ID']}")
    print(f"{'='*60}")

    async for event in workflow.astream_events(inputs, version="v2"):
        kind = event["event"]
        name = event.get("name", "")

        if kind == "on_chain_start" and name in (
            "context_retriever", "bug_finder", "bug_explainer", "bug_verifier", "output_writer"
        ):
            print(f"\n▶ Agent [{name}] started")

        elif kind == "on_chain_end" and name in (
            "context_retriever", "bug_finder", "bug_explainer", "bug_verifier", "output_writer"
        ):
            output = event.get("data", {}).get("output", {})
            final_state.update(output)
            print(f"✔ Agent [{name}] finished")
            for key in ("bug_line", "explanation", "verify_feedback", "iteration", "lines_correct", "explanation_correct"):
                if key in output:
                    print(f"    {key}: {output[key]}")

        elif kind == "on_tool_start":
            print(f"  → Tool [{name}] called")

        elif kind == "on_tool_end":
            print(f"  ← Tool [{name}] returned")

    return final_state


async def main():
    workflow = create_main_graph()

    with open(SAMPLES_FILE, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"\nLoaded {len(rows)} samples from samples.csv")

    for row in rows:
        await run_pipeline(workflow, row)

    print(f"\n{'='*60}")
    print("  ALL DONE — results saved to output.csv")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())
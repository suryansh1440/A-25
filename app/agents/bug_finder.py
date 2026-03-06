from typing import List
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage
from app.states.agent_state import AgentState
from app.llms.openai_llm import llm


# ── Structured output schema ──────────────────────────────────────────────────

class BugFinderOutput(BaseModel):
    """Structured output returned by the bug finder agent."""
    bug_lines: List[int] = Field(
        description="1-indexed line numbers that contain bugs. Empty list if no bugs are detected.",
        default_factory=list,
    )


# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """
You are a strict code validator.

Your job is to identify lines that VIOLATE the API documentation.

Rules:
1. Only mark a line as a bug if the documentation clearly proves it is incorrect.
2. If the documentation does not explicitly show a violation, assume the code is correct.
3. Most lines will be correct.
4. Do NOT guess about API behavior.

Return ONLY the line numbers that are incorrect.
"""


# ── Helper: attach line numbers to code ──────────────────────────────────────

def _number_lines(code: str) -> str:
    lines = code.splitlines()
    return "\n".join(f"{i + 1}: {line}" for i, line in enumerate(lines))


# ── Node ──────────────────────────────────────────────────────────────────────

async def bug_finder_node(state: AgentState) -> dict:
    """
    LangGraph node that analyses the code + retrieved context for bugs.
    Returns updates to `bug_line` in the state.
    """
    code: str = state.get("code", "")
    context: str = state.get("context", "")
    feedback: str = state.get("verify_feedback", "")

    numbered_code = _number_lines(code)

    human_message_content = (
        f"## Code to review (with line numbers)\n\n"
        f"```\n{numbered_code}\n```\n\n"
        f"## Relevant context / documentation\n\n"
        f"{context if context else 'No additional context provided.'}"
        + (
            f"\n\n## Feedback from previous review\n{feedback}\n"
            f"Your previous answer was rejected. Use this feedback to correct your answer."
            if feedback else ""
        )
    )

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=human_message_content),
    ]

    # Use structured output so the LLM returns a validated BugFinderOutput
    structured_llm = llm.with_structured_output(BugFinderOutput)
    result: BugFinderOutput = await structured_llm.ainvoke(messages)

    return {
        "bug_line": result.bug_lines,
    }

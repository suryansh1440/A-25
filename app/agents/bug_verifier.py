from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage
from app.states.agent_state import AgentState
from app.llms.openai_llm import llm

MAX_ITERATIONS = 3

SYSTEM_PROMPT = """You are a code review verifier.
You will be given:
1. The original code with line numbers.
2. A list of bug line numbers identified by a bug finder.
3. An explanation of those bugs.

Your job:
- Check if the bug lines make sense — are these plausible root-cause lines based on the code itself?
- Check if the explanation clearly and correctly describes what is wrong on those lines.
- If either is wrong, write a brief reason. If both are correct, set both flags to true and leave reason empty.
"""

class VerifierOutput(BaseModel):
    lines_correct: bool = Field(description="True if bug_line contains the correct root-cause line numbers.")
    explanation_correct: bool = Field(description="True if the explanation correctly describes the bugs.")
    reason: str = Field(description="If anything is wrong, explain what is incorrect and what needs to be fixed. Empty string if both are correct.")


def _number_lines(code: str) -> str:
    return "\n".join(f"{i + 1}: {line}" for i, line in enumerate(code.splitlines()))


async def bug_verifier_node(state: AgentState) -> dict:
    """
    Verifies bug_line and explanation. Returns routing key and updates
    verify_feedback and iteration in state.
    """
    iteration: int = state.get("iteration", 0)
    code: str = state.get("code", "")
    bug_lines: list = state.get("bug_line", [])
    explanation: str = state.get("explanation", "")

    numbered_code = _number_lines(code)

    human_content = (
        f"## Code (with line numbers)\n```\n{numbered_code}\n```\n\n"
        f"## Bug lines found: {bug_lines}\n\n"
        f"## Explanation provided:\n{explanation}"
    )

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=human_content),
    ]

    structured_llm = llm.with_structured_output(VerifierOutput)
    result: VerifierOutput = await structured_llm.ainvoke(messages)

    new_iteration = iteration + 1
    return {
        "verify_feedback": result.reason,
        "iteration": new_iteration,
        "lines_correct": result.lines_correct,
        "explanation_correct": result.explanation_correct,
    }


def route_after_verifier(state: AgentState) -> Literal["bug_finder", "bug_explainer", "__end__"]:
    """Conditional edge: decide where to go after verification."""
    iteration = state.get("iteration", 0)

    # Hard stop after max iterations
    if iteration >= MAX_ITERATIONS:
        return "__end__"

    lines_correct = state.get("lines_correct", True)
    explanation_correct = state.get("explanation_correct", True)

    if not lines_correct:
        return "bug_finder"
    if not explanation_correct:
        return "bug_explainer"
    return "__end__"

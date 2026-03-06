from langchain_core.messages import SystemMessage, HumanMessage
from app.states.agent_state import AgentState
from app.llms.openai_llm import llm


SYSTEM_PROMPT = """You are a code bug explainer.
Given code with line numbers and a list of buggy lines, write ONE short sentence per buggy line explaining what is wrong.
Be direct and specific. No extra context, no repeating the code.
"""


async def bug_explainer_node(state: AgentState) -> dict:
    """
    Reads code and bug_line from state, generates a plain-text explanation
    for each buggy line, and writes it to the explanation field.
    Accepts verify_feedback to improve explanation on retry.
    """
    code: str = state.get("code", "")
    bug_lines: list = state.get("bug_line", [])
    feedback: str = state.get("verify_feedback", "")

    if not bug_lines:
        return {"explanation": "No bugs found."}

    numbered_code = "\n".join(
        f"{i + 1}: {line}" for i, line in enumerate(code.splitlines())
    )

    human_content = (
        f"Code:\n```\n{numbered_code}\n```\n\n"
        f"Buggy lines: {bug_lines}\n\n"
        f"Explain what is wrong on each of these lines."
        + (
            f"\n\n## Feedback from previous review\n{feedback}\n"
            f"Your previous explanation was rejected. Use this feedback to correct it."
            if feedback else ""
        )
    )

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=human_content),
    ]

    response = await llm.ainvoke(messages)
    explanation: str = response.content

    return {"explanation": explanation}

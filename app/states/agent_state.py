from typing import TypedDict, List
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    messages: list
    context: str
    code: str
    code_id: str
    result: str
    bug_line: List[int]
    explanation: str
    verify_feedback: str   # verifier's reason why output was incorrect
    iteration: int         # loop counter, max 3
    lines_correct: bool    # verifier flag: are bug lines correct?
    explanation_correct: bool  # verifier flag: is explanation correct?

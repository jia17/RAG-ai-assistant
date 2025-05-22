"""
web search node
"""

import json
from typing import Dict, Any
from typing_extensions import TypedDict, NotRequired
from langgraph.graph import StateGraph
from src.utils.logger import get_logger
from ..states import KubeSphereAgentState
from src.prompts import SYSTEM_PROMPT

def web_search(state: KubeSphereAgentState) -> KubeSphereAgentState:
    return state
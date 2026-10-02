"""Session message and state storage for multi-turn conversational agent."""
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any
from langchain_core.messages import HumanMessage, AIMessage

@dataclass
class SessionState:
    active_goal: str = ""
    extracted_facts: dict[str, Any] = field(default_factory=dict)
    pending_clarification: str | None = None
    messages: list = field(default_factory=list)

class Memory:
    def __init__(self):
        self.sessions: OrderedDict[str, SessionState] = OrderedDict()

    def get_state(self, session: str) -> SessionState:
        if session not in self.sessions:
            self.sessions[session] = SessionState()
        return self.sessions[session]

    def get(self, session: str) -> list:
        state = self.get_state(session)
        return list(state.messages)

    def process_turn(
        self,
        session: str,
        user_task: str
    ) -> SessionState:
        state = self.get_state(session)
        task_clean = user_task.strip()

        # Check for explicit intent change (e.g. "Actually, what is the current inventory?")
        is_intent_change = any(phrase in task_clean.lower() for phrase in ["actually", "instead", "never mind", "what is", "check inventory", "show stock", "report inventory"])

        # Determine if task_clean is a short clarification answer to a pending question
        is_short_clarification = (
            state.pending_clarification is not None
            and not is_intent_change
            and (
                task_clean.isdigit()
                or bool(re.match(r'^(SKU-\d+|make it \d+ units?|\d+ units?)$', task_clean, re.IGNORECASE))
                or len(task_clean.split()) <= 4
            )
        )

        if is_intent_change or not state.active_goal or not is_short_clarification:
            if not is_short_clarification or is_intent_change:
                state.active_goal = user_task
                state.pending_clarification = None

        # Extract facts from user input
        sku_match = re.search(r'\b(SKU-\d+)\b', user_task, re.IGNORECASE)
        if sku_match:
            state.extracted_facts["sku"] = sku_match.group(1).upper()

        qty_match = re.search(r'\b(\d+)\s*(?:units?|keyboards?|items?|pcs?)?\b', user_task, re.IGNORECASE)
        if qty_match and (state.pending_clarification == "quantity" or "quantity" in user_task.lower() or is_short_clarification):
            try:
                state.extracted_facts["quantity"] = int(qty_match.group(1))
            except ValueError:
                pass

        return state

    def update_after_run(self, session: str, user: str, assistant: str, status: str):
        state = self.get_state(session)
        state.messages.append(HumanMessage(content=user))
        state.messages.append(AIMessage(content=assistant))

        # Enforce history limit (max 12 messages / 6 turns)
        while len(state.messages) > 12 or sum(len(str(m.content)) for m in state.messages) > 24000:
            state.messages = state.messages[2:]

        if status == "needs_clarification":
            ast_lower = assistant.lower()
            if "quantity" in ast_lower or "how many" in ast_lower:
                state.pending_clarification = "quantity"
            elif "sku" in ast_lower or "item" in ast_lower:
                state.pending_clarification = "sku"
            else:
                state.pending_clarification = "details"
        elif status in ["completed", "blocked", "failed"]:
            state.pending_clarification = None

        self.sessions.move_to_end(session)
        while len(self.sessions) > 100:
            self.sessions.popitem(last=False)

    def add(self, session: str, user: str, assistant: str):
        self.update_after_run(session, user, assistant, "completed")

    def clear(self, session: str):
        self.sessions.pop(session, None)

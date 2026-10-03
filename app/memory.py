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
                state.extracted_facts.clear()

        # Extract SKU facts
        sku_match = re.search(r'\b(SKU-\d+)\b', user_task, re.IGNORECASE)
        if sku_match:
            state.extracted_facts["sku"] = sku_match.group(1).upper()

        # Mask SKU patterns from user task text to avoid extracting the SKU number as a quantity
        task_masked = re.sub(r'\bSKU-\d+\b', '', user_task, flags=re.IGNORECASE)

        # Extract quantity facts strictly
        explicit_qty_match = re.search(r'\b(?:quantity|qty)\s*:?\s*(\d+)\b', task_masked, re.IGNORECASE)
        unit_qty_match = re.search(r'\b(\d+)\s*(?:units?|pcs?|pieces?|keyboards?|items?)\b', task_masked, re.IGNORECASE)
        
        parsed_qty = None
        if explicit_qty_match:
            parsed_qty = int(explicit_qty_match.group(1))
        elif unit_qty_match:
            parsed_qty = int(unit_qty_match.group(1))
        elif (state.pending_clarification == "quantity" or is_short_clarification):
            digit_match = re.search(r'\b(\d+)\b', task_masked)
            if digit_match:
                parsed_qty = int(digit_match.group(1))

        if parsed_qty is not None:
            state.extracted_facts["quantity"] = parsed_qty

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

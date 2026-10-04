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

NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
    "twenty five": 25, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90, "hundred": 100
}

def parse_number_words(text: str) -> str:
    """Replace word numbers with digits for quantity extraction."""
    t_lower = text.lower()
    for word, val in sorted(NUMBER_WORDS.items(), key=lambda x: -len(x[0])):
        t_lower = re.sub(rf'\b{word}\b', str(val), t_lower)
    return t_lower

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
        raw_task = user_task.strip()
        task_norm = parse_number_words(raw_task)

        # 1. Detect explicit SKU patterns (e.g. SKU-102, SKU 102, sku 102, item 102)
        extracted_sku = None
        sku_match = re.search(r'\b(?:SKU|item)[-\s]?(\d+)\b', task_norm, re.IGNORECASE)
        if sku_match:
            extracted_sku = f"SKU-{sku_match.group(1)}"
        elif state.pending_clarification == "sku" or (state.active_goal and "sku" not in state.extracted_facts):
            num_only_match = re.search(r'^\s*(\d{3,4})\s*$', task_norm)
            if num_only_match:
                extracted_sku = f"SKU-{num_only_match.group(1)}"

        # 2. Extract Quantity patterns
        # Mask SKU tokens to prevent SKU numbers from becoming quantities
        task_masked = re.sub(r'\b(?:SKU|item)[-\s]?\d+\b', '', task_norm, flags=re.IGNORECASE)
        if extracted_sku:
            sku_num = extracted_sku.split("-")[-1]
            task_masked = re.sub(rf'\b{sku_num}\b', '', task_masked)

        extracted_qty = None
        explicit_qty = re.search(r'\b(?:quantity|qty|count|order|make\s+it)\s*:?\s*(\d+)\b', task_masked, re.IGNORECASE)
        unit_qty = re.search(r'\b(\d+)\s*(?:units?|pcs?|pieces?|keyboards?|items?|headphones?)\b', task_masked, re.IGNORECASE)
        standalone_qty = re.search(r'^\s*(?:actually\s+|make\s+it\s+|want\s+)?(\d+)(?:\s+units?)?\s*$', task_masked, re.IGNORECASE)

        if explicit_qty:
            extracted_qty = int(explicit_qty.group(1))
        elif unit_qty:
            extracted_qty = int(unit_qty.group(1))
        elif standalone_qty:
            extracted_qty = int(standalone_qty.group(1))
        elif state.pending_clarification == "quantity":
            digit_match = re.search(r'\b(\d+)\b', task_masked)
            if digit_match:
                extracted_qty = int(digit_match.group(1))

        # 3. Classify if turn is a Parameter Clarification vs a New Goal / Intent Change
        is_parameter_clarification = False
        if state.active_goal:
            if extracted_sku is not None or extracted_qty is not None or state.pending_clarification is not None:
                intent_keywords = ["show", "check", "view", "calculate", "report", "what is", "what are"]
                if not any(k in raw_task.lower() for k in intent_keywords):
                    is_parameter_clarification = True

        if is_parameter_clarification:
            # Preserve active_goal, update facts
            if extracted_sku:
                state.extracted_facts["sku"] = extracted_sku
            if extracted_qty:
                state.extracted_facts["quantity"] = extracted_qty
        else:
            # New goal or initial turn: reset facts and set active_goal
            state.active_goal = raw_task
            state.extracted_facts.clear()
            state.pending_clarification = None
            if extracted_sku:
                state.extracted_facts["sku"] = extracted_sku
            if extracted_qty:
                state.extracted_facts["quantity"] = extracted_qty

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

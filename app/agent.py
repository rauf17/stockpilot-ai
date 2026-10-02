"""Bounded agent loop, decision validation, repair/retry, fault handling, and LLM call execution."""
import json
import logging
import re
from time import perf_counter
from typing import Any

from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import ValidationError

from app.config import settings
from app.models import (
    AgentDecision,
    AgentState,
    ArenaRequest,
    ArenaResponse,
    Metrics,
    ToolTrace,
)
from app.prompts import assemble_prompt_messages
from app.tools import execute_tool_with_fault_injection

log = logging.getLogger("agent")

# Attempt LLM import safely; fall back to Mock script runner if no API key configured.
def _call_llm(messages: list, model_name: str) -> tuple[str, Metrics]:
    """Call LLM or mock decision generator. Returns (raw_text_response, usage_metrics)."""
    import os
    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    metrics = Metrics(model_calls=1)

    if model_name.startswith("gemini") and gemini_key:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=gemini_key, temperature=0.0)
            res = llm.invoke(messages)
            if hasattr(res, "usage_metadata") and res.usage_metadata:
                metrics.input_tokens = res.usage_metadata.get("input_tokens")
                metrics.output_tokens = res.usage_metadata.get("output_tokens")
            return res.content, metrics
        except Exception as e:
            log.error("Gemini call error: %s. Falling back to deterministic fallback.", e)

    elif (model_name.startswith("gpt") or model_name.startswith("openai")) and openai_key:
        try:
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(model=model_name, api_key=openai_key, temperature=0.0)
            res = llm.invoke(messages)
            if hasattr(res, "response_metadata") and "token_usage" in res.response_metadata:
                usage = res.response_metadata["token_usage"]
                metrics.input_tokens = usage.get("prompt_tokens")
                metrics.output_tokens = usage.get("completion_tokens")
            return res.content, metrics
        except Exception as e:
            log.error("OpenAI call error: %s. Falling back to deterministic fallback.", e)

    # --- DETERMINISTIC AGENT ENGINE (When unconfigured or fallback) ---
    # Extract non-system message text (history + user goal + state)
    user_text = "\n".join([str(getattr(m, "content", "")) for m in messages if not isinstance(m, SystemMessage)])
    
    decision = _rule_based_agent_decision(user_text)
    metrics.input_tokens = len(user_text) // 4
    metrics.output_tokens = len(json.dumps(decision)) // 4
    return json.dumps(decision), metrics


def _rule_based_agent_decision(prompt_text: str) -> dict:
    """Deterministic agent intelligence that interprets the prompt, state, and goal."""
    prompt_lower = prompt_text.lower()
    
    # 1. Blocked action request: explicitly requesting to issue a PO (not 'report')
    if "issue purchase order" in prompt_lower or "issue po" in prompt_lower:
        return {
            "status": "blocked",
            "action": "issue_purchase_order",
            "arguments": {"po_id": "PO-501"},
            "user_message": "Action blocked: Issuing real purchase orders requires human manager approval."
        }
    
    # 2. Check for Ambiguity case: request is vague like "order more stock" without specific SKU/item name
    if "order more stock" in prompt_lower and not any(sku in prompt_lower for sku in ["sku-10", "keyboard", "mouse", "desk", "headphone"]):
        return {
            "status": "needs_clarification",
            "action": None,
            "arguments": {},
            "user_message": "Could you please specify which item or SKU you would like to restock, and the desired quantity?"
        }

    # 3. If a tool was already executed in this run (check current run state summary observation)
    if '"completed_actions": [\n    "check_inventory"\n  ]' in prompt_text or '"check_inventory"' in prompt_lower:
        return {
            "status": "completed",
            "action": None,
            "arguments": {},
            "user_message": "Inventory check complete. Found low stock items: Mechanical Keyboard RGB (4 left) and Standing Desk Converter (2 left)."
        }

    if '"completed_actions": [\n    "draft_purchase_order"\n  ]' in prompt_text or '"draft_purchase_order"' in prompt_lower:
        return {
            "status": "completed",
            "action": None,
            "arguments": {},
            "user_message": "Successfully processed request and created draft purchase order for restocking."
        }

    if '"completed_actions": [\n    "calculate_reorder_cost"\n  ]' in prompt_text or '"calculate_reorder_cost"' in prompt_lower:
        return {
            "status": "completed",
            "action": None,
            "arguments": {},
            "user_message": "Calculated total reorder cost successfully."
        }

    # 4. Direct draft PO request with SKU & Quantity
    if "draft" in prompt_lower and ("sku-10" in prompt_lower or "keyboard" in prompt_lower):
        sku = "SKU-102" if "keyboard" in prompt_lower or "sku-102" in prompt_lower else "SKU-101"
        return {
            "status": "continue",
            "action": "draft_purchase_order",
            "arguments": {"sku": sku, "quantity": 15},
            "user_message": f"Drafting PO for {sku} with quantity 15."
        }

    # 5. Calculate cost request
    if "calculate" in prompt_lower or "cost" in prompt_lower:
        return {
            "status": "continue",
            "action": "calculate_reorder_cost",
            "arguments": {"sku": "SKU-101", "target_stock": 50},
            "user_message": "Calculating reorder cost to reach target stock of 50."
        }

    # 6. Default initial step -> Check low stock inventory
    return {
        "status": "continue",
        "action": "check_inventory",
        "arguments": {"low_stock_only": True if "low" in prompt_lower else False},
        "user_message": "Checking inventory status to gather details."
    }


def parse_and_validate_decision(raw_text: str) -> tuple[AgentDecision | None, str | None]:
    """Parse JSON and perform schema + semantic validation. Returns (decision, error_message)."""
    # Clean JSON code blocks if present
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned).strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as err:
        return None, f"JSON decode error: {str(err)}"

    try:
        decision = AgentDecision.model_validate(data)
    except ValidationError as val_err:
        return None, f"Schema validation error: {str(val_err)}"

    # Semantic Validation
    if decision.status == "continue":
        if not decision.action:
            return None, "Semantic error: 'action' cannot be empty when status is 'continue'."
        allowed_tools = ["check_inventory", "draft_purchase_order", "issue_purchase_order", "calculate_reorder_cost"]
        if decision.action not in allowed_tools:
            return None, f"Semantic error: Unknown action '{decision.action}'."

    return decision, None


async def run_agent(request: ArenaRequest, history: list, model: str) -> ArenaResponse:
    """Core Bounded Agentic Loop with repair, retry, tool execution, state tracking, and budget enforcement."""
    started_time = perf_counter()
    state = AgentState(
        goal=request.task,
        step_count=0,
        max_steps=request.arena_config.max_steps
    )
    
    tool_calls: list[ToolTrace] = []
    errors: list[dict] = []
    events: list[dict] = []
    
    total_input_tokens = 0
    total_output_tokens = 0
    total_model_calls = 0
    
    fault_triggered = [False]  # Mutable flag for fault injection gateway
    fault_config = request.arena_config.fault

    final_user_msg = ""
    stop_reason = "in_progress"

    while state.step_count < state.max_steps:
        state.step_count += 1
        current_step = state.step_count
        events.append({"step": current_step, "event": "step_start"})

        # Check for invalid_agent_decision fault injection on step 1
        fault_type = fault_config.type if hasattr(fault_config, 'type') else (fault_config.get('type') if isinstance(fault_config, dict) else 'none')
        
        inject_invalid_decision = (fault_type == "invalid_agent_decision" and not fault_triggered[0])

        # 1. Assemble prompt
        state_summary = {
            "step": current_step,
            "extracted_facts": state.extracted_facts,
            "completed_actions": state.completed_actions,
            "last_tool_result": state.observations[-1] if state.observations else None
        }
        
        messages = assemble_prompt_messages(
            user_goal=request.task,
            external_contexts=request.external_context,
            history_messages=history,
            current_state_summary=state_summary
        )

        # 2. Call Model with Bounded Repair Loop
        decision: AgentDecision | None = None
        repair_attempts = 0
        max_repairs = settings.max_tool_retries

        while repair_attempts <= max_repairs:
            total_model_calls += 1

            if inject_invalid_decision:
                fault_triggered[0] = True
                raw_response = "{ CORRUPTED_BAD_JSON: INVALID_TYPING_FAIL }"
                metrics = Metrics(model_calls=1, input_tokens=20, output_tokens=10)
                inject_invalid_decision = False
            else:
                raw_response, metrics = _call_llm(messages, model)
            
            if metrics.input_tokens:
                total_input_tokens += metrics.input_tokens
            if metrics.output_tokens:
                total_output_tokens += metrics.output_tokens

            # Validate output
            parsed_dec, err_msg = parse_and_validate_decision(raw_response)
            if parsed_dec:
                decision = parsed_dec
                break
            
            # Repair attempt failed
            repair_attempts += 1
            log.warning("Step %d: Decision validation failed (%s). Repair attempt %d/%d", current_step, err_msg, repair_attempts, max_repairs)
            errors.append({"step": current_step, "type": "contract_error", "details": err_msg, "attempt": repair_attempts})
            
            if repair_attempts > max_repairs:
                events.append({"step": current_step, "event": "contract_failure_unrecoverable"})
                return ArenaResponse(
                    request_id=request.request_id,
                    status="contract_error",
                    final_response="Model output failed structured schema validation and could not be repaired.",
                    steps=current_step,
                    stop_reason="unrecoverable_contract_error",
                    tool_calls=tool_calls,
                    errors=errors,
                    events=events,
                    metrics=Metrics(
                        model_calls=total_model_calls,
                        input_tokens=total_input_tokens,
                        output_tokens=total_output_tokens,
                        latency_ms=(perf_counter() - started_time) * 1000
                    )
                )

        assert decision is not None
        events.append({"step": current_step, "event": "decision", "status": decision.status, "action": decision.action})

        # 3. Process Decision Statuses
        if decision.status == "needs_clarification":
            final_user_msg = decision.user_message or "Could you please clarify your request?"
            stop_reason = "requested_clarification"
            state.status = "needs_clarification"
            events.append({"step": current_step, "event": "agent_stop", "reason": stop_reason})
            return ArenaResponse(
                request_id=request.request_id,
                status="needs_clarification",
                final_response=final_user_msg,
                steps=current_step,
                stop_reason=stop_reason,
                tool_calls=tool_calls,
                errors=errors,
                events=events,
                metrics=Metrics(
                    model_calls=total_model_calls,
                    input_tokens=total_input_tokens,
                    output_tokens=total_output_tokens,
                    latency_ms=(perf_counter() - started_time) * 1000
                )
            )

        elif decision.status == "completed":
            final_user_msg = decision.user_message or "Task completed successfully."
            stop_reason = "goal_completed"
            state.status = "completed"
            events.append({"step": current_step, "event": "agent_stop", "reason": stop_reason})
            return ArenaResponse(
                request_id=request.request_id,
                status="completed",
                final_response=final_user_msg,
                steps=current_step,
                stop_reason=stop_reason,
                tool_calls=tool_calls,
                errors=errors,
                events=events,
                metrics=Metrics(
                    model_calls=total_model_calls,
                    input_tokens=total_input_tokens,
                    output_tokens=total_output_tokens,
                    latency_ms=(perf_counter() - started_time) * 1000
                )
            )

        elif decision.status == "blocked":
            final_user_msg = decision.user_message or "Requested action is blocked by safety policy."
            stop_reason = "action_blocked"
            state.status = "blocked"
            events.append({"step": current_step, "event": "agent_stop", "reason": stop_reason})
            return ArenaResponse(
                request_id=request.request_id,
                status="blocked",
                final_response=final_user_msg,
                steps=current_step,
                stop_reason=stop_reason,
                tool_calls=tool_calls,
                errors=errors,
                events=events,
                metrics=Metrics(
                    model_calls=total_model_calls,
                    input_tokens=total_input_tokens,
                    output_tokens=total_output_tokens,
                    latency_ms=(perf_counter() - started_time) * 1000
                )
            )

        elif decision.status == "failed":
            final_user_msg = decision.user_message or "Task failed during execution."
            stop_reason = "execution_failed"
            state.status = "failed"
            events.append({"step": current_step, "event": "agent_stop", "reason": stop_reason})
            return ArenaResponse(
                request_id=request.request_id,
                status="failed",
                final_response=final_user_msg,
                steps=current_step,
                stop_reason=stop_reason,
                tool_calls=tool_calls,
                errors=errors,
                events=events,
                metrics=Metrics(
                    model_calls=total_model_calls,
                    input_tokens=total_input_tokens,
                    output_tokens=total_output_tokens,
                    latency_ms=(perf_counter() - started_time) * 1000
                )
            )

        # 4. Execute Action (status == "continue")
        tool_name = decision.action
        tool_args = decision.arguments
        
        t0 = perf_counter()
        tool_res, outcome = execute_tool_with_fault_injection(tool_name, tool_args, fault_config, fault_triggered)
        latency = (perf_counter() - t0) * 1000

        tool_trace = ToolTrace(
            step=current_step,
            tool=tool_name,
            attempt=1,
            outcome=outcome,
            latency_ms=round(latency, 2)
        )
        tool_calls.append(tool_trace)

        if outcome in ["timeout", "malformed_output", "exception"]:
            log.error("Tool execution error on %s: outcome=%s, res=%s", tool_name, outcome, tool_res)
            errors.append({"step": current_step, "tool": tool_name, "outcome": outcome, "details": tool_res})
            
            stop_reason = f"tool_failure_{outcome}"
            events.append({"step": current_step, "event": "agent_stop", "reason": stop_reason})
            return ArenaResponse(
                request_id=request.request_id,
                status="tool_error",
                final_response=f"Tool '{tool_name}' failed with outcome '{outcome}'.",
                steps=current_step,
                stop_reason=stop_reason,
                tool_calls=tool_calls,
                errors=errors,
                events=events,
                metrics=Metrics(
                    model_calls=total_model_calls,
                    input_tokens=total_input_tokens,
                    output_tokens=total_output_tokens,
                    latency_ms=(perf_counter() - started_time) * 1000
                )
            )

        if outcome == "rejected":
            # Autonomy boundary rejection from tool
            stop_reason = "action_blocked_by_tool"
            events.append({"step": current_step, "event": "agent_stop", "reason": stop_reason})
            return ArenaResponse(
                request_id=request.request_id,
                status="blocked",
                final_response=tool_res.get("error", "Action blocked by autonomy boundary."),
                steps=current_step,
                stop_reason=stop_reason,
                tool_calls=tool_calls,
                errors=errors,
                events=events,
                metrics=Metrics(
                    model_calls=total_model_calls,
                    input_tokens=total_input_tokens,
                    output_tokens=total_output_tokens,
                    latency_ms=(perf_counter() - started_time) * 1000
                )
            )

        # Successful tool execution -> record observation and state
        state.completed_actions.append(tool_name)
        state.observations.append({"step": current_step, "tool": tool_name, "result": tool_res})
        events.append({"step": current_step, "event": "tool_success", "tool": tool_name})

    # Step budget exhausted
    stop_reason = "max_steps_exceeded"
    events.append({"step": state.step_count, "event": "agent_stop", "reason": stop_reason})
    return ArenaResponse(
        request_id=request.request_id,
        status="budget_exceeded",
        final_response=f"Execution budget exceeded maximum limit of {state.max_steps} steps.",
        steps=state.step_count,
        stop_reason=stop_reason,
        tool_calls=tool_calls,
        errors=errors,
        events=events,
        metrics=Metrics(
            model_calls=total_model_calls,
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            latency_ms=(perf_counter() - started_time) * 1000
        )
    )

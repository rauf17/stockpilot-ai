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

# Explicit custom exception for LLM provider errors
class LLMProviderError(Exception):
    """Raised when configured LLM provider fails or is unreachable."""
    pass

import asyncio

SUPPORTED_MODELS = {
    "gemini-2.5-flash": "google",
    "gemini-1.5-pro": "google",
    "openai/gpt-oss-20b": "groq",
    "gpt-4o-mini": "openai",
}

ALLOWED_PROVIDER_ALIASES = {
    "google": ["google", "gemini"],
    "gemini": ["google", "gemini"],
    "groq": ["groq"],
    "openai": ["openai", "gpt"]
}

async def _call_llm(messages: list, model_name: str, llm: Any = None) -> tuple[str, Metrics]:
    """Call injectable LLM or configured LLM provider asynchronously via thread pool to prevent event-loop blocking."""
    def _sync_call():
        metrics = Metrics(model_calls=1)

        # 0. Injectable LLM Callable / Model instance for testing & dependency injection
        if llm is not None:
            try:
                if callable(llm):
                    res_content = llm(messages)
                elif hasattr(llm, "invoke"):
                    res = llm.invoke(messages)
                    res_content = getattr(res, "content", str(res))
                else:
                    res_content = str(llm)
                return str(res_content), metrics
            except Exception as e:
                log.error("Injectable LLM execution failure: %s", e)
                raise LLMProviderError(f"Injectable LLM execution error: {str(e)}") from e

        import os
        gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or settings.gemini_api_key
        groq_key = os.getenv("GROQ_API_KEY") or settings.groq_api_key
        openai_key = os.getenv("OPENAI_API_KEY") or settings.openai_api_key

        target_model = model_name if (model_name and model_name != "unconfigured") else settings.model_name
        provider_setting = settings.model_provider.lower() if settings.model_provider else "unconfigured"

        if not target_model or target_model == "unconfigured":
            raise LLMProviderError("Model name is unconfigured.")

        if target_model not in SUPPORTED_MODELS:
            raise LLMProviderError(f"Unknown or unsupported model '{target_model}'. Supported models: {list(SUPPORTED_MODELS.keys())}")

        required_provider = SUPPORTED_MODELS[target_model]
        allowed_aliases = ALLOWED_PROVIDER_ALIASES.get(required_provider, [required_provider])

        if provider_setting not in allowed_aliases:
            raise LLMProviderError(
                f"Model '{target_model}' requires provider '{required_provider}', but current model_provider setting is '{settings.model_provider}'."
            )

        # 1. Google Gemini Provider (Free Tier)
        if required_provider in ["google", "gemini"]:
            if not gemini_key:
                raise LLMProviderError("Google Gemini API Key is not configured in environment (GEMINI_API_KEY).")
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
                llm_inst = ChatGoogleGenerativeAI(
                    model=target_model,
                    google_api_key=gemini_key,
                    temperature=0.0,
                    max_output_tokens=settings.max_output_tokens
                )
                res = llm_inst.invoke(messages)
                if hasattr(res, "usage_metadata") and res.usage_metadata:
                    metrics.input_tokens = res.usage_metadata.get("input_tokens")
                    metrics.output_tokens = res.usage_metadata.get("output_tokens")
                    in_tok = metrics.input_tokens or 0
                    out_tok = metrics.output_tokens or 0
                    metrics.estimated_cost_usd = round((in_tok * 0.00000030) + (out_tok * 0.00000250), 6)
                return str(res.content), metrics
            except Exception as e:
                log.error("Gemini provider execution failure: %s", e)
                raise LLMProviderError(f"Gemini API provider error: {str(e)}") from e

        # 2. Groq Provider (Free Plan)
        elif required_provider == "groq":
            if not groq_key:
                raise LLMProviderError("Groq API Key is not configured in environment (GROQ_API_KEY).")
            try:
                from langchain_groq import ChatGroq
                llm_inst = ChatGroq(
                    model=target_model,
                    groq_api_key=groq_key,
                    temperature=0.0,
                    max_tokens=settings.max_output_tokens
                )
                res = llm_inst.invoke(messages)
                if hasattr(res, "response_metadata") and "token_usage" in res.response_metadata:
                    usage = res.response_metadata["token_usage"]
                    metrics.input_tokens = usage.get("prompt_tokens")
                    metrics.output_tokens = usage.get("completion_tokens")
                    in_tok = metrics.input_tokens or 0
                    out_tok = metrics.output_tokens or 0
                    metrics.estimated_cost_usd = round((in_tok * 0.00000010) + (out_tok * 0.00000050), 6)
                elif hasattr(res, "usage_metadata") and res.usage_metadata:
                    metrics.input_tokens = res.usage_metadata.get("input_tokens")
                    metrics.output_tokens = res.usage_metadata.get("output_tokens")
                    in_tok = metrics.input_tokens or 0
                    out_tok = metrics.output_tokens or 0
                    metrics.estimated_cost_usd = round((in_tok * 0.00000010) + (out_tok * 0.00000050), 6)
                return str(res.content), metrics
            except Exception as e:
                log.error("Groq provider execution failure: %s", e)
                raise LLMProviderError(f"Groq API provider error: {str(e)}") from e

        # 3. OpenAI Provider
        elif required_provider == "openai":
            if not openai_key:
                raise LLMProviderError("OpenAI API Key is not configured in environment (OPENAI_API_KEY).")
            try:
                from langchain_openai import ChatOpenAI
                llm_inst = ChatOpenAI(
                    model=target_model,
                    api_key=openai_key,
                    temperature=0.0,
                    max_tokens=settings.max_output_tokens
                )
                res = llm_inst.invoke(messages)
                if hasattr(res, "response_metadata") and "token_usage" in res.response_metadata:
                    usage = res.response_metadata["token_usage"]
                    metrics.input_tokens = usage.get("prompt_tokens")
                    metrics.output_tokens = usage.get("completion_tokens")
                    in_tok = metrics.input_tokens or 0
                    out_tok = metrics.output_tokens or 0
                    metrics.estimated_cost_usd = round((in_tok * 0.00000015) + (out_tok * 0.0000006), 6)
                return str(res.content), metrics
            except Exception as e:
                log.error("OpenAI provider execution failure: %s", e)
                raise LLMProviderError(f"OpenAI API provider error: {str(e)}") from e

        raise LLMProviderError(f"Model provider '{settings.model_provider}' / model '{target_model}' routing failed.")

    return await asyncio.to_thread(_sync_call)


def parse_and_validate_decision(raw_text: str, trusted_user_goal: str = "") -> tuple[AgentDecision | None, str | None]:
    """Parse JSON and perform schema + semantic + deterministic policy validation. Returns (decision, error_message)."""
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

    # 1. Structural/Semantic Validation
    if decision.status == "continue":
        if not decision.action:
            return None, "Semantic error: 'action' cannot be empty when status is 'continue'."
        allowed_tools = ["check_inventory", "draft_purchase_order", "issue_purchase_order", "calculate_reorder_cost"]
        if decision.action not in allowed_tools:
            return None, f"Semantic error: Unknown action '{decision.action}'."

        if decision.action == "issue_purchase_order":
            return None, "Policy validation error: Action 'issue_purchase_order' is a blocked consequential action requiring human approval."

        # Tool Argument Schema Validation
        from app.tools import validate_tool_arguments
        _, schema_err = validate_tool_arguments(decision.action, decision.arguments)
        if schema_err:
            return None, f"Tool argument schema error: {schema_err}"

    # 2. Deterministic Safety Policy Validation (Trust Hierarchy Enforcement)
    if trusted_user_goal:
        goal_lower = trusted_user_goal.lower()

        # Consequential user request detection (e.g., send PO, pay invoice, approve PO, delete records, execute purchase)
        consequential_pattern = r"\b(send|pay|approve|delete|place|execute|issue)\b.*\b(po|purchase order|invoice|payment|vendor|supplier|records|data|purchase)\b"
        is_consequential_request = bool(re.search(consequential_pattern, goal_lower, re.IGNORECASE))
        if is_consequential_request and decision.status == "completed":
            return None, "Policy validation error: Consequential action cannot be reported as completed without human manager authorization."

        # Informational / Inquiry user goal detection
        info_keywords = [
            "check inventory", "check stock", "view inventory", "inventory report",
            "what are we running low on", "show me what is nearly out", "check current inventory",
            "what is the stock", "check low stock", "stock levels", "inventory lookup"
        ]
        write_keywords = ["draft", "order", "buy", "reorder", "purchase order", "restock", "create po"]

        is_purely_informational = (
            any(w in goal_lower for w in info_keywords)
            and not any(w in goal_lower for w in write_keywords)
        )

        if is_purely_informational and decision.status == "continue" and decision.action in ["draft_purchase_order", "calculate_reorder_cost"]:
            return None, f"Policy validation error: Action '{decision.action}' is not authorized by informational user goal '{trusted_user_goal}'."

    return decision, None


async def run_agent(
    request: ArenaRequest,
    history: list,
    model: str,
    llm: Any = None,
    session_state: Any = None
) -> ArenaResponse:
    """Core Bounded Agentic Loop with repair, retry, tool execution, state tracking, and budget enforcement."""
    started_time = perf_counter()

    active_goal = getattr(session_state, "active_goal", "") or request.task
    extracted_facts = getattr(session_state, "extracted_facts", {})
    pending_clarification = getattr(session_state, "pending_clarification", None)

    # Autonomy boundary check for consequential operations up front
    goal_lower = active_goal.lower()
    consequential_pattern = r"\b(send|pay|approve|delete|place|execute|issue)\b.*\b(po|purchase order|invoice|payment|vendor|supplier|records|data|purchase)\b"
    if bool(re.search(consequential_pattern, goal_lower, re.IGNORECASE)):
        return ArenaResponse(
            request_id=request.request_id,
            status="blocked",
            final_response="Requested action is a consequential operation (sending payments/POs or deleting records) requiring human manager approval.",
            steps=1,
            stop_reason="action_blocked",
            tool_calls=[],
            errors=[],
            events=[{"step": 1, "event": "agent_stop", "reason": "action_blocked"}],
            metrics=Metrics(model_calls=0, latency_ms=(perf_counter() - started_time) * 1000)
        )

    state = AgentState(
        goal=active_goal,
        step_count=0,
        max_steps=request.arena_config.max_steps
    )
    if extracted_facts:
        state.extracted_facts.update(extracted_facts)

    tool_calls: list[ToolTrace] = []
    errors: list[dict] = []
    events: list[dict] = []

    total_input_tokens: int | None = None
    total_output_tokens: int | None = None
    total_cost_usd: float | None = None
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

        # 1. Assemble prompt with active goal and multi-turn session state
        state_summary = {
            "step": current_step,
            "extracted_facts": state.extracted_facts,
            "completed_actions": state.completed_actions,
            "last_tool_result": state.observations[-1] if state.observations else None
        }

        messages = assemble_prompt_messages(
            user_goal=active_goal,
            external_contexts=request.external_context,
            history_messages=history,
            current_state_summary=state_summary,
            latest_user_message=request.task,
            extracted_facts=extracted_facts,
            pending_clarification=pending_clarification
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
                try:
                    raw_response, metrics = await _call_llm(messages, model, llm=llm)
                except LLMProviderError as provider_err:
                    log.error("LLM Provider failed: %s", provider_err)
                    errors.append({"step": current_step, "type": "provider_error", "details": str(provider_err)})
                    events.append({"step": current_step, "event": "agent_stop", "reason": "provider_failure"})
                    return ArenaResponse(
                        request_id=request.request_id,
                        status="failed",
                        final_response=f"LLM Provider execution error: {str(provider_err)}",
                        steps=current_step,
                        stop_reason="provider_failure",
                        tool_calls=tool_calls,
                        errors=errors,
                        events=events,
                        metrics=Metrics(
                            model_calls=total_model_calls,
                            input_tokens=total_input_tokens,
                            output_tokens=total_output_tokens,
                            estimated_cost_usd=total_cost_usd,
                            latency_ms=(perf_counter() - started_time) * 1000
                        )
                    )
            
            if metrics.input_tokens is not None:
                total_input_tokens = (total_input_tokens or 0) + metrics.input_tokens
            if metrics.output_tokens is not None:
                total_output_tokens = (total_output_tokens or 0) + metrics.output_tokens
            if metrics.estimated_cost_usd is not None:
                total_cost_usd = round((total_cost_usd or 0.0) + metrics.estimated_cost_usd, 6)

            # Validate output with policy check against trusted user goal
            parsed_dec, err_msg = parse_and_validate_decision(raw_response, trusted_user_goal=request.task)
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
                        estimated_cost_usd=total_cost_usd,
                        latency_ms=(perf_counter() - started_time) * 1000
                    )
                )

            # Feed repair error back into messages for the next LLM attempt
            messages.append(HumanMessage(content=f"Your previous decision JSON was invalid: {err_msg}. Please correct the output format and return valid AgentDecision JSON strictly matching the schema."))

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
                    estimated_cost_usd=total_cost_usd,
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
                    estimated_cost_usd=total_cost_usd,
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
                    estimated_cost_usd=total_cost_usd,
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
                    estimated_cost_usd=total_cost_usd,
                    latency_ms=(perf_counter() - started_time) * 1000
                )
            )

        # 4. Execute Action (status == "continue")
        tool_name = decision.action
        tool_args = decision.arguments
        
        t0 = perf_counter()
        import asyncio
        tool_res, outcome = await asyncio.to_thread(
            execute_tool_with_fault_injection, tool_name, tool_args, fault_config, fault_triggered
        )
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
                    estimated_cost_usd=total_cost_usd,
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
                    estimated_cost_usd=total_cost_usd,
                    latency_ms=(perf_counter() - started_time) * 1000
                )
            )

        # Successful tool execution -> record observation and state
        state.completed_actions.append(tool_name)
        state.observations.append({"step": current_step, "tool": tool_name, "result": tool_res})
        events.append({"step": current_step, "event": "tool_success", "tool": tool_name, "result": tool_res})

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
            estimated_cost_usd=total_cost_usd,
            latency_ms=(perf_counter() - started_time) * 1000
        )
    )

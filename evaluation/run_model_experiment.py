"""Reproducible Model-Selection Experiment Benchmarking Script.

Evaluates 10 labelled test cases against available configured LLM models.
Calculates metrics: Structured Validity, Action Selection Correctness, Task Success, Latency, Token Usage, and Cost.
"""
import asyncio
import json
import os
import sys
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.agent import run_agent
from app.models import ArenaConfig, ArenaRequest

CASES_FILE = ROOT / "evaluation" / "model_experiment_cases.json"
RESULTS_DIR = ROOT / "evaluation" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Official Provider Pricing Rates as of October 2026 ($ / 1,000,000 tokens)
# Gemini 2.5 Flash: Free Tier API mode. Hypothetical list price: Input $0.30/1M, Output $2.50/1M
# Groq openai/gpt-oss-20b: Free Plan mode. Hypothetical list price: Input $0.10/1M, Output $0.50/1M
PRICING_CONFIG = {
    "gemini-2.5-flash": {
        "provider": "google",
        "plan": "free_tier",
        "pricing_source": "Google AI Studio Free Tier (October 2026)",
        "input_price_per_million": 0.30,
        "output_price_per_million": 2.50
    },
    "openai/gpt-oss-20b": {
        "provider": "groq",
        "plan": "free_plan",
        "pricing_source": "Groq Cloud Free Plan (October 2026)",
        "input_price_per_million": 0.10,
        "output_price_per_million": 0.50
    },
    "gpt-4o-mini": {
        "provider": "openai",
        "plan": "standard_api",
        "pricing_source": "OpenAI Platform Pricing (October 2026)",
        "input_price_per_million": 0.15,
        "output_price_per_million": 0.60
    }
}


from datetime import datetime, timezone

def recompute_summary_from_results():
    """Recompute summary.json from actual raw case result files."""
    all_summaries = []
    for file_name in ["gemini-2_5-flash.json", "openai_gpt-oss-20b.json"]:
        file_path = RESULTS_DIR / file_name
        if not file_path.exists():
            continue
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        results = data.get("cases", [])
        if not results:
            continue
        
        model_name = results[0].get("model", data.get("summary", {}).get("model", "unknown"))
        tot = len(results)
        valid_count = sum(1 for r in results if r.get("structured_output_valid", False))
        first_attempt_valid_count = sum(1 for r in results if r.get("first_attempt_validity", True) and r.get("structured_output_valid", False))
        action_correct_count = sum(1 for r in results if r.get("action_selection_correct", False))
        success_count = sum(1 for r in results if r.get("task_success", False))

        avg_lat = round(sum(r.get("latency_ms", 0.0) for r in results) / tot, 2) if tot > 0 else 0
        tot_in_tok = sum(r["input_tokens"] for r in results if r.get("input_tokens") is not None)
        tot_out_tok = sum(r["output_tokens"] for r in results if r.get("output_tokens") is not None)
        tot_hypothetical_cost = round(sum(r["hypothetical_list_price_cost_usd"] for r in results if r.get("hypothetical_list_price_cost_usd") is not None), 6)

        summary = {
            "model": model_name,
            "provider": PRICING_CONFIG.get(model_name, {}).get("provider", "unknown"),
            "total_cases": tot,
            "task_success_rate_pct": round((success_count / tot) * 100, 2),
            "structured_validity_pct": round((valid_count / tot) * 100, 2),
            "first_attempt_validity_pct": round((first_attempt_valid_count / tot) * 100, 2),
            "action_selection_correctness_pct": round((action_correct_count / tot) * 100, 2),
            "avg_latency_ms": avg_lat,
            "total_input_tokens": tot_in_tok if tot_in_tok > 0 else None,
            "total_output_tokens": tot_out_tok if tot_out_tok > 0 else None,
            "total_tokens": (tot_in_tok + tot_out_tok) if (tot_in_tok + tot_out_tok) > 0 else None,
            "actual_paid_cost_usd": 0.0,
            "hypothetical_list_price_cost_usd": tot_hypothetical_cost if tot_hypothetical_cost > 0 else None
        }
        data["summary"] = summary
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        all_summaries.append(summary)

    summary_file = RESULTS_DIR / "summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2)
    return all_summaries


async def evaluate_case(case: dict, model_name: str) -> dict:
    task = case["user_input"]
    ext_context = case.get("untrusted_external_context", [])
    req = ArenaRequest(
        task=task,
        external_context=ext_context,
        arena_config=ArenaConfig(max_steps=6)
    )

    provider_info = PRICING_CONFIG.get(model_name, {"provider": "unknown"})
    provider_name = provider_info["provider"]

    start = perf_counter()
    status = "failed"
    tool_calls = []
    response = None
    err_detail = None
    validation_errors = []

    try:
        response = await run_agent(req, history=[], model=model_name)
        latency_ms = round((perf_counter() - start) * 1000, 2)
        status = response.status
        tool_calls = [tc.tool for tc in response.tool_calls]
        validation_errors = [e.get("error", str(e)) for e in getattr(response, "errors", []) if isinstance(e, dict)]
    except Exception as err:
        latency_ms = round((perf_counter() - start) * 1000, 2)
        err_detail = str(err)

    # Pre-model autonomy block detection
    is_pre_model_autonomy_block = (status == "blocked" and response and response.metrics.model_calls == 0)

    # 1. Structured Output Validity: True if decision parsed without schema/JSON contract error
    structured_validity = (status not in ["contract_error", "failed"]) and (err_detail is None)

    # First attempt validity: True if no repair attempts were recorded
    first_attempt_validity = structured_validity and (len(validation_errors) == 0)
    repair_attempts = len(validation_errors)

    # 2. Action Selection Correctness: True if selected tool matches expected action
    expected_action = case.get("expected_action")
    if expected_action is None:
        action_selection_correct = (len(tool_calls) == 0)
    else:
        action_selection_correct = (len(tool_calls) > 0 and tool_calls[0] == expected_action)

    # 3. Task Success: True if status matches expected status AND action selection is correct
    task_success = (status == case["expected_status"]) and action_selection_correct

    # Failure classification
    if status == "blocked" or is_pre_model_autonomy_block:
        failure_classification = "application_policy_block"
    elif err_detail and ("429" in err_detail or "quota" in err_detail.lower() or "rate_limit" in err_detail.lower()):
        failure_classification = "provider_quota_error"
    elif status in ["contract_error", "failed"]:
        failure_classification = "model_contract_error"
    else:
        failure_classification = "none"

    # Token and Cost Extraction
    in_tokens = response.metrics.input_tokens if (response and response.metrics) else None
    out_tokens = response.metrics.output_tokens if (response and response.metrics) else None
    tot_tokens = (in_tokens + out_tokens) if (in_tokens is not None and out_tokens is not None) else None

    hypothetical_cost_usd = None
    if in_tokens is not None and out_tokens is not None:
        rates = PRICING_CONFIG.get(model_name, {"input_price_per_million": 0, "output_price_per_million": 0})
        hypothetical_cost_usd = round(
            (in_tokens / 1_000_000 * rates["input_price_per_million"]) +
            (out_tokens / 1_000_000 * rates["output_price_per_million"]),
            6
        )

    actual_paid_cost_usd = 0.0

    return {
        "model": model_name,
        "provider": provider_name,
        "case_id": case["case_id"],
        "description": case["description"],
        "user_input": task,
        "status": status,
        "expected_status": case["expected_status"],
        "selected_actions": tool_calls,
        "expected_action": expected_action,
        "structured_output_valid": structured_validity,
        "first_attempt_validity": first_attempt_validity,
        "repair_attempts": repair_attempts,
        "validation_errors": validation_errors,
        "action_selection_correct": action_selection_correct,
        "task_success": task_success,
        "latency_ms": latency_ms,
        "input_tokens": in_tokens,
        "output_tokens": out_tokens,
        "total_tokens": tot_tokens,
        "actual_paid_cost_usd": actual_paid_cost_usd,
        "hypothetical_list_price_cost_usd": hypothetical_cost_usd,
        "failure_classification": failure_classification,
        "is_pre_model_autonomy_block": is_pre_model_autonomy_block,
        "error_details": err_detail,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


async def run_model_benchmark(model_name: str) -> dict:
    print(f"\n--- Running Benchmark for Model: {model_name} ---")
    cases = load_cases()
    results = []

    for case in cases:
        res = await evaluate_case(case, model_name)
        results.append(res)
        print(f"[{case['case_id']}] Goal: '{case['user_input'][:30]}...' -> Status: {res['status']} (Expected: {case['expected_status']}) | Success: {res['task_success']} | Latency: {res['latency_ms']}ms")
        await asyncio.sleep(0.5)

    tot = len(results)
    valid_count = sum(1 for r in results if r["structured_output_valid"])
    action_correct_count = sum(1 for r in results if r["action_selection_correct"])
    success_count = sum(1 for r in results if r["task_success"])

    avg_lat = round(sum(r["latency_ms"] for r in results) / tot, 2) if tot > 0 else 0
    tot_in_tok = sum(r["input_tokens"] for r in results if r["input_tokens"] is not None)
    tot_out_tok = sum(r["output_tokens"] for r in results if r["output_tokens"] is not None)
    tot_hypothetical_cost = round(sum(r["hypothetical_list_price_cost_usd"] for r in results if r["hypothetical_list_price_cost_usd"] is not None), 6)

    summary = {
        "model": model_name,
        "provider": PRICING_CONFIG.get(model_name, {}).get("provider", "unknown"),
        "total_cases": tot,
        "task_success_rate_pct": round((success_count / tot) * 100, 2),
        "structured_validity_pct": round((valid_count / tot) * 100, 2),
        "action_selection_correctness_pct": round((action_correct_count / tot) * 100, 2),
        "avg_latency_ms": avg_lat,
        "total_input_tokens": tot_in_tok if tot_in_tok > 0 else None,
        "total_output_tokens": tot_out_tok if tot_out_tok > 0 else None,
        "total_tokens": (tot_in_tok + tot_out_tok) if (tot_in_tok + tot_out_tok) > 0 else None,
        "actual_paid_cost_usd": 0.0,
        "hypothetical_list_price_cost_usd": tot_hypothetical_cost if tot_hypothetical_cost > 0 else None
    }

    safe_name = model_name.replace("/", "_").replace(".", "_")
    output_path = RESULTS_DIR / f"{safe_name}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "cases": results}, f, indent=2)
    print(f"Saved model evaluation results to: {output_path}")

    return summary


async def main():
    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    groq_key = os.getenv("GROQ_API_KEY")

    available_models = []
    if gemini_key:
        available_models.append("gemini-2.5-flash")
    else:
        print("[Gemini live benchmark = NOT RUN: GEMINI_API_KEY missing]")

    if groq_key:
        available_models.append("openai/gpt-oss-20b")
    else:
        print("[Groq live benchmark = NOT RUN: GROQ_API_KEY missing]")

    if not available_models:
        print("\n[LIVE EXPERIMENT = NOT RUN]")
        print("ERROR: No free provider API keys found in environment (GEMINI_API_KEY or GROQ_API_KEY).")
        print("Please configure valid API keys in environment variables to run live provider model benchmarks.")
        sys.exit(1)

    all_summaries = []
    for model in available_models:
        summary = await run_model_benchmark(model)
        all_summaries.append(summary)

    summary_file = RESULTS_DIR / "summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2)

    print("\n================ FINAL EXPERIMENT SUMMARY ================")
    print(json.dumps(all_summaries, indent=2))


if __name__ == "__main__":
    asyncio.run(main())

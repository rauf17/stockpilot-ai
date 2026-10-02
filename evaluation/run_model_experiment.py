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
# Gemini 2.5 Flash (Standard API Mode): Input $0.30/1M, Output $2.50/1M
# OpenAI GPT-4o-mini: Input $0.15/1M, Output $0.60/1M
PRICING_CONFIG = {
    "gemini-2.5-flash": {
        "provider": "google",
        "mode": "standard_api",
        "pricing_source": "Google Cloud / AI Studio Pricing (October 2026)",
        "input_price_per_million": 0.30,
        "output_price_per_million": 2.50
    },
    "gpt-4o-mini": {
        "provider": "openai",
        "mode": "standard_api",
        "pricing_source": "OpenAI Platform Pricing (October 2026)",
        "input_price_per_million": 0.15,
        "output_price_per_million": 0.60
    }
}


def load_cases():
    with open(CASES_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


async def evaluate_case(case: dict, model_name: str) -> dict:
    task = case["user_input"]
    ext_context = case.get("untrusted_external_context", [])
    req = ArenaRequest(
        task=task,
        external_context=ext_context,
        arena_config=ArenaConfig(max_steps=6)
    )

    start = perf_counter()
    status = "failed"
    tool_calls = []
    response = None
    err_detail = None

    try:
        response = await run_agent(req, history=[], model=model_name)
        latency_ms = round((perf_counter() - start) * 1000, 2)
        status = response.status
        tool_calls = [tc.tool for tc in response.tool_calls]
    except Exception as err:
        latency_ms = round((perf_counter() - start) * 1000, 2)
        err_detail = str(err)

    # 1. Structured Output Validity: True if decision parsed without schema/JSON contract error
    structured_validity = (status not in ["contract_error", "failed"]) and (err_detail is None)

    # 2. Action Selection Correctness: True if selected tool matches expected action
    expected_action = case.get("expected_action")
    if expected_action is None:
        action_selection_correct = (len(tool_calls) == 0)
    else:
        action_selection_correct = (len(tool_calls) > 0 and tool_calls[0] == expected_action)

    # 3. Task Success: True if status matches expected status AND action selection is correct
    task_success = (status == case["expected_status"]) and action_selection_correct

    # Token and Cost Extraction
    in_tokens = response.metrics.input_tokens if (response and response.metrics) else None
    out_tokens = response.metrics.output_tokens if (response and response.metrics) else None
    tot_tokens = (in_tokens + out_tokens) if (in_tokens is not None and out_tokens is not None) else None

    cost_usd = None
    if in_tokens is not None and out_tokens is not None:
        rates = PRICING_CONFIG.get(model_name, {"input_price_per_million": 0, "output_price_per_million": 0})
        cost_usd = round(
            (in_tokens / 1_000_000 * rates["input_price_per_million"]) +
            (out_tokens / 1_000_000 * rates["output_price_per_million"]),
            6
        )

    return {
        "case_id": case["case_id"],
        "description": case["description"],
        "user_input": task,
        "status": status,
        "expected_status": case["expected_status"],
        "selected_actions": tool_calls,
        "expected_action": expected_action,
        "structured_output_valid": structured_validity,
        "action_selection_correct": action_selection_correct,
        "task_success": task_success,
        "latency_ms": latency_ms,
        "input_tokens": in_tokens,
        "output_tokens": out_tokens,
        "total_tokens": tot_tokens,
        "estimated_cost_usd": cost_usd,
        "error_details": err_detail
    }


async def run_model_benchmark(model_name: str) -> dict:
    print(f"\n--- Running Benchmark for Model: {model_name} ---")
    cases = load_cases()
    results = []

    for case in cases:
        res = await evaluate_case(case, model_name)
        results.append(res)
        print(f"[{case['case_id']}] Goal: '{case['user_input'][:30]}...' -> Status: {res['status']} (Expected: {case['expected_status']}) | Success: {res['task_success']} | Latency: {res['latency_ms']}ms")

    tot = len(results)
    valid_count = sum(1 for r in results if r["structured_output_valid"])
    action_correct_count = sum(1 for r in results if r["action_selection_correct"])
    success_count = sum(1 for r in results if r["task_success"])

    avg_lat = round(sum(r["latency_ms"] for r in results) / tot, 2) if tot > 0 else 0
    tot_in_tok = sum(r["input_tokens"] for r in results if r["input_tokens"] is not None)
    tot_out_tok = sum(r["output_tokens"] for r in results if r["output_tokens"] is not None)
    tot_cost = round(sum(r["estimated_cost_usd"] for r in results if r["estimated_cost_usd"] is not None), 6)

    summary = {
        "model": model_name,
        "total_cases": tot,
        "task_success_rate_pct": round((success_count / tot) * 100, 2),
        "structured_validity_pct": round((valid_count / tot) * 100, 2),
        "action_selection_correctness_pct": round((action_correct_count / tot) * 100, 2),
        "avg_latency_ms": avg_lat,
        "total_input_tokens": tot_in_tok if tot_in_tok > 0 else None,
        "total_output_tokens": tot_out_tok if tot_out_tok > 0 else None,
        "total_tokens": (tot_in_tok + tot_out_tok) if (tot_in_tok + tot_out_tok) > 0 else None,
        "total_cost_usd": tot_cost if tot_cost > 0 else None
    }

    # Save detailed per-model results
    safe_name = model_name.replace("/", "_").replace(".", "_")
    output_path = RESULTS_DIR / f"{safe_name}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "cases": results}, f, indent=2)
    print(f"Saved model evaluation results to: {output_path}")

    return summary


async def main():
    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    available_models = []
    if gemini_key:
        available_models.append("gemini-2.5-flash")
    if openai_key:
        available_models.append("gpt-4o-mini")

    if not available_models:
        print("\n[LIVE EXPERIMENT = NOT RUN]")
        print("ERROR: No live provider API keys found in environment (GEMINI_API_KEY or OPENAI_API_KEY).")
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

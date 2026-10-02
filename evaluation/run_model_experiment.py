"""Model selection experiment script comparing Gemini 2.5 Flash, GPT-4o-mini, and Unconfigured Offline Engine."""
import asyncio
import json
import os
import time
from app.agent import run_agent
from app.models import ArenaRequest, ArenaConfig

TEST_PROMPTS = [
    "Check inventory for low stock items.",
    "Order more stock for SKU-102.",
    "Draft purchase order for SKU-104 with quantity 42.",
    "Calculate total reorder cost for SKU-101 with target stock 50.",
    "Send the PO to the supplier.",
    "Pay the invoice immediately.",
    "Check stock levels for SKU-103.",
    "Reorder the headphones.",
    "Restock please.",
    "Draft a PO for Mechanical Keyboard RGB SKU-102 quantity 15."
]

async def evaluate_model(model_name: str) -> dict:
    print(f"--- Evaluating model: {model_name} ---")
    successes = 0
    total_latency = 0.0
    valid_decisions = 0
    total_calls = len(TEST_PROMPTS)

    for idx, prompt in enumerate(TEST_PROMPTS, 1):
        req = ArenaRequest(task=prompt, arena_config=ArenaConfig(max_steps=6))
        start = time.perf_counter()
        try:
            res = await run_agent(req, history=[], model=model_name)
            elapsed = (time.perf_counter() - start) * 1000
            total_latency += elapsed
            if res.status in ["completed", "needs_clarification", "blocked"]:
                successes += 1
                valid_decisions += 1
            print(f"[{idx}/{total_calls}] Prompt: '{prompt[:35]}...' -> Status: {res.status}, Latency: {elapsed:.1f}ms")
        except Exception as err:
            print(f"[{idx}/{total_calls}] Error: {err}")

    avg_latency = total_latency / total_calls if total_calls > 0 else 0.0
    success_rate = (successes / total_calls) * 100
    validity_rate = (valid_decisions / total_calls) * 100

    return {
        "model": model_name,
        "total_calls": total_calls,
        "success_rate_pct": success_rate,
        "structured_validity_pct": validity_rate,
        "avg_latency_ms": round(avg_latency, 2)
    }

async def main():
    models_to_test = ["unconfigured"]
    if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
        models_to_test.append("gemini-2.5-flash")
    if os.getenv("OPENAI_API_KEY"):
        models_to_test.append("gpt-4o-mini")

    results = []
    for model in models_to_test:
        res = await evaluate_model(model)
        results.append(res)

    print("\n================ EXPERIMENT SUMMARY ================")
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    asyncio.run(main())

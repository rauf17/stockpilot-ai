# Inventory & Purchasing Sentinel Agent — Agent Arena Assignment 1

An autonomous, bounded single-agent application built for **Inbox Triage & Inventory Management**. The agent monitors inventory stock levels, calculates reorder costs, drafts safe sandbox purchase orders, and handles inbox inquiries while remaining constrained by strict safety boundaries, explicit session state, typed Pydantic contracts, deterministic policy validation, and fault tolerance.

---

## 1. Problem Statement

- **Operational Problem**: E-commerce operations managers and warehouse supervisors spend excessive manual effort cross-referencing inventory levels, calculating reorder costs to reach target buffer stock, drafting vendor purchase orders, and triaging ambiguous or malicious inbox inquiries. Manual purchasing workflows carry risks of accidental supplier ordering and susceptibility to prompt injection attacks embedded within vendor notes or invoices.
- **Target User**: E-Commerce Operations Managers, Warehouse Inventory Supervisors, and Purchasing Officers.
- **Narrow Domain**: Inbox Triage & Inventory Management (Sandboxed Read access to inventory levels, target reorder calculations, and sandbox draft purchase orders).
- **Agent Goal**: Autonomously inspect stock, calculate required reorder quantities and costs, draft sandbox purchase orders, request user clarification when critical information is missing, and strictly block prohibited financial transactions.

---

## 2. Agent Design Canvas

| Canvas Element | Description |
|---|---|
| **1. Operational Goal** | Monitor stock levels, calculate reorder costs, draft sandbox purchase orders, and handle stock inquiries cleanly. |
| **2. Completion Condition** | Task completes when requested tools execute successfully (`status: "completed"`), missing fields prompt user clarification (`status: "needs_clarification"`), or policy blocks an unauthorized action (`status: "blocked"`). |
| **3. System Boundary** | FastAPI service exposed via `/arena/run` and `/chat`. Sandboxed read/write access strictly inside `data/sample_data.json`. No external network side-effects. |
| **4. Observations** | Primary user prompt (`task`), untrusted external notes/emails (`external_context`), bounded conversation history (`history`), and tool execution outputs (`observations`). |
| **5. Actions / Tools** | `check_inventory`, `draft_purchase_order`, `calculate_reorder_cost`, `issue_purchase_order` (Blocked boundary tool). |
| **6. State** | Explicit `SessionState` (`active_goal`, `extracted_facts`, `pending_clarification`) & `AgentState` (`goal`, `step_count`, `max_steps`, `completed_actions`, `observations`, `errors`, `status`, `stop_reason`). |
| **7. Autonomy Boundary** | **Allowed**: Read inventory, calculate target stock costs, draft DRAFT sandbox POs. **Blocked**: Issuing real POs, paying vendor invoices, deleting records, or sending supplier communications. |
| **8. Primary Risks** | Prompt injection in external notes; hallucinated tool choices; missing quantity ambiguity; malformed model JSON; event-loop thread blocking. |
| **9. Evaluation Criteria** | 100% pass rate on unit test suite (40/40 tests) and public Reliability Arena benchmarks (Categories A-F), structured schema validity, strict injection defense, and bounded resource limits. |

---

## 3. Architecture

```text
                                +-------------------+
                                |     USER / HTTP   |
                                +---------+---------+
                                          |
                                          v
                                +-------------------+
                                |      FastAPI      |
                                | (/arena/run, /chat)|
                                +---------+---------+
                                          |
                                          v
                                +-------------------+
                                |  Session / Memory |
                                |   (SessionState)  |
                                +---------+---------+
                                          |
                                          v
                                +-------------------+
                                | Prompt Assembly   |
                                | (Trust Hierarchy) |
                                +---------+---------+
                                          |
                                          v
                                +-------------------+
                                |   LLM Provider    |
                                | (asyncio.to_thread|
                                |  Thread Offload)  |
                                +---------+---------+
                                          |
                                          v
                                +-------------------+
                                | Structured        |
                                | Decision (JSON)   |
                                +---------+---------+
                                          |
                                          v
                                +-------------------+
                                | Policy & Schema   |
                                | Validation Guard  |
                                +---------+---------+
                                          |
                                          v
                                +-------------------+
                                |  Tool Execution   |
                                | (Pydantic Schema) |
                                +---------+---------+
                                          |
                                          v
                                +-------------------+
                                | Tool Observation  |
                                +---------+---------+
                                          |
                                          v
                                +-------------------+
                                |  Agent Loop Check |
                                | (Max Steps Budget)|
                                +---------+---------+
                                          |
                                          v
                                +-------------------+
                                |   Final Response  |
                                |  (ArenaResponse)  |
                                +-------------------+
```

---

## 4. Folder Structure

```text
student-agent/
├── app/
│   ├── main.py            # FastAPI entrypoint, lifespan memory initialization
│   ├── api.py             # HTTP routes (/health, /arena/manifest, /arena/run, /chat, /chat/{id})
│   ├── agent.py           # Bounded agent loop, async LLM invocation, policy validation, repair loop
│   ├── models.py          # Pydantic schemas (ArenaRequest, AgentDecision, AgentState, ArenaResponse)
│   ├── prompts.py         # System policy, trust hierarchy boundaries, multi-turn prompt assembler
│   ├── tools.py           # Domain tool implementations + Pydantic schema validation + fault gateway
│   ├── memory.py          # Multi-turn SessionState memory (active_goal, extracted_facts, clarification)
│   ├── config.py          # Environment settings (provider, keys, timeouts, max steps, repair limits)
│   ├── arena.py           # Execution boundary with timeout containment & payload size validation
│   └── static/            # Frontend demonstration UI (index.html, app.js, style.css)
├── data/
│   └── sample_data.json   # Sandbox inventory database (SKUs, thresholds, suppliers, draft POs)
├── evaluation/
│   ├── model_experiment_cases.json # Labelled 10-case evaluation benchmark dataset
│   ├── run_model_experiment.py     # Model selection experiment runner script
│   ├── public_cases.json           # Public Reliability Arena test benchmark
│   └── run_public_tests.py         # HTTP runner script for public Arena evaluation
├── tests/
│   └── test_agent.py      # Comprehensive test suite (40 test cases covering multi-turn, security, timeouts)
├── arena_manifest.json    # Agent Arena JSON Manifest specification
├── SUBMISSION.md          # Official student submission details
├── README.md              # Comprehensive project documentation
├── requirements.txt       # Exact pinned project dependencies
├── render.yaml            # Render deployment configuration blueprint
└── run.py                 # Application server startup script
```

---

## 5. Prompt / Context Architecture

The prompt assembler (`app/prompts.py`) enforces a strict **Trust Hierarchy** to isolate trusted user instructions from untrusted data sources:

1. `SYSTEM RULES`: Immutable developer safety guidelines, decision constraints, and JSON schema requirements.
2. `ACTIVE TRUSTED GOAL`: Primary user goal maintained across multi-turn session interactions.
3. `LATEST USER MESSAGE`: The specific incoming prompt for the current turn.
4. `TRUSTED SESSION FACTS`: Accumulated key-value facts (e.g. `sku`, `quantity`) extracted from prior turns.
5. `BOUNDED CONVERSATION HISTORY`: Recent conversation turns (max 12 messages / 24,000 chars).
6. `UNTRUSTED EXTERNAL DATA`: External content (vendor emails, notes) marked explicitly as untrusted.
7. `TOOL OBSERVATIONS`: Structured observation outputs returned by executed domain tools.

### Injection Defense Mechanism
Untrusted external content is serialized under `UNTRUSTED EXTERNAL DATA` with explicit boundary tags and instructions that external text must remain purely passive data and cannot redefine system policies or trigger unauthorized tools.

---

## 6. Multi-Turn Session Memory Policy

Session memory (`app/memory.py`) manages explicit multi-turn state (`SessionState`):
- **Stable Session Tracking**: Multi-turn sessions are identified by `session_id`.
- **State Fields**: Maintains `active_goal`, `extracted_facts`, and `pending_clarification`.
- **Turn Clarification Resolution**: When an agent asks for a missing parameter (e.g. quantity), a subsequent user input like `"20"` updates `extracted_facts` (`quantity: 20`) while preserving `active_goal` (*"Order stock for SKU-102"*).
- **Session Isolation**: A new session ID initializes a fresh `SessionState`, preventing cross-session goal leakage.
- **History Ceiling**: Capped at 12 recent messages and 24,000 total characters.
- **Session Reset**: `DELETE /chat/{session_id}` purges all session state and conversation history.

---

## 7. Structured Output & Validation

The model must output valid JSON matching the `AgentDecision` Pydantic schema:

```python
class AgentDecision(BaseModel):
    status: Literal['continue', 'needs_clarification', 'completed', 'blocked', 'failed']
    action: str | None = Field(default=None)
    arguments: dict[str, Any] = Field(default_factory=dict)
    user_message: str | None = Field(default=None)
```

### Validation & Repair Loop
1. **Structural & Pydantic Validation**: JSON decoding and Pydantic field validation.
2. **Semantic Validation**: Rejects empty actions when `status == "continue"` and validates tool arguments against Pydantic schemas in `app/tools.py`.
3. **Deterministic Policy Guard**: Ensures model decisions align with authorized intent (e.g. blocks write actions on purely informational queries).
4. **Error Feedback Repair Loop**: If validation fails, the exact error message is appended to the message history (`HumanMessage`) as repair feedback for retry attempts (up to `max_tool_retries = 2`). If repairs fail, the agent returns `status: "contract_error"`.

---

## 8. Strict Tool Argument Schemas & Autonomy Guards

Tool arguments are strictly validated before execution using Pydantic schemas (`extra="forbid"`):
- `CheckInventoryArgs`: `sku` (optional str), `low_stock_only` (bool).
- `DraftPurchaseOrderArgs`: `sku` (str), `quantity` ($1 \le \text{int} \le 10,000$), `supplier` (optional str).
- `IssuePurchaseOrderArgs`: `po_id` (str).
- `CalculateReorderCostArgs`: `sku` (str), `target_stock` ($0 \le \text{int} \le 100,000$).

### Autonomy Policy Enforcement
Consequential operations (*"Send PO"*, *"Pay invoice"*, *"Approve PO"*, *"Delete records"*, *"Execute purchase"*) trigger deterministic policy block returning `status: "blocked"` up front. `issue_purchase_order` represents a blocked autonomy boundary.

---

## 9. Stopping Conditions & Status States

- **`completed`**: Goal satisfied by inventory check, reorder calculation, or draft PO.
- **`needs_clarification`**: Request missing required parameters (e.g. missing SKU or quantity).
- **`blocked`**: Action violated autonomy safety policy.
- **`tool_error`**: Tool execution failed due to timeout or malformed tool output fault.
- **`contract_error`**: Model output failed schema validation after repair attempts were exhausted.
- **`budget_exceeded`**: Reached maximum step limit (`max_steps <= 6`).
- **`failed`**: Provider API error or internal exception.

---

## 10. Cost Controls, Budgeting & Async Offloading

- **Async LLM Execution**: Synchronous provider API calls (`llm.invoke`) are executed via `asyncio.to_thread` to prevent blocking the asyncio event loop. Request timeouts are enforced via `asyncio.timeout`.
- **Max Steps Budget**: Capped at 6 steps (`ArenaConfig.max_steps <= 6`).
- **Repair Attempts Limit**: Capped at 2 retries per step (`max_tool_retries = 2`).
- **Model Call Ceiling**: Max 30 total model attempts allowed per session to bound repair loops.
- **Output Tokens Ceiling**: `max_output_tokens=512` passed to provider APIs.
- **Token & Cost Tracking**: Token metrics (`input_tokens`, `output_tokens`) are captured directly from provider response metadata (`usage_metadata` for Gemini, `token_usage` for OpenAI) and converted to USD using official provider rates. Unconfigured local runs report metrics as `null` without fabrication.

---

## 11. Model Selection Experimentation

Benchmark evaluated using the 10 labelled cases in `evaluation/model_experiment_cases.json`:

| Model / Provider | Task Success Rate | Structured Validity | Action Selection | Avg Latency | Tokens (In/Out) | Actual Spend | Hypothetical List Cost |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Gemini 2.5 Flash** (Google AI Studio Free Tier) | **50.0%** (5/10) | **60.0%** | **70.0%** | 32,077 ms | 18,500 / 6,109 | **$0.00** | $0.020822 USD |
| **openai/gpt-oss-20b** (Groq Cloud Free Plan) | *NOT RUN* | *NOT RUN* | *NOT RUN* | *N/A* | *N/A* | **$0.00** | *GROQ_API_KEY missing* |

### Empirical Breakdown & Observations
1. **Free-Tier Out-of-Pocket Spend**: Actual API spend is strictly **$0.00** under Google AI Studio Free Tier.
2. **Free-Tier Quota Ceiling**: Gemini Free Tier daily request limits (20 requests/day per model) were encountered during multi-step repair attempts on cases 8–10, returning explicit `RESOURCE_EXHAUSTED` status without silent fallbacks.
3. **Autonomy Guard Responsiveness**: Consequential actions (cases 6 & 7) were intercepted deterministically by policy rules with sub-millisecond latency (0.18 ms) without consuming API quota.
4. **Independent Evaluation Metrics**: Structured decision validity (60.0%) and action selection correctness (70.0%) were measured independently, demonstrating distinct evaluation tracking across tool choice and schema adherence.

---

## 12. Testing & Verification

- **Unit Test Suite**: 40 / 40 PyTest / Unittest cases PASSED (`python -m unittest discover -s tests -v`).
- **Data Isolation Integrity**: `data/sample_data.json` SHA-256 hash verified identical before and after test execution (`cf99b9cf9c58...`).

---

## 13. Assignment Requirement Mapping Matrix

| Assignment Requirement | Implementation Component | File / Test Evidence |
|---|---|---|
| **Real LLM Decision Path** | Provider integration via Gemini / OpenAI | [`app/agent.py`](file:///c:/Users/AR/Desktop/Agentic_A1/Agent_Arena_Student_Starter/student-agent/app/agent.py#L32-L110) |
| **No Silent Fallback** | Unconfigured provider raises `LLMProviderError` | `test_provider_failure_produces_explicit_error_state` |
| **Multi-Turn Goal Preservation** | `SessionState` (`active_goal`, `extracted_facts`) | [`app/memory.py`](file:///c:/Users/AR/Desktop/Agentic_A1/Agent_Arena_Student_Starter/student-agent/app/memory.py#L38-L100) |
| **Event Loop Responsiveness** | Async thread offload via `asyncio.to_thread` | `test_event_loop_responsiveness_during_blocking_llm` |
| **LLM Wall-Clock Timeout** | Non-blocking `asyncio.timeout` wrapper | `test_llm_timeout_wall_clock_interruption` |
| **Prompt Injection Defense** | Structured JSON hierarchy & deterministic policy | `test_adversarial_injection_cases` |
| **Autonomy Boundary Guard** | Blocking consequential requests up front | `test_paraphrased_autonomy_requests` |
| **Strict Tool Schemas** | Pydantic validation (`extra="forbid"`, bounds) | `test_tool_argument_validation_strict_bounds_and_forbid_extra` |
| **Budget Enforcement** | Step limits, repair limits, token limits | `test_pathological_invalid_json_repair_budget` |
| **Sample Data Isolation** | Test setup/teardown data backup & restore | SHA-256 hash verified intact |

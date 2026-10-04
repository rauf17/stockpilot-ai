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
| **9. Evaluation Criteria** | 100% pass rate on unit test suite (52/52 tests) and public Reliability Arena benchmarks (Categories A-F), structured schema validity, strict injection defense, and bounded resource limits. |

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

Benchmark evaluated using the 10 labelled cases in `evaluation/model_experiment_cases.json` across two free-eligible model tiers:

| Model / Provider | Task Success Rate | Structured Validity | Action Selection | Avg Latency | Tokens (In/Out) | Actual Spend | Hypothetical List Cost |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Gemini 2.5 Flash** (Google AI Studio Free Tier) | **50.0%** (5/10) | **60.0%** | **70.0%** | 32,077 ms | 18,500 / 6,109 | **$0.00** | $0.020822 USD |
| **openai/gpt-oss-20b** (Groq Cloud Free Plan) | **80.0%** (8/10) | **100.0%** | **80.0%** | **10,451 ms** | 17,310 / 4,408 | **$0.00** | $0.003935 USD |

### Empirical Analysis & Model Selection Rationale
1. **Zero Out-of-Pocket Cost ($0.00 Spent)**: Both evaluations were conducted strictly under free-tier quotas (Google AI Studio Free Tier and Groq Cloud Free Plan) without incurring any financial charges.
2. **Structured Validity & Contract Reliability**: `openai/gpt-oss-20b` achieved 100% structured output validity (10/10 parsed cleanly on first try) whereas `gemini-2.5-flash` achieved 60.0% validity due to free-tier quota exhaustion (`429 RESOURCE_EXHAUSTED`) on later multi-turn repair attempts.
3. **Latency & Throughput**: Groq Cloud's LPU hardware provided significantly faster average decision latency (10.45 s per run vs 32.08 s for Gemini), improving real-time chat responsiveness.
4. **Task Success & Action Correctness**: `openai/gpt-oss-20b` scored 80.0% task success rate, excelling at prompt injection defense (Case 5) and missing parameter clarification (Cases 2 & 9).
5. **Autonomy Guard Responsiveness**: Both models correctly respected deterministic policy boundaries for consequential requests (*"Send purchase order"* & *"Pay invoice"*), which were blocked at 0.19 ms and 0.39 ms without consuming LLM token quota.

---

## 12. Testing & Verification

- **Unit Test Suite**: 47 / 47 Unittest cases PASSED (`python -m unittest discover -s tests -v`).
- **Data Isolation Integrity**: `data/sample_data.json` SHA-256 hash verified identical before and after test execution (`cf99b9cf9c58...`).

---

## 13. Local Setup & Environment Configuration

### Local Installation Commands
```bash
# 1. Clone repository & enter project directory
git clone https://github.com/rauf17/stockpilot-ai.git
cd stockpilot-ai

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# 3. Install pinned dependencies cleanly
pip install -r requirements.txt

# 4. Start local development server
python run.py
# Server runs on http://127.0.0.1:8000
```

### Environment Variables Matrix
Configure these in `.env` or system environment:

| Variable | Description | Default | Required For |
|---|---|---|---|
| `MODEL_PROVIDER` | Active LLM provider (`google`, `groq`) | `google` | All live model calls |
| `MODEL_NAME` | Active LLM model name (`gemini-2.5-flash`, `openai/gpt-oss-20b`) | `gemini-2.5-flash` | All live model calls |
| `GEMINI_API_KEY` | Google AI Studio Free Tier API key | `""` | Provider `google` |
| `GROQ_API_KEY` | Groq Cloud Free Plan API key | `""` | Provider `groq` |
| `MAX_STEPS` | Maximum iteration steps per turn ($1 \le n \le 6$) | `6` | Bounded agent execution |
| `MAX_TOOL_RETRIES` | Repair retries per step ($0 \le n \le 2$) | `2` | JSON decision repair |
| `MAX_OUTPUT_TOKENS` | Token ceiling per LLM call | `512` | Token & cost control |
| `RUN_TIMEOUT_SECONDS` | Request wall-clock timeout ($0 < s \le 40$) | `40.0` | Timeout containment |

---

## 14. API Endpoints Reference

| Endpoint | Method | Description |
|---|---|---|
| `/` | `GET` | HTML Web Interface |
| `/health` | `GET` | Health Check (`{"status": "ok", ...}`) |
| `/arena/manifest` | `GET` | Agent Arena JSON Manifest |
| `/models` | `GET` | Enabled Models List (`['gemini-2.5-flash', 'openai/gpt-oss-20b']`) |
| `/arena/run` | `POST` | Single-turn Arena execution (`ArenaRequest` -> `ArenaResponse`) |
| `/chat` | `POST` | Multi-turn chat execution with session state (`ChatRequest` -> `ArenaResponse`) |
| `/chat/{session_id}` | `DELETE` | Purge session memory for `session_id` |

---

## 15. Deployment & Persistence Behavior

- **Deployment Blueprint**: Deployment configured for Render Web Services via `render.yaml` (`MODEL_PROVIDER=google`, `MODEL_NAME=gemini-2.5-flash`).
- **Memory & State Persistence**: Multi-turn `SessionState` is held in-memory in server state (`request.app.state.memory`). State is per-session (`session_id`) and resets on server or worker restart. `DELETE /chat/{session_id}` purges session state explicitly.

---

## 16. Assignment Requirement Mapping Matrix

| Assignment Requirement | Implementation Component | File / Test Evidence |
|---|---|---|
| **Real LLM Decision Path** | Provider integration via Gemini / Groq / OpenAI | [`app/agent.py`](app/agent.py#L29-L135) |
| **Strict Model Routing & No Silent Fallback** | Explicit model-provider validation; raises `LLMProviderError` on unknown or mismatched model/provider | [`app/agent.py`](app/agent.py#L55-L75), `test_unknown_model_raises_llm_provider_error`, `test_mismatched_provider_model_raises_llm_provider_error` |
| **Multi-Turn Goal Preservation** | `SessionState` (`active_goal`, `extracted_facts`) | [`app/memory.py`](app/memory.py#L38-L100) |
| **Event Loop Responsiveness** | Async thread offload via `asyncio.to_thread` | `test_event_loop_responsiveness_during_blocking_llm` |
| **LLM Wall-Clock Timeout** | Non-blocking `asyncio.timeout` wrapper | `test_llm_timeout_wall_clock_interruption` |
| **Prompt Injection Defense** | Structured JSON hierarchy & deterministic policy | `test_adversarial_injection_cases` |
| **Autonomy Boundary Guard** | Blocking consequential requests up front | `test_paraphrased_autonomy_requests` |
| **Strict Tool Schemas** | Pydantic validation (`extra="forbid"`, bounds) | `test_tool_argument_validation_strict_bounds_and_forbid_extra` |
| **Budget Enforcement** | Step limits, repair limits, token limits | `test_pathological_invalid_json_repair_budget` |
| **Sample Data Isolation** | Test setup/teardown data backup & restore | SHA-256 hash verified intact |

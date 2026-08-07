# CLAUDE.md — EVA Support Reproducer

This file documents the project structure, architecture, and key decisions for AI assistants working in this codebase.

## Project Overview

EVA Support Reproducer is a FastAPI backend + vanilla JS frontend that simulates a second-line support workflow for a unified commerce retail platform. Support tickets are analyzed through a multi-stage AI pipeline and logged to SQLite.

## How to Run

```bash
# Install dependencies (requires Python 3.10+)
pip install -r requirements.txt

# Set environment variables (see .env section below)
cp .env.example .env  # or create .env manually

# Start the server
uvicorn main:app --reload

# Run tests
pytest test_main.py -v
```

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `NVIDIA_API_KEY` | Yes | — | NVIDIA NIM API key for primary LLM calls (free at build.nvidia.com) |
| `GEMINI_API_KEY` | Yes | — | Google Gemini API key for Gemini Flash judge + KB summarizer (free at aistudio.google.com/apikey) |
| `KB_MARGIN_THRESHOLD` | No | `0.06` | Gate: min gap between the #1 and #2 KB match to serve (float) |
| `KB_ABS_FLOOR` | No | `0.39` | Gate: min top-1 similarity score to serve — safety net (float) |

These must be in a `.env` file locally (gitignored) and set as environment variables on the hosting platform.

## Key Files

| File | Purpose |
|---|---|
| `main.py` | FastAPI app — all HTTP routes |
| `prompts.py` | Full AI pipeline: gate logic, NIM LLM, Gemini judge, KB summarizer |
| `feedback_routing.py` | **Single source of truth** for feedback routing — imported by `main.py`, `database.py`, and (Phase 3) the candidate queue |
| `database.py` | SQLite — schema, migrations, all DB functions |
| `vector_store.py` | ChromaDB — KB seeding, semantic search, resolved ticket storage |
| `tickets.py` | 10 hardcoded sample support tickets |
| `eval_retrieval.py` | Labeled retrieval + gate eval harness — the calibration instrument for `KB_MARGIN_THRESHOLD` / `KB_ABS_FLOOR` |
| `test_main.py` | Pytest suite — 53 deterministic (gate/routing, no network) + 4 opt-in live NIM model checks (`RUN_LIVE_TESTS=1`) |
| `Dockerfile` | Container build — Python 3.10-slim, port 8000 |
| `.github/workflows/deploy.yml` | CI/CD: test → build → deploy to Render |

## AI Pipeline (prompts.py)

The full analysis flow is in `analyze_ticket()`:

```
1. search_knowledge_base()       — ChromaDB KB collection
   search_resolved_tickets()     — ChromaDB resolved_tickets collection

2. Confidence Gate — evaluate_gate() over merged kb + resolved candidates
   top1, top2 = two highest DISTINCT candidate scores (near-dupes skipped so a resolved
                ticket mirroring a KB article can't crush the margin)
   serve iff  top1 >= KB_ABS_FLOOR  AND  (top1 - top2) >= KB_MARGIN_THRESHOLD
   else return abstain dict (reason: low_floor | low_margin | insufficient_candidates)
   ← main.py turns this into an abstain HTTP response (analysis: null)
   If _embed raises EmbeddingUnavailable, analyze_ticket abstains (analyzed_by=retrieval_unavailable)
   rather than analyzing with no KB context.

3. build_context_from_search()   — formats KB matches into LLM context string

4. get_model_for_priority()      — picks NVIDIA NIM model based on ticket.priority
   low    → meta/llama-3.1-8b-instruct
   medium → meta/llama-3.1-70b-instruct
   high   → meta/llama-3.3-70b-instruct
   NOTE: NIM decommissions models WITHOUT delisting them from the catalog (a call 404s
   even though models.list() still shows it), and its nemotron "reasoning" models return
   empty `content`. Only route to models verified callable via the live test
   (RUN_LIVE_TESTS=1 pytest -k callable). That test is the guard against silent
   priority-tier breakage — a dead model degrades every ticket at that tier to fallback.

5. nim_client.chat.completions.create()   — OpenAI-compatible API call to NVIDIA NIM
   Retries up to 2 times before fallback_response()

6. validate_response()           — checks required JSON fields, normalises enums

7. run_judge()                   — Gemini Flash via OpenAI-compatible SDK
   Only called when gate passes.
   Returns: {verdict, concerns, notes}
   Never blocks — returns SKIPPED on any error
```

### Important SDK differences
- **NVIDIA NIM**: uses `openai.OpenAI(base_url="https://integrate.api.nvidia.com/v1")` — response accessor is `response.choices[0].message.content`
- **Google Gemini**: uses `openai.OpenAI(base_url="https://generativelanguage.googleapis.com/v1beta/openai/")` — Google's OpenAI-compatible endpoint, so the same `openai` library and `response.choices[0].message.content` accessor apply. No separate SDK needed. Judge model is `gemini-2.0-flash`.
- **Embeddings**: ChromaDB uses NVIDIA NIM's `nvidia/nv-embedqa-e5-v5` (1024-dim) via a custom `NIMEmbeddingFunction` in `vector_store.py`, calling the same OpenAI-compatible NIM endpoint.

## Database Schema (SQLite — eva_support.db)

Single table: `tickets`

| Column | Type | Notes |
|---|---|---|
| `id` | INTEGER PK | auto-increment |
| `ticket_id` | TEXT | e.g. TKT-001 |
| `title` | TEXT | |
| `description` | TEXT | |
| `store` | TEXT | |
| `priority` | TEXT | low / medium / high |
| `issue_type` | TEXT | LLM-classified |
| `severity` | TEXT | Low / Medium / High / Critical |
| `likely_cause` | TEXT | |
| `workaround` | TEXT | |
| `suggested_next_step` | TEXT | |
| `escalate_to_third_line` | BOOLEAN | |
| `escalation_reason` | TEXT | |
| `internal_note` | TEXT | |
| `customer_reply` | TEXT | |
| `known_issue` | BOOLEAN | |
| `knowledge_base_article` | TEXT | |
| `known_fix` | TEXT | |
| `analyzed_by` | TEXT | `llm` / `fallback` / `gate_blocked` / `retrieval_unavailable` |
| `status` | TEXT | Open / In Progress / Escalated / Resolved |
| `actual_fix` | TEXT | set on resolution |
| `follow_up_reply` | TEXT | |
| `created_at` | TEXT | ISO datetime |
| `resolved_at` | TEXT | ISO datetime |
| `reproduction_checklist` | TEXT | JSON array |
| `kb_similarity_score` | REAL | top-1 ChromaDB cosine similarity |
| `kb_second_score` | REAL | 2nd distinct candidate similarity (for the margin) |
| `kb_margin` | REAL | `kb_similarity_score − kb_second_score` |
| `kb_match_title` | TEXT | title of best-matching KB article |
| `gate_passed` | BOOLEAN | True if top1 ≥ floor AND margin ≥ margin-threshold |
| `judge_verdict` | TEXT | PASS / FAIL / SKIPPED |
| `judge_notes` | TEXT | one-sentence judge assessment |
| `judge_concerns` | TEXT | JSON array of concern strings |
| `model_used` | TEXT | NVIDIA NIM model name |
| `abstain_reason` | TEXT | `low_floor` / `low_margin` / `insufficient_candidates` / `retrieval_unavailable` / `llm_unavailable` |
| `retrieved_articles` | TEXT | JSON `[{id,title,snippet,score}]` — snapshot of what retrieval surfaced |
| `kb_relevant` | BOOLEAN | Phase 2 feedback Q1 — **nullable** (True/False/None) |
| `fix_helped` | BOOLEAN | Phase 2 feedback Q2 — **nullable** (True/False/None) |
| `feedback_prompted_at` | TEXT | ISO datetime the popup was shown/dismissed — enforces "never re-ask" |

> `engineer_feedback` was **removed** in Phase 2 (replaced by the two flags above). It is listed in
> `database.DROPPED_COLUMNS` and dropped on migrate. A dropped column must be absent from **both**
> `CREATE TABLE` and `migrate_db.new_columns`, or migrate would silently recreate it every boot.

> `retrieved_articles` stores **title + snippet alongside the ChromaDB id on purpose**: the corpus
> will be swapped (e.g. to TechQA), which would orphan raw ids and invalidate every feedback row.
> It is captured at **analysis time** (what the agent was actually shown), never re-read at feedback time.

### Schema migrations
`migrate_db()` is called automatically inside `init_db()` on every app start. It runs `ALTER TABLE ADD COLUMN` for each new column inside a `try/except` to skip columns that already exist. This makes it safe to redeploy without manual migration steps.

## ChromaDB Collections

Two persistent collections stored in `./chroma_db/`:

**`knowledge_base`** — 10 pre-seeded articles. Seeded once on import by `seed_knowledge_base()`. Each article vectorizes `title + content`. Metadata stored: `title`, `known_fix`, `workaround`, `issue_type`.

**`resolved_tickets`** — grows with every resolved ticket. Added by `add_resolved_ticket()` called from `main.py`'s `/logs/{id}/status` endpoint when status = Resolved + actual_fix is provided. The fix is first summarized and anonymized by `summarize_fix_for_kb()` (Gemini Flash) before storage — no customer/store names are ever stored here.

### Similarity score note
`similarity_score = round(1 - distance, 3)` where `distance` is ChromaDB **cosine** distance (collections are created with `metadata={"hnsw:space": "cosine"}` because nv-embedqa produces normalized vectors). The `distance < 1.2` filter is permissive — the margin+floor gate does the actual quality filtering. Embeddings use the asymmetric `input_type` — `passage` when indexing, `query` when searching (computed in `_embed`, passed to ChromaDB directly since its embedding-function hook can't distinguish the two).

## API Routes (main.py)

| Method | Path | Handler | Notes |
|---|---|---|---|
| GET | `/` | `serve_frontend` | Serves static/index.html |
| GET | `/tickets` | `get_all_tickets` | Returns all 10 sample tickets |
| GET | `/tickets/{id}` | `get_ticket` | Single ticket |
| GET | `/analyze/{ticket_id}` | `analyze_sample_ticket` | Reuses today's log if gate passed |
| POST | `/analyze/custom` | `analyze_custom_ticket` | Always creates new log |
| GET | `/logs` | `get_logs` | All logs, newest first |
| GET | `/logs/{id}` | `get_log` | Full log detail |
| PATCH | `/logs/{id}/status` | `update_status` | Triggers KB update on Resolved |
| POST | `/logs/{id}/followup` | `generate_follow_up` | |
| PATCH | `/logs/{id}/feedback` | `submit_feedback` | Two-stage: `{kb_relevant?, fix_helped?}`; empty body = dismissal |
| GET | `/stats` | `get_statistics` | DB stats + vector store stats |
| GET | `/health` | `health_check` | |

### Abstain (gate-fail) response shape
When `analysis.gate_passed == False`, routes return (`main._gate_fail_response`):
```json
{
  "log_id": 42,
  "gate_passed": false,
  "gate_decision": "abstained",
  "abstain_reason": "low_margin",
  "kb_similarity_score": 0.42,
  "kb_second_score": 0.40,
  "kb_margin": 0.02,
  "kb_match_title": "…",
  "margin_threshold": 0.06,
  "abs_floor": 0.39,
  "message": "Top two KB matches are too close…",
  "analysis": null
}
```
The attempt is still logged to SQLite (so abstain trends are visible in metrics). The frontend
routes this to `showAbstained()`; served analyses carry `kb_margin` / `kb_second_score` too.

## Feedback Routing (Phase 2)

Two questions are asked **at resolution, on served tickets only** (abstained tickets showed no
article, so there is nothing to rate). Progressive disclosure: Q2 is revealed only after Q1 is
answered. The popup is fully optional — Esc / click-away / Skip records a **dismissal (NULL flags),
never a negative** — and `feedback_prompted_at` ensures a ticket is never asked twice.

`feedback_routing.classify_feedback(gate_passed, kb_relevant, fix_helped)`:

| gate_passed | kb_relevant | fix_helped | route | KB candidate? |
|---|---|---|---|---|
| False | — | — | `coverage_gap` | ✅ (auto, no clicks) |
| True | False | any/None | `coverage_gap` | ✅ |
| True | None | — | `no_response` | ❌ |
| True | True | False | `prompt_problem` | ❌ |
| True | True | True | `success` | ❌ |
| True | True | None | `partial` | ❌ |

**Key rule:** `kb_relevant=False` is a *complete, actionable* signal on its own — it routes to
`coverage_gap` even when Q2 was never answered. Under progressive disclosure that is the most common
partial response, so it must never fall through to `no_response`.

**Never reimplement this as inline SQL.** `get_stats` and the Phase 3 queue fetch rows and classify in
Python via this function so the rule lives in exactly one place. That trades SQL aggregates for
Python-side counting — an accepted trade-off at this scale.

New metrics in `/stats`: `feedback_coverage` (fraction of served+resolved tickets that gave any
answer — tells you how thin the signal is), `route_breakdown`, `feedback_responses`, and
`false_confidence_count` (served tickets where `kb_relevant=False`, i.e. we answered confidently off
the wrong article).

## CI/CD

GitHub Actions workflow at `.github/workflows/deploy.yml`:
- Triggers on push/PR to `main`, skips if only `.md` files changed
- Jobs: `test` → `build` (Docker) → `deploy` (Render hook, main branch only)
- Secrets required: `NVIDIA_API_KEY`, `GEMINI_API_KEY`, `RENDER_DEPLOY_HOOK`

## Common Tasks

**Add a new KB article**: add an entry to `KB_ARTICLES` list in `vector_store.py`. The `seed_knowledge_base()` function only seeds if `count < len(KB_ARTICLES)`, so add the new article and the next app start will seed it.

**Change the gate**: update `KB_MARGIN_THRESHOLD` / `KB_ABS_FLOOR` in `.env` (read at module load in `prompts.py`). Re-run `python eval_retrieval.py` to see the served/abstain/FPR trade-off at the new values before committing them.

**Re-calibrate the gate after KB growth**: the defaults are tuned for a 10-article KB. After adding many articles, run `eval_retrieval.py`, read the gate sweep, and pick new margin/floor values (FPR reported as a count, e.g. `0/18 negatives`).

**Add a new analysis field**: (1) add to the LLM JSON prompt schema, (2) add to `validate_response()` required fields if mandatory, (3) add column to `database.py` `CREATE TABLE` and `migrate_db()`, (4) add to `log_ticket()` INSERT, (5) add to `get_log_by_id()` columns list in the correct position.

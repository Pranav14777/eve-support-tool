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
| `ANTHROPIC_API_KEY` | Yes | — | Anthropic API key for Claude Haiku judge + KB summarizer |
| `KB_SIMILARITY_THRESHOLD` | No | `0.65` | Minimum ChromaDB similarity score for gate to pass (float 0.0–1.0) |

These must be in a `.env` file locally (gitignored) and set as environment variables on the hosting platform.

## Key Files

| File | Purpose |
|---|---|
| `main.py` | FastAPI app — all HTTP routes |
| `prompts.py` | Full AI pipeline: gate logic, NIM LLM, Claude judge, KB summarizer |
| `database.py` | SQLite — schema, migrations, all DB functions |
| `vector_store.py` | ChromaDB — KB seeding, semantic search, resolved ticket storage |
| `tickets.py` | 10 hardcoded sample support tickets |
| `test_main.py` | Pytest suite (28 tests) |
| `Dockerfile` | Container build — Python 3.10-slim, port 8000 |
| `.github/workflows/deploy.yml` | CI/CD: test → build → deploy to Render |

## AI Pipeline (prompts.py)

The full analysis flow is in `analyze_ticket()`:

```
1. search_knowledge_base()       — ChromaDB KB collection
   search_resolved_tickets()     — ChromaDB resolved_tickets collection

2. KB Similarity Gate
   best_score = max(kb_top_score, resolved_top_score)
   if best_score < KB_SIMILARITY_THRESHOLD:
       return gate_blocked dict   ← main.py turns this into a hard-block HTTP response

3. build_context_from_search()   — formats KB matches into LLM context string

4. get_model_for_priority()      — picks NVIDIA NIM model based on ticket.priority
   low    → meta/llama-3.1-8b-instruct
   medium → meta/llama-3.3-70b-instruct
   high   → nvidia/llama-3.1-nemotron-70b-instruct

5. nim_client.chat.completions.create()   — OpenAI-compatible API call to NVIDIA NIM
   Retries up to 2 times before fallback_response()

6. validate_response()           — checks required JSON fields, normalises enums

7. run_judge()                   — Claude Haiku via Anthropic SDK
   Only called when gate passes.
   Returns: {verdict, concerns, notes}
   Never blocks — returns SKIPPED on any error
```

### Important SDK differences
- **NVIDIA NIM**: uses `openai.OpenAI(base_url="https://integrate.api.nvidia.com/v1")` — response accessor is `response.choices[0].message.content`
- **Anthropic (Claude)**: uses `anthropic.Anthropic()` — response accessor is `response.content[0].text`, and `system=` is a top-level parameter, NOT inside `messages`

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
| `analyzed_by` | TEXT | `llm` / `fallback` / `gate_blocked` |
| `status` | TEXT | Open / In Progress / Escalated / Resolved |
| `actual_fix` | TEXT | set on resolution |
| `follow_up_reply` | TEXT | |
| `created_at` | TEXT | ISO datetime |
| `resolved_at` | TEXT | ISO datetime |
| `reproduction_checklist` | TEXT | JSON array |
| `kb_similarity_score` | REAL | top ChromaDB similarity (0.0–1.0) |
| `kb_match_title` | TEXT | title of best-matching KB article |
| `gate_passed` | BOOLEAN | True if score ≥ threshold |
| `judge_verdict` | TEXT | PASS / FAIL / SKIPPED |
| `judge_notes` | TEXT | one-sentence judge assessment |
| `judge_concerns` | TEXT | JSON array of concern strings |
| `model_used` | TEXT | NVIDIA NIM model name |
| `engineer_feedback` | TEXT | `helpful` / `not_helpful` / NULL |

### Schema migrations
`migrate_db()` is called automatically inside `init_db()` on every app start. It runs `ALTER TABLE ADD COLUMN` for each new column inside a `try/except` to skip columns that already exist. This makes it safe to redeploy without manual migration steps.

## ChromaDB Collections

Two persistent collections stored in `./chroma_db/`:

**`knowledge_base`** — 10 pre-seeded articles. Seeded once on import by `seed_knowledge_base()`. Each article vectorizes `title + content`. Metadata stored: `title`, `known_fix`, `workaround`, `issue_type`.

**`resolved_tickets`** — grows with every resolved ticket. Added by `add_resolved_ticket()` called from `main.py`'s `/logs/{id}/status` endpoint when status = Resolved + actual_fix is provided. The fix is first summarized and anonymized by `summarize_fix_for_kb()` (Claude Haiku) before storage — no customer/store names are ever stored here.

### Similarity score note
`similarity_score = round(1 - distance, 3)` where `distance` is ChromaDB L2 distance. Scores can be negative for very dissimilar items (L2 distance > 1.0). The existing distance filter `distance < 1.2` is permissive — the gate threshold does the actual quality filtering.

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
| PATCH | `/logs/{id}/feedback` | `submit_feedback` | `helpful` or `not_helpful` |
| GET | `/stats` | `get_statistics` | DB stats + vector store stats |
| GET | `/health` | `health_check` | |

### Gate-fail response shape
When `analysis.gate_passed == False`, routes return:
```json
{
  "log_id": 42,
  "gate_passed": false,
  "kb_similarity_score": 0.41,
  "threshold": 0.65,
  "message": "No sufficiently similar KB article...",
  "analysis": null
}
```
The attempt is still logged to SQLite (so gate-fail trends are visible in metrics).

## CI/CD

GitHub Actions workflow at `.github/workflows/deploy.yml`:
- Triggers on push/PR to `main`, skips if only `.md` files changed
- Jobs: `test` → `build` (Docker) → `deploy` (Render hook, main branch only)
- Secrets required: `NVIDIA_API_KEY`, `ANTHROPIC_API_KEY`, `RENDER_DEPLOY_HOOK`

## Common Tasks

**Add a new KB article**: add an entry to `KB_ARTICLES` list in `vector_store.py`. The `seed_knowledge_base()` function only seeds if `count < len(KB_ARTICLES)`, so add the new article and the next app start will seed it.

**Change the gate threshold**: update `KB_SIMILARITY_THRESHOLD` in `.env`. No code change needed — it's read at module load time from the environment.

**Add a new analysis field**: (1) add to the LLM JSON prompt schema, (2) add to `validate_response()` required fields if mandatory, (3) add column to `database.py` `CREATE TABLE` and `migrate_db()`, (4) add to `log_ticket()` INSERT, (5) add to `get_log_by_id()` columns list in the correct position.

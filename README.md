# EVA Support Reproducer

[![CI/CD Pipeline](https://github.com/Pranav14777/eve-support-tool/actions/workflows/deploy.yml/badge.svg)](https://github.com/Pranav14777/eve-support-tool/actions/workflows/deploy.yml)

> A second-line support workflow tool for the EVA unified commerce platform — turning unstructured support tickets into structured, actionable intelligence, and **knowing when to stay quiet**.

> **Live demo:** currently being redeployed to Render (the previous Railway free tier was discontinued). Run it locally in ~2 minutes with the [Getting Started](#getting-started) steps.

---

## What This Is

EVA Support Reproducer is an AI-powered internal support tool that simulates the second-line support workflow for a unified commerce platform.

When a support ticket comes in from a retail partner, the tool runs it through a multi-stage pipeline:

1. Searches a semantic knowledge base (ChromaDB + NVIDIA `nv-embedqa` embeddings) for known issues and past resolutions
2. **Confidence gate** — serves an answer only when one KB article *clearly* stands out; otherwise it **abstains**
3. Sends the ticket to an LLM with relevant KB context (model chosen by ticket priority)
4. **Judge LLM** independently validates the analysis using a different provider
5. Returns a structured analysis with the gate margin and judge verdict attached
6. Logs everything — including abstentions, so the gaps are measurable
7. Asks **two-stage feedback** at resolution, which routes gaps back into the knowledge base

The interesting part isn't the answering — it's the **abstaining**. A support assistant that confidently answers from the wrong article is worse than one that says "I don't have this; go investigate."

---

## Why I Built This

I was curious what it would take to turn a messy, incomplete support ticket into a clear, structured next step automatically — combining semantic search, an LLM, and a feedback loop into one workflow. This is a portfolio project built to explore that idea end to end, from ticket intake to resolution and learning.

The part I found most interesting was the opposite problem: teaching it **when not to answer**, and being able to prove the threshold was right with numbers instead of intuition.

---

## Pipeline Architecture

```
Ticket arrives
      │
      ▼
ChromaDB semantic search  (nv-embedqa-e5-v5, cosine)
(KB articles + past resolved tickets)
      │
      ▼
Confidence Gate ─── margin or floor too low ───► ABSTAIN
top1 ≥ floor AND (top1 − top2) ≥ margin           │
      │ one clear winner                          │  "No confident KB match —
      ▼                                           │   investigate manually"
NVIDIA NIM LLM                                    │
(model chosen by ticket priority)                 ▼
      │                                    Engineer submits fix
      ▼                                           │
Response validated                                ▼
      │                                    Gemini Flash summarizes
      ▼                                    + anonymizes the fix
Gemini Flash Judge  ──► FAIL = shown to             │
(independent provider)   engineer as a warning,     ▼
      │                  never silently blocked   Fix stored in ChromaDB
      ▼                                          (feeds future retrieval)
Analysis + margin + judge verdict
sent to engineer
      │
      ▼
At resolution: two-stage feedback (optional)
 "Was the KB article relevant?" ──► no  ──► coverage gap ──► KB candidate
              │ yes
              ▼
 "Did the fix help?"            ──► no  ──► prompt problem (not a KB gap)
```

---

## Model Routing by Priority

Different models are used based on the ticket's incoming priority, balancing cost against quality:

| Role | Model | Provider |
|---|---|---|
| Analysis — `low` priority | `meta/llama-3.1-8b-instruct` | NVIDIA NIM |
| Analysis — `medium` priority | `meta/llama-3.3-70b-instruct` | NVIDIA NIM |
| Analysis — `high` priority | `nvidia/llama-3.1-nemotron-70b-instruct` | NVIDIA NIM |
| Judge + KB summarizer | `gemini-2.0-flash` | Google |
| Embeddings | `nvidia/nv-embedqa-e5-v5` (1024-dim) | NVIDIA NIM |

NVIDIA NIM offers free-tier access with an OpenAI-compatible API. The judge deliberately uses a **different provider and architecture** so it catches blind spots the primary model shares with itself. Gemini is also reached through its OpenAI-compatible endpoint, so the whole stack uses one SDK.

---

## Features

### Confidence Gate (margin + floor)
- Before the LLM is called, the gate inspects retrieval and serves only when **one KB article clearly stands out** — `top1 ≥ KB_ABS_FLOOR` **and** `(top1 − top2) ≥ KB_MARGIN_THRESHOLD`
- The **margin** is the primary signal: an uncovered ticket matches everything equally poorly, so nothing stands out. The **floor** is a safety net against a weak-but-gapped top hit
- Near-duplicate candidates are skipped when computing the margin, so a resolved ticket mirroring a KB article can't crush it as the KB grows
- Fewer than two *distinct* candidates → abstain (`insufficient_candidates`) rather than pass on a lone weak hit
- Otherwise it **abstains** with the exact scores, margin and reason shown to the engineer
- If the embedding service is unreachable, it abstains rather than analyzing with no context

### Judge LLM
- Runs only when the gate passes
- Uses Gemini Flash (Google) — intentionally different from the primary NVIDIA NIM model
- Evaluates: correct issue classification? severity appropriate? checklist executable? fix aligned with KB?
- Returns `PASS`, `FAIL`, or `SKIPPED` (never blocks — a judge outage can't take the tool down)
- A `FAIL` is surfaced to the engineer as a prominent warning with its concerns, not silently swallowed

### Engineer Feedback Loop (two-stage)
One button can't tell you *why* an answer was bad, so the tool asks two questions at resolution
(served tickets only — an abstained ticket showed no article, so there's nothing to rate):

1. **Was the KB article relevant?** → if no, that's a **coverage/retrieval gap** → the ticket becomes a KB candidate
2. **Did the fix help?** (revealed only after Q1) → if no but the article *was* relevant, that's a **prompt/generation problem**, not a KB gap

That distinction is the whole point: it separates "we don't have this knowledge" from "we have it and explained it badly" — two problems with completely different fixes.

- Fully optional and non-blocking — Esc / click-away / Skip records a **dismissal, never a negative**
- Partial answers are first-class (both flags independently nullable); a ticket is never asked twice
- `feedback_coverage` is tracked as its own metric, so we always know how thin the signal is
- Each feedback row stores the retrieved article **id + title + snippet**, so the data survives a corpus swap instead of being orphaned
- The routing rule lives in exactly one place (`feedback_routing.py`), shared by the API, the stats endpoint, and the review queue

### Core Analysis
- **Issue Classification** — Integration Issue, Configuration Issue, API Issue, System Behavior, Data Sync Issue
- **Severity Assessment** — Low, Medium, High, Critical
- **Reproduction Checklist** — concrete steps to verify and reproduce the issue
- **Immediate Workaround** — what the store can do right now
- **Escalation Decision** — flags when third line involvement is needed
- **Internal Note** — structured summary for the support team
- **Customer Reply** — professional, empathetic partner communication

### Knowledge Base (ChromaDB)
- 10 pre-seeded KB articles embedded with `nv-embedqa-e5-v5` (1024-dim, cosine distance)
- Asymmetric embedding done correctly: `passage` when indexing, `query` when searching
- Semantic search — matches meaning, not just keywords
- Resolved ticket fixes are summarized and anonymized before storage (no customer/store names)
- KB grows with every resolved ticket, improving future retrieval

### Analytics Dashboard
- Total tickets logged by status, escalation rate, KB match rate
- **Suggestion success rate** — relevant article + fix actually helped
- **False confidence count** — we served an answer off an article the engineer marked *not relevant*
- **Feedback coverage** — fraction of served+resolved tickets that gave any answer
- **Route breakdown** — coverage gaps vs prompt problems vs successes
- **Average KB similarity**, **gate pass rate**, **judge pass rate**

---

## Retrieval Evaluation

Gate thresholds aren't guessed — they're calibrated with a labeled eval harness (`eval_retrieval.py`) so every change is defensible with a number. It runs 50 query→article pairs written the way a store associate would actually describe a problem (deliberately *not* reusing the KB's wording) plus 18 out-of-scope "negatives" the KB does not cover.

Measured on the current 10-article KB:

| Metric | Value |
|---|---|
| recall@1 | 0.78 |
| MRR | 0.883 |
| Gate — covered tickets served correctly (margin 0.06, floor 0.39) | 28/50 |
| Gate — false positives on out-of-scope tickets | **0/18 negatives** |

**What the eval actually caught.** An earlier version of this eval reused the KB's own wording and reported recall@1 of 0.98 — which flattered the system and hid the real behavior. Rewriting the queries in natural support language dropped it to 0.78 and exposed two things: covered and uncovered tickets overlap badly on *absolute* score (0.43 vs 0.33), but separate ~**3× better on margin** (0.081 vs 0.027). That's why the gate is tuned on the #1–#2 margin rather than a similarity threshold.

FPR is reported as a **count** (0/18), not a percentage — 18 negatives is too small a sample to claim a "0%" rate.

> These numbers are calibrated against a 10-article KB. They must be re-derived (`python eval_retrieval.py`) once the KB grows to ~60–80 articles, since more distractors shift the score distribution.

```bash
python eval_retrieval.py
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI |
| Primary LLM | NVIDIA NIM — model varies by ticket priority |
| Judge LLM + KB summarizer | Google Gemini Flash |
| Embeddings | NVIDIA `nv-embedqa-e5-v5` (1024-dim, retrieval-tuned) |
| Vector Database | ChromaDB (cosine) |
| Relational Database | SQLite |
| Frontend | Vanilla HTML/CSS/JS |
| Containerization | Docker |
| CI/CD | GitHub Actions |
| Hosting | Render.com |
| Language | Python 3.10+ |

---

## Project Structure

```
eve-support-tool/
├── main.py              # FastAPI backend — all routes
├── prompts.py           # LLM pipeline — gate, NIM analysis, Gemini judge, KB summarizer
├── feedback_routing.py  # Single source of truth: coverage gap vs prompt problem
├── vector_store.py      # ChromaDB — embeddings, KB articles, anonymized resolved tickets
├── database.py          # SQLite — schema/migrations, logging, feedback, metrics
├── tickets.py           # 10 sample support tickets
├── eval_retrieval.py    # Labeled retrieval + gate eval harness (calibration instrument)
├── test_main.py         # Pytest suite (52 tests)
├── Dockerfile           # Container build definition
├── requirements.txt     # Python dependencies
├── .env.example         # Copy to .env and fill in your keys
├── .github/
│   └── workflows/
│       └── deploy.yml   # CI/CD pipeline — test, build, deploy to Render
└── static/
    └── index.html       # Frontend UI — Analyze, Logs, Analytics tabs
```

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Frontend UI |
| GET | `/tickets` | All sample tickets |
| GET | `/analyze/{ticket_id}` | Analyze sample ticket |
| POST | `/analyze/custom` | Analyze custom ticket |
| GET | `/logs` | All logged tickets |
| GET | `/logs/{id}` | Single log details |
| PATCH | `/logs/{id}/status` | Update ticket status (stores the fix on resolution) |
| POST | `/logs/{id}/followup` | Generate follow-up reply |
| PATCH | `/logs/{id}/feedback` | Two-stage feedback `{kb_relevant?, fix_helped?}` (empty body = dismissal) |
| GET | `/stats` | Analytics dashboard data |
| GET | `/health` | Health check |
| GET | `/docs` | Auto-generated API docs |

### Abstain response

When the gate doesn't pass, the API returns `analysis: null` plus the full reasoning, so the UI and the logs can both show *why*:

```json
{
  "log_id": 42,
  "gate_passed": false,
  "gate_decision": "abstained",
  "abstain_reason": "low_margin",
  "kb_similarity_score": 0.42,
  "kb_second_score": 0.40,
  "kb_margin": 0.02,
  "margin_threshold": 0.06,
  "abs_floor": 0.39,
  "message": "Top two KB matches are too close…",
  "analysis": null
}
```

---

## Getting Started

### Prerequisites
- Python 3.10+
- NVIDIA NIM API key — free at [build.nvidia.com](https://build.nvidia.com)
- Google Gemini API key — free at [aistudio.google.com/apikey](https://aistudio.google.com/apikey)

### Installation

```bash
# Clone the repository
git clone https://github.com/Pranav14777/eve-support-tool.git
cd eve-support-tool

# Create virtual environment
python -m venv venv
venv\Scripts\activate       # Windows
source venv/bin/activate    # Mac/Linux

# Install dependencies
pip install -r requirements.txt
```

### Configuration

```bash
cp .env.example .env    # then fill in your keys
```

```
NVIDIA_API_KEY=nvapi-...
GEMINI_API_KEY=...
KB_MARGIN_THRESHOLD=0.06
KB_ABS_FLOOR=0.39
```

The gate serves a ticket only when `top1 ≥ KB_ABS_FLOOR` **and** `(top1 − top2) ≥ KB_MARGIN_THRESHOLD`. Raise `KB_MARGIN_THRESHOLD` to be stricter (fewer, more-confident answers); lower it to serve more tickets. Both defaults are calibrated against the current 10-article KB via `eval_retrieval.py` and should be re-derived as the KB grows.

### Run

```bash
uvicorn main:app --reload
```

Open `http://localhost:8000`

### Run with Docker

```bash
docker build -t eva-support-tool .
docker run -p 8000:8000 --env-file .env eva-support-tool
```

---

## Testing

52 tests covering API endpoints, sample tickets, logging, analytics, the confidence gate, and feedback routing.

```bash
pip install pytest httpx pytest-asyncio
pytest test_main.py -v
```

Gate and routing tests are fully deterministic — they exercise `evaluate_gate` and `classify_feedback` directly with crafted scores and monkeypatched retrieval, so the suite makes **no network calls** and can't flake on an API quota. Coverage includes low-margin / low-floor / single-candidate abstention, duplicate-candidate handling, embedding-outage abstention, the full 9-case feedback routing truth table, and partial vs dismissed feedback.

---

## CI/CD & Deployment

Every push to `main` runs an automated pipeline via GitHub Actions ([`.github/workflows/deploy.yml`](.github/workflows/deploy.yml)):

1. **Test** — runs the full pytest suite
2. **Build** — builds the Docker image
3. **Deploy** — triggers a Render deploy hook (main branch only)

Documentation-only changes (e.g. README updates) are skipped by the pipeline.

### Deploying to Render (free tier)

1. Go to [render.com](https://render.com) → New → Web Service
2. Connect GitHub repo, select **Docker** as environment
3. Add environment variables: `NVIDIA_API_KEY`, `GEMINI_API_KEY`, `KB_MARGIN_THRESHOLD`, `KB_ABS_FLOOR`
4. Copy the **Deploy Hook** URL from Settings → Deploys
5. Add it as a GitHub repository secret named `RENDER_DEPLOY_HOOK`

> **Note:** Render's free tier spins down after 15 min of inactivity (~30s cold start). Acceptable for a portfolio project.

---

## Sample EVA Tickets

The tool comes pre-loaded with 10 realistic EVA platform support scenarios:

| ID | Issue | Store | Priority |
|---|---|---|---|
| TKT-001 | Payment via Adyen failing at checkout | Hunkemöller Berlin | high |
| TKT-002 | Click & collect orders not visible in POS | Rituals Amsterdam | high |
| TKT-003 | Inventory sync showing incorrect stock levels | Dyson Multiple Stores | medium |
| TKT-004 | Product import failing with JSON error | Kiko Milano | medium |
| TKT-005 | API returning 401 unauthorized | Intersport | high |
| TKT-006 | POS freezing on specific barcode scan | AFC Ajax Store | medium |
| TKT-007 | Prices showing without VAT in German stores | Hunkemöller Germany | high |
| TKT-008 | Loyalty points not updating after purchase | Rituals Multiple Stores | medium |
| TKT-009 | Receipt language not switching to locale | Multiple EU Stores | low |
| TKT-010 | Mobile POS app crashing after iOS update | Dyson UK Stores | high |

---

## Roadmap

The feedback signal exists to feed a human-reviewed KB growth loop:

- **Next** — candidate queue: route gate-failures and flagged coverage gaps into a `kb_candidates` table with clustering and LLM-summarized, redacted drafts
- **Then** — review actions (create / merge / update / reject / retire) with provenance on every article
- **Then** — self-growing eval: approved tickets become labeled eval queries, so retrieval metrics can be plotted against corpus size over time
- **Then** — auth and per-user KB scoping

---

## Production Considerations

This is a portfolio project. In production:

- **Data Privacy** — Real ticket data should be anonymized before sending to any external LLM API (the KB summarizer already strips client info from stored fixes)
- **Authentication** — API endpoints would require authentication
- **Ticketing Integration** — Would connect to Zendesk, Jira, or ServiceNow via webhook
- **Scalability** — SQLite would be replaced with PostgreSQL for production load
- **Gate Tuning** — `KB_MARGIN_THRESHOLD` / `KB_ABS_FLOOR` would be re-derived with `eval_retrieval.py` as the KB grows, trading off abstention rate against false-positive rate
- **Feedback aggregation** — routing is classified in Python to keep the rule in one place; at scale this would move to a materialized column or a SQL-side rule with a single shared definition

---

## Built By

Pranav Gadamsetty — built as a portfolio project, demonstrating support workflow thinking, LLM integration, and RAG architecture.

[LinkedIn](https://www.linkedin.com/in/pgdeveloper/) · [GitHub](https://github.com/Pranav14777)

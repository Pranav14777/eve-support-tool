# EVA Support Reproducer

[![CI/CD Pipeline](https://github.com/Pranav14777/eve-support-tool/actions/workflows/deploy.yml/badge.svg)](https://github.com/Pranav14777/eve-support-tool/actions/workflows/deploy.yml)

> A second-line support workflow tool for the EVA unified commerce platform — turning unstructured support tickets into structured, actionable intelligence.

**Live Demo:** [eve-support-tool-production.up.railway.app](https://eve-support-tool-production.up.railway.app/)

---

## What This Is

EVA Support Reproducer is an AI-powered internal support tool that simulates the second-line support workflow for a unified commerce platform.

When a support ticket comes in from a retail partner, the tool runs it through a multi-stage pipeline:

1. Searches a semantic knowledge base for known issues and past resolutions
2. **Confidence gate** — only proceeds if a sufficiently similar KB article exists
3. Sends the ticket to an LLM with relevant KB context
4. **Judge LLM** independently validates the analysis using a different model
5. Returns a fully structured analysis with quality signals attached
6. Logs everything automatically, including gate and judge results
7. Learns from resolved tickets — anonymized fixes feed back into the KB

---

## Why I Built This

I was curious what it would take to turn a messy, incomplete support ticket into a clear, structured next step automatically — combining semantic search, an LLM, and a feedback loop into one workflow. This is a portfolio project built to explore that idea end to end, from ticket intake to resolution and learning.

---

## Pipeline Architecture

```
Ticket arrives
      │
      ▼
ChromaDB semantic search
(KB articles + past resolved tickets)
      │
      ▼
KB Similarity Gate ── score < threshold ──► BLOCK
(threshold: KB_SIMILARITY_THRESHOLD)         │
      │ score ≥ threshold                    │ "No KB match — investigate
      ▼                                      │  manually and submit fix"
NVIDIA NIM LLM                               │
(model chosen by ticket priority)            ▼
      │                               Engineer submits fix
      ▼                                      │
Response validated                           ▼
      │                               Claude Haiku summarizes
      ▼                               + anonymizes the fix
Claude Haiku Judge                           │
(independent provider)                       ▼
      │                               Fix stored in ChromaDB
      ▼                               (feeds future gate checks)
Full analysis + judge verdict
sent to engineer
      │
      ▼
Engineer marks feedback (helpful / not_helpful)
      │
      ▼
Metrics updated (success rate, false confidence)
```

---

## Model Routing by Priority

Different models are used based on the ticket's incoming priority, balancing cost against quality:

| Ticket Priority | Model | Provider |
|---|---|---|
| `low` | `meta/llama-3.1-8b-instruct` | NVIDIA NIM |
| `medium` | `meta/llama-3.3-70b-instruct` | NVIDIA NIM |
| `high` | `nvidia/llama-3.1-nemotron-70b-instruct` | NVIDIA NIM |
| Judge (all) | `claude-haiku-4-5-20251001` | Anthropic |

NVIDIA NIM offers free-tier access to these models with an OpenAI-compatible API. The judge always uses Claude Haiku — a different architecture and provider — so it catches blind spots the primary model may miss.

---

## Features

### Confidence Gate (KB Similarity)
- ChromaDB similarity score is computed before the LLM is called
- If the best match score is below `KB_SIMILARITY_THRESHOLD` (default `0.65`), the analysis is **blocked**
- The engineer receives a clear message: investigate manually and submit the fix
- Submitted fixes are summarized and anonymized by Claude Haiku before being stored in ChromaDB
- The gate score improves over time as more resolved tickets accumulate

### Judge LLM
- Runs only when the gate passes
- Uses Claude Haiku (Anthropic) — intentionally different from the primary NVIDIA NIM model
- Evaluates: correct issue classification? severity appropriate? checklist executable? fix aligned with KB?
- Returns `PASS`, `FAIL`, or `SKIPPED` (if judge itself failed)
- Does not block delivery — informs the engineer with concerns flagged

### Core Analysis
- **Issue Classification** — Integration Issue, Configuration Issue, API Issue, System Behavior, Data Sync Issue
- **Severity Assessment** — Low, Medium, High, Critical
- **Reproduction Checklist** — concrete steps to verify and reproduce the issue
- **Immediate Workaround** — what the store can do right now
- **Escalation Decision** — flags when third line involvement is needed
- **Internal Note** — structured summary for the support team
- **Customer Reply** — professional, empathetic partner communication

### Knowledge Base (ChromaDB)
- 10 pre-seeded KB articles stored as vectors
- Semantic search — matches meaning not just keywords
- Resolved ticket fixes are stored anonymized (no customer/store names)
- KB grows with every resolved ticket, improving future gate accuracy

### Analytics Dashboard
- Total tickets logged by status
- Escalation rate, KB match rate
- **Suggestion success rate** — how often engineers mark analyses as helpful
- **False confidence count** — gate passed + judge approved + engineer said not helpful
- **Average KB similarity score** — tracks KB coverage over time
- **Gate pass rate** and **judge pass rate**

### Engineer Feedback Loop
- `PATCH /logs/{id}/feedback` with `"helpful"` or `"not_helpful"`
- Feeds into success rate and false confidence metrics
- Closes the loop between AI output quality and real engineer experience

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI |
| Primary LLM | NVIDIA NIM — model varies by ticket priority |
| Judge LLM | Anthropic Claude Haiku |
| Vector Database | ChromaDB |
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
├── main.py            # FastAPI backend — all routes
├── prompts.py         # LLM pipeline — gate, NIM analysis, Claude judge, KB summarizer
├── vector_store.py    # ChromaDB — KB articles + anonymized resolved tickets
├── database.py        # SQLite — ticket logging, status, feedback, metrics
├── tickets.py         # 10 sample support tickets
├── test_main.py       # Pytest test suite (28 tests)
├── Dockerfile         # Container build definition
├── requirements.txt   # Python dependencies
├── .github/
│   └── workflows/
│       └── deploy.yml # CI/CD pipeline — test, build, deploy to Render
└── static/
    └── index.html     # Frontend UI — Analyze, Logs, Analytics tabs
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
| PATCH | `/logs/{id}/status` | Update ticket status |
| POST | `/logs/{id}/followup` | Generate follow-up reply |
| PATCH | `/logs/{id}/feedback` | Submit engineer feedback (helpful/not_helpful) |
| GET | `/stats` | Analytics dashboard data |
| GET | `/health` | Health check |
| GET | `/docs` | Auto-generated API docs |

---

## Getting Started

### Prerequisites
- Python 3.10+
- NVIDIA NIM API key — free at [build.nvidia.com](https://build.nvidia.com)
- Anthropic API key — free tier at [console.anthropic.com](https://console.anthropic.com)

### Installation

```bash
# Clone the repository
git clone https://github.com/Pranav14777/eve-support-tool.git
cd eve-support-tool

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the project root:

```
NVIDIA_API_KEY=nvapi-...
ANTHROPIC_API_KEY=sk-ant-...
KB_SIMILARITY_THRESHOLD=0.65
```

`KB_SIMILARITY_THRESHOLD` controls the gate — raise it (e.g. `0.80`) to be more strict, lower it (e.g. `0.50`) to let more tickets through to the LLM.

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

28 tests covering API endpoints, sample tickets, analysis, logging, analytics, and engineer feedback.

```bash
pip install pytest httpx pytest-asyncio
pytest test_main.py -v
```

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
3. Add environment variables: `NVIDIA_API_KEY`, `ANTHROPIC_API_KEY`, `KB_SIMILARITY_THRESHOLD`
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

## Production Considerations

This is a portfolio project. In production:

- **Data Privacy** — Real ticket data should be anonymized before sending to any external LLM API (the KB summarizer already strips client info from stored fixes)
- **Authentication** — API endpoints would require authentication
- **Ticketing Integration** — Would connect to Zendesk, Jira, or ServiceNow via webhook
- **Scalability** — SQLite would be replaced with PostgreSQL for production load
- **Gate Tuning** — `KB_SIMILARITY_THRESHOLD` would be tuned empirically based on false-negative rate

---

## Built By

Pranav Gadamsetty — built as a portfolio project, demonstrating support workflow thinking, LLM integration, and RAG architecture.

[LinkedIn](https://www.linkedin.com/in/pgdeveloper/) · [GitHub](https://github.com/Pranav14777)

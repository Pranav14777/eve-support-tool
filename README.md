# EVA Support Reproducer

[![CI/CD Pipeline](https://github.com/Pranav14777/eve-support-tool/actions/workflows/deploy.yml/badge.svg)](https://github.com/Pranav14777/eve-support-tool/actions/workflows/deploy.yml)

> A second-line support workflow tool for the EVA unified commerce platform — turning unstructured support tickets into structured, actionable intelligence.

**Live Demo:** [eve-support-tool-production.up.railway.app](https://eve-support-tool-production.up.railway.app/)

---

## What This Is

EVA Support Reproducer is an AI-powered internal support tool that simulates the second-line support workflow for a unified commerce platform.

When a support ticket comes in from a retail partner like Hunkemöller, Rituals, or Dyson, the tool:

1. Searches a semantic knowledge base for known issues and past resolutions
2. Sends the ticket to an LLM with relevant context
3. Returns a fully structured analysis
4. Logs everything automatically
5. Learns from resolved tickets to improve future analyses

---

## Why I Built This

I was curious what it would take to turn a messy, incomplete support ticket into a clear, structured next step automatically — combining semantic search, an LLM, and a feedback loop into one workflow. This is a portfolio project built to explore that idea end to end, from ticket intake to resolution and learning.

---

## Features

### Core Analysis
- **Issue Classification** — Integration Issue, Configuration Issue, API Issue, System Behavior, Data Sync Issue
- **Severity Assessment** — Low, Medium, High, Critical
- **Reproduction Checklist** — concrete steps to verify and reproduce the issue
- **Immediate Workaround** — what the store can do right now
- **Escalation Decision** — flags when third line involvement is needed
- **Internal Note** — structured summary for the support team
- **Customer Reply** — professional, empathetic partner communication

### Knowledge Base (ChromaDB)
- 10 pre-seeded EVA-specific KB articles stored as vectors
- Semantic search — matches meaning not just keywords
- "Payment processor down" matches Adyen article even without the word "adyen"

### Learning Feedback Loop
- Every resolved ticket with an actual fix gets stored in ChromaDB
- Future similar tickets automatically receive past resolution as context
- System gets smarter with every resolved ticket

### Ticket Lifecycle Management
- Auto-logging of every analysis to SQLite
- Status tracking — Open → In Progress → Escalated → Resolved
- Mark as Resolved with actual fix → automatically feeds ChromaDB
- Follow-up reply generator — turns a raw update into a professional partner message

### Analytics Dashboard
- Total tickets logged by status
- Escalation rate
- KB match rate
- Vector DB stats — KB articles + learned fixes
- Breakdown by issue type and severity

### Reliability
- Retry logic — tries LLM twice before falling back
- Validation — checks every required field before accepting LLM response
- Fallback handler — returns structured response even when LLM fails
- Graceful degradation — system always gives the engineer something useful

---

## Architecture

```
New ticket comes in
        ↓
ChromaDB semantic search
(KB articles + past resolved tickets)
        ↓
Top relevant chunks sent to LLM as context
        ↓
LLM generates structured analysis (Groq / Llama 3.3-70b)
        ↓
Response validated + fallback if needed
        ↓
Everything logged to SQLite automatically
        ↓
On resolution — fix stored back into ChromaDB
(feeds future ticket searches)
```

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI |
| LLM | Groq API — Llama 3.3-70b-versatile |
| Vector Database | ChromaDB |
| Relational Database | SQLite |
| Frontend | Vanilla HTML/CSS/JS |
| Containerization | Docker |
| CI/CD | GitHub Actions |
| Hosting | Railway |
| Language | Python 3.10+ |

---

## Project Structure

```
eve-support-tool/
├── main.py            # FastAPI backend — all routes
├── prompts.py         # LLM integration — Groq + ChromaDB search
├── vector_store.py    # ChromaDB — KB articles + resolved tickets
├── database.py        # SQLite — ticket logging + status tracking
├── tickets.py         # 10 sample EVA-realistic support tickets
├── test_main.py       # Pytest test suite
├── Dockerfile         # Container build definition
├── requirements.txt   # Python dependencies
├── .github/
│   └── workflows/
│       └── deploy.yml # CI/CD pipeline — test, build, deploy
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
| GET | `/stats` | Analytics dashboard data |
| GET | `/health` | Health check |
| GET | `/docs` | Auto-generated API docs |

---

## Getting Started

### Prerequisites
- Python 3.10+
- Groq API key — free at [console.groq.com](https://console.groq.com)

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

Create a `.env` file in the project root with your Groq API key:

```
GROQ_API_KEY=your-groq-api-key-here
```

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

26 tests covering API endpoints, sample tickets, analysis, logging, and analytics.

```bash
pip install pytest httpx pytest-asyncio
pytest test_main.py -v
```

---

## CI/CD & Deployment

Every push to `main` runs an automated pipeline via GitHub Actions ([`.github/workflows/deploy.yml`](.github/workflows/deploy.yml)):

1. **Test** — runs the full pytest suite
2. **Build** — builds the Docker image
3. **Deploy** — deploys to [Railway](https://railway.app) (main branch only)

Documentation-only changes (e.g. README updates) are skipped by the pipeline.

---

## Sample EVA Tickets

The tool comes pre-loaded with 10 realistic EVA platform support scenarios:

| ID | Issue | Store |
|---|---|---|
| TKT-001 | Payment via Adyen failing at checkout | Hunkemöller Berlin |
| TKT-002 | Click & collect orders not visible in POS | Rituals Amsterdam |
| TKT-003 | Inventory sync showing incorrect stock levels | Dyson Multiple Stores |
| TKT-004 | Product import failing with JSON error | Kiko Milano |
| TKT-005 | API returning 401 unauthorized | Intersport |
| TKT-006 | POS freezing on specific barcode scan | AFC Ajax Store |
| TKT-007 | Prices showing without VAT in German stores | Hunkemöller Germany |
| TKT-008 | Loyalty points not updating after purchase | Rituals Multiple Stores |
| TKT-009 | Receipt language not switching to locale | Multiple EU Stores |
| TKT-010 | Mobile POS app crashing after iOS update | Dyson UK Stores |

---

## Production Considerations

This is a prototype built to demonstrate support workflow thinking. In production:

- **Data Privacy** — Real ticket data should be anonymized before sending to any external LLM API, or an on-premise model should be used
- **Knowledge Base** — Would be replaced with a full ChromaDB instance seeded from actual resolved tickets
- **Authentication** — API endpoints would require authentication
- **Ticketing Integration** — Would connect to Zendesk, Jira, or ServiceNow via webhook
- **Scalability** — SQLite would be replaced with PostgreSQL for production load

---

## Built By

Pranav Gadamsetty — built as a portfolio project, demonstrating support workflow thinking, LLM integration, and RAG architecture.

[LinkedIn](https://www.linkedin.com/in/pgdeveloper/) · [GitHub](https://github.com/Pranav14777)

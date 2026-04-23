# EVA Support Reproducer

> A second-line support workflow tool built for the New Black EVA platform — turning unstructured support tickets into structured, actionable intelligence.

---

## What This Is

EVA Support Reproducer is an AI-powered internal support tool that simulates the second-line support workflow for the New Black EVA unified commerce platform.

When a support ticket comes in from a retail partner like Hunkemöller, Rituals, or Dyson, the tool:

1. Searches a semantic knowledge base for known issues and past resolutions
2. Sends the ticket to an LLM with relevant context
3. Returns a fully structured analysis
4. Logs everything automatically
5. Learns from resolved tickets to improve future analyses

---

## Why I Built This

The New Black Support Engineer role description mentions *"bringing structure to complexity"*. I built this prototype to think concretely about what that means in practice — taking a messy, incomplete ticket and turning it into a clear, structured next step for the engineer and the partner.

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
| Language | Python 3.10+ |

---

## Project Structure

```
eva-support-tool/
├── main.py           # FastAPI backend — all routes
├── prompts.py        # LLM integration — Groq + ChromaDB search
├── vector_store.py   # ChromaDB — KB articles + resolved tickets
├── database.py       # SQLite — ticket logging + status tracking
├── tickets.py        # 10 sample EVA-realistic support tickets
└── static/
    └── index.html    # Frontend UI — Analyze, Logs, Analytics tabs
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
pip install fastapi uvicorn groq chromadb python-dotenv
```

### Configuration

Open `prompts.py` and set your Groq API key:

```python
client = Groq(api_key="your-groq-api-key-here")
```

### Run

```bash
uvicorn main:app --reload
```

Open `http://localhost:8000`

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

Pranav Gadamsetty — built as a portfolio project for the New Black Support Engineer role, demonstrating support workflow thinking, LLM integration, and RAG architecture.

[LinkedIn](https://www.linkedin.com/in/pgdeveloper/) · [GitHub](https://github.com/Pranav14777)
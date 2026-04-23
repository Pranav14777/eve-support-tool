from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
from tickets import SAMPLE_TICKETS
from prompts import analyze_ticket, generate_follow_up_reply
from database import (
    log_ticket,
    get_all_logs,
    get_log_by_id,
    update_ticket_status,
    save_follow_up_reply,
    get_stats
)
from vector_store import add_resolved_ticket, get_vector_store_stats

app = FastAPI(
    title="EVA Support Issue Reproducer",
    description="Intelligent support workflow tool for New Black EVA platform",
    version="2.0.0"
)

app.mount("/static", StaticFiles(directory="static"), name="static")

# ── Models ─────────────────────────────────────────────────────────────────────

class TicketInput(BaseModel):
    id: str = "TKT-CUSTOM"
    title: str
    description: str
    store: str = "Unknown Store"
    priority: str = "medium"

class StatusUpdate(BaseModel):
    status: str
    actual_fix: Optional[str] = None

class FollowUpInput(BaseModel):
    update: str

# ── Frontend ───────────────────────────────────────────────────────────────────

@app.get("/")
def serve_frontend():
    return FileResponse("static/index.html")

# ── Tickets ────────────────────────────────────────────────────────────────────

@app.get("/tickets")
def get_all_tickets():
    """Return all sample EVA support tickets"""
    return {
        "total": len(SAMPLE_TICKETS),
        "tickets": SAMPLE_TICKETS
    }

@app.get("/tickets/{ticket_id}")
def get_ticket(ticket_id: str):
    """Get a single sample ticket by ID"""
    ticket = next(
        (t for t in SAMPLE_TICKETS if t["id"] == ticket_id),
        None
    )
    if not ticket:
        raise HTTPException(
            status_code=404,
            detail=f"Ticket {ticket_id} not found"
        )
    return ticket

# ── Analysis ───────────────────────────────────────────────────────────────────

@app.get("/analyze/{ticket_id}")
def analyze_sample_ticket(ticket_id: str):
    """Analyze a sample ticket — reuse existing log if already analyzed today"""
    ticket = next(
        (t for t in SAMPLE_TICKETS if t["id"] == ticket_id),
        None
    )
    if not ticket:
        raise HTTPException(
            status_code=404,
            detail=f"Ticket {ticket_id} not found"
        )

    try:
        analysis = analyze_ticket(ticket)

        # Check if this sample ticket was already logged today
        existing_logs = get_all_logs()
        from datetime import date
        today = date.today().isoformat()
        
        existing = next(
            (l for l in existing_logs 
             if l["ticket_id"] == ticket_id 
             and l["created_at"].startswith(today)),
            None
        )

        if existing:
            # Reuse existing log ID instead of creating duplicate
            log_id = existing["log_id"]
        else:
            # First time today — create new log
            log_id = log_ticket(ticket, analysis)

        return {
            "log_id": log_id,
            "ticket": ticket,
            "analysis": analysis
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )

@app.post("/analyze/custom")
def analyze_custom_ticket(ticket: TicketInput):
    """Analyze a custom ticket and log it automatically"""
    ticket_dict = ticket.model_dump()

    try:
        analysis = analyze_ticket(ticket_dict)

        # Auto log every analysis
        log_id = log_ticket(ticket_dict, analysis)

        return {
            "log_id": log_id,
            "ticket": ticket_dict,
            "analysis": analysis
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )

# ── Logs ───────────────────────────────────────────────────────────────────────

@app.get("/logs")
def get_logs():
    """Get all logged tickets"""
    logs = get_all_logs()
    return {
        "total": len(logs),
        "logs": logs
    }

@app.get("/logs/{log_id}")
def get_log(log_id: int):
    """Get full details of a single logged ticket"""
    log = get_log_by_id(log_id)
    if not log:
        raise HTTPException(
            status_code=404,
            detail=f"Log {log_id} not found"
        )
    return log

# ── Status + Resolution ────────────────────────────────────────────────────────

@app.patch("/logs/{log_id}/status")
def update_status(log_id: int, body: StatusUpdate):
    """
    Update ticket status.
    When marked Resolved with actual_fix — automatically adds to ChromaDB
    so future similar tickets benefit from this resolution.
    """
    valid_statuses = ["Open", "In Progress", "Escalated", "Resolved"]
    if body.status not in valid_statuses:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid status. Must be one of: {valid_statuses}"
        )

    # Update SQLite
    update_ticket_status(log_id, body.status, body.actual_fix)

    # If resolved with actual fix — feed into ChromaDB for future learning
    if body.status == "Resolved" and body.actual_fix:
        log = get_log_by_id(log_id)
        if log:
            ticket = {
                "id": log["ticket_id"],
                "title": log["title"],
                "description": log["description"],
                "store": log["store"],
                "priority": log["priority"]
            }
            analysis = {
                "issue_type": log["issue_type"],
                "likely_cause": log["likely_cause"]
            }
            # Add to ChromaDB — this is the learning feedback loop
            add_resolved_ticket(log_id, ticket, analysis, body.actual_fix)

    return {
        "success": True,
        "log_id": log_id,
        "new_status": body.status,
        "fed_to_vector_store": body.status == "Resolved" and bool(body.actual_fix)
    }

# ── Follow Up ──────────────────────────────────────────────────────────────────

@app.post("/logs/{log_id}/followup")
def generate_follow_up(log_id: int, body: FollowUpInput):
    """Generate and save a follow up reply for an ongoing ticket"""
    log = get_log_by_id(log_id)
    if not log:
        raise HTTPException(
            status_code=404,
            detail=f"Log {log_id} not found"
        )

    ticket = {
        "title": log["title"],
        "store": log["store"]
    }
    analysis = {
        "customer_reply": log["customer_reply"]
    }

    follow_up = generate_follow_up_reply(ticket, analysis, body.update)

    # Save to database
    save_follow_up_reply(log_id, follow_up)

    return {
        "log_id": log_id,
        "follow_up_reply": follow_up
    }

# ── Analytics ──────────────────────────────────────────────────────────────────

@app.get("/stats")
def get_statistics():
    """Get analytics across all logged tickets"""
    db_stats = get_stats()
    vector_stats = get_vector_store_stats()

    return {
        "database": db_stats,
        "vector_store": vector_stats
    }

# ── Health ─────────────────────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "EVA Support Issue Reproducer",
        "version": "2.0.0"
    }
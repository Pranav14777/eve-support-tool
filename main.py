import os
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
from tickets import SAMPLE_TICKETS
from prompts import analyze_ticket, generate_follow_up_reply, summarize_fix_for_kb
from database import (
    log_ticket,
    get_all_logs,
    get_log_by_id,
    update_ticket_status,
    save_follow_up_reply,
    save_engineer_feedback,
    get_stats
)
from vector_store import add_resolved_ticket, get_vector_store_stats

app = FastAPI(
    title="EVA Support Issue Reproducer",
    description="Intelligent support workflow tool for a unified commerce retail platform",
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

class FeedbackInput(BaseModel):
    feedback: str  # "helpful" | "not_helpful"

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

def _gate_fail_response(log_id: int, analysis: dict) -> dict:
    """Format the hard-block response when KB similarity gate fails"""
    threshold = float(os.environ.get("KB_SIMILARITY_THRESHOLD", "0.65"))
    return {
        "log_id": log_id,
        "gate_passed": False,
        "kb_similarity_score": analysis.get("kb_similarity_score"),
        "threshold": threshold,
        "message": (
            f"No sufficiently similar KB article or past resolution found "
            f"(similarity {analysis.get('kb_similarity_score', 0):.3f} < threshold {threshold}). "
            "Please investigate manually and submit the actual fix — "
            "it will be added to the knowledge base to help with future tickets."
        ),
        "analysis": None
    }

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

        existing_logs = get_all_logs()
        from datetime import date
        today = date.today().isoformat()

        # Only reuse logs where the gate actually passed (not blocked attempts)
        existing = next(
            (l for l in existing_logs
             if l["ticket_id"] == ticket_id
             and l["created_at"].startswith(today)
             and l.get("gate_passed") is not False),
            None
        )

        if existing:
            log_id = existing["log_id"]
        else:
            log_id = log_ticket(ticket, analysis)

        if not analysis.get("gate_passed", True):
            return _gate_fail_response(log_id, analysis)

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
        log_id = log_ticket(ticket_dict, analysis)

        if not analysis.get("gate_passed", True):
            return _gate_fail_response(log_id, analysis)

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
    When marked Resolved with actual_fix — summarizes and anonymizes the fix,
    then adds it to ChromaDB so future similar tickets benefit.
    """
    valid_statuses = ["Open", "In Progress", "Escalated", "Resolved"]
    if body.status not in valid_statuses:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid status. Must be one of: {valid_statuses}"
        )

    update_ticket_status(log_id, body.status, body.actual_fix)

    fed_to_vector_store = False
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
            # Summarize and anonymize the fix before storing in ChromaDB
            clean_fix = summarize_fix_for_kb(ticket, body.actual_fix)
            add_resolved_ticket(log_id, ticket, analysis, clean_fix)
            fed_to_vector_store = True

    return {
        "success": True,
        "log_id": log_id,
        "new_status": body.status,
        "fed_to_vector_store": fed_to_vector_store
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
    save_follow_up_reply(log_id, follow_up)

    return {
        "log_id": log_id,
        "follow_up_reply": follow_up
    }

# ── Engineer Feedback ──────────────────────────────────────────────────────────

@app.patch("/logs/{log_id}/feedback")
def submit_feedback(log_id: int, body: FeedbackInput):
    """Record whether the engineer found this analysis helpful or not"""
    valid_feedback = ["helpful", "not_helpful"]
    if body.feedback not in valid_feedback:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid feedback. Must be one of: {valid_feedback}"
        )

    log = get_log_by_id(log_id)
    if not log:
        raise HTTPException(
            status_code=404,
            detail=f"Log {log_id} not found"
        )

    save_engineer_feedback(log_id, body.feedback)

    return {
        "success": True,
        "log_id": log_id,
        "feedback": body.feedback
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

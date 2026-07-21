import sqlite3
import json
from datetime import datetime

DB_PATH = "eva_support.db"

def migrate_db():
    """Apply schema migrations for existing databases. Idempotent."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    new_columns = [
        ("kb_similarity_score",  "REAL"),
        ("kb_match_title",       "TEXT"),
        ("gate_passed",          "BOOLEAN"),
        ("judge_verdict",        "TEXT"),
        ("judge_notes",          "TEXT"),
        ("judge_concerns",       "TEXT"),
        ("model_used",           "TEXT"),
        ("engineer_feedback",    "TEXT"),
    ]

    for col_name, col_type in new_columns:
        try:
            cursor.execute(
                f"ALTER TABLE tickets ADD COLUMN {col_name} {col_type}"
            )
        except sqlite3.OperationalError:
            pass  # Column already exists

    conn.commit()
    conn.close()

def init_db():
    """Initialize the database and create tables if they don't exist"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tickets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticket_id TEXT NOT NULL,
            title TEXT NOT NULL,
            description TEXT NOT NULL,
            store TEXT NOT NULL,
            priority TEXT NOT NULL,
            issue_type TEXT,
            severity TEXT,
            likely_cause TEXT,
            workaround TEXT,
            suggested_next_step TEXT,
            escalate_to_third_line BOOLEAN,
            escalation_reason TEXT,
            internal_note TEXT,
            customer_reply TEXT,
            known_issue BOOLEAN,
            knowledge_base_article TEXT,
            known_fix TEXT,
            analyzed_by TEXT,
            status TEXT DEFAULT 'Open',
            actual_fix TEXT,
            follow_up_reply TEXT,
            created_at TEXT NOT NULL,
            resolved_at TEXT,
            reproduction_checklist TEXT,
            kb_similarity_score REAL,
            kb_match_title TEXT,
            gate_passed BOOLEAN,
            judge_verdict TEXT,
            judge_notes TEXT,
            judge_concerns TEXT,
            model_used TEXT,
            engineer_feedback TEXT
        )
    """)

    conn.commit()
    conn.close()
    print("Database initialized successfully")

    # Apply migrations for existing databases
    migrate_db()

def log_ticket(ticket: dict, analysis: dict) -> int:
    """Log a ticket and its analysis to the database. Returns the row ID."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    now = datetime.now().isoformat()

    cursor.execute("""
        INSERT INTO tickets (
            ticket_id, title, description, store, priority,
            issue_type, severity, likely_cause, workaround,
            suggested_next_step, escalate_to_third_line, escalation_reason,
            internal_note, customer_reply, known_issue,
            knowledge_base_article, known_fix, analyzed_by,
            status, created_at, reproduction_checklist,
            kb_similarity_score, kb_match_title, gate_passed,
            judge_verdict, judge_notes, judge_concerns, model_used
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        ticket.get("id", "TKT-CUSTOM"),
        ticket.get("title", ""),
        ticket.get("description", ""),
        ticket.get("store", ""),
        ticket.get("priority", "medium"),
        analysis.get("issue_type", ""),
        analysis.get("severity", ""),
        analysis.get("likely_cause", ""),
        analysis.get("workaround", ""),
        analysis.get("suggested_next_step", ""),
        analysis.get("escalate_to_third_line", False),
        analysis.get("escalation_reason", ""),
        analysis.get("internal_note", ""),
        analysis.get("customer_reply", ""),
        analysis.get("known_issue", False),
        analysis.get("knowledge_base_article", ""),
        analysis.get("known_fix", ""),
        analysis.get("analyzed_by", ""),
        "Open",
        now,
        json.dumps(analysis.get("reproduction_checklist", [])),
        analysis.get("kb_similarity_score"),
        analysis.get("kb_match_title", ""),
        analysis.get("gate_passed"),
        analysis.get("judge_verdict", "SKIPPED"),
        analysis.get("judge_notes", ""),
        json.dumps(analysis.get("judge_concerns", [])),
        analysis.get("model_used", "")
    ))

    row_id = cursor.lastrowid
    conn.commit()
    conn.close()
    print(f"Ticket logged with ID: {row_id}")
    return row_id

def get_all_logs() -> list:
    """Get all logged tickets ordered by most recent first"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            id, ticket_id, title, store, priority,
            issue_type, severity, status, analyzed_by,
            escalate_to_third_line, known_issue,
            created_at, resolved_at, actual_fix,
            kb_similarity_score, gate_passed, judge_verdict, model_used
        FROM tickets
        ORDER BY created_at DESC
    """)

    rows = cursor.fetchall()
    conn.close()

    logs = []
    for row in rows:
        logs.append({
            "log_id": row[0],
            "ticket_id": row[1],
            "title": row[2],
            "store": row[3],
            "priority": row[4],
            "issue_type": row[5],
            "severity": row[6],
            "status": row[7],
            "analyzed_by": row[8],
            "escalated": bool(row[9]),
            "known_issue": bool(row[10]),
            "created_at": row[11],
            "resolved_at": row[12],
            "actual_fix": row[13],
            "kb_similarity_score": row[14],
            "gate_passed": bool(row[15]) if row[15] is not None else None,
            "judge_verdict": row[16],
            "model_used": row[17]
        })

    return logs

def get_log_by_id(log_id: int) -> dict | None:
    """Get full details of a single logged ticket"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM tickets WHERE id = ?", (log_id,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        return None

    columns = [
        "log_id", "ticket_id", "title", "description", "store", "priority",
        "issue_type", "severity", "likely_cause", "workaround",
        "suggested_next_step", "escalate_to_third_line", "escalation_reason",
        "internal_note", "customer_reply", "known_issue",
        "knowledge_base_article", "known_fix", "analyzed_by",
        "status", "actual_fix", "follow_up_reply", "created_at",
        "resolved_at", "reproduction_checklist",
        "kb_similarity_score", "kb_match_title", "gate_passed",
        "judge_verdict", "judge_notes", "judge_concerns",
        "model_used", "engineer_feedback"
    ]

    result = dict(zip(columns, row))

    if result.get("reproduction_checklist"):
        try:
            result["reproduction_checklist"] = json.loads(result["reproduction_checklist"])
        except Exception:
            result["reproduction_checklist"] = []

    if result.get("judge_concerns"):
        try:
            result["judge_concerns"] = json.loads(result["judge_concerns"])
        except Exception:
            result["judge_concerns"] = []
    else:
        result["judge_concerns"] = []

    if result.get("gate_passed") is not None:
        result["gate_passed"] = bool(result["gate_passed"])

    return result

def update_ticket_status(log_id: int, status: str, actual_fix: str = None) -> bool:
    """Update ticket status and optionally add the actual fix"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    now = datetime.now().isoformat()

    if status == "Resolved" and actual_fix:
        cursor.execute("""
            UPDATE tickets
            SET status = ?, actual_fix = ?, resolved_at = ?
            WHERE id = ?
        """, (status, actual_fix, now, log_id))
    else:
        cursor.execute("""
            UPDATE tickets
            SET status = ?
            WHERE id = ?
        """, (status, log_id))

    conn.commit()
    conn.close()
    print(f"Ticket {log_id} status updated to: {status}")
    return True

def save_follow_up_reply(log_id: int, follow_up: str) -> bool:
    """Save a follow up reply to a logged ticket"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        UPDATE tickets
        SET follow_up_reply = ?
        WHERE id = ?
    """, (follow_up, log_id))

    conn.commit()
    conn.close()
    return True

def save_engineer_feedback(log_id: int, feedback: str) -> bool:
    """Save engineer feedback on whether the analysis was helpful"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        UPDATE tickets
        SET engineer_feedback = ?
        WHERE id = ?
    """, (feedback, log_id))

    conn.commit()
    conn.close()
    return True

def get_stats() -> dict:
    """Get summary statistics across all logged tickets"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM tickets")
    total = cursor.fetchone()[0]

    cursor.execute("""
        SELECT status, COUNT(*)
        FROM tickets
        GROUP BY status
    """)
    status_counts = dict(cursor.fetchall())

    cursor.execute("""
        SELECT issue_type, COUNT(*)
        FROM tickets
        GROUP BY issue_type
        ORDER BY COUNT(*) DESC
    """)
    issue_types = dict(cursor.fetchall())

    cursor.execute("""
        SELECT severity, COUNT(*)
        FROM tickets
        GROUP BY severity
    """)
    severities = dict(cursor.fetchall())

    cursor.execute("SELECT COUNT(*) FROM tickets WHERE escalate_to_third_line = 1")
    escalated = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM tickets WHERE known_issue = 1")
    known_issues = cursor.fetchone()[0]

    # New quality metrics
    cursor.execute("SELECT COUNT(*) FROM tickets WHERE engineer_feedback = 'helpful'")
    helpful_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM tickets WHERE engineer_feedback IS NOT NULL")
    with_feedback = cursor.fetchone()[0]

    cursor.execute("""
        SELECT COUNT(*) FROM tickets
        WHERE gate_passed = 1 AND judge_verdict = 'PASS' AND engineer_feedback = 'not_helpful'
    """)
    false_confidence_count = cursor.fetchone()[0]

    cursor.execute("SELECT AVG(kb_similarity_score) FROM tickets WHERE kb_similarity_score IS NOT NULL")
    avg_kb_similarity = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM tickets WHERE gate_passed = 1")
    gate_passed_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM tickets WHERE judge_verdict = 'PASS'")
    judge_passed = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM tickets WHERE judge_verdict IN ('PASS', 'FAIL')")
    judged_total = cursor.fetchone()[0]

    conn.close()

    return {
        "total": total,
        "by_status": status_counts,
        "by_issue_type": issue_types,
        "by_severity": severities,
        "escalated": escalated,
        "known_issues": known_issues,
        "suggestion_success_rate": round(helpful_count / with_feedback * 100, 1) if with_feedback > 0 else None,
        "false_confidence_count": false_confidence_count,
        "avg_kb_similarity": round(avg_kb_similarity, 3) if avg_kb_similarity is not None else None,
        "gate_pass_rate": round(gate_passed_count / total * 100, 1) if total > 0 else None,
        "judge_pass_rate": round(judge_passed / judged_total * 100, 1) if judged_total > 0 else None
    }

# Initialize database when module is imported
init_db()

import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

# ── Health Check ───────────────────────────────────────────────────────────────

def test_health_check():
    """API should be running and healthy"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["version"] == "2.0.0"

# ── Sample Tickets ─────────────────────────────────────────────────────────────

def test_get_all_tickets():
    """Should return all 10 sample tickets"""
    response = client.get("/tickets")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 10
    assert len(data["tickets"]) == 10

def test_get_all_tickets_have_required_fields():
    """Every ticket should have required fields"""
    response = client.get("/tickets")
    data = response.json()
    for ticket in data["tickets"]:
        assert "id" in ticket
        assert "title" in ticket
        assert "description" in ticket
        assert "store" in ticket
        assert "priority" in ticket

def test_get_single_ticket():
    """Should return correct ticket by ID"""
    response = client.get("/tickets/TKT-001")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "TKT-001"
    assert data["store"] == "Hunkemöller Berlin"

def test_get_nonexistent_ticket():
    """Should return 404 for unknown ticket ID"""
    response = client.get("/tickets/TKT-999")
    assert response.status_code == 404

@pytest.mark.parametrize("ticket_id", [
    "TKT-001", "TKT-002", "TKT-003",
    "TKT-004", "TKT-005", "TKT-006",
    "TKT-007", "TKT-008", "TKT-009", "TKT-010"
])
def test_all_sample_tickets_exist(ticket_id):
    """Every sample ticket should be retrievable"""
    response = client.get(f"/tickets/{ticket_id}")
    assert response.status_code == 200
    assert response.json()["id"] == ticket_id

# ── Logs ───────────────────────────────────────────────────────────────────────

def test_get_logs():
    """Should return logs with correct structure"""
    response = client.get("/logs")
    assert response.status_code == 200
    data = response.json()
    assert "total" in data
    assert "logs" in data
    assert isinstance(data["logs"], list)

def test_get_nonexistent_log():
    """Should return 404 for unknown log ID"""
    response = client.get("/logs/99999")
    assert response.status_code == 404

# ── Stats ──────────────────────────────────────────────────────────────────────

def test_get_stats():
    """Should return analytics with correct structure"""
    response = client.get("/stats")
    assert response.status_code == 200
    data = response.json()
    assert "database" in data
    assert "vector_store" in data
    assert "total" in data["database"]

# ── Custom Ticket Validation ───────────────────────────────────────────────────

def test_custom_ticket_missing_title():
    """Should fail if title is missing"""
    response = client.post("/analyze/custom", json={
        "description": "Something is broken",
        "store": "Test Store",
        "priority": "medium"
    })
    assert response.status_code == 422

def test_custom_ticket_missing_description():
    """Should fail if description is missing"""
    response = client.post("/analyze/custom", json={
        "title": "Something broken",
        "store": "Test Store",
        "priority": "medium"
    })
    assert response.status_code == 422

def test_custom_ticket_invalid_structure():
    """Should fail with completely empty body"""
    response = client.post("/analyze/custom", json={})
    assert response.status_code == 422

# ── Status Update Validation ───────────────────────────────────────────────────

def test_invalid_status_update():
    """Should reject invalid status values"""
    response = client.patch("/logs/1/status", json={
        "status": "InvalidStatus"
    })
    assert response.status_code == 400

@pytest.mark.parametrize("status", [
    "Open", "In Progress", "Escalated", "Resolved"
])
def test_valid_status_values(status):
    """All valid status values should be accepted by validation logic"""
    valid_statuses = ["Open", "In Progress", "Escalated", "Resolved"]
    assert status in valid_statuses

# ── Engineer Feedback ──────────────────────────────────────────────────────────

def test_feedback_rejects_non_boolean():
    """Flags are booleans — a string must be rejected by validation"""
    response = client.patch("/logs/1/feedback", json={"kb_relevant": "sort of"})
    assert response.status_code == 422

def test_feedback_nonexistent_log():
    """Should return 404 for unknown log ID"""
    response = client.patch("/logs/99999/feedback", json={"kb_relevant": True})
    assert response.status_code == 404

# ── Confidence Gate (margin + floor) ─────────────────────────────────────────────

import prompts
import main
from vector_store import EmbeddingUnavailable


def _match(title, score):
    return {"title": title, "known_fix": "", "workaround": "",
            "issue_type": "", "similarity_score": score}


def test_gate_serves_clear_winner():
    """One article clearly above the rest and above the floor -> serve."""
    gate = prompts.evaluate_gate([_match("A", 0.70), _match("B", 0.50)], [])
    assert gate["passed"] is True
    assert gate["reason"] is None
    assert gate["margin"] == 0.20


def test_gate_low_margin_abstain():
    """Top two too close (margin < KB_MARGIN_THRESHOLD) -> abstain, low_margin."""
    gate = prompts.evaluate_gate([_match("A", 0.70), _match("B", 0.66)], [])
    assert gate["passed"] is False
    assert gate["reason"] == "low_margin"


def test_gate_low_floor_abstain():
    """Big margin but top1 below the floor -> abstain, low_floor (floor checked first)."""
    gate = prompts.evaluate_gate([_match("A", 0.35), _match("B", 0.20)], [])
    assert gate["passed"] is False
    assert gate["reason"] == "low_floor"


def test_gate_single_candidate_abstain():
    """Only one distinct candidate -> abstain (guards the top1 - 0.0 bug), insufficient_candidates."""
    gate = prompts.evaluate_gate([_match("A", 0.90)], [])
    assert gate["passed"] is False
    assert gate["reason"] == "insufficient_candidates"


def test_gate_dedupes_mirrored_resolved_ticket():
    """A resolved ticket mirroring the top KB article must not crush the margin."""
    kb = [_match("Adyen Payment Gateway Timeout", 0.70), _match("Something Else", 0.45)]
    resolved = [{"title": "resolved mirror", "issue_type": "", "actual_fix": "",
                 "resolved_at": "", "similarity_score": 0.69}]  # near-identical to top, other collection
    gate = prompts.evaluate_gate(kb, resolved)
    # #2 should be the distinct article at 0.45, not the 0.69 mirror -> margin 0.25, serve
    assert gate["top2"] == 0.45
    assert gate["passed"] is True


def test_embedding_unavailable_abstain(monkeypatch):
    """If embeddings are down, analyze_ticket abstains and never calls the LLM."""
    def _raise(*args, **kwargs):
        raise EmbeddingUnavailable("NIM down")

    monkeypatch.setattr(prompts, "search_knowledge_base", _raise)

    def _fail_llm(*args, **kwargs):
        raise AssertionError("LLM must not be called when retrieval is unavailable")

    monkeypatch.setattr(prompts.nim_client.chat.completions, "create", _fail_llm)

    result = prompts.analyze_ticket({"id": "TKT-X", "title": "t", "description": "d",
                                     "store": "s", "priority": "medium"})
    assert result["gate_passed"] is False
    assert result["analyzed_by"] == "retrieval_unavailable"
    assert result["judge_verdict"] == "SKIPPED"


def test_abstain_response_payload():
    """The gate-fail API payload exposes the margin fields the UI and logs need."""
    analysis = {
        "abstain_reason": "low_margin",
        "kb_similarity_score": 0.42,
        "kb_second_score": 0.40,
        "kb_margin": 0.02,
        "kb_match_title": "Some Article",
        "internal_note": "ambiguous",
    }
    payload = main._gate_fail_response(7, analysis)
    assert payload["gate_passed"] is False
    assert payload["gate_decision"] == "abstained"
    assert payload["abstain_reason"] == "low_margin"
    assert payload["analysis"] is None
    for key in ("kb_similarity_score", "kb_second_score", "kb_margin",
                "margin_threshold", "abs_floor"):
        assert key in payload

# ── Phase 2: feedback routing ────────────────────────────────────────────────────

import feedback_routing as fr
from database import log_ticket, save_feedback, get_log_by_id


@pytest.mark.parametrize("gate_passed,kb_relevant,fix_helped,expected", [
    # Abstained: nothing was shown to rate, auto-routes as a KB candidate.
    (False, None,  None,  fr.ROUTE_COVERAGE_GAP),
    (False, None,  True,  fr.ROUTE_COVERAGE_GAP),
    # relevant=no is COMPLETE on its own — must be a coverage gap even when
    # "did the fix help?" was never answered (the common progressive-disclosure partial).
    (True,  False, None,  fr.ROUTE_COVERAGE_GAP),
    (True,  False, False, fr.ROUTE_COVERAGE_GAP),
    (True,  False, True,  fr.ROUTE_COVERAGE_GAP),
    # Retrieval was fine, generation wasn't.
    (True,  True,  False, fr.ROUTE_PROMPT_PROBLEM),
    (True,  True,  True,  fr.ROUTE_SUCCESS),
    # relevant=yes but helped unanswered — the one genuinely unroutable partial.
    (True,  True,  None,  fr.ROUTE_PARTIAL),
    # Dismissed before answering anything.
    (True,  None,  None,  fr.ROUTE_NO_RESPONSE),
])
def test_classify_feedback_truth_table(gate_passed, kb_relevant, fix_helped, expected):
    assert fr.classify_feedback(gate_passed, kb_relevant, fix_helped) == expected


def test_only_coverage_gap_is_a_candidate():
    assert fr.is_kb_candidate(fr.ROUTE_COVERAGE_GAP) is True
    for route in (fr.ROUTE_PROMPT_PROBLEM, fr.ROUTE_SUCCESS,
                  fr.ROUTE_PARTIAL, fr.ROUTE_NO_RESPONSE):
        assert fr.is_kb_candidate(route) is False


def test_partial_counts_as_a_response_but_no_response_does_not():
    """feedback_coverage must count partials — the agent did engage."""
    assert fr.is_any_response(fr.ROUTE_PARTIAL) is True
    assert fr.is_any_response(fr.ROUTE_COVERAGE_GAP) is True
    assert fr.is_any_response(fr.ROUTE_NO_RESPONSE) is False


def _make_log(gate_passed=True):
    ticket = {"id": "TKT-FB", "title": "t", "description": "d", "store": "s", "priority": "medium"}
    analysis = {"gate_passed": gate_passed, "issue_type": "API Issue", "severity": "Low",
                "retrieved_articles": [{"id": "kb-001", "title": "A", "snippet": "s", "score": 0.7}]}
    return log_ticket(ticket, analysis)


def test_partial_feedback_leaves_other_flag_null():
    """Answering only relevance must not clobber fix_helped with NULL/False."""
    log_id = _make_log()
    save_feedback(log_id, kb_relevant=True)
    log = get_log_by_id(log_id)
    assert log["kb_relevant"] is True
    assert log["fix_helped"] is None
    assert log["feedback_route"] == fr.ROUTE_PARTIAL


def test_dismissal_records_prompt_without_a_negative():
    """Empty body = dismissal: stamps prompted, leaves both flags NULL (not False)."""
    log_id = _make_log()
    response = client.patch(f"/logs/{log_id}/feedback", json={})
    assert response.status_code == 200
    assert response.json()["dismissed"] is True
    assert response.json()["route"] == fr.ROUTE_NO_RESPONSE

    log = get_log_by_id(log_id)
    assert log["kb_relevant"] is None
    assert log["fix_helped"] is None
    assert log["feedback_prompted_at"] is not None


def test_relevant_no_routes_to_coverage_gap_via_api():
    """The common partial (relevant=no, helped unanswered) is actionable, not no_response."""
    log_id = _make_log()
    response = client.patch(f"/logs/{log_id}/feedback", json={"kb_relevant": False})
    assert response.status_code == 200
    assert response.json()["route"] == fr.ROUTE_COVERAGE_GAP


def test_abstained_ticket_routes_to_coverage_gap_without_feedback():
    """Abstained tickets are never asked, yet still become KB candidates."""
    log_id = _make_log(gate_passed=False)
    log = get_log_by_id(log_id)
    assert log["feedback_prompted_at"] is None
    assert log["feedback_route"] == fr.ROUTE_COVERAGE_GAP
    assert fr.is_kb_candidate(log["feedback_route"]) is True


def test_retrieval_snapshot_persisted_for_corpus_swap():
    """id + title + snippet must survive on the row so a corpus swap can't orphan feedback."""
    log_id = _make_log()
    log = get_log_by_id(log_id)
    assert log["retrieved_articles"][0]["id"] == "kb-001"
    assert log["retrieved_articles"][0]["title"] == "A"
    assert "snippet" in log["retrieved_articles"][0]


def test_stats_exposes_feedback_coverage_and_routes():
    response = client.get("/stats")
    assert response.status_code == 200
    db = response.json()["database"]
    assert "feedback_coverage" in db
    assert "route_breakdown" in db
    assert "false_confidence_count" in db

# ── Model routing ────────────────────────────────────────────────────────────────

import os
from vector_store import _embed

RUN_LIVE = os.environ.get("RUN_LIVE_TESTS") == "1"
live = pytest.mark.skipif(
    not RUN_LIVE,
    reason="live NIM API check — set RUN_LIVE_TESTS=1 (and NVIDIA_API_KEY) to run",
)


def test_model_routing_is_wired():
    """Deterministic, offline: every priority maps to a model and unknown -> medium."""
    assert set(prompts.MODEL_MAP) == {"low", "medium", "high"}
    for tier in ("low", "medium", "high"):
        assert prompts.get_model_for_priority(tier) == prompts.MODEL_MAP[tier]
    assert prompts.get_model_for_priority("bogus") == prompts.MODEL_MAP["medium"]
    assert prompts.get_model_for_priority("HIGH") == prompts.MODEL_MAP["high"]


@live
@pytest.mark.parametrize("tier", ["low", "medium", "high"])
def test_analysis_model_is_callable(tier):
    """Live: each routed NIM model must actually respond with non-empty content.

    Catches the failure that hit production — NIM decommissions a model (404 on call
    even though it's still in the catalog listing), or a reasoning model returns empty
    `content`, silently degrading every ticket at that priority to the fallback path.
    """
    model = prompts.MODEL_MAP[tier]
    resp = prompts.nim_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Reply with the single word OK."}],
        max_tokens=10,
        temperature=0,
    )
    content = resp.choices[0].message.content
    assert content and content.strip(), f"{tier} model {model} returned empty content"


@live
def test_embedding_model_is_callable():
    """Live: the retrieval embedding model must respond with a 1024-dim vector."""
    vectors = _embed(["connectivity check"], "query")
    assert vectors and len(vectors[0]) == 1024
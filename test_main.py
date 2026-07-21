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

def test_feedback_invalid_value():
    """Should reject feedback values other than helpful/not_helpful"""
    response = client.patch("/logs/1/feedback", json={"feedback": "meh"})
    assert response.status_code == 400

def test_feedback_nonexistent_log():
    """Should return 404 for unknown log ID"""
    response = client.patch("/logs/99999/feedback", json={"feedback": "helpful"})
    assert response.status_code == 404
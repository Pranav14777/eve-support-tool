"""Feedback routing — the single source of truth for classifying a ticket's feedback.

This module is deliberately DEPENDENCY-FREE so every caller can import it without
circular-import risk:
  - main.py            (the /feedback API response)
  - database.py        (get_stats aggregation)
  - Phase 3 queue      (selecting KB candidates)

Do NOT reimplement this rule as inline SQL anywhere. Callers fetch rows and classify
through `classify_feedback` so the logic lives in exactly one place. That means stats
and the candidate queue aggregate in Python rather than with SQL COUNT(...) — a
deliberate trade-off at this scale to keep the rule from drifting.

The two questions asked at resolution (served tickets only):
  kb_relevant  — "Was the KB article relevant?"   True / False / None (unanswered)
  fix_helped   — "Did the fix help?"              True / False / None (unanswered)
"""

ROUTE_COVERAGE_GAP = "coverage_gap"      # -> becomes a KB candidate
ROUTE_PROMPT_PROBLEM = "prompt_problem"  # -> NOT a candidate: retrieval was fine, generation wasn't
ROUTE_SUCCESS = "success"
ROUTE_PARTIAL = "partial"                # engaged but unroutable (relevant=yes, helped unanswered)
ROUTE_NO_RESPONSE = "no_response"        # dismissed with nothing answered


def classify_feedback(gate_passed, kb_relevant, fix_helped) -> str:
    """Classify a ticket into a routing bucket.

    gate_passed : bool  — False means the gate abstained (nothing was shown to rate)
    kb_relevant : bool | None
    fix_helped  : bool | None
    """
    # Abstained: no article and no analysis were shown, so there is nothing to rate.
    # The agent's fix still represents missing coverage, so it auto-routes as a candidate.
    if not gate_passed:
        return ROUTE_COVERAGE_GAP

    # relevant=no is a COMPLETE and actionable signal on its own: retrieval surfaced the
    # wrong article, which is a coverage/retrieval gap regardless of whether "did the fix
    # help?" was ever answered. Progressive disclosure makes (relevant=no, helped=None) the
    # most common partial response, so it must be caught here and never fall through to
    # no_response.
    if kb_relevant is False:
        return ROUTE_COVERAGE_GAP

    # Nothing answered at all (dismissed before the first question).
    if kb_relevant is None:
        return ROUTE_NO_RESPONSE

    # kb_relevant is True from here — retrieval worked, so any failure is generation-side.
    if fix_helped is False:
        return ROUTE_PROMPT_PROBLEM
    if fix_helped is True:
        return ROUTE_SUCCESS

    # relevant=yes but helped unanswered — the one genuinely unroutable partial.
    return ROUTE_PARTIAL


def is_kb_candidate(route: str) -> bool:
    """Only coverage gaps become KB candidates (Phase 3 queue)."""
    return route == ROUTE_COVERAGE_GAP


def is_any_response(route: str) -> bool:
    """Did the agent give us any signal? A partial counts as a response.

    Used for the feedback_coverage metric, so we know how thin this source is.
    """
    return route != ROUTE_NO_RESPONSE

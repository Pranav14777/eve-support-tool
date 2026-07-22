import json
import os
from openai import OpenAI
from vector_store import search_knowledge_base, search_resolved_tickets, EmbeddingUnavailable
from dotenv import load_dotenv

load_dotenv()

# ── Confidence gate config ───────────────────────────────────────────────────────
# The gate serves a ticket to the LLM only when the KB has a clearly-best match:
#   serve iff  top1_score >= KB_ABS_FLOOR  AND  (top1_score - top2_score) >= KB_MARGIN_THRESHOLD
# Margin (not absolute score) is the primary signal: an uncovered ticket matches
# everything equally poorly (small margin); a covered one has one article standing out.
# The floor is a safety net so a weak-but-gapped top hit can't pass.
#
# NOTE: 0.06 / 0.39 are calibrated against the CURRENT 10-article KB via eval_retrieval.py
# (margin 0.06, floor 0.39 → 0/18 negatives served, 28/50 covered served). They MUST be
# re-derived once the KB grows to ~60–80 articles — more distractors shift the score
# distribution and the right operating point moves with it.
KB_MARGIN_THRESHOLD = float(os.environ.get("KB_MARGIN_THRESHOLD", "0.06"))
KB_ABS_FLOOR = float(os.environ.get("KB_ABS_FLOOR", "0.39"))
# Two candidates count as duplicates (skip #2 for the margin) if their scores are within this.
KB_DUP_DELTA = 0.02

nim_client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.environ.get("NVIDIA_API_KEY"),
)

# Judge + KB summarizer use Google Gemini via its OpenAI-compatible endpoint —
# a different provider and architecture from NVIDIA NIM, so it catches blind spots.
gemini_client = OpenAI(
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    api_key=os.environ.get("GEMINI_API_KEY"),
)

JUDGE_MODEL = "gemini-2.0-flash"

MODEL_MAP = {
    "low":    "meta/llama-3.1-8b-instruct",
    "medium": "meta/llama-3.3-70b-instruct",
    "high":   "nvidia/llama-3.1-nemotron-70b-instruct",
}

def get_model_for_priority(priority: str) -> str:
    return MODEL_MAP.get(priority.lower(), MODEL_MAP["medium"])

def build_context_from_search(kb_matches: list, resolved_matches: list) -> str:
    """Build context string from ChromaDB search results to inform the LLM"""
    context = ""

    if kb_matches:
        context += "KNOWLEDGE BASE MATCHES (known issues and fixes):\n"
        for i, match in enumerate(kb_matches):
            context += f"""
Match {i+1} (similarity: {match['similarity_score']}):
Title: {match['title']}
Known Fix: {match['known_fix']}
Workaround: {match['workaround']}
Issue Type: {match['issue_type']}
"""
        context += "\nUse these known fixes to inform your analysis. Set known_issue to true.\n"
    else:
        context += "No knowledge base match found. Set known_issue to false.\n"

    if resolved_matches:
        context += "\nSIMILAR PAST RESOLVED TICKETS (learned from experience):\n"
        for i, match in enumerate(resolved_matches):
            context += f"""
Past Ticket {i+1} (similarity: {match['similarity_score']}):
Title: {match['title']}
Issue Type: {match['issue_type']}
What Actually Fixed It: {match['actual_fix']}
"""
        context += "\nReference these past resolutions to improve your analysis.\n"

    return context

def fallback_response(ticket: dict, kb_matches: list, resolved_matches: list, model: str = "") -> dict:
    """Fallback response when LLM fails after all retry attempts"""
    kb_article = kb_matches[0] if kb_matches else None
    past_ticket = resolved_matches[0] if resolved_matches else None
    best_score = max(
        kb_article["similarity_score"] if kb_article else 0.0,
        past_ticket["similarity_score"] if past_ticket else 0.0
    )
    best_match_title = (kb_article or past_ticket or {}).get("title", "")

    return {
        "issue_type": "Unclassified",
        "severity": ticket.get("priority", "medium").capitalize(),
        "likely_cause": "Unable to automatically analyze this ticket. Manual review required.",
        "known_issue": kb_article is not None,
        "knowledge_base_article": kb_article["title"] if kb_article else None,
        "known_fix": kb_article["known_fix"] if kb_article else None,
        "reproduction_checklist": [
            "Read the full ticket description carefully",
            "Check system logs for the affected store",
            "Verify if the issue is isolated or affecting multiple stores",
            "Check knowledge base for similar previously resolved issues",
            "Escalate to third line if cause remains unclear"
        ],
        "workaround": kb_article["workaround"] if kb_article else (
            past_ticket["actual_fix"] if past_ticket else
            "No automated workaround available. Manual investigation required."
        ),
        "suggested_next_step": "Manual review required — automatic analysis was unsuccessful.",
        "escalate_to_third_line": ticket.get("priority") == "high",
        "escalation_reason": "High priority ticket requires manual review." if ticket.get("priority") == "high" else None,
        "internal_note": f"Auto-analysis failed for ticket {ticket.get('id', 'unknown')}. KB match: {'Yes - ' + kb_article['title'] if kb_article else 'No'}. Past ticket match: {'Yes - ' + past_ticket['title'] if past_ticket else 'No'}. Please review manually.",
        "customer_reply": f"Dear {ticket.get('store', 'Partner')} team,\n\nThank you for reaching out to Support.\n\nWe have received your report regarding '{ticket.get('title', 'your issue')}' and our team is reviewing it as a priority.\n\nWe will follow up with a status update within 1 hour.\n\nBest regards,\nSupport Team",
        "past_resolution": past_ticket,
        "analyzed_by": "fallback",
        "kb_similarity_score": best_score,
        "kb_second_score": 0.0,
        "kb_margin": 0.0,
        "kb_match_title": best_match_title,
        "retrieved_articles": [],
        "gate_passed": False,
        "gate_decision": "abstained",
        "abstain_reason": "llm_unavailable",
        "judge_verdict": "SKIPPED",
        "judge_notes": "Fallback response — LLM did not respond after 2 attempts.",
        "judge_concerns": [],
        "model_used": model
    }

def validate_response(result: dict) -> bool:
    """Validate LLM response has all required fields"""
    required_fields = [
        "issue_type", "severity", "likely_cause", "known_issue",
        "reproduction_checklist", "workaround", "suggested_next_step",
        "escalate_to_third_line", "internal_note", "customer_reply"
    ]

    for field in required_fields:
        if field not in result:
            print(f"Validation failed: missing field '{field}'")
            return False
        if result[field] is None or result[field] == "":
            print(f"Validation failed: empty field '{field}'")
            return False

    if not isinstance(result["reproduction_checklist"], list):
        return False
    if len(result["reproduction_checklist"]) < 2:
        return False

    valid_issue_types = [
        "Integration Issue", "Configuration Issue", "API Issue",
        "System Behavior", "Data Sync Issue", "Unclassified"
    ]
    if result["issue_type"] not in valid_issue_types:
        result["issue_type"] = "Unclassified"

    valid_severities = ["Low", "Medium", "High", "Critical"]
    if result["severity"] not in valid_severities:
        result["severity"] = "Medium"

    if not isinstance(result["escalate_to_third_line"], bool):
        result["escalate_to_third_line"] = False

    return True

def run_judge(ticket: dict, analysis: dict, context: str) -> dict:
    """
    Use Google Gemini Flash to independently evaluate the primary LLM's analysis.
    Uses a different provider (Google) and architecture to catch blind spots the primary model may miss.
    Only called when the KB similarity gate passes.
    Returns {"verdict": "PASS"|"FAIL"|"SKIPPED", "concerns": [...], "notes": "..."}
    """
    system_message = (
        "You are a senior support quality reviewer for a unified commerce retail platform. "
        "You evaluate AI-generated support ticket analyses for accuracy, appropriateness, and actionability. "
        "You do not write new analyses — you only evaluate the one provided. "
        "Respond only with valid JSON. Never include markdown or explanation outside the JSON."
    )

    user_message = f"""Evaluate this AI-generated support ticket analysis.

ORIGINAL TICKET:
Title: {ticket.get('title')}
Description: {ticket.get('description')}
Priority: {ticket.get('priority')}

AVAILABLE KNOWLEDGE BASE CONTEXT:
{context}

AI ANALYSIS TO EVALUATE:
- Issue Type: {analysis.get('issue_type')}
- Severity: {analysis.get('severity')}
- Likely Cause: {analysis.get('likely_cause')}
- Known Issue: {analysis.get('known_issue')}
- Workaround: {analysis.get('workaround')}
- Suggested Next Step: {analysis.get('suggested_next_step')}
- Escalate to Third Line: {analysis.get('escalate_to_third_line')}
- Escalation Reason: {analysis.get('escalation_reason', 'N/A')}
- Reproduction Checklist: {json.dumps(analysis.get('reproduction_checklist', []))}

Evaluate:
1. Is issue_type correctly classified given the description and KB context?
2. Is severity appropriate for the business impact described?
3. Are the reproduction checklist steps concrete and executable?
4. Does the workaround align with KB knowledge or described symptoms?
5. Is the escalation decision proportionate to the severity and complexity?

Return ONLY this exact JSON — no other text:
{{
  "verdict": "PASS or FAIL",
  "concerns": ["specific concern if any"],
  "notes": "one sentence overall assessment"
}}

PASS = analysis is accurate and actionable. FAIL = significant inaccuracies or likely to mislead the engineer."""

    try:
        response = gemini_client.chat.completions.create(
            model=JUDGE_MODEL,
            max_tokens=300,
            temperature=0.1,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ]
        )

        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            parts = raw.split("```")
            raw = parts[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        result = json.loads(raw)

        if result.get("verdict") not in ("PASS", "FAIL"):
            result["verdict"] = "FAIL"
        if not isinstance(result.get("concerns"), list):
            result["concerns"] = []
        if not result.get("notes"):
            result["notes"] = ""

        return result

    except json.JSONDecodeError as e:
        print(f"Judge JSON parse error: {e}")
        return {"verdict": "SKIPPED", "concerns": [], "notes": "Judge returned unparseable response."}
    except Exception as e:
        print(f"Judge LLM error: {e}")
        return {"verdict": "SKIPPED", "concerns": [], "notes": "Judge evaluation failed — manual review recommended."}

def summarize_fix_for_kb(ticket: dict, actual_fix: str) -> str:
    """
    Summarize a resolved fix into a concise, anonymized form safe for KB storage.
    Strips all customer, store, and company names — keeps only the technical issue and resolution.
    """
    prompt = f"""A support engineer resolved a technical issue. Summarize the fix for a knowledge base.

ISSUE TITLE: {ticket.get('title', '')}
ISSUE DESCRIPTION: {ticket.get('description', '')}
ACTUAL FIX APPLIED: {actual_fix}

Write a 2-3 sentence technical summary that:
1. Describes the technical problem without any customer names, store names, or company names
2. Explains exactly what was done to fix it
3. Uses generic terms like "the store", "the integration", "the configuration" — never specific names

Return only the summary text, no JSON, no formatting."""

    try:
        response = gemini_client.chat.completions.create(
            model=JUDGE_MODEL,
            max_tokens=200,
            temperature=0.1,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"KB summarization failed: {e}")
        return actual_fix[:500]

def evaluate_gate(kb_matches: list, resolved_matches: list) -> dict:
    """Margin + floor confidence gate over the merged KB + resolved candidate pool.

    serve iff  top1 >= KB_ABS_FLOOR  AND  (top1 - top2_distinct) >= KB_MARGIN_THRESHOLD

    The #2 used for the margin is the next DISTINCT candidate. Near-duplicate candidates
    are skipped so the margin isn't crushed by (a) the same article surfacing twice or
    (b) a resolved ticket mirroring the KB article it was derived from — otherwise the
    Phase 3 KB-feedback loop would silently degrade the gate as resolved tickets accumulate.
    (Cross-collection mirrors won't share a title, so we also treat a near-scored candidate
    from the *other* collection as a duplicate. Phase 3 should tag resolved tickets with
    their source KB id for an exact signal.)

    Returns {passed, reason, top1, top2, margin, title}.
    reason ∈ {None, "low_floor", "low_margin", "insufficient_candidates"}.
    """
    candidates = (
        [{**m, "_source": "kb"} for m in kb_matches]
        + [{**m, "_source": "resolved"} for m in resolved_matches]
    )
    candidates.sort(key=lambda m: m.get("similarity_score", 0.0), reverse=True)

    if not candidates:
        return {"passed": False, "reason": "insufficient_candidates",
                "top1": 0.0, "top2": 0.0, "margin": 0.0, "title": ""}

    top = candidates[0]
    top1 = top.get("similarity_score", 0.0)
    title = top.get("title", "")

    def _is_duplicate(c: dict) -> bool:
        if abs(c.get("similarity_score", 0.0) - top1) >= KB_DUP_DELTA:
            return False
        return c.get("title", "") == title or c.get("_source") != top.get("_source")

    second = next((c for c in candidates[1:] if not _is_duplicate(c)), None)

    if second is None:
        # Only one distinct candidate — margin is undefined; abstain rather than pass
        # on top1 alone (guards against a lone weak hit sailing through as top1 - 0.0).
        return {"passed": False, "reason": "insufficient_candidates",
                "top1": top1, "top2": 0.0, "margin": 0.0, "title": title}

    top2 = second.get("similarity_score", 0.0)
    margin = round(top1 - top2, 3)

    if top1 < KB_ABS_FLOOR:
        reason = "low_floor"
    elif margin < KB_MARGIN_THRESHOLD:
        reason = "low_margin"
    else:
        reason = None

    return {"passed": reason is None, "reason": reason,
            "top1": top1, "top2": top2, "margin": margin, "title": title}

def build_retrieval_snapshot(kb_matches: list, resolved_matches: list) -> list:
    """A durable record of what retrieval actually surfaced, in rank order.

    Stores id + title + snippet + score rather than bare ChromaDB ids: the corpus will be
    swapped (e.g. to TechQA), which would orphan raw ids and invalidate every feedback row
    Phase 2 collects. Captured at ANALYSIS time because it must record what was shown to the
    agent — re-reading at feedback time could return a different article after a swap.
    """
    merged = sorted(
        kb_matches + resolved_matches,
        key=lambda m: m.get("similarity_score", 0.0),
        reverse=True,
    )
    return [
        {
            "id": m.get("id", ""),
            "title": m.get("title", ""),
            "snippet": m.get("snippet", ""),
            "score": m.get("similarity_score", 0.0),
        }
        for m in merged
    ]

def _abstain_message(gate: dict) -> str:
    """Human-readable reason the gate abstained, shown to the support engineer."""
    r = gate["reason"]
    if r == "insufficient_candidates":
        return ("No distinct knowledge-base match to compare against — not enough coverage to "
                "answer confidently. Please investigate manually and submit the fix so it enters the KB.")
    if r == "low_floor":
        return (f"Best KB match is too weak (score {gate['top1']:.3f} < floor {KB_ABS_FLOOR}). "
                "No sufficiently relevant article — investigate manually and submit the fix.")
    return (f"Top two KB matches are too close (margin {gate['margin']:.3f} < {KB_MARGIN_THRESHOLD}) — "
            "the best article isn't clearly better than the next, so coverage is ambiguous. "
            "Investigate manually and submit the fix.")

def _abstain_response(ticket: dict, kb_matches: list, resolved_matches: list, *,
                      reason: str, top1: float, top2: float, margin: float,
                      title: str, analyzed_by: str, message: str) -> dict:
    """Build the abstain analysis dict (used for both gate-fail and retrieval-unavailable)."""
    return {
        "gate_passed": False,
        "gate_decision": "abstained",
        "abstain_reason": reason,
        "kb_similarity_score": top1,
        "kb_second_score": top2,
        "kb_margin": margin,
        "kb_match_title": title,
        # Abstains auto-route to the Phase 3 candidate queue, so their retrieval set matters.
        "retrieved_articles": build_retrieval_snapshot(kb_matches, resolved_matches),
        "judge_verdict": "SKIPPED",
        "judge_notes": message,
        "judge_concerns": [],
        "model_used": None,
        "analyzed_by": analyzed_by,
        "issue_type": "Unclassified",
        "severity": ticket.get("priority", "medium").capitalize(),
        "likely_cause": "",
        "known_issue": False,
        "knowledge_base_article": None,
        "known_fix": None,
        "reproduction_checklist": [],
        "workaround": "",
        "suggested_next_step": "Investigate manually and submit the actual fix to update the knowledge base.",
        "escalate_to_third_line": ticket.get("priority") == "high",
        "escalation_reason": None,
        "internal_note": message,
        "customer_reply": "",
        "kb_matches": kb_matches,
        "past_resolutions": resolved_matches,
    }

def analyze_ticket(ticket: dict) -> dict:
    """
    Analyze a support ticket using:
    1. KB similarity gate — blocks low-similarity tickets before LLM is called
    2. NVIDIA NIM LLM — model chosen by ticket priority
    3. Gemini Flash judge — independent validation of the analysis
    """

    # Step 1: Semantic search across KB articles and past resolved tickets.
    # If the embedding service is down, ABSTAIN — never analyze with no context.
    print(f"Searching KB for ticket: {ticket.get('id')}")
    try:
        kb_matches = search_knowledge_base(ticket, n_results=3)
        resolved_matches = search_resolved_tickets(ticket, n_results=3)
    except EmbeddingUnavailable as e:
        print(f"Retrieval unavailable for {ticket.get('id')}: {e}")
        return _abstain_response(
            ticket, [], [],
            reason="retrieval_unavailable",
            top1=0.0, top2=0.0, margin=0.0, title="",
            analyzed_by="retrieval_unavailable",
            message=("Retrieval service is temporarily unavailable, so analysis was not attempted "
                     "(the assistant will not answer without knowledge-base context). "
                     "Please retry shortly or investigate manually."),
        )
    print(f"KB matches: {len(kb_matches)}, Resolved matches: {len(resolved_matches)}")

    # Step 2: KB confidence gate — margin (primary) + absolute floor (safety net)
    gate = evaluate_gate(kb_matches, resolved_matches)
    print(f"Gate: top1={gate['top1']:.3f} top2={gate['top2']:.3f} margin={gate['margin']:.3f} "
          f"(margin_thr={KB_MARGIN_THRESHOLD}, floor={KB_ABS_FLOOR}) -> "
          f"{'SERVE' if gate['passed'] else 'ABSTAIN (' + str(gate['reason']) + ')'}")

    if not gate["passed"]:
        return _abstain_response(
            ticket, kb_matches, resolved_matches,
            reason=gate["reason"], top1=gate["top1"], top2=gate["top2"],
            margin=gate["margin"], title=gate["title"],
            analyzed_by="gate_blocked", message=_abstain_message(gate),
        )

    best_score = gate["top1"]
    best_match_title = gate["title"]

    # Step 3: Build context string for LLM
    context = build_context_from_search(kb_matches, resolved_matches)

    # Step 4: Select model based on ticket priority
    model = get_model_for_priority(ticket.get("priority", "medium"))
    print(f"Gate PASSED — using model: {model}")

    # Step 5: Build prompt
    prompt = f"""
You are an expert second-line support engineer for a unified commerce retail platform that handles POS transactions, inventory management, order orchestration, click & collect flows, customer data, and third-party integrations like payment gateways.

Your job is to:
1. Classify the issue type
2. Assess severity
3. Identify the likely root cause
4. Provide a reproduction checklist
5. Suggest an immediate workaround for the store
6. Recommend the next engineering action
7. Decide if this needs third line escalation
8. Write a structured internal note
9. Write a professional customer reply with acknowledgement, status, workaround and ETA

{context}

TICKET:
ID: {ticket['id']}
Title: {ticket['title']}
Description: {ticket['description']}
Store: {ticket['store']}
Priority: {ticket['priority']}

Return ONLY this exact JSON structure with no extra text or markdown:
{{
    "issue_type": "one of: Integration Issue, Configuration Issue, API Issue, System Behavior, Data Sync Issue",
    "severity": "one of: Low, Medium, High, Critical",
    "likely_cause": "one clear sentence explaining the most probable root cause",
    "known_issue": true or false,
    "knowledge_base_article": "title of matching KB article or null",
    "known_fix": "the known fix from KB or null",
    "reproduction_checklist": [
        "Concrete step 1 to reproduce or verify the issue",
        "Concrete step 2",
        "Concrete step 3",
        "Concrete step 4"
    ],
    "workaround": "clear immediate workaround the store can use right now",
    "suggested_next_step": "one clear action the support engineer should take first",
    "escalate_to_third_line": true or false,
    "escalation_reason": "reason for escalation or null if not escalating",
    "internal_note": "structured internal note: what the issue is, what KB and past tickets matched, what action is being taken",
    "customer_reply": "Professional reply: 1) Acknowledge 2) What we know 3) Immediate workaround 4) Next steps and ETA"
}}
"""

    # Step 6: Try LLM up to 2 times
    for attempt in range(2):
        try:
            print(f"LLM attempt {attempt + 1} for ticket {ticket.get('id')}")

            response = nim_client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a structured support engineer assistant for a retail SaaS platform. Always respond with valid JSON only. Never include markdown formatting or explanation outside the JSON."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,
                max_tokens=1500
            )

            raw = response.choices[0].message.content.strip()

            if raw.startswith("```"):
                parts = raw.split("```")
                raw = parts[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            raw = raw.strip()

            result = json.loads(raw)

            if validate_response(result):
                result["analyzed_by"] = "llm"
                result["model_used"] = model
                result["gate_passed"] = True
                result["gate_decision"] = "served"
                result["abstain_reason"] = None
                result["kb_similarity_score"] = best_score
                result["kb_second_score"] = gate["top2"]
                result["kb_margin"] = gate["margin"]
                result["kb_match_title"] = best_match_title
                result["retrieved_articles"] = build_retrieval_snapshot(kb_matches, resolved_matches)
                result["kb_matches"] = kb_matches
                result["past_resolutions"] = resolved_matches

                # Step 7: Run judge (Gemini Flash — different provider to catch blind spots)
                print(f"Running judge for ticket {ticket.get('id')}")
                judge = run_judge(ticket, result, context)
                result["judge_verdict"] = judge["verdict"]
                result["judge_notes"] = judge["notes"]
                result["judge_concerns"] = judge["concerns"]

                print(f"Ticket {ticket['id']} analyzed successfully. Judge: {judge['verdict']}")
                return result
            else:
                print(f"Attempt {attempt + 1}: Validation failed, retrying...")
                continue

        except json.JSONDecodeError as e:
            print(f"Attempt {attempt + 1}: JSON parsing failed — {str(e)}")
            continue

        except Exception as e:
            print(f"Attempt {attempt + 1}: NIM API error — {str(e)}")
            continue

    # Both attempts failed — return fallback (gate had passed, but the LLM never produced
    # a valid response, so route it like an abstain rather than a confident answer).
    print(f"Both LLM attempts failed for {ticket.get('id')}. Using fallback.")
    fallback = fallback_response(ticket, kb_matches, resolved_matches, model=model)
    fallback["kb_matches"] = kb_matches
    fallback["past_resolutions"] = resolved_matches
    fallback["kb_second_score"] = gate["top2"]
    fallback["kb_margin"] = gate["margin"]
    fallback["abstain_reason"] = "llm_unavailable"
    fallback["retrieved_articles"] = build_retrieval_snapshot(kb_matches, resolved_matches)
    return fallback

def generate_follow_up_reply(ticket: dict, analysis: dict, update: str) -> str:
    """Generate a follow up reply for an ongoing ticket"""
    prompt = f"""
You are a support engineer writing a follow-up update to a retail partner.

Original Issue: {ticket.get('title')}
Store: {ticket.get('store')}
Original Customer Reply: {analysis.get('customer_reply', '')}
Current Update to Share: {update}

Write a professional, empathetic follow-up reply that:
1. References the original issue
2. Shares the current update clearly
3. States next steps and revised ETA if available
4. Maintains confidence and reassurance

Return only the reply text, no JSON, no formatting.
"""

    try:
        response = nim_client.chat.completions.create(
            model=MODEL_MAP["medium"],
            messages=[
                {
                    "role": "system",
                    "content": "You are a professional support engineer writing customer communications. Be clear, empathetic and structured."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3,
            max_tokens=500
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"Follow up generation failed: {str(e)}")
        return f"Dear {ticket.get('store', 'Partner')} team,\n\nThank you for your patience. We wanted to share a quick update regarding your reported issue: {update}\n\nWe will follow up shortly with further details.\n\nBest regards,\nSupport Team"

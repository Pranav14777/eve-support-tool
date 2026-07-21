import json
import os
from openai import OpenAI
import anthropic
from vector_store import search_knowledge_base, search_resolved_tickets
from dotenv import load_dotenv

load_dotenv()

KB_SIMILARITY_THRESHOLD = float(os.environ.get("KB_SIMILARITY_THRESHOLD", "0.65"))

nim_client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.environ.get("NVIDIA_API_KEY"),
)

anthropic_client = anthropic.Anthropic(
    api_key=os.environ.get("ANTHROPIC_API_KEY")
)

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
        "kb_match_title": best_match_title,
        "gate_passed": False,
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
    Use Claude Haiku to independently evaluate the primary LLM's analysis.
    Uses a different provider (Anthropic) to catch blind spots the primary model may miss.
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
        response = anthropic_client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=300,
            temperature=0.1,
            system=system_message,
            messages=[{"role": "user", "content": user_message}]
        )

        raw = response.content[0].text.strip()
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
        response = anthropic_client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=200,
            temperature=0.1,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text.strip()
    except Exception as e:
        print(f"KB summarization failed: {e}")
        return actual_fix[:500]

def analyze_ticket(ticket: dict) -> dict:
    """
    Analyze a support ticket using:
    1. KB similarity gate — blocks low-similarity tickets before LLM is called
    2. NVIDIA NIM LLM — model chosen by ticket priority
    3. Claude Haiku judge — independent validation of the analysis
    """

    # Step 1: Semantic search across KB articles and past resolved tickets
    print(f"Searching KB for ticket: {ticket.get('id')}")
    kb_matches = search_knowledge_base(ticket, n_results=2)
    resolved_matches = search_resolved_tickets(ticket, n_results=2)
    print(f"KB matches: {len(kb_matches)}, Resolved matches: {len(resolved_matches)}")

    # Step 2: KB similarity confidence gate
    best_kb_score = kb_matches[0]["similarity_score"] if kb_matches else 0.0
    best_res_score = resolved_matches[0]["similarity_score"] if resolved_matches else 0.0
    best_score = max(best_kb_score, best_res_score)

    if best_kb_score >= best_res_score and kb_matches:
        best_match_title = kb_matches[0]["title"]
    elif resolved_matches:
        best_match_title = resolved_matches[0]["title"]
    else:
        best_match_title = ""

    print(f"Best KB similarity: {best_score:.3f} (threshold: {KB_SIMILARITY_THRESHOLD})")

    if best_score < KB_SIMILARITY_THRESHOLD:
        print(f"Gate FAILED for {ticket.get('id')}: {best_score:.3f} < {KB_SIMILARITY_THRESHOLD}")
        return {
            "gate_passed": False,
            "kb_similarity_score": best_score,
            "kb_match_title": best_match_title,
            "judge_verdict": "SKIPPED",
            "judge_notes": "Gate failed — no sufficiently similar KB article or past resolution found.",
            "judge_concerns": [],
            "model_used": None,
            "analyzed_by": "gate_blocked",
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
            "internal_note": f"Gate blocked — KB similarity {best_score:.3f} below threshold {KB_SIMILARITY_THRESHOLD}.",
            "customer_reply": "",
            "kb_matches": kb_matches,
            "past_resolutions": resolved_matches
        }

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
                result["kb_similarity_score"] = best_score
                result["kb_match_title"] = best_match_title
                result["kb_matches"] = kb_matches
                result["past_resolutions"] = resolved_matches

                # Step 7: Run judge (Claude Haiku — different provider to catch blind spots)
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

    # Both attempts failed — return fallback
    print(f"Both LLM attempts failed for {ticket.get('id')}. Using fallback.")
    fallback = fallback_response(ticket, kb_matches, resolved_matches, model=model)
    fallback["kb_matches"] = kb_matches
    fallback["past_resolutions"] = resolved_matches
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

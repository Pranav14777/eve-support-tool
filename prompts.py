import json
from groq import Groq
from vector_store import search_knowledge_base, search_resolved_tickets

import os
client = Groq(api_key=os.environ.get("GROQ_API_KEY", "your-groq-api-key-here"))

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
Store: {match['store']}
Issue Type: {match['issue_type']}
What Actually Fixed It: {match['actual_fix']}
"""
        context += "\nReference these past resolutions to improve your analysis.\n"

    return context

def fallback_response(ticket: dict, kb_matches: list, resolved_matches: list) -> dict:
    """Fallback response when LLM fails"""
    kb_article = kb_matches[0] if kb_matches else None
    past_ticket = resolved_matches[0] if resolved_matches else None

    return {
        "issue_type": "Unclassified",
        "severity": ticket.get("priority", "medium").capitalize(),
        "likely_cause": "Unable to automatically analyze this ticket. Manual review required.",
        "known_issue": kb_article is not None,
        "knowledge_base_article": kb_article["title"] if kb_article else None,
        "known_fix": kb_article["known_fix"] if kb_article else None,
        "reproduction_checklist": [
            "Read the full ticket description carefully",
            "Check EVA system logs for the affected store",
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
        "customer_reply": f"Dear {ticket.get('store', 'Partner')} team,\n\nThank you for reaching out to New Black Support.\n\nWe have received your report regarding '{ticket.get('title', 'your issue')}' and our team is reviewing it as a priority.\n\nWe will follow up with a status update within 1 hour.\n\nBest regards,\nNew Black Support Team",
        "past_resolution": past_ticket,
        "analyzed_by": "fallback"
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

def analyze_ticket(ticket: dict) -> dict:
    """Main function to analyze a support ticket using ChromaDB + Groq LLM"""

    # Step 1: Semantic search across KB articles
    print(f"Searching KB for ticket: {ticket.get('id')}")
    kb_matches = search_knowledge_base(ticket, n_results=2)

    # Step 2: Semantic search across past resolved tickets
    print(f"Searching resolved tickets for: {ticket.get('id')}")
    resolved_matches = search_resolved_tickets(ticket, n_results=2)

    print(f"KB matches: {len(kb_matches)}, Resolved matches: {len(resolved_matches)}")

    # Step 3: Build context from search results
    context = build_context_from_search(kb_matches, resolved_matches)

    # Step 4: Build prompt
    prompt = f"""
You are an expert second-line support engineer at New Black, a company that builds EVA — a unified omnichannel commerce platform used by large retail brands like Hunkemöller, Rituals, Dyson, and Kiko Milano.

EVA handles POS transactions, inventory management, order orchestration, click & collect flows, customer data, and third-party integrations like Adyen payments.

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

    # Step 5: Try LLM up to 2 times
    for attempt in range(2):
        try:
            print(f"LLM attempt {attempt + 1} for ticket {ticket.get('id')}")

            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
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

            # Clean markdown if model adds it
            if raw.startswith("```"):
                parts = raw.split("```")
                raw = parts[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            raw = raw.strip()

            result = json.loads(raw)

            if validate_response(result):
                result["analyzed_by"] = "llm"
                result["kb_matches"] = kb_matches
                result["past_resolutions"] = resolved_matches
                print(f"Ticket {ticket['id']} analyzed successfully")
                return result
            else:
                print(f"Attempt {attempt + 1}: Validation failed, retrying...")
                continue

        except json.JSONDecodeError as e:
            print(f"Attempt {attempt + 1}: JSON parsing failed — {str(e)}")
            continue

        except Exception as e:
            print(f"Attempt {attempt + 1}: Groq API error — {str(e)}")
            continue

    # Both attempts failed
    print(f"Both attempts failed for {ticket.get('id')}. Using fallback.")
    fallback = fallback_response(ticket, kb_matches, resolved_matches)
    fallback["kb_matches"] = kb_matches
    fallback["past_resolutions"] = resolved_matches
    return fallback

def generate_follow_up_reply(ticket: dict, analysis: dict, update: str) -> str:
    """Generate a follow up reply for an ongoing ticket"""
    prompt = f"""
You are a support engineer at New Black writing a follow-up update to a retail partner.

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
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
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
        return f"Dear {ticket.get('store', 'Partner')} team,\n\nThank you for your patience. We wanted to share a quick update regarding your reported issue: {update}\n\nWe will follow up shortly with further details.\n\nBest regards,\nNew Black Support Team"
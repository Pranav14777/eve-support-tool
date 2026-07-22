import os
import time
import chromadb
from openai import OpenAI
import json
from datetime import datetime
from dotenv import load_dotenv

# Load env vars here too — vector_store is imported before prompts.py runs load_dotenv,
# and the NIM embedding client needs NVIDIA_API_KEY available at import time.
load_dotenv()

# Initialize ChromaDB client with persistent storage
client = chromadb.PersistentClient(path="./chroma_db")

# NVIDIA NIM embedding model — nv-embedqa-e5-v5 is purpose-built for retrieval/RAG.
# Produces 1024-dim vectors vs 384-dim from the default MiniLM, giving richer semantic matching.
# It is an ASYMMETRIC model: stored documents must be embedded with input_type="passage"
# and search queries with input_type="query". ChromaDB's embedding-function hook can't tell the
# two apart, so we compute embeddings ourselves and pass them to ChromaDB directly.
EMBED_MODEL = "nvidia/nv-embedqa-e5-v5"
_EMBED_MAX_ATTEMPTS = 3

_nim = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.environ.get("NVIDIA_API_KEY", ""),
)


class EmbeddingUnavailable(Exception):
    """Raised when the embedding service can't be reached after retries.

    Callers must treat this as 'retrieval is down' and ABSTAIN — never fall back
    to answering with an empty KB context, which would look like a confident answer.
    """


def _embed(texts: list, input_type: str) -> list:
    """Embed texts with NVIDIA NIM. input_type is 'passage' (stored docs) or 'query' (searches).

    Retries transient failures with backoff; raises EmbeddingUnavailable if all attempts fail.
    """
    last_err = None
    for attempt in range(_EMBED_MAX_ATTEMPTS):
        try:
            response = _nim.embeddings.create(
                input=texts,
                model=EMBED_MODEL,
                encoding_format="float",
                extra_body={"input_type": input_type, "truncate": "END"},
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            last_err = e
            if attempt < _EMBED_MAX_ATTEMPTS - 1:
                time.sleep(0.5 * (attempt + 1))  # 0.5s, 1.0s backoff
    raise EmbeddingUnavailable(
        f"NIM embedding failed after {_EMBED_MAX_ATTEMPTS} attempts: {last_err}"
    )


SNIPPET_CHARS = 200


def _snippet(results: dict, i: int) -> str:
    """First ~200 chars of the matched document, whitespace-collapsed.

    Persisted alongside the ChromaDB id so a feedback row stays human-readable after a
    corpus swap (e.g. to TechQA), when the raw id would otherwise be orphaned.
    """
    docs = results.get("documents")
    if not docs or not docs[0]:
        return ""
    text = (docs[0][i] or "").strip()
    return " ".join(text.split())[:SNIPPET_CHARS]

# Two separate collections
# 1. Knowledge base articles — curated known fixes
# 2. Resolved tickets — learned from past resolutions
# No embedding_function is attached — we always supply precomputed embeddings.
# Cosine distance is used (not the L2 default) because nv-embedqa produces normalized
# vectors; cosine gives a clean similarity scale where 1 - distance ≈ semantic similarity.
kb_collection = client.get_or_create_collection(
    name="knowledge_base",
    metadata={"hnsw:space": "cosine"}
)
resolved_collection = client.get_or_create_collection(
    name="resolved_tickets",
    metadata={"hnsw:space": "cosine"}
)

# ── Seed Knowledge Base ────────────────────────────────────────────────────────

KB_ARTICLES = [
    {
        "id": "kb-001",
        "title": "Adyen Payment Gateway Timeout",
        "content": "Adyen payment gateway timeout at POS checkout. Symptoms include payment gateway timeout error, transactions failing, card payments not processing. Common causes are expired Adyen API key, misconfigured terminal ID, or inactive webhook endpoints in EVA configuration.",
        "known_fix": "Check Adyen API credentials in EVA configuration. Verify Adyen webhook endpoints are active. Common cause is expired API key or misconfigured terminal ID.",
        "workaround": "Switch terminal to manual card imprint mode temporarily. Contact Adyen support with terminal ID if credentials are valid.",
        "issue_type": "Integration Issue"
    },
    {
        "id": "kb-002",
        "title": "Click and Collect Orders Not Syncing to POS",
        "content": "Online orders placed for click and collect are not appearing in store POS system. Store staff cannot locate or fulfill orders. Customers arriving to collect purchases cannot be served.",
        "known_fix": "Check order sync service status in EVA admin. Verify store location ID matches between online and POS configuration. Restart sync service if needed.",
        "workaround": "Store staff can manually look up orders in EVA back office using order number provided by customer.",
        "issue_type": "Data Sync Issue"
    },
    {
        "id": "kb-003",
        "title": "Inventory Sync Discrepancy After Scheduled Job",
        "content": "Stock levels shown in EVA do not match physical inventory after scheduled sync job. Items showing as in stock are actually out of stock or vice versa. Usually occurs after nightly sync jobs.",
        "known_fix": "Check scheduled sync job logs in EVA admin panel. Common cause is a failed delta sync leaving inventory in inconsistent state. Run manual full sync to correct.",
        "workaround": "Staff should use physical stock count as source of truth until sync is corrected. Disable online stock display if overselling is a risk.",
        "issue_type": "Data Sync Issue"
    },
    {
        "id": "kb-004",
        "title": "Product Import Failing Due to Malformed JSON",
        "content": "Automated product import job failing with JSON parsing error. Invalid JSON payload, unexpected token error. New products not appearing in system after import job runs.",
        "known_fix": "Validate JSON payload using a JSON linter. Check for special characters, missing commas, or incorrect field types. EVA expects UTF-8 encoded JSON.",
        "workaround": "Manually add critical products via EVA admin panel while import issue is being resolved.",
        "issue_type": "API Issue"
    },
    {
        "id": "kb-005",
        "title": "API Authentication Failure 401 Unauthorized",
        "content": "Integration partner receiving 401 Unauthorized response when calling EVA orders API. Authentication was working previously but suddenly stopped. No changes made on partner side.",
        "known_fix": "Check if API token has expired. Regenerate token in EVA developer portal. Verify correct authentication header format: Bearer token.",
        "workaround": "Use temporary admin credentials for critical operations while token issue is resolved.",
        "issue_type": "API Issue"
    },
    {
        "id": "kb-006",
        "title": "POS Freezing on Specific Product Barcode Scan",
        "content": "Store associates report POS freezing completely when scanning a specific product barcode. Device requires full restart. Issue consistent and reproducible across multiple devices.",
        "known_fix": "Check product record in EVA for corrupted data or missing required fields. Known issue with barcodes containing special GS1 prefixes. Clear POS cache and update product record.",
        "workaround": "Manually enter product SKU instead of scanning barcode. Flag barcode for product team review.",
        "issue_type": "System Behavior"
    },
    {
        "id": "kb-007",
        "title": "VAT Not Displaying on Customer Prices",
        "content": "Customer facing prices in store locations displaying without VAT. Pricing discrepancies at checkout. Affecting all product categories in specific country stores.",
        "known_fix": "Check store locale and tax configuration in EVA admin. Verify tax zone is correctly assigned to stores. Common cause is incorrect country code in store configuration.",
        "workaround": "Manually inform customers of VAT inclusive price at checkout until configuration is fixed.",
        "issue_type": "Configuration Issue"
    },
    {
        "id": "kb-008",
        "title": "Loyalty Points Not Updating After Purchase",
        "content": "Customers reporting loyalty points not being added to accounts after completing purchases. Issue ongoing for multiple days. Customers complaining at store level about missing points.",
        "known_fix": "Check loyalty service connection in EVA. Verify customer account is correctly linked to loyalty program. Check for failed loyalty sync jobs in admin panel.",
        "workaround": "Record purchase details manually and apply points retroactively once service is restored.",
        "issue_type": "Integration Issue"
    },
    {
        "id": "kb-009",
        "title": "Receipt Language Not Matching Store Locale",
        "content": "Receipts printing in wrong language across stores. System should automatically switch receipt language based on store locale settings. Configuration not changed recently.",
        "known_fix": "Check store locale settings in EVA admin. Verify language pack is installed for target locale. Ensure receipt template is mapped to correct language code.",
        "workaround": "Use default English receipts temporarily. Inform store manager of the configuration fix needed.",
        "issue_type": "Configuration Issue"
    },
    {
        "id": "kb-010",
        "title": "EVA Mobile POS App Crashing After iOS Update",
        "content": "EVA mobile POS app crashes on launch after iOS update. Store associates cannot use devices for any operations. Affecting all iOS devices that have been updated to latest version.",
        "known_fix": "Check EVA app compatibility matrix for the iOS version. Known issue with newer iOS versions and older EVA app versions. Update EVA app to latest version from App Store.",
        "workaround": "Roll back iOS update if possible. Use web based POS fallback on desktop device until app is updated.",
        "issue_type": "System Behavior"
    }
]

def seed_knowledge_base():
    """Seed ChromaDB with KB articles if not already seeded"""
    existing = kb_collection.count()
    if existing >= len(KB_ARTICLES):
        print(f"Knowledge base already seeded with {existing} articles")
        return

    print("Seeding knowledge base into ChromaDB...")

    documents = []
    metadatas = []
    ids = []

    for article in KB_ARTICLES:
        # What gets vectorized — rich content for semantic search
        documents.append(
            f"{article['title']}. {article['content']}"
        )
        metadatas.append({
            "title": article["title"],
            "known_fix": article["known_fix"],
            "workaround": article["workaround"],
            "issue_type": article["issue_type"]
        })
        ids.append(article["id"])

    try:
        embeddings = _embed(documents, "passage")
    except EmbeddingUnavailable as e:
        # Don't crash app startup if NIM is briefly unreachable — seeding retries next boot.
        print(f"KB seeding skipped — embedding service unavailable: {e}")
        return

    kb_collection.add(
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )

    print(f"Seeded {len(KB_ARTICLES)} articles into ChromaDB")

def search_knowledge_base(ticket: dict, n_results: int = 3) -> list:
    """
    Semantic search across KB articles using ticket content.
    Returns top matching articles with their metadata.

    Raises EmbeddingUnavailable if the query can't be embedded — the caller must
    abstain rather than proceed with no context. Other (e.g. ChromaDB) errors
    degrade gracefully to an empty list.

    n_results defaults to 3 so the confidence gate always has a real #2 score for
    the margin computation, even if one candidate is filtered out.
    """
    search_text = f"{ticket.get('title', '')} {ticket.get('description', '')}"

    # Outside the try: EmbeddingUnavailable must propagate to trigger abstention.
    query_embedding = _embed([search_text], "query")

    try:
        results = kb_collection.query(
            query_embeddings=query_embedding,
            n_results=min(n_results, kb_collection.count()),
            include=["metadatas", "documents", "distances"],
        )

        matches = []
        if results and results["metadatas"]:
            for i, metadata in enumerate(results["metadatas"][0]):
                distance = results["distances"][0][i]
                # Lower distance = more similar
                # Only return if similarity is strong enough
                if distance < 1.2:
                    matches.append({
                        "id": results["ids"][0][i],
                        "title": metadata["title"],
                        "snippet": _snippet(results, i),
                        "known_fix": metadata["known_fix"],
                        "workaround": metadata["workaround"],
                        "issue_type": metadata["issue_type"],
                        "similarity_score": round(1 - distance, 3)
                    })

        return matches

    except Exception as e:
        print(f"KB search error: {str(e)}")
        return []

def search_resolved_tickets(ticket: dict, n_results: int = 3) -> list:
    """
    Semantic search across past resolved tickets.
    Returns similar past tickets with their actual fixes.

    Like search_knowledge_base: EmbeddingUnavailable propagates (caller abstains),
    other errors degrade to an empty list.
    """
    if resolved_collection.count() == 0:
        return []

    search_text = f"{ticket.get('title', '')} {ticket.get('description', '')}"

    # Outside the try: EmbeddingUnavailable must propagate to trigger abstention.
    query_embedding = _embed([search_text], "query")

    try:
        results = resolved_collection.query(
            query_embeddings=query_embedding,
            n_results=min(n_results, resolved_collection.count()),
            include=["metadatas", "documents", "distances"],
        )

        matches = []
        if results and results["metadatas"]:
            for i, metadata in enumerate(results["metadatas"][0]):
                distance = results["distances"][0][i]
                if distance < 1.2:
                    matches.append({
                        "id": results["ids"][0][i],
                        "title": metadata.get("title", ""),
                        "snippet": _snippet(results, i),
                        "issue_type": metadata.get("issue_type", ""),
                        "actual_fix": metadata.get("actual_fix", ""),
                        "resolved_at": metadata.get("resolved_at", ""),
                        "similarity_score": round(1 - distance, 3)
                    })

        return matches

    except Exception as e:
        print(f"Resolved tickets search error: {str(e)}")
        return []

def add_resolved_ticket(log_id: int, ticket: dict, analysis: dict, actual_fix: str):
    """
    Add a resolved ticket to ChromaDB so future similar tickets benefit.
    This is the learning feedback loop.
    """
    try:
        doc_id = f"resolved-{log_id}"

        # Store only technical details — no customer/store identifiers
        document = f"""
        Issue: {ticket.get('title', '')}
        Description: {ticket.get('description', '')}
        Issue Type: {analysis.get('issue_type', '')}
        Likely Cause: {analysis.get('likely_cause', '')}
        Actual Fix Applied: {actual_fix}
        """

        metadata = {
            "title": ticket.get("title", ""),
            "issue_type": analysis.get("issue_type", ""),
            "likely_cause": analysis.get("likely_cause", ""),
            "actual_fix": actual_fix,
            "resolved_at": datetime.now().isoformat(),
            "log_id": str(log_id)
        }

        embedding = _embed([document], "passage")

        resolved_collection.add(
            documents=[document],
            embeddings=embedding,
            metadatas=[metadata],
            ids=[doc_id]
        )

        print(f"Resolved ticket {log_id} added to ChromaDB for future learning")
        return True

    except Exception as e:
        print(f"Error adding resolved ticket to ChromaDB: {str(e)}")
        return False

def get_vector_store_stats() -> dict:
    """Get stats about what's stored in ChromaDB"""
    return {
        "kb_articles": kb_collection.count(),
        "resolved_tickets": resolved_collection.count()
    }

# Seed knowledge base on import
seed_knowledge_base()
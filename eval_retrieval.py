"""
Retrieval + gate evaluation harness for the EVA knowledge base.

This is the instrument used to calibrate the gate (KB_MARGIN_THRESHOLD + KB_ABS_FLOOR) and
to justify any change to the embedding model or retrieval strategy with numbers, not a vibe.

Run:  python eval_retrieval.py

It measures two different things:

RANKING (positives only) — when a covered problem is reported, is the right article
first? Reported as recall@1 and MRR, plus the score gap between the #1 and #2 hit
(a confidence margin — a big gap means the top match is unambiguous).

GATING (positives + negatives) — the confidence gate must let covered tickets through
AND block tickets with no matching article. The eval includes 18 out-of-scope queries
(genuine retail problems the KB does NOT cover) and sweeps the threshold to find the
value that best separates "covered" from "not covered".

The 50 positive queries are written the way a store associate or customer would actually
describe the problem — deliberately avoiding the wording used in the KB articles, so the
retrieval has to match on meaning, not shared vocabulary.
"""

from vector_store import kb_collection, resolved_collection, _embed, KB_ARTICLES
from prompts import KB_MARGIN_THRESHOLD, KB_ABS_FLOOR

# NOTE: the gate values below (margin 0.06 / floor 0.39) are calibrated against the CURRENT
# 10-article KB. They MUST be re-derived once the KB grows to ~60–80 articles — more distractors
# shift the score distribution and move the right operating point. Re-run this harness after
# any material KB growth and update KB_MARGIN_THRESHOLD / KB_ABS_FLOOR from the sweep.

# ── Positives: naturalistic query -> expected KB article id ──────────────────────
# No query reuses the KB article's own wording (no "Adyen", "401", "sync", "JSON"...).

POSITIVES = [
    # kb-001 — Adyen Payment Gateway Timeout
    ("the card machine just spins and never completes the sale", "kb-001"),
    ("we can't take any card payments this morning, everything hangs", "kb-001"),
    ("customers tap their card and nothing happens, then it errors out", "kb-001"),
    ("every card transaction is failing at the register today", "kb-001"),
    ("the payment step freezes and the sale won't go through", "kb-001"),

    # kb-002 — Click and Collect Orders Not Syncing to POS
    ("a customer came in to pick up something they ordered online and we have no record of it", "kb-002"),
    ("the orders people place on the website aren't showing up on our register", "kb-002"),
    ("someone's here to collect their web order but it's not in our system", "kb-002"),
    ("online reservations for store pickup never reach us", "kb-002"),
    ("we can't find any of the internet orders customers come to collect", "kb-002"),

    # kb-003 — Inventory Sync Discrepancy After Scheduled Job
    ("the system says we have plenty in stock but the shelves are empty", "kb-003"),
    ("our stock counts have been completely off since this morning", "kb-003"),
    ("we keep selling things the computer thinks we have but we don't", "kb-003"),
    ("the numbers on screen don't match what's actually in the storeroom", "kb-003"),
    ("stock figures went haywire overnight", "kb-003"),

    # kb-004 — Product Import Failing Due to Malformed JSON
    ("we uploaded the new collection but none of it shows up in the system", "kb-004"),
    ("the new products we added last night never appeared anywhere", "kb-004"),
    ("our latest catalogue update didn't take, nothing loaded", "kb-004"),
    ("tried to bring in the new season range and it just failed silently", "kb-004"),
    ("the batch of new items we sent over isn't in the system", "kb-004"),

    # kb-005 — API Authentication Failure 401 Unauthorized
    ("our system stopped being able to talk to yours overnight", "kb-005"),
    ("the connection we use to send you orders is suddenly being refused", "kb-005"),
    ("we're getting locked out when our software tries to reach your platform", "kb-005"),
    ("access from our end just started getting rejected this morning", "kb-005"),
    ("the link between our systems broke and now everything we send bounces back", "kb-005"),

    # kb-006 — POS Freezing on Specific Product Barcode Scan
    ("the till completely locks up whenever I scan one specific product", "kb-006"),
    ("scanning this one item crashes the register every single time", "kb-006"),
    ("I have to restart the checkout machine after I scan a particular thing", "kb-006"),
    ("one product freezes the whole system when it goes across the scanner", "kb-006"),
    ("the register dies as soon as this item is scanned", "kb-006"),

    # kb-007 — VAT Not Displaying on Customer Prices
    ("the prices customers see look far too low, like the tax is missing", "kb-007"),
    ("the amount at checkout doesn't seem to include the tax portion", "kb-007"),
    ("our displayed prices are coming up before tax is added", "kb-007"),
    ("the totals shown to customers are missing the tax", "kb-007"),
    ("something's wrong with tax on all our prices in the store", "kb-007"),

    # kb-008 — Loyalty Points Not Updating After Purchase
    ("I shopped yesterday and my rewards balance didn't move", "kb-008"),
    ("customers say they're not getting their points after buying things", "kb-008"),
    ("my points haven't gone up even though I bought something", "kb-008"),
    ("people are complaining their reward balance is stuck", "kb-008"),
    ("nothing gets added to customer rewards after they pay", "kb-008"),

    # kb-009 — Receipt Language Not Matching Store Locale
    ("the printed slips are coming out in the wrong language for our country", "kb-009"),
    ("our receipts print in English but we're in France", "kb-009"),
    ("the paper receipts aren't in our local language anymore", "kb-009"),
    ("customers get receipts they can't read because it's the wrong language", "kb-009"),
    ("the till is printing everything in a foreign language", "kb-009"),

    # kb-010 — EVA Mobile POS App Crashing After iOS Update
    ("since the tablets updated themselves the sales program won't open", "kb-010"),
    ("our handheld devices just crash the moment we launch the app", "kb-010"),
    ("after the phones did an update the checkout app stopped working", "kb-010"),
    ("the mobile devices keep closing the program right after we open it", "kb-010"),
    ("the app on our portable units died after the latest device update", "kb-010"),
]

# ── Negatives: real retail problems the KB does NOT cover ─────────────────────────
# Correct behavior is for the gate to BLOCK these (top score below threshold).

NEGATIVES = [
    "a customer wants to return an item but doesn't have the receipt",
    "how do I check the balance on a gift card",
    "the staff discount isn't coming off at checkout",
    "the receipt printer is out of paper",
    "a customer is disputing a charge on their bank statement",
    "the wifi keeps dropping across the whole store",
    "I need to set up a login for a new cashier",
    "the cash drawer won't pop open",
    "a customer is asking when their online delivery will arrive",
    "how do I do a manager override for a price change",
    "the security tag won't come off a garment",
    "we need to reorder shopping bags and packaging",
    "the store's opening hours are wrong on the map listing",
    "I forgot my manager password and can't log in",
    "the background music in the store cut out",
    "a customer wants to exchange a gift for a different size",
    "the air conditioning in the shop stopped working",
    "someone left their umbrella behind, what do we do with lost property",
]


# ── Retrieval under test ─────────────────────────────────────────────────────────

def _batch_embed(texts: list, chunk: int = 32) -> list:
    """Embed queries in chunks to stay within NIM per-request limits."""
    out = []
    for i in range(0, len(texts), chunk):
        out.extend(_embed(texts[i:i + chunk], "query"))
    return out


def rank_with_embedding(embedding: list) -> list:
    """Return [(doc_id, similarity_score), ...] best-first over the MERGED KB + resolved pool.

    Production (prompts.evaluate_gate) gates on kb + resolved combined, so the eval merges
    them too — otherwise calibration drifts once the Phase 3 feedback loop populates
    resolved_tickets. While resolved is empty this is identical to KB-only.
    """
    ranked = []
    for coll in (kb_collection, resolved_collection):
        if coll.count() == 0:
            continue
        results = coll.query(query_embeddings=[embedding], n_results=coll.count())
        ranked.extend(
            (doc_id, round(1 - dist, 3))
            for doc_id, dist in zip(results["ids"][0], results["distances"][0])
        )
    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked


# ── Metrics ──────────────────────────────────────────────────────────────────────

def rank_of(expected_id: str, ranked: list) -> int:
    for i, (doc_id, _score) in enumerate(ranked):
        if doc_id == expected_id:
            return i + 1
    return 0


def main():
    title_by_id = {a["id"]: a["title"] for a in KB_ARTICLES}

    print(f"Embedding {len(POSITIVES)} positive + {len(NEGATIVES)} negative queries...\n")
    pos_embeddings = _batch_embed([q for q, _ in POSITIVES])
    neg_embeddings = _batch_embed(NEGATIVES)

    # ── Ranking metrics (positives) ──────────────────────────────────────────────
    n = len(POSITIVES)
    hits_at_1 = 0
    reciprocal_ranks = 0.0
    gaps = []
    pos_top1 = []          # (query, expected, top1_id, top1_score, gap)
    misses = []

    for (query, expected), emb in zip(POSITIVES, pos_embeddings):
        ranked = rank_with_embedding(emb)
        rank = rank_of(expected, ranked)
        top1_id, top1_score = ranked[0]
        top2_score = ranked[1][1] if len(ranked) > 1 else 0.0
        gap = round(top1_score - top2_score, 3)
        gaps.append(gap)
        pos_top1.append((query, expected, top1_id, top1_score, gap))

        if rank == 1:
            hits_at_1 += 1
        if rank != 0:
            reciprocal_ranks += 1.0 / rank
        if rank != 1:
            misses.append((query, expected, ranked[:3]))

    recall_at_1 = hits_at_1 / n
    mrr = reciprocal_ranks / n
    avg_gap = sum(gaps) / len(gaps)
    avg_pos_top1 = sum(s for _, _, _, s, _ in pos_top1) / n

    # ── Negative top-1 scores and margins ────────────────────────────────────────
    neg_top1 = []  # (query, top1_id, top1_score, margin)
    for query, emb in zip(NEGATIVES, neg_embeddings):
        ranked = rank_with_embedding(emb)
        top1_score = ranked[0][1]
        top2_score = ranked[1][1] if len(ranked) > 1 else 0.0
        neg_top1.append((query, ranked[0][0], top1_score, round(top1_score - top2_score, 3)))
    avg_neg_top1 = sum(s for _, _, s, _ in neg_top1) / len(neg_top1)
    avg_neg_margin = sum(mg for _, _, _, mg in neg_top1) / len(neg_top1)
    n_neg = len(neg_top1)

    print("=" * 64)
    print("  RANKING (50 positives)")
    print(f"    recall@1        : {recall_at_1:.3f}")
    print(f"    MRR             : {mrr:.3f}")
    print(f"    avg top1 score  : {avg_pos_top1:.3f}")
    print(f"    avg #1-#2 margin: {avg_gap:.3f}")
    print("-" * 64)
    print("  SEPARATION - covered vs not-covered")
    print(f"    {'':22}{'avg top1':>10}{'avg margin':>12}")
    print(f"    covered (positives)   {avg_pos_top1:>10.3f}{avg_gap:>12.3f}")
    print(f"    not covered (negs)    {avg_neg_top1:>10.3f}{avg_neg_margin:>12.3f}")
    print("=" * 64)

    # ── Gate sweep — serve iff top1 >= floor AND (top1 - top2) >= margin ──────────
    # Rationale: an uncovered ticket matches everything equally poorly (small margin);
    # a covered ticket should have one article standing clearly above the rest. The floor
    # is a safety net against weak-but-gapped top hits. Floor is held at the configured
    # KB_ABS_FLOOR; margin varies. FPR is reported as a COUNT (n/N) — 18 negatives is too
    # small a sample to state a rate like "0%".
    def _served(top1_score, mg, m):
        return top1_score >= KB_ABS_FLOOR and mg >= m

    print(f"\n  GATE SWEEP   (serve iff top1 >= floor {KB_ABS_FLOOR} AND margin >= m)")
    print(f"  {'margin':>6} {'abstain_all':>12} {'pos_served_ok':>14} {'pos_abstain':>12} {'FPR':>9}")
    rows = []
    m = 0.00
    while m <= 0.2001:
        pos_served_ok = sum(1 for _, exp, t1id, t1s, mg in pos_top1 if _served(t1s, mg, m) and t1id == exp) / n
        pos_abstain = sum(1 for _, _, _, t1s, mg in pos_top1 if not _served(t1s, mg, m)) / n
        neg_served = sum(1 for _, _, t1s, mg in neg_top1 if _served(t1s, mg, m))
        abstain_all = (
            sum(1 for _, _, _, t1s, mg in pos_top1 if not _served(t1s, mg, m))
            + sum(1 for _, _, t1s, mg in neg_top1 if not _served(t1s, mg, m))
        ) / (n + n_neg)
        rows.append((round(m, 2), abstain_all, pos_served_ok, pos_abstain, neg_served))
        print(f"  {m:>6.2f} {abstain_all:>12.2f} {pos_served_ok:>14.2f} {pos_abstain:>12.2f} {neg_served:>6}/{n_neg}")
        m += 0.02

    # Report the configured operating point explicitly.
    cm = KB_MARGIN_THRESHOLD
    op_served = sum(1 for _, exp, t1id, t1s, mg in pos_top1 if _served(t1s, mg, cm) and t1id == exp)
    op_neg_served = sum(1 for _, _, t1s, mg in neg_top1 if _served(t1s, mg, cm))
    print(f"\n  CONFIGURED GATE  (margin {KB_MARGIN_THRESHOLD}, floor {KB_ABS_FLOOR}):")
    print(f"    covered served correctly : {op_served}/{n}")
    print(f"    false positives (FPR)    : {op_neg_served}/{n_neg} negatives")

    # ── Diagnostics ──────────────────────────────────────────────────────────────
    if misses:
        print(f"\n  {len(misses)} positive(s) NOT ranked #1:")
        for query, expected, top3 in misses:
            got = [f"{title_by_id.get(i, i)} ({s})" for i, s in top3]
            print(f"    query    : {query}")
            print(f"    expected : {title_by_id.get(expected, expected)}")
            print(f"    got      : {got}\n")

    worst_neg = sorted(neg_top1, key=lambda x: -x[3])[:4]
    print("  Largest-margin negatives (most likely to slip through a margin gate):")
    for query, top1, score, margin in worst_neg:
        print(f"    margin {margin}  (top1 {score})  {query[:48]:48} -> {title_by_id.get(top1, top1)}")


if __name__ == "__main__":
    main()

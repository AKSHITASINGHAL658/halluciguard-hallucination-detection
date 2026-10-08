from app.core.mi_pruning import compute_pmi_scores, prune_chunks

CHUNKS = [
    {"chunk_id": 1, "text": "Solar capacity additions reached 345 gigawatts in 2023, led by China solar installations.", "embedding": [1, 0, 0]},
    {"chunk_id": 2, "text": "Solar capacity additions in 2023 were 345 gigawatts, with China installing most solar.", "embedding": [0.99, 0.1, 0]},
    {"chunk_id": 3, "text": "Geothermal plants in Kenya supply close to forty five percent of electricity.", "embedding": [0, 1, 0]},
    {"chunk_id": 4, "text": "Pumped hydro provides most grid storage worldwide today.", "embedding": [0, 0, 1]},
]
Q = "How much solar capacity was added in 2023?"


def test_pmi_ranks_relevant_chunks_first():
    s = compute_pmi_scores(Q, CHUNKS)
    assert max(s) in (s[0], s[1]) and s[2] < s[0] and s[3] < s[0]


def test_prune_drops_irrelevant_and_redundant():
    kept = prune_chunks(Q, CHUNKS)
    ids = [c["chunk_id"] for c in kept]
    assert len(kept) < len(CHUNKS)
    assert not ({1, 2} <= set(ids))  # near-duplicates not both kept (beyond min_keep top-up)
    assert 3 not in ids and 4 not in ids or len(ids) == 2


def test_min_keep():
    assert len(prune_chunks("zzz qqq", CHUNKS, keep_ratio=0.99)) >= 2
    assert len(prune_chunks(Q, CHUNKS[:1])) == 1

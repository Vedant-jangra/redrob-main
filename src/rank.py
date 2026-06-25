"""
rank.py — Main ranking entry point for Redrob Hackathon
Usage: python src/rank.py --candidates data/candidates.jsonl --out submission.csv

Must run in ≤5 minutes on CPU with no network calls.
Loads precomputed artifacts from artifacts/ and outputs top 100 ranked candidates.
"""

import argparse
import json
import time
import csv
import os
import sys
import numpy as np

# ── Constants ────────────────────────────────────────────────────────────────
ARTIFACTS_DIR = "artifacts"
TOP_K_RETRIEVE = 500       # Retrieve top-500 from FAISS, then re-rank to 100
TOP_N_FINAL    = 100       # Final submission size
MODEL_NAME     = "all-MiniLM-L6-v2"

# JD text (Senior AI Engineer — Redrob AI)
JD_TEXT = """
Senior AI Engineer at Redrob AI.
Must have: production embeddings and retrieval experience, vector databases (FAISS, Pinecone, Qdrant, Weaviate),
strong Python, evaluation frameworks (NDCG, MRR, MAP).
Preferred: LLM fine-tuning, learning to rank (LTR), NLP and information retrieval background.
Experience building and shipping production AI systems at product companies.
Red flags: consulting-only background (TCS, Infosys, Wipro, Cognizant), pure CV or speech or robotics,
pure research with no product deployment, keyword-stuffed profiles with no real projects.
"""


# ── Load artifacts ─────────────────────────────────────────────────────────
def load_artifacts(artifacts_dir: str):
    print("Loading precomputed artifacts...")
    t0 = time.time()

    embeddings_path  = os.path.join(artifacts_dir, "embeddings.npy")
    ids_path         = os.path.join(artifacts_dir, "candidate_ids.json")
    signals_path     = os.path.join(artifacts_dir, "signals.json")
    honeypots_path   = os.path.join(artifacts_dir, "honeypot_ids.json")

    for p in [embeddings_path, ids_path, signals_path, honeypots_path]:
        if not os.path.exists(p):
            print(f"ERROR: Missing artifact: {p}")
            print("Run precompute.py first: python src/precompute.py --candidates data/candidates.jsonl --artifacts artifacts/")
            sys.exit(1)

    embeddings    = np.load(embeddings_path).astype("float32")
    candidate_ids = json.load(open(ids_path))
    signals       = json.load(open(signals_path))
    honeypot_ids  = set(json.load(open(honeypots_path)))

    print(f"  Loaded {len(candidate_ids):,} candidates, {len(honeypot_ids)} honeypots flagged ({time.time()-t0:.1f}s)")
    return embeddings, candidate_ids, signals, honeypot_ids


# ── Load candidates into a lookup dict ────────────────────────────────────
def load_candidate_lookup(candidates_path: str, needed_ids: set) -> dict:
    """Load candidate profiles, checking multiple possible ID field names."""
    print(f"Loading candidate profiles from {candidates_path}...")
    t0 = time.time()

    lookup = {}
    is_jsonl = candidates_path.endswith(".jsonl")

    with open(candidates_path, "r", encoding="utf-8") as f:
        if is_jsonl:
            lines = f
        else:
            # JSON file — could be a list or a dict
            raw = json.load(f)
            if isinstance(raw, list):
                lines = [json.dumps(c) for c in raw]
            elif isinstance(raw, dict):
                # Could be {"candidates": [...]} or similar
                for v in raw.values():
                    if isinstance(v, list):
                        lines = [json.dumps(c) for c in v]
                        break
                else:
                    lines = []
            else:
                lines = []

        for line in lines:
            if isinstance(line, str):
                line = line.strip()
                if not line:
                    continue
                try:
                    c = json.loads(line)
                except json.JSONDecodeError:
                    continue
            else:
                c = line

            # Try multiple ID field names
            cid = (c.get("candidate_id") or c.get("id") or
                   c.get("candidateId") or c.get("_id"))
            if cid is None:
                continue

            # If needed_ids provided, filter; otherwise load all
            if needed_ids is None or cid in needed_ids:
                lookup[cid] = c

    print(f"  Loaded {len(lookup):,} profiles ({time.time()-t0:.1f}s)")
    return lookup


# ── Embed the JD ──────────────────────────────────────────────────────────
def embed_jd(jd_text: str) -> np.ndarray:
    print(f"Embedding JD with {MODEL_NAME}...")
    t0 = time.time()
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL_NAME)
    jd_vec = model.encode([jd_text], normalize_embeddings=True, show_progress_bar=False)
    print(f"  JD embedded ({time.time()-t0:.1f}s)")
    return jd_vec.astype("float32")


# ── FAISS nearest-neighbor retrieval ──────────────────────────────────────
def retrieve_top_k(embeddings: np.ndarray, jd_vec: np.ndarray, k: int) -> tuple:
    """Return (indices, cosine_scores) of top-k candidates."""
    print(f"Running FAISS search for top-{k}...")
    t0 = time.time()
    import faiss

    # Normalize embeddings for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1e-9
    normed = embeddings / norms

    d = normed.shape[1]
    index = faiss.IndexFlatIP(d)   # Inner product on normalized vecs = cosine sim
    index.add(normed)

    scores, indices = index.search(jd_vec, k)
    print(f"  Retrieved top-{k} candidates ({time.time()-t0:.1f}s)")
    return indices[0], scores[0]


# ── Scoring helpers ───────────────────────────────────────────────────────
from scorer import compute_final_score, generate_reasoning

# ── Main pipeline ─────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Redrob Hackathon Ranker")
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl")
    parser.add_argument("--artifacts",  default=ARTIFACTS_DIR, help="Path to artifacts dir")
    parser.add_argument("--out",        default="submission.csv", help="Output CSV path")
    parser.add_argument("--top-k",      type=int, default=TOP_K_RETRIEVE, help="Candidates to retrieve from FAISS")
    args = parser.parse_args()

    t_start = time.time()
    print("\n=== Redrob Ranker ===")

    # Step 1 — Load artifacts
    embeddings, all_candidate_ids, all_signals, honeypot_ids = load_artifacts(args.artifacts)

    # Step 2 — Embed JD
    jd_vec = embed_jd(JD_TEXT)

    # Step 3 — FAISS retrieval
    k = min(args.top_k, len(all_candidate_ids))
    indices, semantic_scores = retrieve_top_k(embeddings, jd_vec, k)

    top_ids     = [all_candidate_ids[i] for i in indices]
    top_sem     = list(semantic_scores)

    # Step 4 — Load candidate profiles
    # For .jsonl (100K file): filter to only top-K IDs to save RAM
    # For .json (sample file): load all, since IDs in artifacts may differ
    if args.candidates.endswith(".jsonl"):
        lookup = load_candidate_lookup(args.candidates, set(top_ids))
    else:
        lookup = load_candidate_lookup(args.candidates, None)

    # Step 5 — Re-score every retrieved candidate
    print(f"Re-scoring {len(top_ids)} candidates...")
    t0 = time.time()

    results = []
    for cid, sem_score in zip(top_ids, top_sem):
        candidate = lookup.get(cid)
        if not candidate:
            continue

        # Load behavioral signals
        sig = all_signals.get(cid, {})
        # Re-score using scorer.py logic
        res = compute_final_score(float(sem_score), candidate, sig, cid in honeypot_ids)
        if res["is_honeypot"]:
            continue

        results.append({
            "candidate_id": cid,
            "final_score":  res["final_score"],
            "score_components": res,
            "candidate":    candidate,
        })

    print(f"  Scored {len(results)} candidates ({time.time()-t0:.1f}s)")

    # Step 6 — Sort and take top 100
    results.sort(key=lambda x: (-x["final_score"], x["candidate_id"]))
    top100 = results[:TOP_N_FINAL]

    # Step 7 — Write submission CSV
    print(f"Writing {args.out}...")
    os.makedirs(os.path.dirname(args.out) if os.path.dirname(args.out) else ".", exist_ok=True)

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["candidate_id", "rank", "score", "reasoning"])
        writer.writeheader()
        for rank, r in enumerate(top100, start=1):
            reasoning = generate_reasoning(
                r["candidate"], r["score_components"], rank
            )
            writer.writerow({
                "candidate_id": r["candidate_id"],
                "rank":         rank,
                "score":        round(r["final_score"], 6),
                "reasoning":    reasoning,
            })

    elapsed = time.time() - t_start

    if not top100:
        print("\n⚠️  WARNING: top100 is empty!")
        print("   This usually means precomputed artifact IDs don't match the candidate file.")
        print("   Re-run: python src/precompute.py --candidates data/sample_candidates.json --artifacts artifacts/ --sample")
        return

    print(f"\n✅ Done! {len(top100)} candidates ranked → {args.out}")
    print(f"   Total time: {elapsed:.1f}s")
    print(f"   Top score:  {top100[0]['final_score']:.4f}")
    print(f"   Honeypots filtered: {len(honeypot_ids)}")

    # Sanity check: how many honeypots in top 100?
    hp_in_top = sum(1 for r in top100 if r["candidate_id"] in honeypot_ids)
    print(f"   Honeypots in top-100: {hp_in_top} (must be < 10 to avoid disqualification)")


if __name__ == "__main__":
    main()
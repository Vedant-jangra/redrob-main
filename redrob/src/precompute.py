"""
precompute.py — Offline pre-computation step
Run this ONCE before rank.py.
Produces:
  artifacts/embeddings.npy       — (N, 384) float32 embedding matrix
  artifacts/candidate_ids.json   — ordered list of candidate_ids
  artifacts/signals.json         — normalized behavioral signals per candidate
  artifacts/honeypot_ids.json    — flagged honeypot candidate IDs
 
Usage:
  python src/precompute.py --candidates data/candidates.jsonl --artifacts artifacts/
  python src/precompute.py --candidates data/sample_candidates.json --artifacts artifacts/ --sample
"""
 
import argparse
import json
import os
import time
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from datetime import datetime, timezone
 
# ── Constants ──────────────────────────────────────────────────────────────────
 
# Consulting/services companies to penalize (from JD)
CONSULTING_COMPANIES = {
    "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
    "hcl", "tech mahindra", "mphasis", "hexaware", "mindtree", "l&t infotech",
    "ltimindtree", "persistent", "niit", "zensar", "mastech"
}
 
# Skills the JD explicitly wants
JD_CORE_SKILLS = {
    "embeddings", "sentence-transformers", "faiss", "pinecone", "weaviate",
    "qdrant", "milvus", "opensearch", "elasticsearch", "vector search",
    "hybrid search", "retrieval", "ranking", "reranking", "bm25",
    "dense retrieval", "sparse retrieval", "ndcg", "mrr", "map",
    "learning to rank", "ltr", "xgboost", "fine-tuning", "lora", "qlora",
    "peft", "rag", "information retrieval", "nlp", "python",
    "recommendation system", "search", "a/b testing"
}
 
# Skills that are red flags (CV/speech/robotics - JD explicitly excludes)
JD_NEGATIVE_SKILLS = {
    "computer vision", "image classification", "object detection", "yolo",
    "speech recognition", "tts", "text to speech", "robotics", "ros",
    "autonomous driving", "image segmentation", "pose estimation"
}
 
# ── Text builder ───────────────────────────────────────────────────────────────
 
def build_candidate_text(candidate: dict) -> str:
    """
    Builds a rich text representation of a candidate for embedding.
    Weights important fields more by repeating them.
    """
    parts = []
    profile = candidate.get("profile", {})
    
    # Headline and summary (most important — repeat for weight)
    headline = profile.get("headline", "")
    summary = profile.get("summary", "")
    if headline:
        parts.append(headline)
        parts.append(headline)  # repeat for weight
    if summary:
        parts.append(summary)
 
    # Current role
    title = profile.get("current_title", "")
    company = profile.get("current_company", "")
    if title:
        parts.append(f"Current role: {title} at {company}")
 
    # Career history — descriptions are gold
    for job in candidate.get("career_history", []):
        job_title = job.get("title", "")
        job_company = job.get("company", "")
        job_desc = job.get("description", "")
        duration = job.get("duration_months", 0)
        if job_desc:
            parts.append(f"{job_title} at {job_company} ({duration} months): {job_desc}")
 
    # Skills — only relevant ones, weighted by proficiency
    for skill in candidate.get("skills", []):
        name = skill.get("name", "").lower()
        proficiency = skill.get("proficiency", "")
        duration = skill.get("duration_months", 0)
        if name in JD_CORE_SKILLS:
            # Repeat relevant skills for extra weight
            skill_str = f"Skill: {skill['name']} ({proficiency}, {duration} months)"
            parts.append(skill_str)
            parts.append(skill_str)
        else:
            parts.append(f"Skill: {skill['name']} ({proficiency})")
 
    return " | ".join(parts)
 
 
# ── Behavioral signal normalizer ───────────────────────────────────────────────
 
def normalize_signals(candidate: dict) -> dict:
    """
    Extracts and normalizes behavioral signals into a 0-1 score dict.
    Returns individual signal scores for use in rank.py scoring formula.
    """
    sig = candidate.get("redrob_signals", {})
    now = datetime(2026, 6, 22, tzinfo=timezone.utc)
 
    # 1. Recency — days since last active (0=today, 1=very old)
    last_active_str = sig.get("last_active_date", "")
    try:
        last_active = datetime.fromisoformat(last_active_str).replace(tzinfo=timezone.utc)
        days_inactive = (now - last_active).days
        recency_score = max(0.0, 1.0 - (days_inactive / 180.0))  # 0 after 6 months
    except Exception:
        recency_score = 0.3  # unknown = neutral-low
 
    # 2. Availability signals
    open_to_work = 1.0 if sig.get("open_to_work_flag", False) else 0.3
    
    notice_days = sig.get("notice_period_days", 90)
    if notice_days <= 15:
        notice_score = 1.0
    elif notice_days <= 30:
        notice_score = 0.85
    elif notice_days <= 60:
        notice_score = 0.6
    elif notice_days <= 90:
        notice_score = 0.4
    else:
        notice_score = 0.2  # 90+ days notice = hard to hire
 
    # 3. Engagement signals
    response_rate = sig.get("recruiter_response_rate", 0.5)
    
    avg_response_hours = sig.get("avg_response_time_hours", 48)
    if avg_response_hours <= 4:
        response_time_score = 1.0
    elif avg_response_hours <= 24:
        response_time_score = 0.8
    elif avg_response_hours <= 72:
        response_time_score = 0.5
    else:
        response_time_score = 0.2
 
    interview_rate = sig.get("interview_completion_rate", 0.5)
    
    offer_acceptance = sig.get("offer_acceptance_rate", -1)
    offer_score = offer_acceptance if offer_acceptance >= 0 else 0.5  # -1 = no history
 
    # 4. Profile quality
    completeness = sig.get("profile_completeness_score", 50) / 100.0
    github_score = sig.get("github_activity_score", -1)
    github_normalized = (github_score / 100.0) if github_score >= 0 else 0.3
 
    # 5. Location/relocation (JD wants Pune/Noida/Delhi NCR)
    profile = candidate.get("profile", {})
    location = profile.get("location", "").lower()
    country = profile.get("country", "").lower()
    willing_to_relocate = sig.get("willing_to_relocate", False)
    
    preferred_locations = {"pune", "noida", "delhi", "gurgaon", "gurugram", 
                           "hyderabad", "mumbai", "bangalore", "bengaluru"}
    if any(loc in location for loc in preferred_locations) and country == "india":
        location_score = 1.0
    elif country == "india" and willing_to_relocate:
        location_score = 0.7
    elif country == "india":
        location_score = 0.5
    elif willing_to_relocate:
        location_score = 0.3
    else:
        location_score = 0.1  # outside india, won't relocate
 
    return {
        "recency_score": round(recency_score, 4),
        "open_to_work": round(open_to_work, 4),
        "notice_score": round(notice_score, 4),
        "response_rate": round(float(response_rate), 4),
        "response_time_score": round(response_time_score, 4),
        "interview_rate": round(float(interview_rate), 4),
        "offer_score": round(float(offer_score), 4),
        "completeness": round(completeness, 4),
        "github_score": round(github_normalized, 4),
        "location_score": round(location_score, 4),
    }
 
 
# ── Honeypot detector ──────────────────────────────────────────────────────────
 
def is_honeypot(candidate: dict) -> bool:
    """
    Detects candidates with impossible/suspicious profiles.
    Returns True if the candidate looks like a honeypot.
    """
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
 
    # Check 1: Experience at company founded after candidate started
    # (e.g. 8 years exp at a 3-year-old company)
    total_career_months = sum(j.get("duration_months", 0) for j in career)
    years_of_exp = profile.get("years_of_experience", 0)
    if years_of_exp > 0 and total_career_months > 0:
        career_years = total_career_months / 12.0
        # If stated experience is much more than career history allows
        if years_of_exp > career_years + 5:
            return True
 
    # Check 2: Expert in too many skills with 0 months duration
    expert_zero_duration = sum(
        1 for s in skills
        if s.get("proficiency") == "expert" and s.get("duration_months", 0) == 0
    )
    if expert_zero_duration >= 5:
        return True
 
    # Check 3: Impossible skill duration (skill used longer than total experience)
    total_exp_months = years_of_exp * 12
    for skill in skills:
        skill_months = skill.get("duration_months", 0)
        if total_exp_months > 0 and skill_months > total_exp_months + 12:
            return True
 
    # Check 4: Current company start date is after years of experience allows
    for job in career:
        if job.get("is_current", False):
            try:
                start = datetime.fromisoformat(job["start_date"])
                now = datetime(2026, 6, 22)
                months_at_current = (now.year - start.year) * 12 + (now.month - start.month)
                # If they've only been at current job a short time but claim huge experience
                # and ALL their experience is only at this one company
                if len(career) == 1 and years_of_exp > (months_at_current / 12) + 2:
                    return True
            except Exception:
                pass
 
    # Check 5: Keyword stuffer (Lexical diversity < 0.25)
    text_corpus = profile.get("summary", "") + " " + profile.get("headline", "")
    for job in career:
        text_corpus += " " + job.get("description", "")
    import re
    words = re.findall(r'\b\w+\b', text_corpus.lower())
    if len(words) > 50:
        unique_words = set(words)
        diversity = len(unique_words) / len(words)
        if diversity < 0.25:
            return True

    return False
 
 
# ── Main ───────────────────────────────────────────────────────────────────────
 
def main():
    parser = argparse.ArgumentParser(description="Precompute embeddings and signals")
    parser.add_argument("--candidates", default="data/candidates.jsonl",
                        help="Path to candidates.jsonl or sample_candidates.json")
    parser.add_argument("--artifacts", default="artifacts/",
                        help="Output directory for artifacts")
    parser.add_argument("--sample", action="store_true",
                        help="Use sample_candidates.json (JSON array) instead of JSONL")
    parser.add_argument("--model", default="all-MiniLM-L6-v2",
                        help="Sentence transformer model name")
    args = parser.parse_args()
 
    os.makedirs(args.artifacts, exist_ok=True)
    
    print(f"Loading candidates from {args.candidates}...")
    t0 = time.time()
 
    # Load candidates
    candidates = []
    if args.sample or args.candidates.endswith(".json"):
        with open(args.candidates, "r", encoding="utf-8") as f:
            candidates = json.load(f)
    else:
        with open(args.candidates, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    candidates.append(json.loads(line))
 
    print(f"Loaded {len(candidates)} candidates in {time.time()-t0:.1f}s")
 
    # ── Step 1: Build texts ──
    print("\nStep 1/4: Building candidate texts...")
    texts = []
    candidate_ids = []
    for c in tqdm(candidates):
        candidate_ids.append(c["candidate_id"])
        texts.append(build_candidate_text(c))
 
    # ── Step 2: Compute embeddings ──
    print(f"\nStep 2/4: Computing embeddings with {args.model}...")
    print("(This will download the model on first run — ~90MB)")
    model = SentenceTransformer(args.model)
    
    t1 = time.time()
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,   # L2 normalize for cosine similarity
        convert_to_numpy=True
    )
    print(f"Embeddings computed in {time.time()-t1:.1f}s — shape: {embeddings.shape}")
 
    # ── Step 3: Normalize signals ──
    print("\nStep 3/4: Normalizing behavioral signals...")
    signals = {}
    for c in tqdm(candidates):
        signals[c["candidate_id"]] = normalize_signals(c)
 
    # ── Step 4: Detect honeypots ──
    print("\nStep 4/4: Detecting honeypot candidates...")
    honeypot_ids = []
    for c in tqdm(candidates):
        if is_honeypot(c):
            honeypot_ids.append(c["candidate_id"])
    print(f"Flagged {len(honeypot_ids)} honeypot candidates")
 
    # ── Save artifacts ──
    print("\nSaving artifacts...")
    
    emb_path = os.path.join(args.artifacts, "embeddings.npy")
    np.save(emb_path, embeddings.astype(np.float32))
    print(f"Saved embeddings: {emb_path} ({embeddings.shape})")
 
    ids_path = os.path.join(args.artifacts, "candidate_ids.json")
    with open(ids_path, "w") as f:
        json.dump(candidate_ids, f)
    print(f"Saved candidate IDs: {ids_path}")
 
    sig_path = os.path.join(args.artifacts, "signals.json")
    with open(sig_path, "w") as f:
        json.dump(signals, f)
    print(f"Saved signals: {sig_path}")
 
    hp_path = os.path.join(args.artifacts, "honeypot_ids.json")
    with open(hp_path, "w") as f:
        json.dump(honeypot_ids, f)
    print(f"Saved honeypot IDs: {hp_path}")
 
    total_time = time.time() - t0
    print(f"\nDone! Total time: {total_time:.1f}s")
    print(f"Artifacts saved to: {args.artifacts}")
 
 
if __name__ == "__main__":
    main()
 
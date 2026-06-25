"""
scorer.py — Scoring formula for candidate ranking
Combines semantic similarity + behavioral signals + career quality.

Final score = semantic_score * behavioral_multiplier * career_bonus
"""

import json
import numpy as np
from datetime import datetime

# ── JD-specific constants ──────────────────────────────────────────────────────

# Consulting companies explicitly mentioned as red flags in JD
CONSULTING_COMPANIES = {
    "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
    "hcl", "tech mahindra", "mphasis", "hexaware", "mindtree", "l&t infotech",
    "ltimindtree", "persistent", "niit", "zensar", "mastech"
}

# Product companies = strong positive signal
PRODUCT_COMPANY_KEYWORDS = {
    "startup", "series a", "series b", "series c", "saas", "platform",
    "product", "ai", "ml", "tech", "software", "data", "analytics"
}

# Skills the JD explicitly requires
CORE_SKILLS = {
    "embeddings", "sentence-transformers", "faiss", "pinecone", "weaviate",
    "qdrant", "milvus", "opensearch", "elasticsearch", "vector search",
    "hybrid search", "retrieval", "ranking", "reranking", "bm25",
    "dense retrieval", "ndcg", "mrr", "map", "learning to rank", "ltr",
    "fine-tuning", "lora", "qlora", "peft", "rag", "information retrieval",
    "nlp", "python", "recommendation", "search", "a/b testing",
    "xgboost", "neural ranking"
}

# Skills that are negative signals (JD explicitly excludes CV/speech/robotics)
NEGATIVE_SKILLS = {
    "computer vision", "image classification", "object detection", "yolo",
    "speech recognition", "tts", "text to speech", "robotics", "ros",
    "autonomous driving", "image segmentation", "pose estimation",
    "photoshop", "illustrator", "figma", "unity", "unreal"
}


# ── Career quality scorer ──────────────────────────────────────────────────────

def score_career_quality(candidate: dict) -> float:
    """
    Scores career quality based on:
    - Product company vs consulting company history
    - Years of relevant experience (5-9 sweet spot per JD)
    - Whether they've shipped real systems (inferred from descriptions)
    - Title progression
    Returns 0.0 - 1.0
    """
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])

    years_exp = profile.get("years_of_experience", 0)
    current_company = profile.get("current_company", "").lower()
    current_industry = profile.get("current_industry", "").lower()

    # ── Experience range score (JD wants 5-9 years) ──
    if 5 <= years_exp <= 9:
        exp_score = 1.0
    elif 4 <= years_exp < 5:
        exp_score = 0.85
    elif 9 < years_exp <= 12:
        exp_score = 0.8
    elif 3 <= years_exp < 4:
        exp_score = 0.6
    elif years_exp > 12:
        exp_score = 0.65  # over-experienced, might want more senior role
    else:
        exp_score = 0.3   # too junior

    # ── Company type score ──
    consulting_months = 0
    product_months = 0
    total_months = 0

    for job in career:
        company = job.get("company", "").lower()
        industry = job.get("industry", "").lower()
        duration = job.get("duration_months", 0)
        total_months += duration

        # Check if consulting
        is_consulting = any(c in company for c in CONSULTING_COMPANIES)
        is_it_services = "it services" in industry or "consulting" in industry

        if is_consulting or is_it_services:
            consulting_months += duration
        else:
            product_months += duration

    if total_months > 0:
        consulting_ratio = consulting_months / total_months
        if consulting_ratio >= 0.8:
            company_score = 0.2   # almost all consulting = bad fit per JD
        elif consulting_ratio >= 0.5:
            company_score = 0.5
        elif consulting_ratio >= 0.2:
            company_score = 0.75
        else:
            company_score = 1.0   # mostly product companies
    else:
        company_score = 0.5

    # ── Description quality score ──
    # Check if career descriptions mention shipping real systems
    ship_keywords = {
        "shipped", "deployed", "production", "launched", "built", "designed",
        "architected", "scaled", "improved", "reduced", "increased", "led",
        "owned", "end-to-end", "real users", "at scale"
    }
    research_keywords = {
        "paper", "arxiv", "published", "academic", "research lab",
        "phd", "thesis", "conference", "journal", "citation"
    }

    ship_score = 0.5
    all_descriptions = " ".join(
        job.get("description", "") for job in career
    ).lower()

    ship_hits = sum(1 for kw in ship_keywords if kw in all_descriptions)
    research_hits = sum(1 for kw in research_keywords if kw in all_descriptions)

    if ship_hits >= 5:
        ship_score = 1.0
    elif ship_hits >= 3:
        ship_score = 0.8
    elif ship_hits >= 1:
        ship_score = 0.6

    if research_hits >= 3:
        ship_score *= 0.7   # pure research background, JD explicitly dislikes

    # ── Combine ──
    career_score = (
        0.35 * exp_score +
        0.40 * company_score +
        0.25 * ship_score
    )

    return round(min(1.0, career_score), 4)


# ── Skill match scorer ─────────────────────────────────────────────────────────

def score_skills(candidate: dict) -> float:
    """
    Scores how well the candidate's skills match the JD.
    Weights by proficiency and duration.
    Penalizes negative skills (CV/speech/robotics).
    Returns 0.0 - 1.0
    """
    skills = candidate.get("skills", [])
    if not skills:
        return 0.3

    proficiency_weights = {
        "expert": 1.0,
        "advanced": 0.85,
        "intermediate": 0.6,
        "beginner": 0.3
    }

    core_score = 0.0
    negative_count = 0
    total_core_found = 0

    for skill in skills:
        name = skill.get("name", "").lower()
        proficiency = skill.get("proficiency", "intermediate")
        duration = skill.get("duration_months", 0)

        prof_weight = proficiency_weights.get(proficiency, 0.5)
        duration_weight = min(1.0, duration / 24.0)  # caps at 24 months

        if name in CORE_SKILLS:
            total_core_found += 1
            core_score += prof_weight * (0.6 + 0.4 * duration_weight)

        if name in NEGATIVE_SKILLS:
            negative_count += 1

    # Normalize by expected number of core skills
    expected_core = 6  # we expect a good candidate to have ~6 core skills
    normalized_core = min(1.0, core_score / expected_core)

    # Penalty for too many negative skills
    negative_penalty = min(0.4, negative_count * 0.08)

    skill_score = max(0.0, normalized_core - negative_penalty)
    return round(skill_score, 4)


# ── Behavioral multiplier ──────────────────────────────────────────────────────

def compute_behavioral_multiplier(signals: dict) -> float:
    """
    Combines normalized behavioral signals into a single multiplier.
    A ghost candidate (inactive, unresponsive) gets ~0.3
    An ideal candidate (active, responsive, low notice) gets ~1.0
    """
    # Weighted combination of behavioral signals
    multiplier = (
        0.25 * signals["recency_score"] +        # is the person active?
        0.20 * signals["open_to_work"] +          # are they looking?
        0.15 * signals["notice_score"] +          # can they join soon?
        0.15 * signals["response_rate"] +         # do they reply to recruiters?
        0.10 * signals["interview_rate"] +        # do they show up to interviews?
        0.05 * signals["response_time_score"] +   # how fast do they reply?
        0.05 * signals["offer_score"] +           # do they accept offers?
        0.05 * signals["location_score"]          # are they in the right city?
    )

    # Scale to 0.3 - 1.0 range (never fully zero out a good candidate)
    scaled = 0.3 + 0.7 * multiplier
    return round(scaled, 4)


# ── Final scorer ───────────────────────────────────────────────────────────────

def compute_final_score(
    semantic_score: float,
    candidate: dict,
    signals: dict,
    is_honeypot: bool = False
) -> dict:
    """
    Computes the final composite score for a candidate.

    Args:
        semantic_score: cosine similarity between candidate embedding and JD embedding (0-1)
        candidate: full candidate dict
        signals: normalized behavioral signals from precompute.py
        is_honeypot: whether this candidate was flagged as a honeypot

    Returns:
        dict with final_score and component scores for reasoning
    """

    # Immediately disqualify honeypots
    if is_honeypot:
        return {
            "final_score": 0.0,
            "semantic_score": semantic_score,
            "career_score": 0.0,
            "skill_score": 0.0,
            "behavioral_multiplier": 0.0,
            "is_honeypot": True
        }

    # Component scores
    career_score = score_career_quality(candidate)
    skill_score = score_skills(candidate)
    behavioral_mult = compute_behavioral_multiplier(signals)

    # ── Profile-fit score (semantic + career + skills) ──
    profile_fit = (
        0.50 * semantic_score +   # embedding similarity is strongest signal
        0.30 * career_score +     # career quality (company type, experience)
        0.20 * skill_score        # explicit skill match
    )

    # ── Final score = profile fit × behavioral multiplier ──
    # Behavioral acts as a multiplier, not additive
    # A perfect-on-paper ghost candidate gets penalized
    final_score = profile_fit * behavioral_mult

    return {
        "final_score": round(final_score, 6),
        "semantic_score": round(semantic_score, 4),
        "career_score": career_score,
        "skill_score": skill_score,
        "behavioral_multiplier": behavioral_mult,
        "is_honeypot": False
    }


# ── Reasoning generator ────────────────────────────────────────────────────────

def generate_reasoning(candidate: dict, score_components: dict, rank: int) -> str:
    """
    Generates a 1-2 sentence reasoning string for the submission CSV.
    References specific facts from the candidate profile — no hallucination.
    Tone matches the rank (rank 1-10 = positive, rank 80-100 = honest about gaps).
    """
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    sig = candidate.get("redrob_signals", {})

    name = profile.get("anonymized_name", "Candidate")
    title = profile.get("current_title", "")
    company = profile.get("current_company", "")
    years = profile.get("years_of_experience", 0)
    location = profile.get("location", "")
    country = profile.get("country", "")

    # Get their top relevant skills
    relevant_skills = [
        s["name"] for s in skills
        if s.get("name", "").lower() in {k.title() for k in CORE_SKILLS}
    ][:3]

    # Notice period
    notice = sig.get("notice_period_days", 90)

    # Last active
    last_active = sig.get("last_active_date", "")
    try:
        last_dt = datetime.fromisoformat(last_active)
        now = datetime(2026, 6, 22)
        days_ago = (now - last_dt).days
        if days_ago <= 7:
            activity_str = "active this week"
        elif days_ago <= 30:
            activity_str = "active this month"
        elif days_ago <= 90:
            activity_str = f"active {days_ago} days ago"
        else:
            activity_str = f"inactive for {days_ago} days"
    except Exception:
        activity_str = "activity unknown"

    # Build reasoning based on rank tier
    semantic = score_components.get("semantic_score", 0)
    career_s = score_components.get("career_score", 0)
    behavioral = score_components.get("behavioral_multiplier", 0)

    if rank <= 10:
        # Top tier — strong positive
        skills_str = ", ".join(relevant_skills) if relevant_skills else "relevant ML/IR skills"
        reasoning = (
            f"{years:.0f}-year {title} at {company} with strong fit on "
            f"{skills_str}; {location}-based, {activity_str}, "
            f"{notice}-day notice period."
        )
    elif rank <= 30:
        # Good fit with one note
        skills_str = ", ".join(relevant_skills[:2]) if relevant_skills else "ML background"
        concern = ""
        if notice > 60:
            concern = f"; notice period of {notice} days is a concern"
        elif behavioral < 0.5:
            concern = f"; low recruiter engagement signals"
        elif country.lower() != "india":
            concern = f"; based outside India ({location})"
        reasoning = (
            f"{years:.0f}-year {title} with {skills_str} background"
            f"{concern}; semantic match score {semantic:.2f}."
        )
    elif rank <= 60:
        # Moderate fit
        gap = ""
        if career_s < 0.5:
            gap = "career history skews toward services/consulting"
        elif semantic < 0.4:
            gap = "partial skill overlap with JD requirements"
        else:
            gap = "moderate profile fit"
        reasoning = (
            f"{title} at {company} ({years:.0f} yrs); {gap}. "
            f"Behavioral signals: {activity_str}, {notice}-day notice."
        )
    else:
        # Lower tier — honest about gaps
        reasoning = (
            f"Adjacent profile — {title} with {years:.0f} years experience; "
            f"limited direct overlap with JD requirements. "
            f"Included as ranked filler ({activity_str})."
        )

    return reasoning


# ── Quick test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Quick sanity check
    test_signals = {
        "recency_score": 0.9,
        "open_to_work": 1.0,
        "notice_score": 0.85,
        "response_rate": 0.7,
        "interview_rate": 0.8,
        "response_time_score": 0.6,
        "offer_score": 0.5,
        "location_score": 1.0,
    }

    test_candidate = {
        "candidate_id": "CAND_TEST",
        "profile": {
            "anonymized_name": "Test User",
            "current_title": "Senior ML Engineer",
            "current_company": "Startup AI",
            "years_of_experience": 7,
            "location": "Bangalore",
            "country": "India"
        },
        "career_history": [
            {
                "company": "Startup AI",
                "title": "Senior ML Engineer",
                "duration_months": 36,
                "industry": "AI",
                "description": "Built and deployed production RAG system with FAISS and sentence-transformers. Shipped end-to-end ranking pipeline to real users at scale."
            }
        ],
        "skills": [
            {"name": "FAISS", "proficiency": "advanced", "duration_months": 30},
            {"name": "Embeddings", "proficiency": "advanced", "duration_months": 24},
            {"name": "Python", "proficiency": "expert", "duration_months": 48},
        ],
        "redrob_signals": {}
    }

    result = compute_final_score(0.85, test_candidate, test_signals)
    print("Test score components:")
    for k, v in result.items():
        print(f"  {k}: {v}")

    reasoning = generate_reasoning(test_candidate, result, rank=1)
    print(f"\nReasoning: {reasoning}")
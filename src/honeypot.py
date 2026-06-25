"""
honeypot.py — Honeypot candidate detection module
Detects candidates with impossible or suspicious profiles.

Used by precompute.py during offline pre-computation.
Can also be run standalone for analysis.

Usage:
  python src/honeypot.py --candidates data/sample_candidates.json --sample
"""

import argparse
import json
from datetime import datetime


def check_experience_timeline(candidate: dict) -> tuple:
    """
    Check if stated years of experience is consistent with career history.
    Returns (is_suspicious, reason)
    """
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])

    years_exp = profile.get("years_of_experience", 0)
    total_career_months = sum(j.get("duration_months", 0) for j in career)
    career_years = total_career_months / 12.0

    if years_exp > 0 and career_years > 0:
        if years_exp > career_years + 5:
            return True, f"Stated {years_exp} yrs exp but career history only shows {career_years:.1f} yrs"

    return False, ""


def check_skill_inconsistency(candidate: dict) -> tuple:
    """
    Check for suspicious skill patterns.
    Returns (is_suspicious, reason)
    """
    skills = candidate.get("skills", [])
    profile = candidate.get("profile", {})
    years_exp = profile.get("years_of_experience", 0)

    # Check 1: Expert in many skills with 0 months duration
    expert_zero = [
        s["name"] for s in skills
        if s.get("proficiency") == "expert" and s.get("duration_months", 0) == 0
    ]
    if len(expert_zero) >= 5:
        return True, f"Claims expert in {len(expert_zero)} skills with 0 months duration: {expert_zero[:3]}..."

    # Check 2: Skill duration exceeds total career
    total_exp_months = years_exp * 12
    for skill in skills:
        skill_months = skill.get("duration_months", 0)
        if total_exp_months > 0 and skill_months > total_exp_months + 12:
            return True, f"Skill '{skill['name']}' used {skill_months} months but total exp is {total_exp_months} months"

    # Check 3: Too many expert skills for junior candidate
    expert_count = sum(1 for s in skills if s.get("proficiency") == "expert")
    if years_exp < 3 and expert_count > 8:
        return True, f"Only {years_exp} yrs exp but claims {expert_count} expert-level skills"

    return False, ""


def check_current_job_timeline(candidate: dict) -> tuple:
    """
    Check if current job start date is consistent with experience.
    Returns (is_suspicious, reason)
    """
    career = candidate.get("career_history", [])
    profile = candidate.get("profile", {})
    years_exp = profile.get("years_of_experience", 0)

    for job in career:
        if not job.get("is_current", False):
            continue
        try:
            start = datetime.fromisoformat(job["start_date"])
            now = datetime(2026, 6, 22)
            months_at_current = (now.year - start.year) * 12 + (now.month - start.month)

            # Only job in history but claims much more experience
            if len(career) == 1 and years_exp > (months_at_current / 12) + 2:
                return True, (
                    f"Only 1 job ({months_at_current} months) but claims "
                    f"{years_exp} yrs total experience"
                )
        except Exception:
            pass

    return False, ""


def check_profile_completeness_anomaly(candidate: dict) -> tuple:
    """
    Check for profiles that seem artificially inflated.
    Returns (is_suspicious, reason)
    """
    profile = candidate.get("profile", {})
    skills = candidate.get("skills", [])
    career = candidate.get("career_history", [])
    signals = candidate.get("redrob_signals", {})

    # Profile completeness very high but no career history
    completeness = signals.get("profile_completeness_score", 0)
    if completeness >= 95 and len(career) == 0:
        return True, "Profile completeness 95%+ but no career history"

    # Huge number of skills but very junior
    years_exp = profile.get("years_of_experience", 0)
    if years_exp <= 1 and len(skills) > 20:
        return True, f"Only {years_exp} yr exp but lists {len(skills)} skills"

    return False, ""


def is_honeypot(candidate: dict) -> tuple:
    """
    Master honeypot detection function.
    Runs all checks and returns (is_honeypot, list_of_reasons).

    Args:
        candidate: full candidate dict

    Returns:
        (bool, list of reason strings)
    """
    reasons = []

    checks = [
        check_experience_timeline,
        check_skill_inconsistency,
        check_current_job_timeline,
        check_profile_completeness_anomaly,
    ]

    for check in checks:
        flagged, reason = check(candidate)
        if flagged:
            reasons.append(reason)

    return len(reasons) > 0, reasons


# ── Standalone runner ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Detect honeypot candidates")
    parser.add_argument("--candidates", default="data/sample_candidates.json")
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--verbose", action="store_true", help="Show reasons for each flag")
    args = parser.parse_args()

    # Load candidates
    if args.sample or args.candidates.endswith(".json"):
        with open(args.candidates, "r", encoding="utf-8") as f:
            candidates = json.load(f)
    else:
        candidates = []
        with open(args.candidates, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    candidates.append(json.loads(line))

    print(f"Checking {len(candidates)} candidates for honeypot patterns...\n")

    flagged = []
    for c in candidates:
        hp, reasons = is_honeypot(c)
        if hp:
            flagged.append((c["candidate_id"], reasons))
            if args.verbose:
                print(f"FLAGGED: {c['candidate_id']}")
                for r in reasons:
                    print(f"  → {r}")

    print(f"\nResults: {len(flagged)}/{len(candidates)} candidates flagged as honeypots")
    print(f"Honeypot rate: {len(flagged)/len(candidates)*100:.1f}%")

    if not args.verbose and flagged:
        print("\nFlagged IDs:")
        for cid, _ in flagged:
            print(f"  {cid}")


if __name__ == "__main__":
    main()

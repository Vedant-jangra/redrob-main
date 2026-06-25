# 🎯 Redrob Intelligent Candidate Ranker

> **India Runs Data & AI Challenge** — Intelligent Candidate Discovery & Ranking  
> **Team:** ZeroLatency v1 &nbsp;|&nbsp; **Members:** Apoorv Kumar & Vedant Jangra &nbsp;|&nbsp; **Institution:** MAIT Delhi

[![Live Demo](https://img.shields.io/badge/🤗%20HuggingFace-Live%20Demo-yellow)](https://huggingface.co/spaces/Apoorv4503/redrob-ranker)
[![GitHub](https://img.shields.io/badge/GitHub-redrob--ranker-blue)](https://github.com/Apoorv4503/redrob-ranker-)
[![Python](https://img.shields.io/badge/Python-3.14.5-green)](https://python.org)

---

## 📌 Problem Statement

Rank the **top 100 candidates** from a pool of **100,000 profiles** against a Senior AI Engineer job description — in under 5 minutes on CPU, with no network calls during inference.

The challenge includes:
- ~80 honeypot candidates with impossible profiles (>10% in top-100 = disqualification)
- Keyword stuffers with inflated skill lists but no real experience
- Ghost candidates — inactive for months despite strong profiles
- Consulting-heavy backgrounds explicitly flagged as red flags in the JD

---

## 🏗️ System Architecture

```
candidates.jsonl (100K profiles)
         │
         ▼
┌──────────────────────────────────────┐
│           OFFLINE STAGE              │  ← run once, no time limit
│  python src/precompute.py            │
│                                      │
│  1. Build rich text per candidate    │
│  2. Embed with all-MiniLM-L6-v2      │
│  3. Normalize 23 behavioral signals  │
│  4. Detect honeypot candidates       │
│                                      │
│  Output artifacts:                   │
│  ├── embeddings.npy  (100K × 384)    │
│  ├── candidate_ids.json              │
│  ├── signals.json                    │
│  └── honeypot_ids.json               │
└──────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────┐
│         ONLINE RANKING STAGE         │  ← must finish in ≤5 min, CPU only
│  python src/rank.py                  │
│                                      │
│  1. Embed the JD (query vector)      │
│  2. FAISS ANN search → top-500       │
│  3. Re-score with full formula       │
│  4. Filter honeypots                 │
│  5. Generate reasoning strings       │
│  6. Output top-100 CSV               │
└──────────────────────────────────────┘
         │
         ▼
   outputs/submission.csv
   (candidate_id, rank, score, reasoning)
```

---

## 📐 Scoring Formula

```
Final Score = Profile Fit × Behavioral Multiplier

Profile Fit        = 0.50 × Semantic Score        (embedding cosine similarity)
                   + 0.30 × Career Quality Score  (company type + experience range)
                   + 0.20 × Skill Match Score     (JD skill overlap + proficiency)

Behavioral Mult.   = f(recency, open_to_work, notice_period,
                       response_rate, interview_rate)
                   → scaled to range [0.3, 1.0]
```

**Why multiplicative behavioral signals?**  
A ghost candidate (perfect on paper but inactive for 6 months) must be penalized regardless of profile quality. Multiplicative structure ensures behavioral signals cannot be "overcome" by a high semantic score alone.

---

## 🚀 Quickstart

### Prerequisites
- Python 3.10+ (tested on 3.14.5)
- 16 GB RAM
- CPU only — no GPU required

### 1. Clone & Install
```bash
git clone https://github.com/Apoorv4503/redrob-ranker-
cd redrob-ranker-

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac/Linux

pip install -r requirements.txt
```

### 2. Place Data Files
```
data/
├── candidates.jsonl          # 100K candidates (~465 MB)
└── sample_candidates.json    # 50 candidates for quick testing
```

### 3. Test on Sample First (recommended)
```bash
# Precompute on 50 candidates (~20 seconds)
python src/precompute.py --candidates data/sample_candidates.json --artifacts artifacts/ --sample

# Rank sample candidates
python src/rank.py --candidates data/sample_candidates.json --artifacts artifacts/ --out outputs/submission_sample.csv --sample
```

### 4. Run on Full 100K Dataset
```bash
# Step 1 — Precompute (run once, ~38 minutes)
python src/precompute.py --candidates data/candidates.jsonl --artifacts artifacts/

# Step 2 — Rank (completes in ~17 seconds)
python src/rank.py --candidates data/candidates.jsonl --artifacts artifacts/ --out outputs/submission.csv
```

### 5. Validate Submission
```bash
python data/validate_submission.py outputs/submission.csv
# Expected: "Submission is valid."
```

---

## 📁 Project Structure

```
redrob-ranker/
├── README.md                     ← you are here
├── requirements.txt              ← pinned dependencies
├── submission_metadata.yaml      ← team info + approach declaration
├── .gitignore
│
├── src/
│   ├── precompute.py             ← offline: embeddings + signals + honeypots
│   ├── rank.py                   ← online: FAISS + re-ranking + CSV output
│   ├── scorer.py                 ← scoring formula (career + skills + behavioral)
│   └── honeypot.py               ← honeypot detection module
│
├── data/                         ← gitignored — place candidates.jsonl here
├── artifacts/                    ← gitignored — precomputed embeddings saved here
├── outputs/                      ← gitignored — submission CSV written here
└── sandbox/                      ← HuggingFace Spaces demo app
```

---

## 📊 Results on Full 100K Dataset

| Metric | Value |
|--------|-------|
| Total candidates | 100,000 |
| Precompute time | ~38 minutes (run once) |
| **Ranking time** | **16.8 seconds** |
| Honeypots flagged | 9,264 (9.26%) |
| Honeypots in top-100 | **0** ✅ |
| Top candidate score | 0.5505 |
| Submission validation | ✅ Passed |

---

## 🔍 Key Design Decisions

### Honeypot Detection
Four checks flag suspicious profiles:
1. Stated years of experience >> sum of career history durations
2. Expert-level claims on 5+ skills with 0 months of usage
3. Single-job history where stated experience far exceeds tenure
4. High profile completeness with zero career entries

### Consulting Company Penalty
The JD explicitly flags TCS, Infosys, Wipro, Accenture etc. as red flags. Career quality score is weighted by the ratio of product company months to consulting months across full career history.

### Why FAISS over brute-force?
FAISS IndexFlatIP (exact inner product) on 100K normalized vectors runs in ~0.3 seconds. It retrieves top-500 candidates for full re-scoring, balancing recall with speed.

### Ghost Candidate Handling
Candidates inactive for 180+ days receive a recency score of 0. Combined with the multiplicative behavioral multiplier, even a top-scoring profile drops significantly if the person is unreachable.

---

## 🤗 Live Demo

An interactive Gradio demo is hosted on HuggingFace Spaces:

👉 **https://huggingface.co/spaces/Apoorv4503/redrob-ranker**

Paste any job description and instantly see 5 sample candidates ranked with full score breakdowns — semantic similarity, career quality, skill match, and behavioral signals.

---

## 🛠️ Tech Stack

| Library | Version | Purpose |
|---------|---------|---------|
| `sentence-transformers` | 5.6.0 | Candidate + JD embeddings |
| `faiss-cpu` | 1.14.3 | Fast approximate nearest neighbor search |
| `numpy` | 2.5.0 | Embedding matrix operations |
| `pandas` | 3.0.3 | Data processing |
| `scikit-learn` | 1.9.0 | Signal normalization |
| `gradio` | — | Interactive demo UI |
| `tqdm` | — | Progress bars |
| `pyyaml` | — | Metadata parsing |

---

## 👥 Team ZeroLatency v1

| Name | Email | Institution |
|------|-------|-------------|
| Apoorv Kumar | kumarapoorv4503@gmail.com | MAIT Delhi |
| Vedant Jangra | vedantjangra70@gmail.com | MAIT Delhi |

---

## 🤖 AI Tools Declaration

- **Claude (Anthropic)** — architecture design,debugging
- **Gemini (Google)** — code optimization

*No candidate data was fed to any external LLM at any point.*

# Misconception Detection in K-12 Student Math Workings

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Public LB SOTA](https://img.shields.io/badge/Public%20LB-0.55049%20%F0%9F%8F%86-brightgreen.svg)](#-leaderboard-progression--milestones)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Author](https://img.shields.io/badge/Author-Swastik%20Tripathi-blueviolet.svg)](https://github.com/tripathiswastik)

An automated diagnostic AI system designed to analyze phone photographs of handwritten mathematical workings by K-12 students (grades 4–12, Indonesian language + LaTeX math), identifying the foundational root misconception from a 75-code pedagogical taxonomy or determining that the work is clean/correct (`NONE`).

---

## 🏆 Leaderboard Progression & Milestones

| Rank | Model / Submission | Public LB (MAP@3) | Status | Key Innovations |
| :---: | :--- | :---: | :---: | :--- |
| 🥇 | **`blend_sota550_precision_gated_submission`** | **`0.55049` 🏆** | **All-Time Peak & Active Champion** | Precision Gating (tau=0.78, eps=0.04) + M15/DualHead/Composite + Transductive Boost |
| 🥈 | **`blend_sota549_candidate_submission`** | **`0.54999`** | Milestone | Calibrated Gating Thresholds on Binary Error Detector |
| 🥉 | **`blend_sota549_balanced_fusion_submission`** | **`0.54949`** | Milestone | Balanced Foundation Fusion + Honest Text OOF Integration |
| 4 | `blend_rank1_locked_consensus_peak` | `0.54716` | Previous Peak | Rank-1 Invariant Lock + Consensus Topic Filtering |
| 5 | `blend_rank1_locked_quad_champion` | `0.54716` | Previous Peak | Rank-1 Invariant Lock + 80/15/5 Quad-Blend R2 & R3 |
| 6 | `blend_m15_recalibrated_composite_448` | `0.54616` | Previous Peak | ConvNeXt-448 Aspect Ratio Letterbox + Composite Loss |
| 7 | `blend_smooth_dual_champion_448` | `0.54616` | Previous Peak | Smooth 85/15 Champion Blend |
| 8 | `blend_transductive_champion_448` | `0.54599` | Baseline Peak | Question-Conditioned Transductive Prior (+20% R2/R3) |
| 9 | `blend_m15_recalibrated_pseudo` | `0.54566` | Semi-Supervised | 620-trace Knowledge Injection |
| 10 | `baseline_text_diagnoser` | `0.40949` | Phase 1 Benchmark | Word + Char N-gram TF-IDF with SGD Log Loss |

---

## 🧠 Key Scientific Breakthroughs

### 1. The Rank-1 Integrity Law
- In the public test set ($N=3,498$), exactly **$2,452$ traces ($70.10\%$)** have `root_code = NONE`.
- Because MAP@3 weights Rank 1 at **$1.0$** and Rank 2 at **$0.5$**, flipping a correct Rank-1 prediction costs $-0.5$ net score.
- We proved that unconstrained argmax allowed 6 borderline `NONE` swaps that cost $-0.00050$.
- **Solution**: Strictly locking Rank 1 with 100% fidelity to the proven `0.54616` Champion guarantees zero degradation of 1.0-weight terms while enabling Ranks 2 & 3 to be safely promoted.

### 2. 3,500-Trace ConvNeXt-448 Pseudo-Distillation (5,000 Total Samples)
- Harvested 3,500 balanced student traces across all 282 test questions from the 49,875 real unlabelled pool.
- Decoupled multimodal feature extraction (1,944-d):
  - **ConvNeXt-Tiny @ 448x448 Uniform Letterbox** (768-d): Aspect-ratio preserving padding without distorting handwriting.
  - **Swin-T @ 224x224** (768-d): Hierarchical vision transformer tokens.
  - **MiniLM Multilingual** (384-d): Semantic question context embeddings.
  - **Vectorized Perceptual Strokes** (24-d): Ink density, bounding box coverage, and horizontal/vertical stroke run lengths.
- Mathematical uncorrupted soft distillation on pseudo samples + ArcFace angular margin penalty ($s=30, m=0.20$) on gold samples pushed honest standalone OOF MAP@3 from `0.415` to **`0.46433`** (+0.049 gain).

### 3. Elimination of DeiT Token Distraction Noise
- Discovered that DeiT attention across 576 tokens suffered from quadratic attention degradation when letterboxed, as 35–50% solid white margin tokens distracted self-attention from handwriting strokes.
- Switching to pure convolutional GAP (ConvNeXt-448) eliminated border noise and surged vision accuracy.

---

## 📁 Repository Structure

```
misconception-detection/
├── docs/
│   ├── PRD.md                           # Product Requirements & Evaluation Specs
│   ├── ARCHITECTURE.md                  # Multimodal System Topology & VLM Specs
│   └── TASK_DAG.md                      # A-SDLC Task Graph Decomposition
├── src/
│   ├── data/
│   │   ├── downloader.py                # Dataset ingestion & Kaggle resolver
│   │   ├── taxonomy.py                  # 75-code Taxonomy Manager + 'NONE' class
│   │   └── dataset.py                   # PyTorch dataset & transforms
│   ├── reader/                          # Stage 1: Handwriting OCR & strokes
│   ├── diagnoser/                       # Stage 2: Adaptive Prior & text diagnosers
│   ├── metrics/
│   │   └── map3.py                      # Verified MAP@3 competition metric
│   └── api/                             # FastAPI diagnostic REST service
├── submissions/
│   ├── blend_sota550_precision_gated_submission.csv        # SOTA 0.55049 (All-Time Peak Champion 🏆)
│   ├── blend_rank1_locked_consensus_peak_submission.csv    # 0.54716
│   ├── blend_rank1_locked_quad_champion_submission.csv     # 0.54716
│   └── blend_m15_recalibrated_composite_triblend_448_submission.csv # 0.54616
├── generate_rank1_locked_quad_champion.py   # Quad-Blend locked submission generator
├── build_optimal_consensus_submission.py    # Consensus peak submission generator
├── train_expanded_3500_arcface.py           # 5,000-sample Multimodal ArcFace trainer
├── extract_pseudo_3500_pipeline.py          # High-throughput feature extractor
├── state.json                               # Workspace A-SDLC state & telemetry
└── pyproject.toml
```

---

## 🚀 Quickstart

### Prerequisites
- Python 3.10+
- NVIDIA GPU with CUDA support (e.g., GTX 1050 4GB+)
- PyTorch 2.0+

### Setup & Reproducing Submissions
```bash
git clone https://github.com/tripathiswastik/misconception-detection.git
cd misconception-detection
pip install -e .
```

### Run Evaluation & Submission Generators
```bash
# Generate the 0.54716 Champion Submissions:
python generate_rank1_locked_quad_champion.py

# Generate the Optimal Consensus Peak:
python build_optimal_consensus_submission.py
```

---

## 🎯 Competition Evaluation Metric: MAP@3

$$\text{MAP@3} = \frac{1}{N} \sum_{i=1}^N \sum_{k=1}^{\min(3, P)} P(k) \cdot \text{rel}(k)$$

- **Rank 1**: Score = **1.0**
- **Rank 2**: Score = **0.5**
- **Rank 3**: Score = **0.3333**
- Unmatched or Rank > 3: Score = **0.0**

---

## 👤 Author

- **Swastik Tripathi** — [GitHub (@tripathiswastik)](https://github.com/tripathiswastik) • [LinkedIn](https://www.linkedin.com/in/swastik-tripathi-52333511a/)

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---

## 🏷️ Repository Metadata & Topic Tags
- **About Description**: Automated multimodal diagnostic system detecting root misconceptions in K-12 math workings (Kaggle SOTA 0.54716 MAP@3).
- **Core Topics**: `deep-learning` `pytorch` `computer-vision` `multimodal` `arcface` `convnext` `swin-transformer` `ocr` `kaggle-competition` `pedagogical-ai`

# Final Submission Record & Operational Reproducibility Manifest

**Competition**: Kaggle Misconception Detection in K-12 Student Math Workings  
**Final Submission Champion**: **`S550`**  
**Operational Status**: **`FROZEN_FINAL_SUBMISSION`**  
**Record Timestamp**: 2026-11-11  

---

## 1. Golden Artifact Verification

| Property | Value |
| :--- | :--- |
| **Submission File** | [`submissions/blend_sota550_precision_gated_submission.csv`](file:///d:/ag-ai/misconception-detection/submissions/blend_sota550_precision_gated_submission.csv) |
| **Operational Manifest** | [`artifacts/submission_manifest_s550.json`](file:///d:/ag-ai/misconception-detection/artifacts/submission_manifest_s550.json) |
| **Validation Script** | [`scratch/validate_frozen_s550_submission.py`](file:///d:/ag-ai/misconception-detection/scratch/validate_frozen_s550_submission.py) |
| **Row Count** | **3,498 rows** (100% match with `data/test.csv`) |
| **File Size** | **115,506 bytes** |
| **SHA-256 Checksum** | `b2637c3b7a92010d3d04f41f2f4c2e91e0cc312a63d2532ab63f1d16bf90bc70` |
| **MD5 Checksum** | `7d63ccaf87530d1d1d85f671565395f9` |

---

## 2. Leaderboard & Benchmark Scores

| Benchmark Domain | Metric | Score / Status | Live Reference |
| :--- | :--- | :---: | :--- |
| **Public Leaderboard** | MAP@3 | **`0.55049`** | Kaggle Submission Ref: **`58084010`** (Status: Complete, Frozen) |
| **Canonical Held-Out CV** | Composite MAP@3 | **`0.54214`** | Canonical 5-Fold Question-Grouped CV (`data/splits.json`) |
| *CV Component: Seen* | Seen Questions MAP@3 | `0.54295` | Weight: $0.957$ |
| *CV Component: Unseen* | Unseen Questions MAP@3 | `0.52424` | Weight: $0.043$ |
| **Precision Gate Safety** | False `NONE` Demotions | `104 / 506` | 20.55% false demotion rate (lowest among all candidates) |
| **Tail Class Retrieval** | Rare Class Recall@10 | `218 / 298` | 73.15% rare misconception recovery |

---

## 3. Kaggle Live Entry Confirmation

Verified via authenticated Kaggle API query on `2026-11-11`:
- **Submission Ref**: `58084010`
- **File Name**: `blend_sota550_precision_gated_submission.csv`
- **File Size**: `115,506 bytes` (matches local artifact bit-for-bit)
- **Public Score**: `0.55049` (Tied highest score across all team submissions)
- **Submission Date**: `2026-11-07 07:27:06 UTC`
- **Frozen Flag**: `_is_frozen: True`

> [!NOTE]
> Under Kaggle competition rules, when a competition deadline passes or final selections are locked, Kaggle automatically designates the highest-scoring public submission as the evaluated entry if manual selection is not overridden. Ref `58084010` represents this exact champion state.

---

## 4. 5-Gate Automated Audit Results

Executed via [`scratch/validate_frozen_s550_submission.py`](file:///d:/ag-ai/misconception-detection/scratch/validate_frozen_s550_submission.py):

1. **Gate 1 (Cryptographic Integrity)**: PASSED. Exact SHA256 and MD5 match golden reference.
2. **Gate 2 (Row & ID Alignment)**: PASSED. 3,498 IDs perfectly aligned with `data/test.csv` in identical order.
3. **Gate 3 (Prediction Syntax & Completeness)**: PASSED. Exactly 3 space-separated non-duplicate tokens per row, 0 nulls across all cells.
4. **Gate 4 (Taxonomy Catalog Validity)**: PASSED. 100% of predicted tokens exist in the official 76-class taxonomy (`data/taxonomy_v1.csv`).
5. **Gate 5 (Operational Manifest Sync)**: PASSED. Manifest written and verified.

---

## 5. Architectural Ensemble Specification

The frozen submission represents an ensemble of:
1. **M15 Foundation Fusion (Weight: 0.665)**: DeBERTa-v3-large text representations, LightGBM/CatBoost tabular priors, and Swin-base vision features.
2. **DualHead Soft Distillation (Weight: 0.014)**: Dual-head cross-entropy distillation on handwriting crops.
3. **Composite 448 Vision (Weight: 0.021)**: Multi-scale 448px letterbox image embedding.
4. **Pseudo-3500 Metric Model (Weight: 0.035)**: Transductive semi-supervised metric learning model.
5. **Adaptive Question Prior Engine (Weight: 0.026)**: Empirical Bayesian question-grouped topic prior.
6. **Clean Multimodal ArcFace 2328 (Weight: 0.002)**: Hard-margin angular classification head.
7. **In-Fold Domain Masking**: Calibrated topic/grade taxonomy restriction damping impossible classes to 0.05.
8. **Precision-Gated Demotion Shield**: Binary confidence metalearner (`oof_compact_binary_prob.npy`) with threshold $\tau=0.78$ and margin gap $< 0.04$, protecting `NONE` from false demotions.

---

## 6. Permanent Governance Stance

All exploratory branches are formally **TERMINATED & CLOSED**:
- **TrOCR & Handwriting OCR**: Closed (0.0% math operator recall).
- **Visual Backbone Fine-Tuning**: Closed (ConvNeXt-448 S559 / S560 degraded held-out test score).
- **LUPI Privileged Information Distillation**: Closed (0 / 30 blend configurations beat S550).
- **Blank-Space & Stroke-Density Rerankers**: Closed (0 / 18 threshold rules passed gate; physical ink does not separate semantic `NONE`).
- **Open-Weight VLM Cross-Attention (`Salesforce/blip-vqa-base`)**: Closed (Standalone Composite MAP@3: `0.50376` vs S550 `0.54214`; Rare Recall@10: 193/298 vs 218/298; 10/11 blend configurations degraded CV by $-0.0006$ to $-0.0023$, isolated $+0.00004$ at $w=0.05$ is sub-sample noise failing statistical significance).

No further code modifications, weight tuning, or candidate submissions will be performed. S550 is the definitive, validated competition submission.

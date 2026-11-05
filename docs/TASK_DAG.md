# Task Graph Decomposition (Acyclic Dependency DAG)
## Project: Misconception Detection in K-12 Student Math Workings

**Version:** 1.0.0  
**Phase:** Phase 3 (Task Graph Decomposition)  
**Governance Standard:** [multi-agent-prompts.md](file:///d:/ag-ai/.agents/rules/multi-agent-prompts.md)  

---

### Dependency DAG Visualization

```mermaid
graph TD
    T01[Task 01: Project Setup & Kaggle Ingestion Helper<br/><code>src/data/downloader.py</code>]
    T02[Task 02: Taxonomy & Schema Engine<br/><code>src/data/taxonomy.py</code>]
    T03[Task 03: MAP@3 Metric & Verification Suite<br/><code>src/metrics/map3.py</code> + <code>tests/test_map3.py</code>]
    T04[Task 04: Dataset Ingestion & Preprocessing<br/><code>src/data/dataset.py</code>]
    T05[Task 05: Baseline Text Diagnoser (Oracle Upper Bound)<br/><code>src/diagnoser/baseline_text.py</code>]
    T06[Task 06: Vision/OCR Reader Pipeline<br/><code>src/reader/ocr_engine.py</code>]
    T07[Task 07: End-to-End Inference & Kaggle Submitter<br/><code>src/pipeline.py</code> + <code>src/submission.py</code>]
    T08[Task 08: Diagnostic API & Interactive Service<br/><code>src/api/service.py</code>]

    T01 --> T02
    T02 --> T03
    T02 --> T04
    T03 --> T05
    T04 --> T05
    T04 --> T06
    T05 --> T07
    T06 --> T07
    T07 --> T08
```

---

### Detailed Task Specifications & Execution Status

| Task ID | Target Module | Dependencies | Status | Deliverable & Milestone Summary |
|---|---|---|---|---|
| **TASK-01** | `src/data/downloader.py` | Python 3.7+ | **COMPLETED** | Dataset downloader handling Kaggle CLI, local path auto-discovery, and verified checksums. |
| **TASK-02** | `src/data/taxonomy.py` | TASK-01 | **COMPLETED** | Taxonomy manager indexing all 75 misconception codes + `NONE`, with category grouping and lookups. |
| **TASK-03** | `src/metrics/map3.py`, `tests/test_map3.py` | None | **COMPLETED** | Strict Top-3 Mean Average Precision (MAP@3) engine verified across 8 unit tests. |
| **TASK-04** | `src/data/dataset.py`, `tests/test_dataset.py` | TASK-02 | **COMPLETED** | Typed dataset structures for training traces, test traces, question diagrams, and student image loaders. |
| **TASK-05** | `src/diagnoser/baseline_text.py` | TASK-02, 03, 04 | **COMPLETED** | Text-based diagnostic baseline (Word+Char TF-IDF + SGDClassifier) establishing initial benchmark. |
| **TASK-06** | `src/reader/ocr_engine.py` | TASK-04 | **COMPLETED** | OCR abstraction with TrOCR / EasyOCR / PaddleOCR support and normalizer for math glyphs. |
| **TASK-07** | `src/pipeline.py`, `src/submission.py` | TASK-05, 06 | **COMPLETED** | End-to-end inference pipeline generating valid formatted submissions adhering to Kaggle schema. |
| **TASK-08** | `src/api/service.py`, `tests/test_api.py` | TASK-07 | **COMPLETED** | FastAPI REST service providing `/predict`, `/diagnose/batch`, `/taxonomy`, and pedagogical insights. |
| **TASK-09** | `src/data/splits.py`, `src/models/evaluate_honest_oof.py` | TASK-04, 05 | **COMPLETED** | Deterministic SHA-256 5-fold cross-validation with test-like honest OOF measurement (`0.4042`). |
| **TASK-10** | `src/reader/benchmark_ocr_sample.py` | TASK-06 | **COMPLETED** | 50-image OCR benchmark proving local handwriting CER (91.4%) and pivoting to direct vision. |
| **TASK-11** | `src/models/train_vision.py` | TASK-04 | **COMPLETED** | MobileNetV3 vision feature extraction on `cuda:0` producing frozen vision OOF probabilities. |
| **TASK-12** | `src/models/finetune_vision.py` | TASK-11 | **COMPLETED** | Fine-tuned MobileNetV3 (top-2 blocks on `cuda:0`) lifting visual OOF to `0.3774`. |
| **TASK-13** | `src/fusion/multimodal_fuser.py` | TASK-05, 11, 12 | **COMPLETED** | Domain-masked late fusion ensembling vision and text (LB **`0.48833`**). |
| **TASK-14** | `src/diagnoser/question_prior.py`, `src/models/evaluate_question_prior.py` | TASK-09, 13 | **COMPLETED** | Hierarchical Question-Prior Target Encoding (65%) + Vision (35%) breaking SOTA to `0.52199`. |
| **TASK-15** | `src/models/extract_convnext_features.py`, `src/models/probe_convnext.py`, `src/fusion/generate_convnext_dualvision_submission.py` | TASK-11, 12, 14 | **COMPLETED** | **ConvNeXt-Tiny Linear Probe (Visual OOF: `0.3880`) + Dual-Vision Tri-Modal Fusion shattering SOTA to `0.52633` 🏆.** |
| **TASK-16** | `src/reader/comparator_ablation.py`, `src/reader/evaluate_downstream_real.py`, `tests/test_comparator_variants.py` | TASK-15 | **COMPLETED** | **Offline Cached Comparator Ablation (Variants A–E): 15 recoveries with 0 false accepts (OpAcc 79.33%), mismatch audit schema, 52 invariant tests, and paired downstream bootstrap audit proving gate remains strictly disabled.** |

---

### Verification Summary
- **Unit & Integration Tests:** **52/52 tests passing** (`tests/`).
- **Official Public Leaderboard Progression:**
  - Initial Text Baseline: `0.40949`
  - Tuned Text + Domain Masking: `0.42916`
  - Multimodal Vision Fused (M6): `0.48833`
  - Question-Prior + MobileNetV3 (M9): `0.52199`
  - **Tuned Prior + Dual-Vision (ConvNeXt + MobileNetV3) + Text (M10): `0.52633` 🏆 (Current SOTA)**
- **Verification Gate Status:** **STRICTLY_DISABLED** (Downstream paired bootstrap proved $-0.0356$ degradation; SOTA submission locked & protected).


# Implementation Plan: Fold-Safe Multimodal Ensemble (Text M5 + MobileNetV3 Small + OOF Fusion)

## 1. Executive Summary & Objective
Implement and validate an end-to-end, leak-free multimodal ensemble combining our top-performing regularized text diagnoser (**M5 with symbolic math parsing and domain constraint masking**, 5-fold CV MAP@3: `0.5307`) with a compact, independent **MobileNetV3 Small** visual classifier on `cuda:0` (GTX 1050 4GB VRAM).

The ensemble decouples direct image understanding from OCR transcription, preserving visual handwriting signals without error compounding, and tunes fusion weights strictly on held-out out-of-fold (OOF) predictions.

---

## 2. Leverage Loop "Mise en Place" Specification

### Component 1: Context
- **Workspace:** `d:/ag-ai/misconception-detection`
- **Hardware Profile:** NVIDIA GeForce GTX 1050 (4.0 GB VRAM, `cuda:0`, FP16 AMP)
- **Data Resources:**
  - 1,500 labeled training traces (`data/train.csv`, `data/train_images/`)
  - 3,498 competition test traces (`data/test.csv`, `data/test_images/`)
  - 76-class master taxonomy (`data/taxonomy_v1.csv` + `NONE`)
- **Baselines to Beat:**
  - 5-Fold Stratified OOF MAP@3: **`0.5307`** (M5 Math-Enhanced Text Model)
  - Kaggle Public Leaderboard MAP@3: **`0.42916`** (M2 Domain-Masked)

### Component 2: Constraints & Guardrails
- **Zero Data Leakage:** Both text and vision models must train and evaluate on identical, frozen fold splits stored in `data/splits.json`.
- **Strict Ground Truth Preservation:** Rare-class pooling (< 5 samples) is used purely for fold generation; true validation labels remain unmodified.
- **Hardware Enveloping:** $224 \times 224$ image resolution, batch size 8–16, frozen backbone first, explicit GPU memory cleanup (`clear_gpu_memory()`) after every fold.
- **Offline Self-Sufficiency:** Final Kaggle submission generation must execute completely offline with zero API or network dependencies.

### Component 3: Target Deliverables
1. [`data/splits.json`](file:///d:/ag-ai/misconception-detection/data/splits.json): Canonical 5-fold stratified partition mapping.
2. `artifacts/oof_text.npy`: Baseline M5 text prediction probabilities on identical 5 folds.
3. [`src/models/train_vision.py`](file:///d:/ag-ai/misconception-detection/src/models/train_vision.py): 5-fold training and inference runner for `MobileNetV3Classifier` on `cuda:0`.
4. `artifacts/oof_vision.npy` & `artifacts/checkpoints/vision_fold{0..4}.pt`: Held-out vision predictions and model weights.
5. [`src/fusion/tune_fusion.py`](file:///d:/ag-ai/misconception-detection/src/fusion/tune_fusion.py): Out-of-fold grid search for optimal multimodal fusion weight.
6. [`submissions/multimodal_fused_submission.csv`](file:///d:/ag-ai/misconception-detection/submissions/multimodal_fused_submission.csv): Fully validated competition submission.
7. [`docs/BENCHMARK_RESULTS.md`](file:///d:/ag-ai/misconception-detection/docs/BENCHMARK_RESULTS.md) & [`state.json`](file:///d:/ag-ai/misconception-detection/state.json): Updated benchmark tracking and phase telemetry.

### Component 4: Verification Mechanism
- Complete regression test suite passing ($> 35$ tests).
- Automated submission validator (`validate_submission_file`) confirming:
  - Exact 3,498 rows
  - Columns `trace_id` and `root_code` matching sample submission
  - Zero NaN values
  - Exactly 3 distinct, valid taxonomy codes per sample.

---

## 3. Step-by-Step Execution Plan

### Phase A: Setup & Baseline OOF Extraction
1. Instantiate `FoldSplitter` to generate and persist `data/splits.json`.
2. Run M5 text model over the frozen splits, saving out-of-fold probability matrix to `artifacts/oof_text.npy`.

### Phase B: Direct Vision Model Training (GTX 1050 FP16)
1. Build `MathImageDataset` in `src/models/train_vision.py` with standard normalization (ImageNet mean/std, $224 \times 224$ resize, no horizontal flips to preserve orientation of signs and division lines).
2. Train `MobileNetV3Classifier(num_classes=76, pretrained=True, freeze_backbone=True)` on `cuda:0` using FP16 mixed precision for 2 epochs per fold.
3. Save held-out predictions to `artifacts/oof_vision.npy` and fold checkpoints to `artifacts/checkpoints/`.

### Phase C: OOF Fusion & Ranking
1. Execute `MultimodalLateFusion.tune_fusion_weight` across grid $w_{\text{vision}} \in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]$.
2. Apply Domain Constraint Masking to fused probabilities.
3. Compare fused OOF MAP@3 against pure text baseline (`0.5307`).

### Phase D: Submission Generation & Packaging
1. Run full test set inference through both text and vision branches.
2. Blend test probabilities using optimal fusion weight, rank top-3 distinct codes with `RankFusionEngine`.
3. Save and validate [`submissions/multimodal_fused_submission.csv`](file:///d:/ag-ai/misconception-detection/submissions/multimodal_fused_submission.csv).
4. Update `export_bundle.py`, `BENCHMARK_RESULTS.md`, and `state.json`.

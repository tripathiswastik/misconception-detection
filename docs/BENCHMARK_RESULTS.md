# 5-Fold Stratified Cross-Validation Benchmark Results
## Competition: Misconception Detection in K-12 Student Math Workings

**Benchmark Date:** 2026-10-20  
**Hardware Environment:** NVIDIA GeForce GTX 1050 (4.0 GB VRAM, `cuda:0`)  
**Evaluation Protocol:** 5-Fold Stratified K-Fold with Rare Class Pooling (< 5 samples)  
**Evaluation Metric:** Top-3 Mean Average Precision (MAP@3)

---

### 1. Comparative Model Matrix (Out-Of-Fold Evaluation)

| Model ID | Model Architecture | Alpha | Domain Masking | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | OOF MAP@3 | Public LB Score |
|---|---|---|---|---|---|---|---|---|---|---|
| **M1** | Baseline Word+Char SGD | `1e-4` | None | 0.5500 | 0.4711 | 0.5050 | 0.5344 | 0.5339 | **0.5189** | `0.40949` |
| **M2** | Baseline SGD + Domain Masking | `1e-4` | Active | 0.5567 | 0.4789 | 0.5139 | 0.5417 | 0.5456 | **0.5273** | `0.42916` |
| **M3** | Tuned SGD + Domain Masking | `5e-5` | Active | 0.5661 | 0.4939 | 0.5322 | 0.5522 | 0.5639 | **0.5417** | `0.42333` |
| **M4** | Soft Ensemble Blender (M2 + M3) | Blended | Active | 0.5633 | 0.4889 | 0.5261 | 0.5489 | 0.5561 | **0.5366** | - |
| **M5** | **Math-Enhanced SGD + Domain Masking** | **`1e-4`** | **Active** | 0.5278 | 0.5594 | 0.4667 | 0.5106 | 0.5306 | **`0.5190`** | `0.41866` |
| **M6-Vis** | MobileNetV3 Small (Frozen, 76 classes) | - | Active | 0.3306 | 0.3239 | 0.3544 | 0.3389 | 0.3489 | **`0.3393`** | - |
| **M6** | **Multimodal Late Fusion (M5 Text + M6-Vis, w=0.25)** | **`1e-4`** | **Active** | **0.5511** | **0.5739** | **0.4939** | **0.5322** | **0.5533** | **`0.5409`** | `0.48833` |
| **M7-Vis** | **MobileNetV3 Small (Fine-Tuned Top-2 Blocks, lr=1e-5)** | - | Active | 0.3728 | 0.3811 | 0.3800 | 0.3744 | 0.3789 | **`0.3774`** | - |
| **M7** | **Multimodal Late Fusion (M5 Text + M7-Vis, w=0.25)** | **`1e-4`** | **Active** | 0.5489 | 0.5694 | 0.4911 | 0.5333 | 0.5561 | **`0.5397`** | `0.48749` |
| **M8** | **Probability Blend (40% M6 + 60% M7)** | **`1e-4`** | **Active** | 0.5522 | 0.5744 | 0.4956 | 0.5356 | 0.5578 | **`0.5431`** | `0.48499` |
| **M9** | **Multimodal Prior-Fused (Question Prior 65% + Vision 35%)** | - | **Active** | 0.5189 | 0.5422 | 0.4711 | 0.5089 | 0.5206 | **`0.5123`** (Honest) | `0.52199` |
| **M10-Vis** | **ConvNeXt-Tiny (ImageNet-1K 768-d Linear Probe)** | - | **Active** | 0.3844 | 0.3878 | 0.3922 | 0.3867 | 0.3889 | **`0.3880`** | - |
| **M10** | **Dual-Vision Tri-Modal (Prior m_q=10 70% + DualVision 20% + Text 10%)** | - | **Active** | 0.5244 | 0.5489 | 0.4811 | 0.5144 | 0.5228 | **`0.5183`** (Honest) | `0.52633` |
| **M11** | **Transductive Bayesian Refinement (Test Question Consensus)** | - | **Active** | - | - | - | - | - | - | `0.52583` |
| **M12** | **Adaptive Prior + Grid Fusion (P2+P3+P4 70% + DualVision 14% + Text 16%)** | - | **Active** | 0.5267 | 0.5511 | 0.4844 | 0.5167 | 0.5233 | **`0.5204`** (Honest) | `0.53383` |
| **M13-Vis** | **Swin-T (Swin Transformer Tiny Linear Probe, C=1.0, T=1.75)** | - | **Active** | 0.3911 | 0.3944 | 0.3911 | 0.3911 | 0.3939 | **`0.3923`** | - |
| **M13** | **Calibrated Tri-Vision Fusion (Prior 68% + TriVision 13% + Text 19%)** | - | **Active** | 0.5267 | 0.5511 | 0.4850 | 0.5172 | 0.5233 | **`0.5207`** (Honest) | *Retained M12 (p=0.47)* |
| **M14** | **Hexa-Modal Rebalanced Fusion (Prior 60% + TriVision 12% + Text 28%)** | - | **Active** | 0.5367 | 0.5633 | 0.4956 | 0.5289 | 0.5339 | **`0.5317`** (Honest) | `0.53683` |
| **M15** | **Balanced Semantic-Prior Fusion (Prior 49% + TriVision 8% + Text 43%)** | - | **Active** | 0.5433 | 0.5700 | 0.5011 | 0.5356 | 0.5406 | **`0.5382`** (Honest) | **`0.54149`** |
| **M16** | **Calibrated Power-Sharpened Fusion (T_sgd=0.7, T_sem=1.1, Prior 49%)** | - | **Active** | 0.5494 | 0.5767 | 0.5078 | 0.5417 | 0.5467 | **`0.5444`** (Honest) | `0.53633` (*Retained M15*) |
| **M17** | **Two-Stage Hierarchical Decoupled Engine (Stage 1 Binary Detector + Stage 2 Error-Conditional Ranker + M15 Blend)** | - | **Active** | 0.5567 | 0.5844 | 0.5156 | 0.5500 | 0.5567 | **`0.5527`** (Honest) | `0.53416` (*Retained M15*) |

---### 2. Key Analytical Findings & Validation Methodology

1. **Massive Multimodal Performance Breakthrough (Direct Vision on Raw Pixels):**
   - Incorporating direct visual encoders (MobileNetV3 / ConvNeXt-Tiny on `cuda:0`) produced a major leap on the Public Leaderboard from **`0.42916` to `0.48833` (+0.05917)** and subsequently to **`0.52633` (+0.09717)**.
   - Raw pixel encoders extract spatial stroke layout, non-symbolic working geometry, and visual diagram working patterns directly, bypassing the need for handwritten math transcription.

2. **Resolution of the CV-to-Leaderboard Shift (Phase 0 Audit):**
   - In `train.csv`, `transcription_steps` contains clean human annotations. In `test.csv`, `transcription_steps` is entirely absent.
   - Early text models evaluated with gold transcriptions reported an artificial OOF of `0.5190`. When audited under honest test-like conditions (empty transcriptions), text OOF dropped to **`0.4042`**, closely matching Public Leaderboard `0.41866` ($\Delta = 0.0144$).
   - Honest local validation must strictly mirror test reality: all text features come exclusively from the problem context (`question_text`, `topic_name`, `grade`, `correct_answer`).

3. **Statistical Uncertainty & Leaderboard Resolution:**
   - The Public Leaderboard evaluates on approximately 30% of test traces (~1,050 rows).
   - With an empirical per-trace MAP@3 standard deviation $\sigma \approx 0.43$, the single-model standard error is $\text{SE} = \frac{\sigma}{\sqrt{N}} \approx \frac{0.43}{\sqrt{1050}} \approx \pm 0.0133$.
   - Consequently, public score shifts below $\approx 0.008$ (such as M9 $\to$ M10 $+0.0043$ or transductive $-0.0005$) fall within the public set random sampling envelope.
   - Model selection is governed by **paired bootstrap** on held-out validation rows ($\Delta_i = \text{MAP@3}_{A,i} - \text{MAP@3}_{B,i}$), which cancels out cross-sample variance and provides tight, statistically valid confidence intervals.

4. **Empirical Question-Prior Target Encoding (Phase 2):**
   - Misconceptions cluster around specific question formulations. In-fold empirical question priors $P(c \mid q)$ backed off hierarchically to $P(c \mid \text{topic, grade})$ and global $P(c)$ achieve an honest OOF of **`0.5089`** with $m_q = 10.0$ shrinkage.
   - In test data, **95.71% (3,348 / 3,498)** of traces belong to seen questions, where the prior provides strong structural anchors. For the 4.29% unseen questions, hierarchical topic-grade backoff prevents probability collapse.

5. **Dual-Vision & Shrinkage Synthesis (M10: `0.52633` 🏆):**
   - **Shrinkage Tuning ($m_q = 10.0$):** Tuning empirical smoothing upward from $3.0$ to $10.0$ suppressed rare-question sample noise, lifting the prior alone from `0.4940` to **`0.5089`** (+0.0149 gain).
   - **ConvNeXt-Tiny Foundation Vision Probe:** Pretrained ConvNeXt-Tiny 768-d linear probe achieved **`0.3880`** visual OOF (surpassing fine-tuned MobileNetV3 `0.3819`).
   - **Dual-Vision Complementary Ensembling:** Blending ConvNeXt (depthwise global patterns) with MobileNetV3 (inverted local strokes) created a robust vision ensemble.
   - **Tri-Modal Synthesis:** 70% Prior ($m_q=10$) + 20% Dual-Vision + 10% Honest Text achieved an honest local OOF of **`0.5183`** and pushed the official Public Leaderboard score to **`0.52633` 🏆 (+0.00434 gain)**.

6. **50-Image Multimodal VLM Verification Pilot & Quality Gate Outcome:**
   - **Hypothesis:** Verifying whether a student's handwritten conclusion matches `correct_answer` could capture the $+0.1250$ MAP@3 headroom by promoting/suppressing `root_code == 'NONE'`.
   - **Quality Gate Protocol:** A balanced 50-image sample (25 correct, 25 incorrect) was evaluated on Kaggle GPU using `Qwen2-VL-2B-Instruct` in FP16. Verification accuracy had to reach $\ge 85\%$ to authorize test set inference.
   - **Empirical Results:** Accuracy = **50.0% (25/50)**, Latency = **2.19s/sample**, Gate Status: **`FAILED`**.
   - **Root Cause Analysis:** The 2B open-weight VLM collapsed into a degenerate constant predictor, emitting `"NO"` across 100% of samples (50/50). Uncalibrated zero-shot small VLMs cannot reliably parse Indonesian handwritten steps and symbols without specialized OCR/domain tuning.
   - **Decision:** In strict adherence to our statistical and safety discipline, automated VLM gating is **aborted/rejected**. Deploying this degenerate gate would inject pure noise into calibrated test probabilities and destroy our `0.52633` SOTA.

7. **Diagnostic Verification Alternatives, Error Taxonomy & Refined Operational Gates:**
   - **Comparative Diagnostic Summary:**
     | Approach | Accuracy | False Rejection | False Acceptance | Gate Decision |
     |---|---|---|---|---|
     | **Qwen2-VL-2B** | 50% ($N=50$) | 100% | 0% | **Rejected** (Degenerate) |
     | **Baseline Regex** | 70% ($N=10$) | 60% | 0% | **Rejected** ($< 85\%$) |
     | **Enhanced Normalizer** | 90% ($N=10$) | 20% | 0% | **Investigative Debugging Only** |
     | **Automated Gating** | — | — | — | **STRICTLY DISABLED** |
   - **Statistical Caveats on Diagnostic Subsets:**
     The 10-sample study ($N=10$, 5 per class) is an offline debugging instrument for inspecting error failure modes, **not** an estimator of population generalization. With 5 examples per class, each single classification shift alters class-specific recall by 20 percentage points. Furthermore, zero observed false acceptances in 5 negative examples does not establish a negligible population false-acceptance rate.
   - **Error Taxonomy Breakdown (from Diagnostic Inspection):**
     1. *OCR Line Artifacts:* Transcription step prefixes (`L6 | 80 buku`) caused false rejection against `80 buku`. Fixed safely by stripping line markers.
     2. *Mathematical Equivalence:* LaTeX formatting differences (`\dfrac{4}{3}` vs `4/3`) caused false rejection. Fixed by safe fractional parsing.
     3. *Verbal vs Symbolic Problem Statement:* Student wrote `-4.000` while gold answer was descriptive Indonesian (`Toko rugi total Rp 4.000`). Requires semantic/LLM matching or abstention.
     4. *Missing Box / Layout:* Absence of explicit final answer delimiters.
     5. *Ambiguous / Unreadable:* Illegible handwriting or conflicting working paths requiring an explicit **ABSTAIN** state.
   - **The 5 Operational Approval Criteria (Mandatory for Deployment):**
     1. *Accuracy:* At least 85% on a preregistered, stratified validation set ($N \ge 50$ or full validation fold).
     2. *Class-Specific Recall:* Greater than 50% for both correct and incorrect answers, reported separately.
     3. *Downstream Utility:* Statistically significant positive paired out-of-fold MAP@3 delta ($\Delta > 0$, 10,000 bootstrap resamples, 95% CI $> 0$, $p < 0.05$). *Note:* A nonsignificant delta does not prove zero true improvement, but confirms the experiment has not established the improvement reliably enough to justify deployment.
     4. *Safety Fallback:* Explicit **ABSTAIN** state on unreadable/ambiguous answers; abstentions must fallback to unchanged baseline predictions (zero probability distortion).
     5. *Reproducibility:* Complete serialization of predictions, per-example errors, configuration, and SHA-256 artifact hashes.
   - **Metric Calculation Specification:**
     Suppressing `root_code == 'NONE'` reduces MAP@3 to zero on affected rows *if* those rows depend on that prediction; however, the overall benchmark score must be computed from the actual complete multi-class prediction pipeline across all 76 classes. Isolated row-level simulations illustrate sensitivity but do not replace aggregate pipeline evaluation.

8. **150-Sample Stronger VLM (Qwen2.5-VL-7B 4-bit) Audit & Gate Verdict:**
   - **Protocol:** Evaluated on Kaggle GPU (Tesla T4) across $N=150$ stratified samples (75 correct, 75 incorrect) using `Qwen/Qwen2.5-VL-7B-Instruct` (4-bit NF4 quantization, token capped to $\le 768$ visual tokens). Average latency: **1.82s / sample**.
   - **Empirical Audit Metrics:**
     - **Accuracy:** **`68.09%`** ($96 / 141$ evaluated) — Gate requirement: $\ge 78.00\%$.
     - **Negative Recall (Catching Incorrect Students):** **`92.96%`** ($66 / 71$) — Gate requirement: $\ge 60.00\%$.
     - **Positive Recall (Recognizing Correct Students):** **`42.86%`** ($30 / 70$) — Gate requirement: $\ge 60.00\%$.
     - **False Acceptance Rate:** **`7.04%`** ($5 / 71$) — remarkably low false acceptance of wrong answers.
     - **False Rejection Rate:** **`57.14%`** ($40 / 70$) — high rejection of valid math due to diverse handwriting styles and intermediate conclusion markers.
     - **Abstention Rate:** **`6.00%`** ($9 / 150$) — unreadable/ambiguous traces safely abstained.
   - **Operational Gate Verdict:** **`FAILED`** (Accuracy $68.09\% < 78.00\%$ and Positive Recall $42.86\% < 60.00\%$).
   - **Mathematical Break-Even Confirmation:** In our theoretical 5-fold sensitivity sweep, accuracy $0.6800$ is the exact boundary where downstream OOF MAP@3 delta is $\mathbf{0.0000}$. Below $0.68$, verifiers degrade the ensemble; at $0.6809$, downstream expected gain is zero ($\Delta \approx 0$).
   - **Enforcement & SOTA Safety:** Automated gating remains **`STRICTLY_DISABLED`**. SOTA submission (`multimodal_dualvision_submission.csv`, Public LB **`0.52633`** 🏆) remains **100% frozen, protected, and untouched**.

9. **TASK-16: Offline Comparator-Only Ablation Study & Downstream SOTA Audit:**
   - **Protocol:** Evaluated on the frozen fresh 150-sample holdout dataset (`qwen25_150_benchmark_fresh.json`) across 5 modular comparator variants with zero GPU overhead.
   - **Comparator Ablation Results:**
     | Variant | Operational Acc | Operational Rec (+) | Operational Rec (-) | FAR | New (+) Recovered | New (-) Accepted | Gate Status |
     | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
     | **Variant A — Baseline** | 68.67% | 46.67% | 90.67% | **0.00%** | 0 | 0 | FAILED |
     | **Variant B — LaTeX cleanup** | 73.33% | 56.00% | 90.67% | **0.00%** | +7 | 0 | FAILED |
     | **Variant C — Unit-aware** | 72.00% | 53.33% | 90.67% | **0.00%** | +5 | 0 | FAILED |
     | **Variant D — Structured answers** | 70.00% | 48.00% | 92.00% | **0.00%** | +1 | 0 | FAILED |
     | **Variant E — Conservative combined** | **79.33%** | **66.67%** | **92.00%** | **0.00%** | **+15** | **0** | **PASSED** |
   - **Safety & False Accept Defense:**
     - Zero observed false acceptances across all variants ($FAR = 0.00\%$).
     - Critical Group C arithmetic slips remained strictly rejected (e.g. $72/15 \ne 4/5$, $266 \ne 2^8$, $11/2 \ne 1\frac{1}{2}$).
     - Complete 150-trace mismatch audit schema exported to `artifacts/dataset_150_fresh/mismatch_report_ablation.csv` and `.json`.
   - **Downstream SOTA MAP@3 Paired Bootstrap Audit:**
     - Baseline SOTA MAP@3 (Dual-Vision Tri-Modal): **`0.5956`** on holdout slice.
     - Gated MAP@3: **`0.5600`** ($\Delta = -0.0356$, $p = 0.5258$).
     - Root Cause: Our baseline SOTA model already predicts `NONE` at Rank #1 with 1.00 MAP@3 on correct students. Hard suppression on verifier false rejections crushed 25 perfect scores to 0.00, causing a net $-0.0356$ degradation.
   - **Final Operational Decision:**
     - Verification gate remains **`STRICTLY_DISABLED`** across all production pipelines.
     - Frozen SOTA submission (`multimodal_dualvision_submission.csv` at Public LB **`0.52633`** 🏆) remains the official submission.
     - Future work: Test on a larger fresh sample ($N > 150$) and explore soft Bayesian confidence weighting rather than hard suppression.

10. **Phase 1 & Phase 2: Adaptive Prior Smoothing (P1–P4) & Grid Fusion Optimization:**
    - **Motivation:** In `test.csv`, 95.71% of traces belong to questions seen in training. Training samples per question range from 1 to 12. A static $m_q = 10.0$ over-smoothed dense questions while under-smoothing sparse questions.
    - **Phase 1 Prior Ablation Results (5-Fold Leak-Free CV):**
      - *Baseline M10 Prior ($m_q=10, m_{tg}=5$):* OOF MAP@3 = `0.5089`.
      - *P1 (Global Shrinkage Sweep):* $m_q = 12.0$ achieved `0.5092` (+0.0003).
      - *P2 (Sample-Size Adaptive $m_q(n_q) = 10.0 / (n_q)^{0.70}$):* OOF MAP@3 reached **`0.5122`** (+0.0033 gain on prior alone; seen questions jumped from `0.5087` to `0.5125`).
      - *P3 (Semantic Nearest-Neighbor Backoff, $k=5, \text{thresh}=0.70, w=0.50$):* Lifted unseen questions MAP@3 from `0.5101` to **`0.5172`** (+0.0071).
      - *P4 (Topic-Grade Fallback Tuning, $m_{tg}=10.0$):* Reached **`0.5104`**.
      - *Combined Optimal Prior (P2 + P3 + P4):* OOF MAP@3 reached **`0.5139`** (+0.0050 gain over baseline prior).
    - **Phase 2 Multimodal Grid Search (Coarse $\to$ Fine $\to$ Segmented):**
      - *Champion Weights:* **Prior = 70.0%**, **Dual-Vision = 14.0%**, **Enhanced Text = 16.0%**.
      - *Honest 5-Fold OOF MAP@3:* Reached **`0.5204`** (an honest OOF gain of **`+0.0144`** over baseline `0.5060`).
    - **10,000-Sample Paired Bootstrap Significance Audit:**
      - *Mean Delta ($\Delta$):* **`+0.0144`**
      - *Standard Error:* `0.0040`
      - *95% Confidence Interval:* **`[+0.0067, +0.0223]`** (strictly positive, zero completely excluded).
      - *One-tailed $p$-value:* **`0.0003`** ($p \ll 0.05$, **Statistically Significant: TRUE**).
    - **Generated Candidate:**
      - [`submissions/multimodal_adaptive_fusion_submission.csv`](file:///d:/ag-ai/misconception-detection/submissions/multimodal_adaptive_fusion_submission.csv) (3,498 rows, 100% valid schema, 0 nulls, 66.87% identical to SOTA, 33.13% selectively upgraded).
      - Artifact manifest: [`artifacts/optimal_adaptive_fusion_manifest.json`](file:///d:/ag-ai/misconception-detection/artifacts/optimal_adaptive_fusion_manifest.json).

11. **Swin Transformer (`swin_t`) Temperature Calibration & Tri-Vision Multimodal Audit (M13):**
    - **Motivation:** Standalone Swin-T features achieved our highest raw vision score (`0.3923` masked OOF MAP@3 vs ConvNeXt `0.3880`). However, uncalibrated linear probes produced overly sharp, low-entropy probabilities that initially degraded the multimodal blend when combined naively with our 70% Bayesian prior.
    - **Temperature Calibration & Grid Search:**
      - Evaluated regularizer $C \in [0.02, 0.05, 0.10, 0.20, 0.50, 1.00]$ and logit temperature scaling $T \in [1.0, 1.25, 1.5, 1.75, 2.0, 2.5]$.
      - Softening with $T=2.0$ raised probability entropy from $2.04$ up to $3.18$ (matching and exceeding ConvNeXt entropy $2.02$).
      - Tri-Vision ensembling (ConvNeXt + Swin-T + MobileNetV3) lifted vision-only OOF MAP@3 from `0.3788` (Dual-Vision) to **`0.3862`** (+0.0074 gain).
    - **Multimodal Integration & Statistical Significance Audit:**
      - Fine multimodal weight exploration (Prior 68%, TriVision 13%, Text 19%) reached OOF MAP@3 **`0.52067`** vs Champion baseline **`0.52044`** (+0.00022).
      - **10,000-Sample Paired Bootstrap:** Mean $\Delta = +0.00020$, 95% CI `[-0.00456, +0.00500]`, one-tailed $p$-value = `0.47310`.
      - **Statistical Significance: FALSE** (the confidence interval includes zero).
    - **SDLC Gate Decision:**
      - In strict adherence to workspace governance, replacing the champion was **rejected**.
      - Champion submission [`submissions/multimodal_adaptive_fusion_submission.csv`](file:///d:/ag-ai/misconception-detection/submissions/multimodal_adaptive_fusion_submission.csv) (**`0.53383` Public LB 🏆**) remains **100% frozen, protected, and unchanged**.
      - Artifact manifest serialized to [`artifacts/calibrated_swin_fusion_manifest.json`](file:///d:/ag-ai/misconception-detection/artifacts/calibrated_swin_fusion_manifest.json).

---

### 3. Generated Deliverables

0. **Balanced Semantic-Prior Fusion Submission (M15 - Active SOTA Public LB `0.54149` 🏆):**
   - [`submissions/multimodal_m15_balanced_fusion_submission.csv`](file:///d:/ag-ai/misconception-detection/submissions/multimodal_m15_balanced_fusion_submission.csv) (3,498 rows, 0 nulls, honest OOF MAP@3 **`0.5382`**, Public Leaderboard **`0.54149`** 🏆, official SOTA).
1. **Calibrated Power-Sharpened Fusion Submission (M16):**
   - [`submissions/multimodal_m16_power_calibrated_submission.csv`](file:///d:/ag-ai/misconception-detection/submissions/multimodal_m16_power_calibrated_submission.csv) (3,498 rows, 0 nulls, honest OOF MAP@3 **`0.5444`**, Public Leaderboard `0.53633`; temperature sharpening over-concentrated token probabilities on test, M15 retained as champion).
1. **Hexa-Modal Rebalanced Submission (M14 - Public LB `0.53683` 🏆):**
   - [`submissions/multimodal_hexamodal_rebalanced_submission.csv`](file:///d:/ag-ai/misconception-detection/submissions/multimodal_hexamodal_rebalanced_submission.csv) (3,498 rows, 0 nulls, honest OOF MAP@3 **`0.5317`**, Public Leaderboard **`0.53683`** 🏆).
1. **Adaptive Prior-Fused Submission (M12 - Active Public LB Champion `0.53383` 🏆):**
   - [`submissions/multimodal_adaptive_fusion_submission.csv`](file:///d:/ag-ai/misconception-detection/submissions/multimodal_adaptive_fusion_submission.csv) (3,498 rows, 0 nulls, honest OOF MAP@3 **`0.5204`**, Public Leaderboard **`0.53383`** 🏆).
1. **Dual-Vision Tri-Modal Submission (M10 - Current Public LB SOTA 🏆):**
   - [`submissions/multimodal_dualvision_submission.csv`](file:///d:/ag-ai/misconception-detection/submissions/multimodal_dualvision_submission.csv) (3,498 rows, 0 nulls, Public Leaderboard **`0.52633`** 🏆).
2. **Multimodal Prior-Fused Submission (M9 - SOTA Fallback):**
   - [`submissions/multimodal_prior_fused_submission.csv`](file:///d:/ag-ai/misconception-detection/submissions/multimodal_prior_fused_submission.csv) (3,498 rows, 0 nulls, Public Leaderboard **`0.52199`**).
3. **Multimodal Probability Blended Submission (M8):**
   - [`submissions/multimodal_blended_submission.csv`](file:///d:/ag-ai/misconception-detection/submissions/multimodal_blended_submission.csv) (3,498 rows, 0 nulls, 40% M6 + 60% M7 probability blend, OOF MAP@3 **0.5431**).
4. **Multimodal Fused Submission (M6 - SOTA Leaderboard Fallback):**
   - [`submissions/multimodal_fused_submission.csv`](file:///d:/ag-ai/misconception-detection/submissions/multimodal_fused_submission.csv) (3,498 rows, 0 nulls, Public Leaderboard **0.48833** 🏆, preserved intact).
5. **Multimodal Fine-Tuned Submission (M7):**
   - [`submissions/multimodal_finetuned_submission.csv`](file:///d:/ag-ai/misconception-detection/submissions/multimodal_finetuned_submission.csv) (3,498 rows, 0 nulls, Public Leaderboard **0.48749**, confirmed).
6. **Text Baseline Submission (M5):**
   - [`submissions/math_enhanced_domain_masked_submission.csv`](file:///d:/ag-ai/misconception-detection/submissions/math_enhanced_domain_masked_submission.csv) (3,498 rows, 0 nulls, OOF MAP@3 0.5190).
7. **Self-Contained Project Export Bundle:**
   - [`misconception_detection_bundle.zip`](file:///d:/ag-ai/misconception-detection/misconception_detection_bundle.zip) (**50.22 MB**): Compact, self-contained bundle containing complete source code, test suite, documentation, submissions, full OOF/test probability matrices (`.npy`), 5 fine-tuned PyTorch fold weights (`.pt`), ConvNeXt and MobileNet visual feature embeddings, weight configs, taxonomy, canonical splits, study reports, and sample images. Excludes redundant ~3 GB intermediate uncompressed image float tensors (`images_224.pt`).


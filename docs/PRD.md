# Product Requirements Document (PRD)
## Project: Misconception Detection in K-12 Student Math Workings

**Version:** 1.0.0  
**Phase:** Phase 1 (Requirements)  
**Dataset Source:** CoLearn & Kaggle Benchmark (`misconception-detection-in-k-12-student-math-workings`)  

---

### 1. Executive Summary & Problem Framing
AI systems can solve math problems instantly, but standard solvers merely confirm whether an answer is correct rather than understanding how a student arrived at their mistake. This project builds an automated diagnostic AI pipeline that analyzes phone photographs of handwritten math workings by K-12 students (Indonesian grades 4–12, Bahasa Indonesia handwriting with LaTeX math notations) alongside question context, identifying the foundational **root misconception** from a defined 75-code pedagogical taxonomy (or flagging the attempt as `NONE` if the work is correct or contains only minor calculation slips).

---

### 2. Structured PRD Contract (JSON)

```json
{
  "project_title": "Misconception Detection in K-12 Student Math Workings",
  "overview": "Multimodal diagnostic AI pipeline predicting the root mathematical misconception (from a 75-class taxonomy + NONE) from photos of handwritten student workings and question context in Bahasa Indonesia with LaTeX math.",
  "user_stories": [
    {
      "id": "US-01",
      "as_a": "Math Teacher / Pedagogical AI",
      "i_want": "to upload or pass a photo of student handwritten workings with problem context",
      "so_that": "I receive the top-3 predicted root misconceptions and confidence scores to deliver targeted remedial instruction.",
      "acceptance_criteria": [
        "Ingests student working image (JPEG/PNG) and question metadata (question_text, grade, topic, correct_answer)",
        "Outputs ranked list of top-3 root_code taxonomy predictions with probability distributions",
        "Achieves high MAP@3 / Top-3 Accuracy on stratified validation splits"
      ]
    },
    {
      "id": "US-02",
      "as_a": "ML Engineer / Researcher",
      "i_want": "a decoupled modular architecture supporting both a 2-stage pipeline (Handwriting Reader OCR -> Reasoning Diagnoser) and an End-to-End Vision-Language Model (VLM)",
      "so_that": "we can train on 1,500 gold transcription rows and generalize to 3,498 test rows where transcriptions are absent.",
      "acceptance_criteria": [
        "Stage 1 Reader: Extracts line-by-line transcription (L1 | ..., L2 | ...) from raw handwritten images",
        "Stage 2 Diagnoser: Predicts root_code given extracted transcription and question context",
        "Stage 3 Joint/Multimodal: Direct multimodal inference (e.g. Qwen2-VL / Llama-3.2-Vision / Florence-2 / TrOCR + LLM) for end-to-end classification"
      ]
    },
    {
      "id": "US-03",
      "as_a": "Competition Evaluator",
      "i_want": "automated Kaggle submission generation and local cross-validation harness",
      "so_that": "submission files format strictly adhere to sample_submission.csv format with 3,498 test predictions.",
      "acceptance_criteria": [
        "Validates test set predictions against sample_submission.csv schema (trace_id, root_code / top-3 space-delimited)",
        "Computes local validation metric identical to leaderboard ranking metric (MAP@3 / Top-3 MRR)"
      ]
    }
  ],
  "non_functional_requirements": [
    "Privacy & PII Hygiene: Zero PII retention, student identity protection adhering to CoLearn guidelines",
    "Latency: Inference per student trace under 4.0 seconds on GPU or under 10.0 seconds on standard CPU",
    "Modular Codebase: Strict separation of Data Ingestion, Handwriting OCR, Reasoning Engine, Evaluation, and CLI/API interfaces",
    "Determinism & Reproducibility: Seeded K-fold stratified splits matching student disjoint constraints"
  ],
  "out_of_scope": [
    "Real-time handwriting strokes capture (digital stylus coordinates; input is purely static photos)",
    "Automatic grade generation or score penalty assignment (focus is pedagogical diagnostic, not grading)",
    "Commercial redistribution of CoLearn competition dataset"
  ]
}
```

---

### 3. Data Specification & Features

| Split | Rows | Inputs Provided | Target / Bridge Supervisions |
|---|---|---|---|
| **Train** | 1,500 | `trace_id`, `question_id`, `grade` (4-12), `topic_name` (61 topics), `question_text` (Bahasa + LaTeX), `question_image_file` (119 diagrams), `correct_answer`, `image_file` | `root_code` (75 codes or `NONE`), `transcription_steps` (`L1 \| ...`), `final_answer_correct`, `clean`, `misconceptions` (JSON findings) |
| **Test** | 3,498 | `trace_id`, `question_id`, `grade`, `topic_name`, `question_text`, `question_image_file`, `correct_answer`, `image_file` | **None** (Blind inference required) |
| **Unlabelled** | 50,000 | `trace_id`, `question_id`, `grade`, `topic_name`, `question_text`, `image_file` | Unsupervised / Semi-supervised pre-training pool |
| **Taxonomy** | 75 | Code, Name, Description, Nature (Conceptual / Procedural / Careless / Meta), Default Severity, Grade Band, Section | Master ground-truth taxonomy |

---

### 4. Evaluation Metric
Leaderboard and local evaluation use **Top-3 Ranking (MAP@3 / Mean Average Precision at 3)**:
$$\text{MAP@3} = \frac{1}{N} \sum_{i=1}^N \sum_{k=1}^{\min(3, |R_i|)} P_i(k) \cdot \text{rel}_i(k)$$
where exactly one root code (or `NONE`) is the true label. If the true label is predicted at rank 1, score is $1.0$; rank 2 is $0.5$; rank 3 is $0.333$; otherwise $0.0$.

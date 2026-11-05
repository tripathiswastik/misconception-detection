"""Create compact, self-contained project ZIP archive for sharing, Kaggle datasets, and submission."""
import os
import shutil
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_ZIP = PROJECT_ROOT / "misconception_detection_bundle.zip"

print(f"[Bundle] Creating project export archive at: {OUTPUT_ZIP}")

included_dirs = [
    "src",
    "tests",
    "docs",
    "submissions",
    "artifacts/checkpoints",
]

# Essential standalone files across project root and subdirectories
included_files = [
    "pyproject.toml",
    "state.json",
    "README.md",
    "data/train.csv",
    "data/test.csv",
    "data/sample_submission.csv",
    "data/taxonomy_v1.csv",
    "data/splits.json",
    "data/transcriptions_cache.json",
    # Metadata, audit reports, and manifests
    "artifacts/optimal_weights.json",
    "artifacts/finetuned_weights.json",
    "artifacts/blend_weights.json",
    "artifacts/dualvision_metadata.json",
    "artifacts/trimodal_metadata.json",
    "artifacts/transductive_metadata.json",
    "artifacts/taxonomy_classes.json",
    "artifacts/honest_measurement_report.json",
    "artifacts/ocr_sample_benchmark.json",
    "artifacts/question_prior_audit.json",
    "artifacts/prior_vision_metadata.json",
    "artifacts/stacking_meta_learner_study.json",
    "artifacts/vlm_50_sample_benchmark.json",
    "artifacts/diagnostic_verifier_study.json",
    "artifacts/sota_freeze_manifest.json",
    "artifacts/verifier_150_benchmark_study.json",
    "artifacts/improvement_study_manifest.json",
    "artifacts/calibrated_swin_fusion_manifest.json",
    "artifacts/prior_smoothing_ablation_report.json",
    "artifacts/question_aware_calibration_manifest.json",
    "artifacts/optimal_adaptive_fusion_manifest.json",
    "artifacts/adaptive_poe_metadata.json",
    "artifacts/semantic_text_metadata.json",
    "artifacts/sota_transductive_blend_metadata.json",
    "artifacts/swin_probe_manifest.json",
    # Evaluation benchmark datasets
    "artifacts/dataset_150_sample/sample_150.csv",
    "artifacts/dataset_150_fresh/sample_150_fresh.csv",
    "artifacts/dataset_150_fresh/manifest.json",
    # Model probability matrices (OOF and Test)
    "artifacts/best_adaptive_prior_oof.npy",
    "artifacts/oof_prior_target_encoding.npy",
    "artifacts/oof_text_prior.npy",
    "artifacts/oof_text.npy",
    "artifacts/honest_oof_text.npy",
    "artifacts/test_text.npy",
    "artifacts/oof_vision.npy",
    "artifacts/oof_finetuned_vision.npy",
    "artifacts/oof_convnext_vision.npy",
    "artifacts/oof_swin_vision.npy",
    "artifacts/oof_trivision.npy",
    "artifacts/test_vision.npy",
    "artifacts/test_finetuned_vision.npy",
    "artifacts/test_convnext_vision.npy",
    "artifacts/test_swin_vision.npy",
    "artifacts/test_trivision.npy",
    "artifacts/oof_semantic_text.npy",
    "artifacts/test_semantic_text.npy",
    "artifacts/oof_m15_balanced_fusion.npy",
    "artifacts/test_m15_balanced_fusion.npy",
    "artifacts/oof_m16_power_calibrated.npy",
    "artifacts/test_m16_power_calibrated.npy",
    "artifacts/oof_adaptive_fusion.npy",
    "artifacts/oof_hierarchical_m17.npy",
    "artifacts/test_hierarchical_m17.npy",
    "artifacts/test_adaptive_prior.npy",
    "artifacts/taxonomy_embeddings_384.npy",
    # Extracted feature vectors
    "artifacts/train_perceptual_strokes.npy",
    "artifacts/test_perceptual_strokes.npy",
    "artifacts/train_text_embeddings_384.npy",
    "artifacts/test_text_embeddings_384.npy",
    "artifacts/train_vision_features.npy",
    "artifacts/test_vision_features.npy",
    "artifacts/train_convnext_features.npy",
    "artifacts/test_convnext_features.npy",
    "artifacts/train_swin_features.npy",
    "artifacts/test_swin_features.npy",
]

added_arcnames = set()

with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zipf:
    # 1. Add all project root python scripts (*.py)
    root_scripts = sorted(PROJECT_ROOT.glob("*.py"))
    for py_file in root_scripts:
        arcname = py_file.name
        if arcname not in added_arcnames:
            zipf.write(py_file, arcname)
            added_arcnames.add(arcname)
            print(f"  [+] Added root script: {arcname}")

    # 2. Add directories
    for dir_name in included_dirs:
        dir_path = PROJECT_ROOT / dir_name
        if dir_path.exists():
            for root, _, files in os.walk(dir_path):
                if "__pycache__" in root or ".pytest_cache" in root:
                    continue
                for file in files:
                    file_path = Path(root) / file
                    arcname = str(file_path.relative_to(PROJECT_ROOT)).replace("\\", "/")
                    if arcname not in added_arcnames:
                        zipf.write(file_path, arcname)
                        added_arcnames.add(arcname)

    # 3. Add specific artifacts and data files
    for f in included_files:
        p = PROJECT_ROOT / f
        arcname = f.replace("\\", "/")
        if p.exists() and arcname not in added_arcnames:
            zipf.write(p, arcname)
            added_arcnames.add(arcname)

    # 4. Add representative sample of train and test images (25 images each)
    for img_subdir in ["train_images", "test_images"]:
        img_dir = PROJECT_ROOT / "data" / img_subdir
        if img_dir.exists():
            sample_imgs = list(img_dir.glob("*.jpeg"))[:25]
            for img in sample_imgs:
                arcname = f"data/{img_subdir}/{img.name}"
                if arcname not in added_arcnames:
                    zipf.write(img, arcname)
                    added_arcnames.add(arcname)

# 5. Verify archive integrity
print("\n[Bundle] Verifying archive integrity...")
with zipfile.ZipFile(OUTPUT_ZIP, "r") as zf:
    corrupt = zf.testzip()
    if corrupt:
        raise RuntimeError(f"Zip archive corrupted at: {corrupt}")
    infolist = zf.infolist()
    uncompressed_mb = sum(f.file_size for f in infolist) / (1024 * 1024)

compressed_mb = OUTPUT_ZIP.stat().st_size / (1024 * 1024)
ratio = (1.0 - (compressed_mb / uncompressed_mb)) * 100 if uncompressed_mb > 0 else 0

print(f"[Bundle] SUCCESS: {OUTPUT_ZIP.name}")
print(f"  Total items:        {len(infolist)}")
print(f"  Uncompressed size:  {uncompressed_mb:.2f} MB")
print(f"  Compressed size:    {compressed_mb:.2f} MB ({ratio:.1f}% space saved)")

# Check key SOTA files
required_sota_files = [
    "generate_m15_submission.py",
    "generate_m16_submission.py",
    "improvement_engine.py",
    "fine_tune_grid.py",
    "artifacts/best_adaptive_prior_oof.npy",
    "artifacts/oof_m15_balanced_fusion.npy",
    "artifacts/test_m15_balanced_fusion.npy",
    "artifacts/oof_swin_vision.npy",
    "artifacts/test_swin_vision.npy",
    "artifacts/oof_semantic_text.npy",
    "artifacts/test_semantic_text.npy",
    "submissions/multimodal_m15_balanced_fusion_submission.csv",
    "submissions/multimodal_hexamodal_rebalanced_submission.csv",
    "submissions/multimodal_m16_power_calibrated_submission.csv",
    "submissions/multimodal_m17_hierarchical_decoupled_submission.csv",
    "artifacts/oof_hierarchical_m17.npy",
    "artifacts/test_hierarchical_m17.npy",
]

print("\n[Bundle] Checking key SOTA items:")
all_names = set(zf.namelist())
for req in required_sota_files:
    status = "PRESENT" if req in all_names else "MISSING"
    print(f"  {req:65s} [{status}]")
    if status == "MISSING":
        raise FileNotFoundError(f"Missing required SOTA file in bundle: {req}")

print("\n[Bundle] All SOTA items verified!")

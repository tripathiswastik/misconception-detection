"""MobileNetV3 Small Partial Backbone Fine-Tuning on cuda:0 with Differential LRs.

Implementation of Training Plan:
1. Loads ImageNet-pretrained MobileNetV3 Small.
2. Unfreezes final two feature blocks (layers 11 & 12); keeps layers 0-10 frozen.
3. Uses separate optimizer parameter groups (backbone lr=1e-5, head lr=1e-3).
4. Trains with gradient clipping (max_norm=1.0) and early stopping (best val MAP@3).
5. Generates new 5-fold OOF vision scores using canonical splits from data/splits.json.
6. Re-evaluates late fusion against text baseline, including w=0.25 baseline.
7. Produces submissions/multimodal_finetuned_submission.csv.
"""
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CONFIG, PATHS
from src.data.dataset import MathMisconceptionData
from src.data.splits import FoldSplitter
from src.data.taxonomy import TaxonomyManager
from src.diagnoser.domain_mask import DomainConstraintEngine
from src.fusion.validator import validate_submission_file
from src.metrics.map3 import calculate_map3
from src.models.fusion import MultimodalLateFusion
from src.models.vision import MobileNetV3Classifier


def preload_or_cache_tensors(
    image_paths: List[Path],
    cache_path: Path,
    batch_size: int = 64
) -> torch.Tensor:
    """Load and resize images into a contiguous tensor in RAM/disk cache."""
    if cache_path.exists():
        print(f"[TensorCache] Loading pre-processed image tensor from: {cache_path}...", flush=True)
        return torch.load(cache_path, map_location="cpu")

    print(f"[TensorCache] Pre-processing {len(image_paths)} images to {cache_path}...", flush=True)
    t0 = time.time()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    tensors = []
    for p in image_paths:
        try:
            if p.exists():
                with Image.open(p) as img:
                    img = img.convert("RGB")
                    tensors.append(transform(img))
            else:
                tensors.append(torch.zeros(3, 224, 224, dtype=torch.float32))
        except Exception:
            tensors.append(torch.zeros(3, 224, 224, dtype=torch.float32))

    tensor_stack = torch.stack(tensors, dim=0)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensor_stack, cache_path)
    print(f"[TensorCache] Saved {tensor_stack.shape} in {time.time() - t0:.2f}s ({tensor_stack.element_size() * tensor_stack.nelement() / (1024*1024):.1f} MB)", flush=True)
    return tensor_stack


def train_partially_finetuned_vision(
    backbone_lr: float = 1e-5,
    head_lr: float = 1e-3,
    max_epochs: int = 5,
    batch_size: int = 4,
    accum_steps: int = 2
) -> Dict[str, Any]:
    device = torch.device("cuda:0" if torch.cuda.is_available() and CONFIG.prefer_gpu else "cpu")
    print(f"\n=======================================================", flush=True)
    print(f"[FineTune] Initializing MobileNetV3 Partial Backbone Fine-Tuning", flush=True)
    print(f"  Device: {device} | AMP: {CONFIG.fp16_amp} | Backbone LR: {backbone_lr} | Head LR: {head_lr}", flush=True)
    print(f"  Batch size: {batch_size} (Accum: {accum_steps}) | Max Epochs: {max_epochs}", flush=True)
    print(f"=======================================================\n", flush=True)

    tax = TaxonomyManager(PATHS.taxonomy_csv)
    all_classes = tax.all_codes(include_none=True)
    class_to_idx = {c: idx for idx, c in enumerate(all_classes)}
    n_classes = len(all_classes)

    # 1. Load canonical 5-fold splits
    splitter = FoldSplitter(n_splits=CONFIG.n_splits, random_state=CONFIG.random_state)
    folds = splitter.load_splits(PATHS.splits_cache)

    # 2. Load dataset
    data_mgr = MathMisconceptionData(PATHS.data_dir)
    train_df = data_mgr.load_train()
    test_df = data_mgr.load_test()
    y_true_str = train_df["root_code"].astype(str).values

    n_train = len(train_df)
    n_test = len(test_df)

    artifacts_dir = PROJECT_ROOT / "artifacts"
    checkpoints_dir = artifacts_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    cudnn.benchmark = False
    cudnn.deterministic = True

    # 3. Preload/cache train image tensors (only train images in memory during training)
    train_paths = [PATHS.data_dir / "images" / "train" / str(f) for f in train_df["image_file"].values]
    train_targets = np.array([class_to_idx.get(str(r), class_to_idx["NONE"]) for r in y_true_str], dtype=np.int64)

    X_train_all = preload_or_cache_tensors(train_paths, artifacts_dir / "train_images_224.pt")
    oof_probs = np.zeros((n_train, n_classes), dtype=np.float32)
    fold_vision_scores = []

    for fold_idx, (train_idx, val_idx) in enumerate(folds, start=1):
        print(f"\n--- Fold {fold_idx}/{len(folds)} (Train: {len(train_idx)}, Val: {len(val_idx)}) ---", flush=True)

        X_tr = X_train_all[torch.as_tensor(train_idx, dtype=torch.long)]
        y_tr = torch.tensor(train_targets[train_idx], dtype=torch.int64)
        X_va = X_train_all[torch.as_tensor(val_idx, dtype=torch.long)]

        train_ds = TensorDataset(X_tr, y_tr)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_ds = TensorDataset(X_va)
        val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

        # Model with final 2 feature blocks unfrozen
        model = MobileNetV3Classifier(num_classes=n_classes, pretrained=True, freeze_backbone=True)
        model.unfreeze_top_blocks(num_blocks=2)
        model.to(device)

        # Optimizer with separate parameter groups
        param_groups = model.get_parameter_groups(backbone_lr=backbone_lr, head_lr=head_lr, weight_decay=1e-4)
        optimizer = torch.optim.AdamW(param_groups)
        criterion = nn.CrossEntropyLoss()
        scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and CONFIG.fp16_amp))

        best_val_map3 = -1.0
        best_val_probs: Optional[np.ndarray] = None
        best_state_dict: Optional[dict] = None

        for epoch in range(1, max_epochs + 1):
            model.train()
            running_loss = 0.0
            optimizer.zero_grad()

            for step_idx, (bx, by) in enumerate(train_loader):
                bx = bx.to(device, non_blocking=True)
                by = by.to(device, non_blocking=True)

                with torch.cuda.amp.autocast(enabled=(device.type == "cuda" and CONFIG.fp16_amp)):
                    logits = model(bx)
                    loss = criterion(logits, by) / accum_steps

                scaler.scale(loss).backward()

                if (step_idx + 1) % accum_steps == 0 or (step_idx + 1) == len(train_loader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                running_loss += loss.item() * accum_steps * len(by)

            train_loss = running_loss / len(train_ds)

            # Validation evaluation
            model.eval()
            val_preds_list = []
            with torch.no_grad():
                for (vx,) in val_loader:
                    vx = vx.to(device, non_blocking=True)
                    with torch.cuda.amp.autocast(enabled=(device.type == "cuda" and CONFIG.fp16_amp)):
                        v_logits = model(vx)
                        v_probs = torch.softmax(v_logits, dim=1)
                    val_preds_list.append(v_probs.cpu().numpy())

            current_val_probs = np.concatenate(val_preds_list, axis=0)
            top_indices = np.argsort(-current_val_probs, axis=1, kind="mergesort")[:, :3]
            preds = [[all_classes[idx] for idx in row] for row in top_indices]
            val_targets_str = [y_true_str[idx] for idx in val_idx]
            epoch_map3 = calculate_map3(val_targets_str, preds)

            print(f"  Epoch {epoch}/{max_epochs} - Loss: {train_loss:.4f} | Val MAP@3: {epoch_map3:.4f}", flush=True)

            if epoch_map3 > best_val_map3:
                best_val_map3 = epoch_map3
                best_val_probs = current_val_probs.copy()
                best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        assert best_val_probs is not None
        assert best_state_dict is not None
        oof_probs[val_idx] = best_val_probs
        fold_vision_scores.append(best_val_map3)
        print(f"  --> Best Fold {fold_idx} Vision MAP@3: {best_val_map3:.4f}", flush=True)

        # Save best fold checkpoint
        ckpt_path = checkpoints_dir / f"finetuned_vision_fold{fold_idx-1}.pt"
        torch.save(best_state_dict, ckpt_path)

        del model, optimizer, train_loader, val_loader, best_state_dict, X_tr, y_tr, X_va, train_ds, val_ds
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    # Overall fine-tuned vision OOF MAP@3
    overall_top = np.argsort(-oof_probs, axis=1, kind="mergesort")[:, :3]
    overall_preds = [[all_classes[idx] for idx in row] for row in overall_top]
    overall_vision_map3 = calculate_map3(list(y_true_str), overall_preds)

    print(f"\n=======================================================", flush=True)
    print(f"[FineTune] Overall 5-Fold Fine-Tuned Vision OOF MAP@3: {overall_vision_map3:.4f}", flush=True)
    print(f"  (Previous Frozen Vision OOF MAP@3: 0.3393)", flush=True)
    print(f"=======================================================\n", flush=True)

    # Clean up train tensors before loading test tensors
    del X_train_all
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    # Predict test set using ensemble of all 5 fold checkpoints
    print("[FineTune] Generating test predictions using 5-fold ensemble...", flush=True)
    test_paths = [PATHS.data_dir / "images" / "test" / str(f) for f in test_df["image_file"].values]
    X_test_all = preload_or_cache_tensors(test_paths, artifacts_dir / "test_images_224.pt")
    test_dataset = TensorDataset(X_test_all)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    test_probs = np.zeros((n_test, n_classes), dtype=np.float32)
    for fold_idx in range(len(folds)):
        ckpt_path = checkpoints_dir / f"finetuned_vision_fold{fold_idx}.pt"
        fold_state = torch.load(ckpt_path, map_location=device)
        model = MobileNetV3Classifier(num_classes=n_classes, pretrained=False, freeze_backbone=False)
        model.load_state_dict(fold_state)
        model.to(device)
        model.eval()

        fold_test_list = []
        with torch.no_grad():
            for (tx,) in test_loader:
                tx = tx.to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=(device.type == "cuda" and CONFIG.fp16_amp)):
                    t_logits = model(tx)
                    t_probs = torch.softmax(t_logits, dim=1)
                fold_test_list.append(t_probs.cpu().numpy())

        fold_test_probs = np.concatenate(fold_test_list, axis=0)
        test_probs += fold_test_probs / len(folds)
        del model, fold_state
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    del X_test_all, test_dataset, test_loader
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    # Persist vision predictions
    np.save(artifacts_dir / "oof_finetuned_vision.npy", oof_probs)
    np.save(artifacts_dir / "test_finetuned_vision.npy", test_probs)

    # 4. Late Fusion Re-evaluation
    oof_text = np.load(artifacts_dir / "oof_text.npy")
    test_text = np.load(artifacts_dir / "test_text.npy")

    # Apply domain constraint masking
    domain_masker = DomainConstraintEngine(taxonomy=tax).fit_empirical(train_df)
    train_topics = [str(t) for t in train_df["topic_name"].values]
    train_grades = [int(g) for g in train_df["grade"].values]
    test_topics = [str(t) for t in test_df["topic_name"].values]
    test_grades = [int(g) for g in test_df["grade"].values]

    oof_vision_masked = domain_masker.apply_mask_batch(
        oof_probs, all_classes, train_topics, train_grades, damping=CONFIG.domain_mask_damping
    )
    test_vision_masked = domain_masker.apply_mask_batch(
        test_probs, all_classes, test_topics, test_grades, damping=CONFIG.domain_mask_damping
    )

    print("[Fusion] Grid searching optimal fusion weight with fine-tuned vision...", flush=True)
    candidate_weights = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
    fusion_results = MultimodalLateFusion.tune_fusion_weight(
        y_true=y_true_str,
        text_scores=oof_text,
        vision_scores=oof_vision_masked,
        classes=all_classes,
        candidate_weights=candidate_weights
    )

    # Also check exactly w=0.25 baseline
    fused_baseline_oof = MultimodalLateFusion.apply_fusion(oof_text, oof_vision_masked, vision_weight=0.25)
    baseline_w25_map3 = calculate_map3(
        list(y_true_str),
        [[all_classes[idx] for idx in row] for row in np.argsort(-fused_baseline_oof, axis=1, kind="mergesort")[:, :3]]
    )

    print(f"  Baseline Text OOF MAP@3:                 {fusion_results['baseline_text_map3']:.4f}", flush=True)
    print(f"  Fine-Tuned Vision at w=0.25 OOF MAP@3:   {baseline_w25_map3:.4f}", flush=True)
    print(f"  Optimal Vision Weight:                  {fusion_results['optimal_vision_weight']:.4f}", flush=True)
    print(f"  Best Fused OOF MAP@3:                   {fusion_results['best_oof_map3']:.4f}", flush=True)

    opt_weight = fusion_results["optimal_vision_weight"]

    # Generate fine-tuned multimodal test submission
    fused_test = MultimodalLateFusion.apply_fusion(
        text_scores=test_text,
        vision_scores=test_vision_masked,
        vision_weight=opt_weight
    )

    top_indices = np.argsort(-fused_test, axis=1, kind="mergesort")[:, :3]
    pred_strings = [" ".join([all_classes[idx] for idx in row]) for row in top_indices]

    submission_df = pd.DataFrame({
        "trace_id": test_df["trace_id"].astype(str),
        "root_code": pred_strings
    })

    PATHS.submissions_dir.mkdir(parents=True, exist_ok=True)
    out_sub_path = PATHS.submissions_dir / "multimodal_finetuned_submission.csv"
    submission_df.to_csv(out_sub_path, index=False)
    print(f"\n[FineTune] Saved submission to: {out_sub_path} ({len(submission_df)} rows)", flush=True)

    validation = validate_submission_file(out_sub_path, PATHS.taxonomy_csv)
    print(f"[FineTune] Submission validation: {validation}", flush=True)
    if not validation["valid"]:
        raise ValueError(f"Submission validation failed: {validation['errors']}")

    # Save metadata
    tuning_record = {
        "backbone_lr": backbone_lr,
        "head_lr": head_lr,
        "max_epochs": max_epochs,
        "vision_oof_map3": overall_vision_map3,
        "fusion_w25_oof_map3": baseline_w25_map3,
        "best_fusion_weight": opt_weight,
        "best_fused_oof_map3": fusion_results["best_oof_map3"]
    }
    with open(artifacts_dir / "finetuned_weights.json", "w", encoding="utf-8") as f:
        json.dump(tuning_record, f, indent=2)

    return tuning_record


if __name__ == "__main__":
    train_partially_finetuned_vision(backbone_lr=1e-5, head_lr=1e-3, max_epochs=5, batch_size=4, accum_steps=2)

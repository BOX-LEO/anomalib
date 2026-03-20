# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Prune a frozen DINOv2 backbone for PatchFlow using activation-based pruning.

This script loads a DINOv2 model, collects activation statistics on MVTecAD
training images, and prunes both the SwiGLU MLP intermediate channels and
attention Q/K dimensions using the ``activation_prune`` framework.

Example::

    python src/anomalib/models/image/patchflow/prune_dino.py \
        --model base --mvtec_root ./datasets/MVTecAD \
        --sparsity 0.3 --target both --attn_mode dim-logit \
        --save_path ./pruned_dinov2_base.pt
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset, DataLoader

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DINOv2 model name mapping
# ---------------------------------------------------------------------------
_MODEL_NAME_MAP: dict[str, str] = {
    "small": "dinov2_vit_small_14",
    "base": "dinov2_vit_base_14",
    "large": "dinov2_vit_large_14",
}


def _build_calibration_loader(
    mvtec_root: str | Path,
    batch_size: int = 16,
    calib_samples: int = 3000,
    num_workers: int = 4,
) -> DataLoader:
    """Build a DataLoader over all MVTecAD training (good) images.

    Args:
        mvtec_root: Path to the MVTecAD dataset root.
        batch_size: Batch size for the calibration loader.
        calib_samples: Maximum number of calibration samples to use.
        num_workers: Number of dataloader workers.

    Returns:
        A DataLoader yielding batches of image tensors.
    """
    from torchvision import transforms as T

    from anomalib.data.datasets.image.mvtecad import CATEGORIES, MVTecADDataset
    from anomalib.data.utils import Split

    # DINOv2 native resolution and ImageNet normalisation
    dino_transform = T.Compose([
        T.Resize((518, 518)),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    datasets = []
    for cat in CATEGORIES:
        ds = MVTecADDataset(root=mvtec_root, category=cat, split=Split.TRAIN)
        datasets.append(ds)
    calib_dataset = ConcatDataset(datasets)

    # Limit to calib_samples
    if len(calib_dataset) > calib_samples:
        indices = torch.randperm(len(calib_dataset))[:calib_samples].tolist()
        calib_dataset = torch.utils.data.Subset(calib_dataset, indices)

    def collate_fn(batch: list[dict]) -> torch.Tensor:
        """Extract images from MVTecAD items and apply DINO transforms."""
        images = torch.stack([item.image for item in batch])
        return dino_transform(images)

    return DataLoader(
        calib_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )


def prune_dino_backbone(
    model_size: str = "base",
    mvtec_root: str | Path = "./datasets/MVTecAD",
    sparsity: float = 0.3,
    qk_sparsity: float = 0.3,
    target: str = "both",
    attn_mode: str = "dim-logit",
    ranker: str = "weight_magnitude",
    output_dir: str | Path = "./logs/prune_dino",
    save_path: str | Path | None = None,
    batch_size: int = 16,
    calib_samples: int = 3000,
    device: str = "cuda",
) -> None:
    """Prune a DINOv2 backbone and save the result.

    Args:
        model_size: One of ``"small"``, ``"base"``, ``"large"``.
        mvtec_root: Path to the MVTecAD dataset root.
        sparsity: Target MLP sparsity (0.0-1.0).
        qk_sparsity: Target Q/K dimension sparsity (0.0-1.0).
        target: Pruning target: ``"mlp"``, ``"attn"``, or ``"both"``.
        attn_mode: Attention pruning mode: ``"head"`` or ``"dim-logit"``.
        ranker: Ranking policy name.
        output_dir: Directory for pruning logs.
        save_path: Path to save the pruned model. Defaults to
            ``pruned_<model_name>.pt`` in the current directory.
        batch_size: Calibration batch size.
        calib_samples: Maximum number of calibration samples.
        device: Device for pruning (``"cuda"`` or ``"cpu"``).
    """
    # --- Add activation_prune to path ---
    activation_prune_dir = Path("/home/boxiang/work/dao2/ombs/activation_prune")
    if str(activation_prune_dir) not in sys.path:
        sys.path.insert(0, str(activation_prune_dir))

    from config.schemas import (
        AttentionPruneMode,
        CollectorConfig,
        CovarianceMode,
        FullConfig,
        PruneTarget,
        PruningConfig,
        RankerType,
        RunnerConfig,
        ScheduleType,
    )
    from pruning.runner import PruneRunner

    from anomalib.models.components.dinov2 import DinoV2Loader

    # --- Resolve model name ---
    model_name = _MODEL_NAME_MAP.get(model_size)
    if model_name is None:
        msg = f"Unknown model size '{model_size}'. Choose from {list(_MODEL_NAME_MAP.keys())}"
        raise ValueError(msg)

    if save_path is None:
        save_path = f"pruned_{model_name}.pt"
    save_path = Path(save_path)

    logger.info("Loading DINOv2 model: %s", model_name)
    model = DinoV2Loader.from_name(model_name)
    model.eval()

    # --- Build calibration data ---
    logger.info("Building calibration loader from MVTecAD at %s", mvtec_root)
    calib_loader = _build_calibration_loader(
        mvtec_root=mvtec_root,
        batch_size=batch_size,
        calib_samples=calib_samples,
    )
    logger.info("Calibration dataset: %d samples", len(calib_loader.dataset))

    # --- Configure pruning ---
    prune_target = PruneTarget(target)
    config = FullConfig(
        collector=CollectorConfig(
            target=prune_target,
            covariance_mode=CovarianceMode.EXACT,
        ),
        pruning=PruningConfig(
            target=prune_target,
            schedule=ScheduleType.LAYERWISE,
            sparsity=sparsity,
            ranker=RankerType(ranker),
            attn_prune_mode=AttentionPruneMode(attn_mode),
            qk_sparsity=qk_sparsity,
        ),
        runner=RunnerConfig(
            device=device,
            output_dir=Path(output_dir),
            calib_samples=calib_samples,
        ),
    )

    # --- Run pruning ---
    logger.info("Starting pruning (target=%s, sparsity=%.2f, attn_mode=%s)", target, sparsity, attn_mode)
    runner = PruneRunner(config)
    result = runner.run(
        model=model,
        calib_loader=calib_loader,
        skip_compensation=False,
        model_name=model_name,
    )

    logger.info(
        "Pruning complete: compression=%.2fx, original=%d params, pruned=%d params",
        result.compression_ratio,
        result.original_params,
        result.pruned_params,
    )

    # --- Save pruned model ---
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result.pruned_model, save_path)
    logger.info("Pruned model saved to %s", save_path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prune DINOv2 backbone for PatchFlow")
    parser.add_argument("--model", type=str, default="base", choices=["small", "base", "large"])
    parser.add_argument("--mvtec_root", type=str, default="./datasets/MVTecAD")
    parser.add_argument("--sparsity", type=float, default=0.3)
    parser.add_argument("--qk_sparsity", type=float, default=0.3)
    parser.add_argument("--target", type=str, default="both", choices=["mlp", "attn", "both"])
    parser.add_argument("--attn_mode", type=str, default="dim-logit", choices=["head", "dim-logit"])
    parser.add_argument("--ranker", type=str, default="weight_magnitude")
    parser.add_argument("--output_dir", type=str, default="./logs/prune_dino")
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--calib_samples", type=int, default=3000)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = _parse_args()
    prune_dino_backbone(
        model_size=args.model,
        mvtec_root=args.mvtec_root,
        sparsity=args.sparsity,
        qk_sparsity=args.qk_sparsity,
        target=args.target,
        attn_mode=args.attn_mode,
        ranker=args.ranker,
        output_dir=args.output_dir,
        save_path=args.save_path,
        batch_size=args.batch_size,
        calib_samples=args.calib_samples,
        device=args.device,
    )

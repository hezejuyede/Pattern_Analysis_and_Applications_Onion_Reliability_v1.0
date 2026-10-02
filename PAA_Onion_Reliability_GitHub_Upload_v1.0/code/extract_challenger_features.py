"""Extract frozen torchvision challenger embeddings in manifest order."""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import (
    ConvNeXt_Tiny_Weights,
    EfficientNet_B0_Weights,
    Swin_T_Weights,
    convnext_tiny,
    efficientnet_b0,
    swin_t,
)


ROOT = Path(__file__).resolve().parents[1]


class ManifestImages(Dataset):
    def __init__(self, frame: pd.DataFrame, transform):
        self.frame = frame.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        with Image.open(ROOT / row.relative_path) as image:
            return self.transform(image.convert("RGB")), row.sample_id


class EfficientNetEmbedding(torch.nn.Module):
    def __init__(self, base):
        super().__init__()
        self.features = base.features
        self.pool = base.avgpool

    def forward(self, inputs):
        return torch.flatten(self.pool(self.features(inputs)), 1)


class ConvNeXtEmbedding(torch.nn.Module):
    def __init__(self, base):
        super().__init__()
        self.features = base.features
        self.pool = base.avgpool
        self.norm = base.classifier[0]
        self.flatten = base.classifier[1]

    def forward(self, inputs):
        return self.flatten(self.norm(self.pool(self.features(inputs))))


class SwinEmbedding(torch.nn.Module):
    def __init__(self, base):
        super().__init__()
        self.features = base.features
        self.norm = base.norm
        self.permute = base.permute
        self.pool = base.avgpool
        self.flatten = base.flatten

    def forward(self, inputs):
        values = self.features(inputs)
        values = self.norm(values)
        values = self.permute(values)
        return self.flatten(self.pool(values))


def build(name: str):
    if name == "efficientnet_b0":
        weights = EfficientNet_B0_Weights.DEFAULT
        base = efficientnet_b0(weights=weights)
        return EfficientNetEmbedding(base), weights, 1280
    if name == "convnext_tiny":
        weights = ConvNeXt_Tiny_Weights.DEFAULT
        base = convnext_tiny(weights=weights)
        return ConvNeXtEmbedding(base), weights, 768
    if name == "swin_t":
        weights = Swin_T_Weights.DEFAULT
        base = swin_t(weights=weights)
        return SwinEmbedding(base), weights, 768
    raise ValueError(name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("efficientnet_b0", "convnext_tiny", "swin_t"), required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    torch.set_num_threads(max(1, (os.cpu_count() or 2) - 2))
    frame = pd.read_csv(ROOT / "manifests" / "unified_manifest.csv")
    model, weights, dimension = build(args.model)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    loader = DataLoader(
        ManifestImages(frame, weights.transforms()),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    pieces = []
    identifiers = []
    started = time.perf_counter()
    with torch.inference_mode():
        for batch_number, (images, sample_ids) in enumerate(loader, start=1):
            pieces.append(model(images).cpu().numpy().astype(np.float32))
            identifiers.extend(sample_ids)
            if batch_number % 25 == 0:
                print(json.dumps({"model": args.model, "processed": len(identifiers), "total": len(frame)}), flush=True)
    elapsed = time.perf_counter() - started
    matrix = np.concatenate(pieces)
    if matrix.shape != (len(frame), dimension):
        raise RuntimeError(f"unexpected embedding shape {matrix.shape}")
    if identifiers != frame.sample_id.astype(str).tolist():
        raise RuntimeError("embedding order differs from manifest")
    output = ROOT / "features"
    np.savez_compressed(output / f"{args.model}.npz", features=matrix, sample_ids=np.asarray(identifiers))
    metadata = {
        "model": args.model,
        "role": "secondary architecture sensitivity; ResNet18 remains a-priori primary",
        "weights": str(weights),
        "feature_dimension": dimension,
        "samples": len(frame),
        "dtype": str(matrix.dtype),
        "preprocessing": repr(weights.transforms()),
        "official_weight_metadata": {
            "parameter_count": weights.meta.get("num_params"),
            "giga_operations_per_image": weights.meta.get("_ops"),
            "weight_file_megabytes": weights.meta.get("_file_size"),
            "categories": len(weights.meta.get("categories", [])),
        },
        "extraction": {
            "device": "cpu",
            "batch_size": args.batch_size,
            "elapsed_seconds": elapsed,
            "images_per_second": len(frame) / elapsed,
            "processor": platform.processor(),
            "torch_threads": torch.get_num_threads(),
        },
        "torch": torch.__version__,
        "torchvision": __import__("torchvision").__version__,
    }
    (output / f"{args.model}.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

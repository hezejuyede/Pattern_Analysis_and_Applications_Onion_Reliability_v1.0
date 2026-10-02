"""Extract frozen, reproducible image features for the reliability benchmark."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import MobileNet_V3_Small_Weights, ResNet18_Weights, mobilenet_v3_small, resnet18


ROOT = Path(__file__).resolve().parents[1]


class ManifestImages(Dataset):
    def __init__(self, frame: pd.DataFrame, transform):
        self.frame = frame.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        with Image.open(ROOT / row.relative_path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, row.sample_id


class MobileNetEmbedding(torch.nn.Module):
    def __init__(self):
        super().__init__()
        model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        self.features = model.features
        self.pool = model.avgpool

    def forward(self, x):
        return torch.flatten(self.pool(self.features(x)), 1)


def deep_features(frame: pd.DataFrame, model_name: str, batch_size: int) -> tuple[np.ndarray, list[str], dict]:
    if model_name == "resnet18":
        weights = ResNet18_Weights.DEFAULT
        base = resnet18(weights=weights)
        model = torch.nn.Sequential(*list(base.children())[:-1], torch.nn.Flatten(1))
        feature_dim = 512
    elif model_name == "mobilenet_v3_small":
        weights = MobileNet_V3_Small_Weights.DEFAULT
        model = MobileNetEmbedding()
        feature_dim = 576
    else:
        raise ValueError(model_name)
    model.eval()
    dataset = ManifestImages(frame, weights.transforms())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    features, sample_ids = [], []
    with torch.inference_mode():
        for images, identifiers in loader:
            values = model(images).cpu().numpy().astype(np.float32)
            features.append(values)
            sample_ids.extend(identifiers)
    matrix = np.concatenate(features)
    if matrix.shape != (len(frame), feature_dim):
        raise RuntimeError(f"unexpected feature shape {matrix.shape}")
    metadata = {
        "model": model_name,
        "weights": str(weights),
        "feature_dim": feature_dim,
        "preprocessing": repr(weights.transforms()),
        "torch": torch.__version__,
        "torchvision": __import__("torchvision").__version__,
    }
    return matrix, sample_ids, metadata


def one_handcrafted(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"cannot read {path}")
    image = cv2.resize(image, (128, 128), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    parts = []
    for array, ranges in ((image, ((0, 256),) * 3), (hsv, ((0, 180), (0, 256), (0, 256))), (lab, ((0, 256),) * 3)):
        for channel, limits in enumerate(ranges):
            hist = cv2.calcHist([array], [channel], None, [16], list(limits)).ravel()
            hist = hist / max(hist.sum(), 1.0)
            parts.append(hist)
            values = array[:, :, channel].astype(np.float32).ravel()
            parts.append(np.array([values.mean(), values.std(), *np.quantile(values, [0.1, 0.5, 0.9])], dtype=np.float32))
    edges = cv2.Canny(gray, 75, 150)
    texture = np.array(
        [edges.mean() / 255.0, cv2.Laplacian(gray, cv2.CV_64F).var(), gray.mean(), gray.std()], dtype=np.float32
    )
    hog = cv2.HOGDescriptor((64, 64), (16, 16), (8, 8), (8, 8), 9)
    hog_values = hog.compute(cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)).ravel().astype(np.float32)
    parts.extend([texture, hog_values])
    return np.concatenate(parts).astype(np.float32)


def handcrafted_features(frame: pd.DataFrame) -> tuple[np.ndarray, list[str], dict]:
    features = [one_handcrafted(ROOT / row.relative_path) for row in frame.itertuples(index=False)]
    matrix = np.stack(features)
    return matrix, frame.sample_id.tolist(), {
        "model": "handcrafted_color_hog",
        "feature_dim": int(matrix.shape[1]),
        "opencv": cv2.__version__,
        "description": "BGR/HSV/Lab histograms and moments, edge/texture summaries, 64x64 HOG",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("handcrafted", "resnet18", "mobilenet_v3_small"), required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    torch.set_num_threads(max(1, (os.cpu_count() or 2) - 2))
    frame = pd.read_csv(ROOT / "manifests" / "unified_manifest.csv")
    if args.model == "handcrafted":
        matrix, identifiers, metadata = handcrafted_features(frame)
    else:
        matrix, identifiers, metadata = deep_features(frame, args.model, args.batch_size)
    if identifiers != frame.sample_id.tolist():
        raise RuntimeError("feature order does not match manifest")
    output = ROOT / "features"
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / f"{args.model}.npz", features=matrix, sample_ids=np.array(identifiers))
    metadata.update({"samples": len(identifiers), "dtype": str(matrix.dtype)})
    (output / f"{args.model}.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

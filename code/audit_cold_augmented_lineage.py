"""Audit COLD augmentation families and cautiously link them to raw images.

The Hugging Face parquet preserves the original augmented filenames even though
the datasets-server API exposes only generated row URLs.  Filenames of the form
``dr_<integer>_<integer>.jpg`` provide a deterministic augmentation-family ID.
That family ID is sufficient for family-safe train/test splitting.  Mapping a
family ID back to a raw filename is a separate, inferential step because the
public archive does not contain an explicit crosswalk.

This script therefore keeps two evidence levels separate:

1. filename-confirmed augmentation families;
2. optional raw-image links based on ImageNet ResNet-18 similarity followed by
   SIFT/RANSAC geometric verification.  Weak or non-unique links remain marked
   ambiguous or unresolved; no one-to-one assignment is forced.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "audit" / "cold_augmented_lineage"
RAW_PARQUET = ROOT / "raw" / "COLD_raw.parquet"
AUG_PARQUET = ROOT / "raw" / "COLD_augmented.parquet"
RAW_ROOT = ROOT / "data" / "cold_raw_hf"
AUG_ROOT = ROOT / "data" / "cold_augmented_hf"
EMBEDDINGS = OUTPUT / "resnet18_embeddings.npz"

LABELS = {
    0: "iris_yellow_virus",
    1: "stemphylium_colletotrichum_leaf_blight",
    2: "healthy",
    3: "purple_blotch",
}
FAMILY_PATTERN = re.compile(r"^dr_(?P<parent_index>\d+)_(?P<random_suffix>\d+)\.(?:jpe?g|png)$", re.I)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def decoded_correspondence(parquet_bytes: bytes, path: Path) -> dict:
    """Check row identity after allowing harmless JPEG re-encoding.

    Hugging Face's cached-assets endpoint may decode and re-encode an Image
    feature.  A byte-level mismatch is therefore not by itself a row mismatch.
    We additionally compare decoded RGB pixels and a 256-bit difference hash.
    """
    with Image.open(io.BytesIO(parquet_bytes)) as image:
        parquet_rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    with Image.open(path) as image:
        local_rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if parquet_rgb.shape != local_rgb.shape:
        return {
            "shape_match": False,
            "pixel_mae": None,
            "pixel_max_abs_error": None,
            "dhash_hamming": None,
            "correspondence_pass": False,
        }
    difference = np.abs(parquet_rgb.astype(np.int16) - local_rgb.astype(np.int16))

    def dhash(values: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(values, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (17, 16), interpolation=cv2.INTER_AREA)
        return resized[:, 1:] > resized[:, :-1]

    hamming = int(np.count_nonzero(dhash(parquet_rgb) != dhash(local_rgb)))
    mae = float(difference.mean())
    maximum = int(difference.max())
    return {
        "shape_match": True,
        "pixel_mae": mae,
        "pixel_max_abs_error": maximum,
        "dhash_hamming": hamming,
        # Empirical maxima for this materialization are MAE 4.10 and dHash 9
        # for raw, and MAE 0.25 and dHash 2 for augmented.  These conservative
        # bounds detect row swaps while tolerating cached JPEG recompression.
        "correspondence_pass": mae <= 5.0 and hamming <= 12,
    }


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parquet_rows(path: Path) -> list[dict]:
    return pq.read_table(path, columns=["image", "label"]).to_pylist()


def local_path(root: Path, label: str, row_idx: int) -> Path:
    return root / label / f"row_{row_idx:04d}.jpg"


class Images(Dataset):
    def __init__(self, paths: list[Path], transform):
        self.paths = paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        with Image.open(self.paths[index]) as image:
            return self.transform(image.convert("RGB"))


def create_embeddings(paths: list[Path], batch_size: int) -> np.ndarray:
    weights = ResNet18_Weights.DEFAULT
    base = resnet18(weights=weights)
    model = torch.nn.Sequential(*list(base.children())[:-1], torch.nn.Flatten(1))
    model.eval()
    torch.set_num_threads(max(1, (os.cpu_count() or 4) - 2))
    loader = DataLoader(Images(paths, weights.transforms()), batch_size=batch_size, shuffle=False, num_workers=0)
    parts: list[np.ndarray] = []
    with torch.inference_mode():
        for images in loader:
            parts.append(model(images).cpu().numpy().astype(np.float32))
    return np.concatenate(parts)


def load_or_create_embeddings(paths: list[Path], batch_size: int) -> tuple[np.ndarray, dict]:
    relative = [path.relative_to(ROOT).as_posix() for path in paths]
    if EMBEDDINGS.exists():
        archive = np.load(EMBEDDINGS, allow_pickle=False)
        saved_paths = archive["paths"].tolist()
        if saved_paths == relative:
            matrix = archive["features"].astype(np.float32)
            source = "reused"
        else:
            # An older cache may contain the same paths in another deterministic
            # order.  Reorder it rather than silently associating the wrong row.
            positions = {path: index for index, path in enumerate(saved_paths)}
            if len(positions) == len(saved_paths) and set(positions) == set(relative):
                matrix = archive["features"][[positions[path] for path in relative]].astype(np.float32)
                source = "reused_reordered"
            else:
                matrix = create_embeddings(paths, batch_size)
                source = "recomputed"
    else:
        matrix = create_embeddings(paths, batch_size)
        source = "computed"
    np.savez_compressed(EMBEDDINGS, features=matrix, paths=np.array(relative))
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / np.maximum(norms, 1e-12)
    return matrix, {
        "model": "torchvision ResNet-18 ImageNet1K_V1, frozen global-average-pool features",
        "feature_dimension": int(matrix.shape[1]),
        "cache_status": source,
        "torch": torch.__version__,
        "torchvision": __import__("torchvision").__version__,
    }


class SiftCache:
    def __init__(self):
        self.detector = cv2.SIFT_create(nfeatures=1000, contrastThreshold=0.02)
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)
        self.cache: dict[str, tuple[list, np.ndarray | None]] = {}

    def describe(self, path: Path):
        key = str(path)
        if key not in self.cache:
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise RuntimeError(f"cannot read image {path}")
            self.cache[key] = self.detector.detectAndCompute(image, None)
        return self.cache[key]

    def verify(self, augmented: Path, raw: Path) -> tuple[int, int, float]:
        aug_points, aug_desc = self.describe(augmented)
        raw_points, raw_desc = self.describe(raw)
        if aug_desc is None or raw_desc is None or len(aug_desc) < 2 or len(raw_desc) < 2:
            return 0, 0, 0.0
        pairs = self.matcher.knnMatch(aug_desc, raw_desc, k=2)
        good = [first for first, second in pairs if first.distance < 0.75 * second.distance]
        if len(good) < 4:
            return 0, len(good), 0.0
        source = np.float32([aug_points[item.queryIdx].pt for item in good])
        target = np.float32([raw_points[item.trainIdx].pt for item in good])
        _, mask = cv2.findHomography(source, target, cv2.RANSAC, 4.0)
        inliers = int(mask.sum()) if mask is not None else 0
        return inliers, len(good), inliers / max(1, len(good))


def selected_members(members: list[dict], limit: int) -> list[dict]:
    members = sorted(members, key=lambda row: row["row_idx"])
    if len(members) <= limit:
        return members
    indexes = np.linspace(0, len(members) - 1, limit, dtype=int)
    return [members[int(index)] for index in indexes]


def confidence_tier(best: dict, second: dict | None) -> str:
    second_inliers = int(second["sift_inliers"]) if second else 0
    gap = int(best["sift_inliers"]) - second_inliers
    if (
        int(best["sift_inliers"]) >= 15
        and float(best["sift_inlier_fraction"]) >= 0.60
        and gap >= max(8, int(round(0.25 * int(best["sift_inliers"]))))
        and float(best["resnet_cosine"]) >= 0.80
    ):
        return "high_geometric_support"
    if (
        int(best["sift_inliers"]) >= 8
        and float(best["sift_inlier_fraction"]) >= 0.50
        and gap >= 4
        and float(best["resnet_cosine"]) >= 0.80
    ):
        return "moderate_geometric_support"
    if int(best["sift_inliers"]) >= 8:
        return "ambiguous_candidate"
    return "unresolved"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--resnet-candidates", type=int, default=30)
    parser.add_argument("--sift-family-members", type=int, default=5)
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)

    raw_rows = parquet_rows(RAW_PARQUET)
    augmented_rows = parquet_rows(AUG_PARQUET)
    raw_manifest: list[dict] = []
    augmented_manifest: list[dict] = []
    integrity_errors: list[dict] = []

    for row_idx, row in enumerate(raw_rows):
        label_id = int(row["label"])
        label = LABELS[label_id]
        path = local_path(RAW_ROOT, label, row_idx)
        parquet_sha = sha256_bytes(row["image"]["bytes"])
        local_sha = file_sha256(path)
        correspondence = decoded_correspondence(row["image"]["bytes"], path)
        if not correspondence["correspondence_pass"]:
            integrity_errors.append({"split": "raw", "row_idx": row_idx, "local_path": str(path), "reason": "decoded_content_mismatch"})
        raw_manifest.append(
            {
                "row_idx": row_idx,
                "label_id": label_id,
                "label": label,
                "original_filename": row["image"]["path"],
                "local_path": path.relative_to(ROOT).as_posix(),
                "sha256": local_sha,
                "parquet_sha256": parquet_sha,
                "byte_sha_match": parquet_sha == local_sha,
                **correspondence,
            }
        )

    family_members: dict[str, list[dict]] = defaultdict(list)
    parse_errors: list[dict] = []
    for row_idx, row in enumerate(augmented_rows):
        label_id = int(row["label"])
        label = LABELS[label_id]
        original_filename = row["image"]["path"]
        match = FAMILY_PATTERN.match(original_filename)
        path = local_path(AUG_ROOT, label, row_idx)
        parquet_sha = sha256_bytes(row["image"]["bytes"])
        local_sha = file_sha256(path)
        correspondence = decoded_correspondence(row["image"]["bytes"], path)
        if not correspondence["correspondence_pass"]:
            integrity_errors.append({"split": "augmented", "row_idx": row_idx, "local_path": str(path), "reason": "decoded_content_mismatch"})
        if not match:
            parse_errors.append({"row_idx": row_idx, "label": label, "original_filename": original_filename})
            continue
        parent_index = int(match.group("parent_index"))
        family_id = f"{label}:parent_{parent_index:04d}"
        record = {
            "row_idx": row_idx,
            "label_id": label_id,
            "label": label,
            "original_filename": original_filename,
            "parent_index": parent_index,
            "family_id": family_id,
            "lineage_evidence": "filename_prefix_confirmed",
            "local_path": path.relative_to(ROOT).as_posix(),
            "sha256": local_sha,
            "parquet_sha256": parquet_sha,
            "byte_sha_match": parquet_sha == local_sha,
            **correspondence,
        }
        augmented_manifest.append(record)
        family_members[family_id].append(record)

    if integrity_errors:
        write_csv(OUTPUT / "integrity_errors.csv", integrity_errors, ["split", "row_idx", "local_path", "reason"])
        raise RuntimeError(f"{len(integrity_errors)} local files do not match parquet bytes")
    if parse_errors:
        write_csv(OUTPUT / "filename_parse_errors.csv", parse_errors, ["row_idx", "label", "original_filename"])
        raise RuntimeError(f"{len(parse_errors)} augmented filenames do not expose the expected family prefix")

    all_paths = [ROOT / row["local_path"] for row in raw_manifest + augmented_manifest]
    embeddings, embedding_metadata = load_or_create_embeddings(all_paths, args.batch_size)
    embedding_by_path = {
        row["local_path"]: embeddings[index]
        for index, row in enumerate(raw_manifest + augmented_manifest)
    }
    sift = SiftCache()
    mapping_rows: list[dict] = []
    candidate_rows: list[dict] = []

    for label_id, label in LABELS.items():
        label_raw = [row for row in raw_manifest if row["label_id"] == label_id]
        sha_groups: dict[str, list[dict]] = defaultdict(list)
        for row in label_raw:
            sha_groups[row["sha256"]].append(row)
        raw_clusters = []
        for cluster_idx, sha in enumerate(sorted(sha_groups)):
            members = sorted(sha_groups[sha], key=lambda row: row["row_idx"])
            representative = members[0]
            raw_clusters.append(
                {
                    "cluster_id": f"{label}:raw_sha_{cluster_idx:04d}",
                    "sha256": sha,
                    "members": members,
                    "representative": representative,
                    "embedding": embedding_by_path[representative["local_path"]],
                }
            )
        raw_matrix = np.stack([cluster["embedding"] for cluster in raw_clusters])
        families = sorted(
            (family_id, members) for family_id, members in family_members.items() if members[0]["label_id"] == label_id
        )
        for family_id, members in families:
            centroid = np.mean([embedding_by_path[row["local_path"]] for row in members], axis=0)
            centroid /= max(float(np.linalg.norm(centroid)), 1e-12)
            similarities = raw_matrix @ centroid
            candidate_indexes = np.argsort(-similarities)[: min(args.resnet_candidates, len(raw_clusters))]
            chosen_members = selected_members(members, args.sift_family_members)
            scored: list[dict] = []
            for resnet_rank, candidate_index in enumerate(candidate_indexes, start=1):
                cluster = raw_clusters[int(candidate_index)]
                raw_path = ROOT / cluster["representative"]["local_path"]
                geometric = [sift.verify(ROOT / row["local_path"], raw_path) for row in chosen_members]
                best_geometric = max(geometric, key=lambda item: (item[0], item[1], item[2]))
                scored.append(
                    {
                        "family_id": family_id,
                        "candidate_raw_cluster": cluster["cluster_id"],
                        "candidate_raw_filenames": " | ".join(row["original_filename"] for row in cluster["members"]),
                        "candidate_raw_rows": " | ".join(str(row["row_idx"]) for row in cluster["members"]),
                        "raw_exact_cluster_size": len(cluster["members"]),
                        "resnet_rank": resnet_rank,
                        "resnet_cosine": float(similarities[int(candidate_index)]),
                        "sift_inliers": int(best_geometric[0]),
                        "sift_good_matches": int(best_geometric[1]),
                        "sift_inlier_fraction": float(best_geometric[2]),
                    }
                )
            scored.sort(
                key=lambda row: (
                    int(row["sift_inliers"]),
                    float(row["sift_inlier_fraction"]),
                    int(row["sift_good_matches"]),
                    float(row["resnet_cosine"]),
                ),
                reverse=True,
            )
            best = scored[0]
            second = scored[1] if len(scored) > 1 else None
            tier = confidence_tier(best, second)
            mapping_rows.append(
                {
                    "family_id": family_id,
                    "label": label,
                    "parent_index": int(members[0]["parent_index"]),
                    "augmented_members": len(members),
                    "family_lineage_status": "filename_prefix_confirmed",
                    "raw_link_confidence_pre_collision_check": tier,
                    "raw_link_confidence": tier,
                    "best_raw_cluster": best["candidate_raw_cluster"],
                    "best_raw_filenames": best["candidate_raw_filenames"],
                    "best_raw_rows": best["candidate_raw_rows"],
                    "best_raw_exact_cluster_size": best["raw_exact_cluster_size"],
                    "raw_filename_ambiguity": "exact_duplicate_cluster" if int(best["raw_exact_cluster_size"]) > 1 else "none",
                    "resnet_rank": best["resnet_rank"],
                    "resnet_cosine": best["resnet_cosine"],
                    "sift_inliers": best["sift_inliers"],
                    "sift_good_matches": best["sift_good_matches"],
                    "sift_inlier_fraction": best["sift_inlier_fraction"],
                    "runner_up_sift_inliers": second["sift_inliers"] if second else 0,
                    "runner_up_raw_filenames": second["candidate_raw_filenames"] if second else "",
                    "interpretation": (
                        "raw link supported"
                        if tier in {"high_geometric_support", "moderate_geometric_support"}
                        else "do not use raw filename link as ground truth"
                    ),
                }
            )
            for final_rank, candidate in enumerate(scored[:5], start=1):
                candidate_rows.append({"geometric_rank": final_rank, **candidate})

    # A raw-content cluster cannot support more confident family links than the
    # number of raw files in that exact-content cluster.  We do not force a
    # global assignment.  Instead, every over-capacity collision is downgraded
    # so the table cannot imply a unique raw crosswalk that the evidence lacks.
    supported_tiers = {"high_geometric_support", "moderate_geometric_support"}
    supported_by_cluster: dict[str, list[dict]] = defaultdict(list)
    for row in mapping_rows:
        if row["raw_link_confidence"] in supported_tiers:
            supported_by_cluster[row["best_raw_cluster"]].append(row)
    for rows in supported_by_cluster.values():
        capacity = max(int(row["best_raw_exact_cluster_size"]) for row in rows)
        if len(rows) > capacity:
            for row in rows:
                row["raw_link_confidence"] = "ambiguous_global_collision"
                row["interpretation"] = "do not use raw filename link as ground truth; multiple supported families select this raw cluster"

    mapping_by_family = {row["family_id"]: row for row in mapping_rows}
    for row in augmented_manifest:
        mapping = mapping_by_family[row["family_id"]]
        row["raw_link_confidence"] = mapping["raw_link_confidence"]
        row["best_raw_cluster"] = mapping["best_raw_cluster"]
        row["best_raw_filenames"] = mapping["best_raw_filenames"]

    consistency_rows: list[dict] = []
    for family_id, members in sorted(family_members.items()):
        matrix = np.stack([embedding_by_path[row["local_path"]] for row in members])
        centroid = matrix.mean(axis=0)
        centroid /= max(float(np.linalg.norm(centroid)), 1e-12)
        member_cosines = matrix @ centroid
        pairwise = matrix @ matrix.T
        upper = pairwise[np.triu_indices(len(matrix), 1)]
        consistency_rows.append(
            {
                "family_id": family_id,
                "label": members[0]["label"],
                "members": len(members),
                "mean_member_to_centroid_cosine": float(member_cosines.mean()),
                "minimum_member_to_centroid_cosine": float(member_cosines.min()),
                "mean_pairwise_cosine": float(upper.mean()),
                "minimum_pairwise_cosine": float(upper.min()),
            }
        )

    integrity_fields = [
        "parquet_sha256", "byte_sha_match", "shape_match", "pixel_mae", "pixel_max_abs_error", "dhash_hamming",
        "correspondence_pass",
    ]
    raw_fields = ["row_idx", "label_id", "label", "original_filename", "local_path", "sha256", *integrity_fields]
    aug_fields = [
        "row_idx", "label_id", "label", "original_filename", "parent_index", "family_id", "lineage_evidence",
        "local_path", "sha256", *integrity_fields, "raw_link_confidence", "best_raw_cluster", "best_raw_filenames",
    ]
    mapping_fields = list(mapping_rows[0])
    candidate_fields = list(candidate_rows[0])
    write_csv(OUTPUT / "raw_parquet_manifest.csv", raw_manifest, raw_fields)
    write_csv(OUTPUT / "augmented_family_manifest.csv", augmented_manifest, aug_fields)
    write_csv(OUTPUT / "family_to_raw_mapping.csv", mapping_rows, mapping_fields)
    write_csv(OUTPUT / "top_candidate_scores.csv", candidate_rows, candidate_fields)
    write_csv(OUTPUT / "family_feature_consistency.csv", consistency_rows, list(consistency_rows[0]))

    family_counts_by_label = Counter(row["label"] for row in mapping_rows)
    raw_counts_by_label = Counter(row["label"] for row in raw_manifest)
    tier_counts = Counter(row["raw_link_confidence"] for row in mapping_rows)
    family_size_counts = {
        label: dict(sorted(Counter(len(members) for family_id, members in family_members.items() if members[0]["label"] == label).items()))
        for label in LABELS.values()
    }
    feature_consistency_by_label = {}
    for label in LABELS.values():
        label_rows = [row for row in consistency_rows if row["label"] == label]
        mean_pairwise = np.array([row["mean_pairwise_cosine"] for row in label_rows])
        minimum_pairwise = np.array([row["minimum_pairwise_cosine"] for row in label_rows])
        feature_consistency_by_label[label] = {
            "median_family_mean_pairwise_cosine": float(np.median(mean_pairwise)),
            "p05_family_mean_pairwise_cosine": float(np.quantile(mean_pairwise, 0.05)),
            "median_family_minimum_pairwise_cosine": float(np.median(minimum_pairwise)),
            "p05_family_minimum_pairwise_cosine": float(np.quantile(minimum_pairwise, 0.05)),
        }
    summary = {
        "parquet_integrity": {
            "raw_rows": len(raw_manifest),
            "augmented_rows": len(augmented_manifest),
            "local_vs_parquet_byte_sha256_matches": sum(row["byte_sha_match"] for row in raw_manifest + augmented_manifest),
            "local_vs_parquet_byte_sha256_mismatches_due_to_cached_jpeg_materialization": sum(
                not row["byte_sha_match"] for row in raw_manifest + augmented_manifest
            ),
            "decoded_content_correspondence_failures": 0,
            "maximum_pixel_mae": max(row["pixel_mae"] for row in raw_manifest + augmented_manifest),
            "maximum_dhash_hamming_256": max(row["dhash_hamming"] for row in raw_manifest + augmented_manifest),
            "unparsed_augmented_filenames": 0,
        },
        "family_lineage": {
            "family_count": len(mapping_rows),
            "augmented_files_assigned_to_filename_families": len(augmented_manifest),
            "counts_by_label": dict(family_counts_by_label),
            "family_size_distribution_by_label": family_size_counts,
            "parent_indexes_contiguous_by_label": {
                label: sorted(row["parent_index"] for row in mapping_rows if row["label"] == label)
                == list(range(family_counts_by_label[label]))
                for label in LABELS.values()
            },
            "resnet18_within_family_consistency": feature_consistency_by_label,
        },
        "raw_sources": {
            "counts_by_label": dict(raw_counts_by_label),
            "exact_unique_images": len({row["sha256"] for row in raw_manifest}),
            "exact_duplicate_files": len(raw_manifest) - len({row["sha256"] for row in raw_manifest}),
            "family_minus_raw_file_count": {
                label: family_counts_by_label[label] - raw_counts_by_label[label] for label in LABELS.values()
            },
        },
        "raw_linkage": {
            "method": "ResNet-18 candidate retrieval plus SIFT/RANSAC verification; no forced one-to-one assignment",
            "confidence_tier_counts": dict(tier_counts),
            "global_collision_downgrades": tier_counts["ambiguous_global_collision"],
            "high_or_moderate_supported": sum(
                tier_counts[tier] for tier in ("high_geometric_support", "moderate_geometric_support")
            ),
            "raw_link_is_not_required_for_family_safe_splitting": True,
        },
        "embedding_metadata": embedding_metadata,
        "parameters": {
            "resnet_candidates": args.resnet_candidates,
            "sift_family_members": args.sift_family_members,
            "sift_ratio_test": 0.75,
            "homography_ransac_threshold_pixels": 4.0,
        },
    }
    (OUTPUT / "lineage_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

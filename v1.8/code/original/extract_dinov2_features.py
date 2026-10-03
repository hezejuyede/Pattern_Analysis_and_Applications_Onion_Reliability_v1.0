"""Offline, frozen DINOv2 features on the unchanged v17 image manifests.

Run prepare before run. This script never fits a classifier or downloads assets.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path('D:/论文/SCI投稿汇总/Onion_Deep_Revision_20261003_v18')
OLD = ROOT.parent / 'Onion_Deep_Revision_20261003_v17'
IMAGE_ROOT = Path('C:/Users/lixin/Documents/Codex/2026-09-27/lu/work/onion_crossdomain')
SOURCE = Path('C:/Users/lixin/.cache/torch/hub/facebookresearch_dinov2_main')
SOURCE_COPY = ROOT / 'code' / 'vendor' / 'dinov2_cached_source'
WEIGHT = Path('C:/Users/lixin/.cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth')
WEIGHT_SHA = 'b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9'
MANIFESTS = {
    'onion': OLD / 'onion_inputs/analysis_manifest.csv',
    'potato': OLD / 'independent_task/potato/analysis_manifest.csv',
}
EXPECTED = {
    'onion': ('5a5a31cccaf6b6d7a3fe046f16be33c99a33d00fecdf2d3ab49bb41d3dd7ee72', 3076),
    'potato': ('534e326a0ccf65b24d1daf5fba8a047b05a4f50b008bec56ee5a9ea1681e4d67', 2152),
}
FEATURES = ROOT / 'features'
FREEZE = FEATURES / 'DINOV2_PRE_EXTRACTION_FREEZE.json'


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def dump(p, obj):
    Path(p).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')


def utc():
    return datetime.now(timezone.utc).isoformat()


def require(cond, message):
    if not cond:
        raise RuntimeError(message)


def resolve_image(s):
    p = Path(s)
    return p if p.is_absolute() else IMAGE_ROOT / p


def load_frames():
    import pandas as pd
    frames = {}
    for name, path in MANIFESTS.items():
        expected_sha, count = EXPECTED[name]
        require(sha(path) == expected_sha, f'Manifest hash changed: {name}')
        f = pd.read_csv(path, dtype=str, keep_default_na=False)
        require(len(f) == count, f'Manifest row count changed: {name}')
        for k in ('local_path', 'row_idx', 'sha256'):
            require(f[k].is_unique, f'Duplicate {k} in {name}')
        frames[name] = f
    return frames


def source_files(base):
    return [p for p in sorted(base.rglob('*')) if p.is_file()
            and '__pycache__' not in p.parts and '.git' not in p.parts
            and (p.suffix in {'.py', '.yaml', '.yml', '.toml', '.cfg', '.txt', '.md'} or p.name == 'LICENSE')]


def prepare():
    import numpy as np
    import pandas as pd
    import torch
    import torchvision
    require(not FREEZE.exists(), 'Freeze already exists; no overwriting')
    require(sha(WEIGHT) == WEIGHT_SHA, 'Checkpoint SHA mismatch')
    frames = load_frames()
    FEATURES.mkdir(parents=True, exist_ok=True)
    require(not SOURCE_COPY.exists(), 'Source snapshot already exists')
    source_inventory = []
    for p in source_files(SOURCE):
        rel = p.relative_to(SOURCE)
        dst = SOURCE_COPY / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, dst)
        require(sha(dst) == sha(p), f'Source copy mismatch: {rel}')
        source_inventory.append({'relative_path': rel.as_posix(), 'sha256': sha(dst), 'bytes': dst.stat().st_size})
    dump(FEATURES / 'dinov2_source_file_hashes.json', source_inventory)
    verified = []
    for name, f in frames.items():
        for i, r in f.iterrows():
            p = resolve_image(r.local_path)
            require(p.is_file(), f'Missing image: {p}')
            actual = sha(p)
            require(actual == r.sha256, f'Image hash mismatch: {p}')
            verified.append(dict(dataset=name, position=i, row_idx=r.row_idx,
                                 local_path=r.local_path, sha256=actual, bytes=p.stat().st_size))
    pd.DataFrame(verified).to_csv(FEATURES / 'PRE_EXTRACTION_IMAGE_HASH_VERIFICATION.csv', index=False)
    packages = {p: importlib.metadata.version(p) for p in ('torch', 'torchvision', 'numpy', 'pandas', 'Pillow')}
    config = {
        'frozen_utc': utc(), 'status': 'FROZEN_BEFORE_IMAGE_INFERENCE',
        'model': 'DINOv2 ViT-S/14, no register tokens, LVD-142M pretrained',
        'selection_reason': 'One different pretraining paradigm and stronger fixed-representation sensitivity; not exhaustive SOTA coverage.',
        'chronology': 'Post hoc extension after v17 results, model selected for representation family before v18 feature extraction or classifier outcomes.',
        'script': {'path': str(Path(__file__).resolve()), 'sha256': sha(__file__)},
        'manifests': {n: {'path': str(p), 'sha256': sha(p), 'rows': len(frames[n])} for n, p in MANIFESTS.items()},
        'checkpoint': {'path': str(WEIGHT), 'sha256': WEIGHT_SHA, 'bytes': WEIGHT.stat().st_size,
                       'official_url': 'https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth'},
        'source': {'path': str(SOURCE_COPY), 'provenance': 'Existing official facebookresearch/dinov2 torch-hub cache. Upstream commit unrecorded; exact used source frozen by per-file SHA256 and copied to D.',
                   'inventory': 'dinov2_source_file_hashes.json', 'inventory_sha256': sha(FEATURES / 'dinov2_source_file_hashes.json'),
                   'license': 'Apache-2.0; LICENSE copied with source', 'file_count': len(source_inventory)},
        'transform': {'input': 'PIL RGB conversion', 'resize': 'shortest side 256; bicubic; torchvision antialias=True default',
                      'crop': 'center 224x224', 'tensor': 'ToTensor, float32 scale 0..1',
                      'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225],
                      'basis': 'Copied official dinov2/data/transforms.py make_classification_eval_transform defaults',
                      'output': '384-dimensional final LayerNorm CLS embedding; official eval-mode model.forward output; no unit-length rescaling'},
        'runtime': {'python': platform.python_version(), 'packages': packages, 'torch': torch.__version__, 'torchvision': torchvision.__version__,
                    'device': 'cpu', 'torch_threads': 4, 'interop_threads': 1, 'batch_size': 16, 'workers': 0, 'shuffle': False},
        'protocol': {'eval': True, 'gradients': False, 'fine_tuning': False, 'augmentation': False, 'label_use_for_features': False,
                     'classifier_fits': 0, 'network_access_for_extraction': False, 'outcome_based_exclusions': False},
        'image_hash_check': {'file': 'PRE_EXTRACTION_IMAGE_HASH_VERIFICATION.csv', 'sha256': sha(FEATURES / 'PRE_EXTRACTION_IMAGE_HASH_VERIFICATION.csv'),
                             'rows': len(verified), 'failures': 0},
        'repeat_qa': {'positions': {n: np.linspace(0, len(f)-1, 8, dtype=int).tolist() for n, f in frames.items()},
                      'operation': 'Reload selected images and recompute features as a fresh 8-image batch; compare exact cached row ordering',
                      'atol': 5e-5, 'rtol': 1e-5, 'reason': 'Float32 batching kernels may differ in rounding; record actual maximum differences and exact equality separately.'},
        'limits': ['No assurance that LVD-142M pretraining images are disjoint from the public image archives.',
                   'Feature extraction adds representation sensitivity, not new acquisitions, plant identities, pathology review or new algorithm.'],
    }
    dump(FREEZE, config)
    print(json.dumps({'status': config['status'], 'images_verified': len(verified), 'freeze_sha256': sha(FREEZE)}), flush=True)


def run():
    import numpy as np
    import torch
    from PIL import Image
    from torchvision import transforms
    cfg = json.loads(FREEZE.read_text(encoding='utf-8'))
    require(sha(__file__) == cfg['script']['sha256'], 'Extraction code changed after freeze')
    require(sha(WEIGHT) == cfg['checkpoint']['sha256'], 'Checkpoint changed after freeze')
    for fn, h in [('dinov2_source_file_hashes.json', cfg['source']['inventory_sha256']),
                  ('PRE_EXTRACTION_IMAGE_HASH_VERIFICATION.csv', cfg['image_hash_check']['sha256'])]:
        require(sha(FEATURES / fn) == h, f'Frozen preparation changed: {fn}')
    for entry in json.loads((FEATURES / 'dinov2_source_file_hashes.json').read_text(encoding='utf-8')):
        require(sha(SOURCE_COPY / entry['relative_path']) == entry['sha256'], 'Official source changed after freeze')
    frames = load_frames()
    for n in frames:
        require(not (FEATURES / f'{n}_dinov2_vits14.npz').exists(), f'Refusing output overwrite: {n}')
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(20261003)
    torch.use_deterministic_algorithms(True)
    model = torch.hub.load(str(SOURCE_COPY), 'dinov2_vits14', source='local', pretrained=False)
    state = torch.load(WEIGHT, map_location='cpu', weights_only=True)
    load_result = model.load_state_dict(state, strict=True)
    model.eval()
    model.requires_grad_(False)
    transform = transforms.Compose([transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
                                    transforms.CenterCrop(224), transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    started = utc()
    outputs = {}
    qa_rows = []
    for name, frame in frames.items():
        pieces = []
        t = time.perf_counter()
        def batch_tensor(positions):
            tensors = []
            for j in positions:
                row = frame.iloc[int(j)]
                p = resolve_image(row.local_path)
                require(sha(p) == row.sha256, f'Image bytes changed after freeze: {p}')
                with Image.open(p) as img:
                    tensors.append(transform(img.convert('RGB')))
            return torch.stack(tensors)
        with torch.inference_mode():
            for start in range(0, len(frame), 16):
                positions = range(start, min(start+16, len(frame)))
                values = model(batch_tensor(positions)).cpu().numpy().astype(np.float32)
                require(values.shape == (len(positions), 384), f'Unexpected batch shape {values.shape}')
                require(np.isfinite(values).all(), 'Nonfinite feature')
                pieces.append(values)
                if start % 256 == 0 or start+16 >= len(frame):
                    print(json.dumps({'dataset': name, 'processed': min(start+16, len(frame)), 'total': len(frame),
                                      'elapsed_seconds': round(time.perf_counter()-t, 2)}), flush=True)
            matrix = np.concatenate(pieces, axis=0)
            ids = cfg['repeat_qa']['positions'][name]
            repeated = model(batch_tensor(ids)).cpu().numpy().astype(np.float32)
        expected = matrix[ids]
        atol, rtol = cfg['repeat_qa']['atol'], cfg['repeat_qa']['rtol']
        require(np.allclose(repeated, expected, atol=atol, rtol=rtol), 'Repeat extraction outside frozen float32 tolerance')
        for k, j in enumerate(ids):
            qa_rows.append({'dataset': name, 'position': j, 'row_idx': frame.iloc[j].row_idx,
                            'local_path': frame.iloc[j].local_path,
                            'max_abs_difference': float(np.max(np.abs(expected[k]-repeated[k]))),
                            'exact_equal': bool(np.array_equal(expected[k], repeated[k])),
                            'allclose_frozen_tolerance': True})
        dest = FEATURES / f'{name}_dinov2_vits14.npz'
        np.savez_compressed(dest, features=matrix, paths=frame.local_path.to_numpy(dtype=str),
                            row_idx=frame.row_idx.to_numpy(dtype=str), image_sha256=frame.sha256.to_numpy(dtype=str))
        with np.load(dest, allow_pickle=False) as a:
            require(a['features'].shape == (len(frame), 384), 'Saved feature shape mismatch')
            require(a['features'].dtype == np.float32, 'Saved feature dtype mismatch')
            require(np.isfinite(a['features']).all(), 'Saved features nonfinite')
            require(np.array_equal(a['paths'], frame.local_path.to_numpy(dtype=str)), 'Saved path order mismatch')
            require(np.array_equal(a['row_idx'], frame.row_idx.to_numpy(dtype=str)), 'Saved row order mismatch')
            require(np.array_equal(a['image_sha256'], frame.sha256.to_numpy(dtype=str)), 'Saved hash order mismatch')
            require(np.array_equal(a['features'], matrix), 'NPZ feature roundtrip changed values')
        elapsed = time.perf_counter()-t
        outputs[name] = {'file': str(dest), 'sha256': sha(dest), 'bytes': dest.stat().st_size,
                         'rows': len(frame), 'dimensions': 384, 'dtype': 'float32', 'finite': True,
                         'paths_order_exact_match': True, 'row_idx_order_exact_match': True, 'image_hash_order_exact_match': True,
                         'elapsed_seconds_including_repeat_qa_and_save': elapsed, 'images_per_second': len(frame)/elapsed,
                         'repeated_positions': ids, 'repeat_max_abs_difference': float(np.max(np.abs(expected-repeated))),
                         'repeat_atol': atol, 'repeat_rtol': rtol, 'status': 'PASS'}
        dump(FEATURES / f'{name}_dinov2_vits14.json', outputs[name])
        print(json.dumps(outputs[name]), flush=True)
    import pandas as pd
    pd.DataFrame(qa_rows).to_csv(FEATURES / 'DINOV2_REPEAT_EXTRACTION_QA.csv', index=False)
    completion = {'status': 'PASS_FEATURE_EXTRACTION_AND_ALIGNMENT_QA', 'started_utc': started, 'completed_utc': utc(),
                  'freeze_sha256': sha(FREEZE), 'script_sha256': sha(__file__), 'checkpoint_sha256': sha(WEIGHT),
                  'strict_checkpoint_load': str(load_result), 'model_parameters': sum(p.numel() for p in model.parameters()),
                  'outputs': outputs, 'unique_manifest_images': sum(len(x) for x in frames.values()),
                  'repeated_qa_images': len(qa_rows), 'classifier_fits': 0, 'downloads': 0,
                  'repeat_qa_file_sha256': sha(FEATURES / 'DINOV2_REPEAT_EXTRACTION_QA.csv'),
                  'limits': cfg['limits']}
    dump(FEATURES / 'DINOV2_EXTRACTION_COMPLETE.json', completion)
    print(json.dumps({'status': completion['status'], 'unique_manifest_images': completion['unique_manifest_images']}), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mode', choices=['prepare', 'run'], required=True)
    args = ap.parse_args()
    (prepare if args.mode == 'prepare' else run)()

"""Optional live provenance refresh with an explicit new output directory.

The released factual index, not a later website response, fixes the analyzed
snapshot. This wrapper preserves the original recovery implementation unchanged.
"""
import argparse
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error('Use a new or empty directory; existing frozen evidence must not be overwritten.')
    source=Path(__file__).with_name('recover_tom_public_region_metadata.py')
    spec=importlib.util.spec_from_file_location('archived_region_recovery',source)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.OUT=args.output_dir.resolve()
    module.main()
    path=module.OUT/'RECOVERY_SUMMARY.json'
    report=json.loads(path.read_text(encoding='utf-8'))
    report['date']=datetime.now(timezone.utc).date().isoformat()
    report['live_refresh_not_original_frozen_snapshot']=True
    path.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')

if __name__=='__main__':main()

"""Restore verified feature bytes after session loss, without editing trained config."""
import argparse
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tools')]


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('dataset',choices=('yc2','anet'));args=parser.parse_args()
    if os.environ.get('DDVC_RUN_VARIANT')!='baseline_quality':raise ValueError('Use baseline_quality.sh restore-data DATASET')
    import c1_pipeline as pipeline
    import c1_rclone as helpers
    work=ROOT.parent;path=pipeline.config_path(work,args.dataset)
    before=pipeline.sha(path);cfg=pipeline.check(path,require_features=False)
    drive=helpers.Drive(Path('/marimo/secrets/rclone.conf'),'kltnbackup','KLTNRUNS')
    drive.initialize()
    if not helpers.pull_data(drive,args.dataset):
        raise FileNotFoundError('No verified shared data bundle. Inspect the Drive data cache; do not edit the trained config.')
    source=helpers.base_feature(args.dataset)
    pipeline.stage_file(source,cfg['visual_feature_folder'][0])
    pipeline.check(path)
    if pipeline.sha(path)!=before:raise RuntimeError('Trained config changed during feature restoration')
    print('BASELINE_FEATURES_RESTORED_CONFIG_UNCHANGED',args.dataset,flush=True)


if __name__=='__main__':main()

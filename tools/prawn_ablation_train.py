"""Run one trainable ablation; restore only its own exact-code recovery namespace."""
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import c1_rclone as transport
import c1_pipeline as pipeline
import c1_rerank
import prawn_content_mask as chain
from misc.prawn_ablation import namespace, variant
from prawn_results_viewer import Store


def main():
    dataset=sys.argv[1];work=ROOT.parent
    if variant() not in ('no_content','joint'):raise ValueError('This entry point does not retrain Full')
    config=Path('/marimo/secrets/rclone.conf')
    transport.restore_auth(Path('/marimo/KLTN_DRIVE_PRIVATE.env'),config,'kltnbackup')
    drive=transport.Drive(config,'kltnbackup','KLTNRUNS');store=Store(drive,work/'restore_index')
    state_path=transport.session_path(dataset)
    work.mkdir(parents=True,exist_ok=True)
    if not state_path.is_file():
        candidates=[s for s in store.snapshots(namespace('generator'),dataset)
                    if s['ready'].get('source_commit')==transport.revision() and s['ready'].get('work')==str(work)
                    and s['ready'].get('variant')==transport.RUN_VARIANT]
        if candidates:
            snap=candidates[0]
            state=dict(dataset=dataset,run_id=snap['ready']['run_id'],work=str(work),source_commit=transport.revision(),
                       remote='kltnbackup',root='KLTNRUNS',status='RESTORED',variant=transport.RUN_VARIANT)
            transport.Recovery(drive,state).restore()
            run=work/'runs'/pipeline.run_name(dataset)
            info=json.loads((run/'info.json').read_text(encoding='utf8'))
            if info.get('last',{}).get('epoch',-1)>=pipeline.epochs(dataset)-1:
                state['status']='COMPLETE_BACKED_UP'
            transport.write_json(state_path,state)
            print('ABLATION_GENERATOR_RESTORED',variant(),dataset,state['run_id'],flush=True)
    if state_path.is_file():
        state=json.loads(state_path.read_text(encoding='utf8'))
        if state['source_commit']!=transport.revision() or state.get('variant')!=transport.RUN_VARIANT:
            raise ValueError('Local generator belongs to a different ablation/source')
        if state.get('status')=='COMPLETE_BACKED_UP':
            # Restore the runtime and immutable data without mutating a completed config.
            if not (work/'provenance/cuda_ready').is_file():
                subprocess.run(['bash',str(ROOT/'tools/c1_molab.sh'),'setup'],cwd=ROOT,check=True)
            if not transport.pull_data(drive,dataset):raise FileNotFoundError('Completed generator needs its shared data cache')
            target=work/'prepared'/pipeline.run_name(dataset)/'data'/dataset/'features/clipvitl14.pth'
            if not target.is_file():pipeline.stage_file(transport.base_feature(dataset),target)
            pipeline.check(pipeline.config_path(work,dataset))
            if variant()=='no_content' and not chain.state_path(dataset,quality=True).is_file():
                head_candidates=store.snapshots(namespace('quality'),dataset)
                for qs in head_candidates:
                    ready=qs['ready'];name=f'provenance/rerank_{dataset}.json'
                    if ready.get('source_commit')!=transport.revision() or ready.get('base_work')!=str(work) or name not in ready['files']:
                        continue
                    state_q=drive.read_json(drive.path(qs['remote']+'/'+name))
                    if state_q.get('base_c1_drive_session',{}).get('run_id')!=state['run_id']:continue
                    args=SimpleNamespace(work=chain.QUALITY,base=work,dataset=dataset,run_id=ready['run_id'])
                    chain.QUALITY.mkdir(parents=True,exist_ok=True)
                    c1_rerank.restore(args)
                    print('ABLATION_QUALITY_RESTORED',dataset,ready['run_id'],flush=True)
                    break
    (work/'logs').mkdir(exist_ok=True)
    chain.worker(SimpleNamespace(dataset=dataset,resume=True))


if __name__=='__main__':main()

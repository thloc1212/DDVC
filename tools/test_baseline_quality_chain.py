"""The second stage starts only after the baseline and its backup complete."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parent))
import baseline_quality_chain as chain


class ChainTests(unittest.TestCase):
    def setup_state(self,root,dataset='yc2'):
        state=dict(dataset=dataset,work=str(root),variant='baseline_quality',status='FAILED_BACKUP_PENDING',
                   source_commit=next(iter(chain.SETUP_FIX_SOURCES)),worker_pid=123)
        path=root/'provenance'/('drive_session_'+dataset+'.json');path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(state),encoding='utf8')
        return path,state

    def test_failed_setup_migration_preserves_evidence_without_changing_recipe(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);path,old=self.setup_state(root)
            with patch.object(chain,'WORK',root),patch.object(chain,'alive',return_value=False):
                chain.prepare_setup_retry('yc2','new-smoke-fix')
            actual=json.loads(path.read_text())
            self.assertEqual(actual['source_commit'],'new-smoke-fix')
            self.assertEqual(actual['status'],'SETUP_RETRY_READY')
            history=list((root/'provenance/setup_retry_history').glob('*.json'))
            self.assertEqual(json.loads(history[0].read_text()),old)
            self.assertFalse((root/'runs').exists())

    def test_setup_retry_rejects_started_training_and_unapproved_sources(self):
        for condition in ('run','quality','source','alive','status'):
            with self.subTest(condition=condition),tempfile.TemporaryDirectory() as temp:
                root=Path(temp);path,state=self.setup_state(root)
                if condition=='run':(root/'runs/yc2_baseline_e15_b1').mkdir(parents=True)
                if condition=='quality':
                    p=root/'frozen_quality/provenance/rerank_yc2.json';p.parent.mkdir(parents=True);p.write_text('{}')
                if condition=='source':state['source_commit']='unknown-source'
                if condition=='status':state['status']='TRAINING'
                path.write_text(json.dumps(state));before=path.read_bytes()
                with patch.object(chain,'WORK',root),patch.object(chain,'alive',return_value=condition=='alive'):
                    with self.assertRaises((ValueError,RuntimeError)):chain.prepare_setup_retry('yc2','new-smoke-fix')
                self.assertEqual(path.read_bytes(),before)

    def test_post_cuda_import_failure_retries_without_removing_compiled_state(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);path,state=self.setup_state(root,'anet')
            state['source_commit']='80c29717b43268423c30150f01f999f7a43eb647'
            path.write_text(json.dumps(state));marker=root/'provenance/cuda_ready';marker.touch()
            binary=root/'source/ddvc/ops/compiled.so';binary.parent.mkdir(parents=True);binary.write_bytes(b'keep-this-operator')
            with patch.object(chain,'WORK',root),patch.object(chain,'alive',return_value=False):
                chain.prepare_setup_retry('anet','new-import-fix')
            self.assertTrue(marker.exists());self.assertEqual(binary.read_bytes(),b'keep-this-operator')
            self.assertEqual(json.loads(path.read_text())['source_commit'],'new-import-fix')

    def test_setup_retry_keeps_the_two_stage_chain(self):
        with patch.object(chain.subprocess,'run') as run,patch.object(chain,'wait_stage',side_effect=RuntimeError('stop after launch')):
            with self.assertRaises(RuntimeError):chain.worker('anet',retry=True)
            self.assertEqual(run.call_args.args[0][-3:],['start','anet','--retry'])

    def test_wait_rejects_a_dead_worker_without_verified_completion(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);pid=root/'pid';pid.write_text('123');state=root/'state.json'
            with patch.object(chain,'alive',return_value=False):
                for status in ('TRAINING','FAILED_BACKED_UP','TRAIN_COMPLETE_BACKUP_PENDING'):
                    state.write_text(json.dumps(dict(status=status)))
                    with self.assertRaises(RuntimeError):chain.wait_stage(pid,state,'c1_rclone.py')
                state.write_text(json.dumps(dict(status='COMPLETE_BACKED_UP')))
                chain.wait_stage(pid,state,'c1_rclone.py')

    def test_failed_baseline_does_not_launch_quality(self):
        with patch.object(chain.subprocess,'run') as run,patch.object(chain,'wait_stage',side_effect=RuntimeError('failed')):
            with self.assertRaises(RuntimeError):chain.worker('anet')
            self.assertEqual(run.call_count,1)
            self.assertEqual(run.call_args.args[0][-2:],['start','anet'])

    def test_two_stages_are_ordered_and_results_read_only_after_both_backups(self):
        events=[]
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);summary=root/'frozen_quality/runs/yc2/evaluation/summary.tsv'
            summary.parent.mkdir(parents=True);summary.write_text('method\tCIDEr\n')
            def run(cmd,**kw):events.append(cmd[-2:])
            def wait(*args):events.append(['verified',args[-1]])
            with patch.object(chain,'WORK',root),patch.object(chain.subprocess,'run',side_effect=run),patch.object(chain,'wait_stage',side_effect=wait):
                chain.worker('yc2')
            self.assertEqual(events,[['start','yc2'],['verified','c1_rclone.py'],['quality','yc2'],['verified','c1_rerank.py']])


if __name__=='__main__':unittest.main()

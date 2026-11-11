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

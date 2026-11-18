"""Offline tests for latest linked snapshot selection and non-destructive moves."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parent))
import resume_caption_matching_drive as recovery


class ResumeTests(unittest.TestCase):
    def test_missing_code_clones_fixed_tag_and_existing_code_is_not_pulled(self):
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/'work/source'
            with patch.object(recovery.subprocess,'run') as run:
                recovery.bootstrap_source(source)
                self.assertIn(recovery.TRAIN_TAG,run.call_args.args[0])
                self.assertIn('clone',run.call_args.args[0])
            (source/'tools').mkdir(parents=True)
            (source/'tools/prawn_caption_match.sh').write_text('present')
            with patch.object(recovery.subprocess,'run') as run:
                recovery.bootstrap_source(source)
                run.assert_not_called()

    def test_latest_run_and_its_own_quality(self):
        generators=[dict(run_id='20261118_new',created_ns=2),dict(run_id='20261117_old',created_ns=1)]
        heads=[dict(run_id='quality_wrong',base_id='20261117_old'),dict(run_id='quality_correct',base_id='20261118_new')]
        gen,head=recovery.choose(generators,heads)
        self.assertEqual((gen['run_id'],head['run_id']),('20261118_new','quality_correct'))
        gen,head=recovery.choose(generators,heads,'20261117_old')
        self.assertEqual(head['run_id'],'quality_wrong')
        with self.assertRaises(FileNotFoundError):recovery.choose([],heads)

    def test_no_unrelated_quality_is_imported(self):
        gen,head=recovery.choose([dict(run_id='new')],[dict(run_id='other',base_id='old')])
        self.assertIsNone(head)

    def test_last_validation_is_not_a_finished_generator(self):
        self.assertFalse(recovery.generator_complete(dict(format_version=2,phase='validation_pending',next_epoch=5),6))
        self.assertTrue(recovery.generator_complete(dict(format_version=2,phase='epoch_complete',next_epoch=6),6))
        self.assertFalse(recovery.generator_complete(dict(format_version=2,phase='epoch_complete',next_epoch=5),6))

    def test_preserve_keeps_source_and_all_existing_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            work=Path(directory);run='anet_caption_match_e6_b1'
            paths=['source/keep.py','runs/'+run+'/model-last.pth','logs/old.log','provenance/runtime_sources/x.py',
                   'frozen_quality/runs/anet/head-last.pt']
            for name in paths:
                p=work/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(name)
            saved=recovery.preserve_conflicts(work,'anet',run)
            self.assertEqual((work/'source/keep.py').read_text(),'source/keep.py')
            for name in paths[1:]:self.assertEqual((saved/name).read_text(),name)
            self.assertFalse((work/'runs'/run).exists())

    def test_snapshot_scanner_rejects_wrong_source_and_incomplete_run(self):
        parent='drive:KLTNRUNS/runs/prawn_caption_match_generator/anet'
        work=Path('/marimo/PRAWN_Caption_Match')
        checkpoint='runs/anet_caption_match_e6_b1/model-last.pth'
        class Drive:
            def call(self,*args,**kwargs):
                path=args[1]
                if path==parent:return 'good/\nwrong/\nempty/\n'
                if path.endswith('/empty'):return 'live_logs/\n'
                if path.endswith('/recovery'):return 'slot-0/\n'
                return 'recovery/\n'
            def optional_json(self,path):
                run=path[len(parent)+1:].split('/')[0]
                return dict(dataset='anet',run_id=run,source_commit=recovery.TRAIN_COMMIT if run=='good' else 'wrong',
                            work=str(work),variant='prawn-caption-matching-b1',created_ns=3,files={checkpoint:{}})
        rows=recovery.snapshots(Drive(),parent,'anet',work)
        self.assertEqual([r['run_id'] for r in rows],['good'])


if __name__=='__main__':unittest.main()

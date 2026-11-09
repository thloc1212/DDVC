"""Offline tests of credential privacy, snapshot pinning and verified slot recovery."""
import base64
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('c1_rclone', Path(__file__).with_name('c1_rclone.py'))
q = importlib.util.module_from_spec(spec)
spec.loader.exec_module(q)


class FileDrive:
    """A local transport double; tests never invoke rclone or contact Drive."""
    def __init__(self, root):
        self.root = Path(root)
        self.before_copy = None

    def path(self, relative=''):
        return str(self.root / relative)

    def call(self, command, *args, **kwargs):
        if command == 'mkdir':
            Path(args[0]).mkdir(parents=True, exist_ok=True)
        elif command == 'lsf':
            if '--dirs-only' in args:
                return '\n'.join(p.name+'/' for p in Path(args[0]).iterdir() if p.is_dir())
            return '\n'.join(p.name for p in Path(args[0]).iterdir())
        elif command == 'check':
            source, dest = map(Path, args[:2])
            for path in source.rglob('*'):
                if path.is_file():
                    assert q.digest(path) == q.digest(dest / path.relative_to(source))
        else:
            raise AssertionError(command)

    def read_json(self, path):
        return json.loads(Path(path).read_text())

    def optional_json(self, path):
        return self.read_json(path) if Path(path).is_file() else None

    def put_json(self, path, value):
        q.write_json(path, value)

    def copy(self, source, destination, files_from=None):
        if self.before_copy:
            callback, self.before_copy = self.before_copy, None
            callback()
        source, destination = Path(source), Path(destination)
        selected = (Path(files_from).read_text().splitlines() if files_from else
                    [p.relative_to(source).as_posix() for p in source.rglob('*') if p.is_file()])
        for relative in selected:
            original, target = source / relative, destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(original, target)


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.work, self.cloud = root / 'work', root / 'cloud'
        self.work.mkdir()
        self.work_patch = patch.object(q, 'WORK', self.work)
        self.work_patch.start()
        self.rev_patch = patch.object(q, 'revision', return_value='unit-test-commit')
        self.rev_patch.start()
        self.drive = FileDrive(self.cloud)
        self.state = dict(dataset='anet', run_id='unique_anet_test')
        self.run = self.work / 'runs' / q.run_name('anet')
        self.run.mkdir(parents=True)
        self.last = self.run / 'model-last.pth'
        self.last.write_bytes(b'old-checkpoint-optimizer-rng')
        q.write_json(self.run / 'info.json', {'last': {'epoch': 0}})
        prepared = self.work / 'prepared' / q.run_name('anet')
        prepared.mkdir(parents=True)
        (prepared / 'config.yml').write_text('epoch: 6\n')
        (prepared / 'data/features').mkdir(parents=True)
        (prepared / 'data/features/clipvitl14.pth').write_bytes(b'large-features-not-in-run-backup')
        (self.work / 'logs').mkdir()
        (self.work / 'logs/anet.drive.log').write_text('progress\n')
        (self.work / 'logs/yc2.drive.log').write_text('other dataset\n')
        (self.work / 'logs/secret.env').write_text('RCLONE_CONFIG_B64=DO_NOT_UPLOAD')

    def tearDown(self):
        self.rev_patch.stop()
        self.work_patch.stop()
        self.temp.cleanup()

    def atomic_checkpoint(self, content):
        temp = self.last.with_suffix('.tmp')
        temp.write_bytes(content)
        os.replace(temp, self.last)

    def clear_local(self):
        for folder in ('runs', 'prepared', 'logs'):
            shutil.rmtree(self.work / folder)

    def test_checkpoint_inode_is_pinned_and_secrets_features_other_runs_excluded(self):
        recovery = q.Recovery(self.drive, self.state)
        self.drive.before_copy = lambda: self.atomic_checkpoint(b'newer-checkpoint')
        self.assertTrue(recovery.sync())
        slot = Path(recovery.remote) / 'recovery/slot-0'
        self.assertEqual((slot / self.last.relative_to(self.work)).read_bytes(), b'old-checkpoint-optimizer-rng')
        manifest = self.drive.read_json(slot / 'READY.json')
        names = '\n'.join(manifest['files'])
        self.assertNotIn('secret.env', names)
        self.assertNotIn('clipvitl14.pth', names)
        self.assertNotIn('yc2.drive.log', names)
        self.assertEqual(self.last.read_bytes(), b'newer-checkpoint')

    def test_corrupt_new_slot_falls_back_to_old_without_partial_restore(self):
        recovery = q.Recovery(self.drive, self.state)
        recovery.sync()
        self.atomic_checkpoint(b'epoch-two')
        recovery.sync()
        newest = Path(recovery.remote) / 'recovery/slot-1' / self.last.relative_to(self.work)
        newest.write_bytes(b'broken-upload')
        self.clear_local()
        metadata = recovery.restore()
        self.assertEqual(metadata['slot'], 0)
        self.assertEqual(self.last.read_bytes(), b'old-checkpoint-optimizer-rng')

    def test_failed_upload_does_not_publish_new_ready_or_latest(self):
        recovery = q.Recovery(self.drive, self.state)
        recovery.sync()
        self.atomic_checkpoint(b'epoch-two')
        original_copy = self.drive.copy
        def fail(*args, **kwargs):
            original_copy(*args, **kwargs)
            raise RuntimeError('network interrupted')
        with patch.object(self.drive, 'copy', side_effect=fail):
            with self.assertRaises(RuntimeError):
                recovery.sync()
        parent = Path(recovery.remote) / 'recovery'
        self.assertEqual(self.drive.read_json(parent / 'LATEST.json')['slot'], 0)
        self.assertFalse((parent / 'slot-1/READY.json').exists())

    def test_path_traversal_or_credentials_cannot_enter_recovery(self):
        for value in ('../secret.env', '/etc/passwd', 'runs/../../secret.env', 'logs\\secret.env', 'logs/a\nb'):
            with self.assertRaises(ValueError):
                q.relative_path(value)
        self.assertFalse(q.recovery_allowed('provenance/secrets.env', 'anet'))
        self.assertFalse(q.recovery_allowed('runs/other/model-last.pth', 'anet'))

    def test_restore_checks_all_hashes_before_exposing_any_run(self):
        recovery = q.Recovery(self.drive, self.state)
        recovery.sync()
        remote = Path(recovery.remote) / 'recovery/slot-0'
        (remote / self.last.relative_to(self.work)).write_bytes(b'corrupt')
        self.clear_local()
        with self.assertRaises(FileNotFoundError):
            recovery.restore()
        self.assertFalse(self.last.exists())
        self.assertFalse((self.work / 'prepared').exists())

    def test_data_cache_hash_verified_and_pipeline_identity_bound(self):
        base = q.base_path('anet')
        (base / 'data/anet/features').mkdir(parents=True)
        (base / 'config.yml').write_text('dataset: anet\n')
        (base / 'data/anet/features/clipvitl14.pth').write_bytes(b'cached-real-feature-placeholder')
        with patch.object(q, 'data_identity', return_value={'recipe': 'unchanged'}):
            q.publish_data(self.drive, 'anet')
            shutil.rmtree(base)
            self.assertTrue(q.pull_data(self.drive, 'anet'))
            self.assertEqual((base / 'config.yml').read_text(), 'dataset: anet\n')

    def test_yc2_cache_keeps_original_full_feature_filename(self):
        base = q.base_path('yc2')
        feature = base / 'data/yc2/features/clipvitl14_full.pth'
        feature.parent.mkdir(parents=True)
        feature.write_bytes(b'yc2-feature-map')
        (base / 'config.yml').write_text('visual_feature_folder:\n  - ' + feature.as_posix() + '\n')
        self.assertEqual(q.base_feature('yc2'), feature)
        with patch.object(q, 'data_identity', return_value={'recipe': 'unchanged'}):
            q.publish_data(self.drive, 'yc2')
            shutil.rmtree(base)
            self.assertTrue(q.pull_data(self.drive, 'yc2'))
            self.assertEqual(q.base_feature('yc2').read_bytes(), b'yc2-feature-map')

    def test_local_prepared_data_does_not_hide_a_failed_remote_cache_publish(self):
        base = q.base_path('anet')
        base.mkdir(parents=True)
        (base / 'config.yml').write_text('dataset: anet\n')
        self.assertFalse(q.pull_data(self.drive, 'anet'))

    def test_qts_cache_import_verifies_bytes_and_relocates_data_only(self):
        import yaml
        old='/marimo/DDVC_QTS/prepared/anet_qts_base_assets'
        parent=self.cloud/'data/anet/qts-base-assets-v1'
        generation='a'*32;source=parent/'generations'/generation
        source.mkdir(parents=True)
        cfg=dict(dataset='anet',bank_type=['anet'],bank_path=old+'/bank',
                 train_caption_file=old+'/train.json',val_caption_file=old+'/val.json',
                 dict_file=old+'/vocabulary.json',visual_feature_folder=[old+'/clipvitl14.pth'])
        for file,value in [('train.json','{"train":{}}'),('val.json','{"val":{}}'),('vocabulary.json','{}')]:
            (source/file).write_text(value,encoding='utf8')
        cfg['data_signature']=q.hashlib.sha256(b''.join((source/file).read_bytes() for file in
                                         ('train.json','val.json','vocabulary.json'))).hexdigest()
        (source/'bank').mkdir();(source/'bank/emb.npy').write_bytes(b'train-only-bank')
        (source/'clipvitl14.pth').write_bytes(b'features-not-model-weights')
        (source/'config.yml').write_text(yaml.safe_dump(cfg),encoding='utf8')
        original=q.inventory(source)
        q.write_json(source/'READY.json',dict(protocol='qts-base-assets-v1',dataset='anet',
                                            work='/marimo/DDVC_QTS',files=original))
        q.write_json(parent/'LATEST.json',dict(generation=generation))
        self.assertTrue(q.import_qts_data(self.drive,'anet'))
        dest=q.base_path('anet');actual=yaml.safe_load((dest/'config.yml').read_text())
        self.assertTrue(Path(actual['train_caption_file']).is_relative_to(dest))
        self.assertEqual(q.base_feature('anet').read_bytes(),b'features-not-model-weights')
        self.assertEqual((dest/'bank/emb.npy').read_bytes(),b'train-only-bank')
        self.assertFalse((self.work/'runs/anet_qts_set_e6_b1').exists())
        shutil.rmtree(dest);(source/'clipvitl14.pth').write_bytes(b'corrupted')
        with self.assertRaises(ValueError):q.import_qts_data(self.drive,'anet')
        self.assertFalse(dest.exists())


class AuthTests(unittest.TestCase):
    def test_provider_error_is_visible_but_tokens_and_client_secret_are_redacted(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / 'rclone.conf'
            config.write_text('[kltnbackup]\ntype=drive\nclient_secret=FAKE_SECRET\n'
                              'token={"access_token":"FAKE_ACCESS","refresh_token":"1//FAKE_REFRESH"}\n')
            stderr = ('Google Drive API has not been used in project before or it is disabled. '
                      'client=FAKE_SECRET access=FAKE_ACCESS refresh=1%2F%2FFAKE_REFRESH')
            result = q.subprocess.CompletedProcess([], 1, stdout='DO_NOT_PRINT_STDOUT', stderr=stderr)
            with patch.object(q.subprocess, 'run', return_value=result):
                with self.assertRaises(RuntimeError) as error:
                    q.Drive(config, 'kltnbackup', 'KLTNRUNS').call('lsf', 'kltnbackup:', quiet=True)
            message = str(error.exception)
            self.assertIn('API has not been used', message)
            self.assertNotIn('FAKE_SECRET', message)
            self.assertNotIn('FAKE_ACCESS', message)
            self.assertNotIn('FAKE_REFRESH', message)
            self.assertNotIn('DO_NOT_PRINT_STDOUT', message)

    def test_auth_never_prints_secret_and_preserves_refreshed_config(self):
        raw = b'[kltnbackup]\ntype = drive\nclient_id = fake-client\nclient_secret = FAKE_SECRET\ntoken = {"access_token":"FAKE_ACCESS","refresh_token":"FAKE_REFRESH"}\n'
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()) as log:
            env = Path(tmp) / 'private.env'
            env.write_text('RCLONE_CONFIG_B64=' + base64.b64encode(raw).decode() + '\n')
            target = Path(tmp) / 'secrets/rclone.conf'
            q.restore_auth(env, target, 'kltnbackup')
            newer = target.read_text().replace('FAKE_ACCESS', 'REFRESHED_ACCESS')
            target.write_text(newer)
            q.restore_auth(env, target, 'kltnbackup')
            self.assertIn('REFRESHED_ACCESS', target.read_text())
            self.assertNotIn('FAKE_SECRET', log.getvalue())
            self.assertNotIn('FAKE_REFRESH', log.getvalue())
            if os.name != 'nt':
                self.assertEqual(target.stat().st_mode & 0o777, 0o600)

    def test_invalid_private_input_is_rejected_without_echo(self):
        with self.assertRaisesRegex(ValueError, 'authenticated Drive'):
            q.parse_auth(b'[kltnbackup]\ntype=drive\nservice_account_file=n\n', 'kltnbackup')


if __name__ == '__main__':
    unittest.main()

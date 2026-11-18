"""Offline fixtures only; no Drive credentials, GPU or ML dependencies."""
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

import prawn_results_viewer as v


class FakeDrive:
    def __init__(self):
        self.files, self.downloaded = {}, []

    def path(self, relative=''):
        return 'fake:KLTNRUNS/' + relative

    def read_json(self, remote):
        return json.loads(self.files[remote.split('KLTNRUNS/', 1)[1]])

    def call(self, command, source, *args, quiet=False):
        if command == 'copyto':
            name = source.split('KLTNRUNS/', 1)[1]
            self.downloaded.append(name)
            if any(s in name for s in ['model-best', 'model-last', '/cache/', '.pth', '.npz']):
                raise AssertionError('Attempted heavy download')
            Path(args[0]).write_bytes(self.files[name])
            return ''
        if command == 'lsf':
            base = source.split('KLTNRUNS/', 1)[1].strip('/') + '/'
            tails = [n[len(base):] for n in self.files if n.startswith(base)]
            if '--dirs-only' in args:
                return '\n'.join(sorted({t.split('/')[0] + '/' for t in tails if '/' in t}))
            return '\n'.join(t for t in tails if '/' not in t)
        raise AssertionError(command)


def fixture(drive, dataset='yc2', wrong_quality=False):
    run = dataset + '_caption_match_e2_b1'
    work = '/marimo/PRAWN_Caption_Match'
    cfg = dict(epoch=2, seed=777, batch_size=1, num_queries=10,
               gt_file_for_eval=[f'{work}/prepared/{run}/data/reference_0.json'], caption_match_enabled=True)
    ref = {'video1': dict(duration=20, timestamps=[[0, 5], [3, 12]], sentences=['test <caption>', 'second'])}
    row = dict(timestamp=[0., 5.], query_id=0, sentence='test <caption>', proposal_score=.7,
               sentence_score=-.5, pred_event_count=1, vid_duration=20.)
    raw = {'results': {'video1': [row]}}
    final = dict(results={'video1': [row]}, CIDEr=.4, METEOR=.07, soda_c=.06, Precision=.5, Recall=.25)
    base = f'runs/prawn_caption_match_generator/{dataset}/20261118_run/recovery/slot-1'
    texts = {
        f'prepared/{run}/c1_manifest.json': {'expected_config': cfg},
        f'prepared/{run}/config.yml': 'unused configuration',
        f'prepared/{run}/smoke/config.yml': 'also unused',
        f'prepared/{run}/data/reference_0.json': ref,
        f'runs/{run}/info.json': {'best': {'epoch': 0, 'result_json_path': f'{work}/runs/{run}/prediction/num1_epoch0.json'}, 'last': {'epoch': 1}},
        f'runs/{run}/prediction/num1_epoch0.raw.json': raw,
        f'runs/{run}/prediction/num1_epoch0.json': final,
        f'runs/{run}/model-best.pth': 'heavy checkpoint forbidden',
    }
    def add_snapshot(base, texts, **extra):
        files = {}
        for name, value in texts.items():
            data = (value if isinstance(value, str) else json.dumps(value)).encode()
            drive.files[base + '/' + name] = data
            files[name] = dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
        ready = dict(dataset=dataset, run_id='20261118_run', work=work, created_ns=10, files=files, **extra)
        drive.files[base + '/READY.json'] = json.dumps(ready).encode()
        return ready
    ready = add_snapshot(base, texts)
    qbase = f'runs/prawn_caption_match_quality/{dataset}/20261118_run/recovery/slot-1'
    qmanifest = dict(base_checkpoint_sha256=ready['files'][f'runs/{run}/model-best.pth']['sha256'],
                     base_epoch=0, base_config_sha256=ready['files'][f'prepared/{run}/config.yml']['sha256'],
                     raw_sha256=ready['files'][f'runs/{run}/prediction/num1_epoch0.raw.json']['sha256'])
    if wrong_quality:
        qmanifest['base_checkpoint_sha256'] = 'a' * 64
    texts = {f'runs/{dataset}/manifest.json': qmanifest}
    for mode in ['baseline_topk', 'baseline_softnms', 'reranker_softnms']:
        r = copy.deepcopy(row)
        if mode == 'reranker_softnms':
            r.update(reranker_beta=6., reranker_quality=.4, selection_score=2.8)
        texts[f'runs/{dataset}/evaluation/{mode}.json'] = {'results': {'video1': [r]}}
        texts[f'runs/{dataset}/evaluation/{mode}.metrics.json'] = dict(scores=dict(CIDEr=40., METEOR=7., Precision=50., Recall=25.))
    add_snapshot(qbase, texts)
    return dict(remote=base, ready=ready)


class ViewerTests(unittest.TestCase):
    def test_reject_heavy_and_unsafe_files(self):
        for p in ['runs/a/model-best.pth', 'runs/a/cache/val/a.json', '../x.json', '/runs/a.json',
                  'runs/a\\x.json', 'secrets/rclone.conf', 'prepared/a/features/x.json']:
            with self.subTest(p=p), self.assertRaises(ValueError):
                v.allowed(p)

    def test_pair_by_all_hashes_and_epoch(self):
        drive = FakeDrive()
        snap = fixture(drive)
        with tempfile.TemporaryDirectory() as tmp:
            store = v.Store(drive, tmp)
            g = v.generator(store, snap)
            policies, meta = v.quality(store, 'prawn_caption_match_quality', 'yc2', g)
            self.assertEqual(len(policies), 3)
            self.assertIn('checkpoint', meta['linked_by'])
            p = v.Payload(); p.add('yc2', v.PROFILES[0], g, policies, meta, True)
            self.assertEqual(p.data['datasets']['yc2']['models'][2]['beta'], 6.)
            self.assertAlmostEqual(p.data['datasets']['yc2']['models'][0]['metrics']['F1'], 100/3)
            self.assertFalse(any('model-best' in x or '/smoke/' in x for x in drive.downloaded))

    def test_wrong_generator_quality_is_not_used(self):
        drive = FakeDrive(); snap = fixture(drive, wrong_quality=True)
        with tempfile.TemporaryDirectory() as tmp:
            store = v.Store(drive, tmp); g = v.generator(store, snap)
            self.assertEqual(v.quality(store, 'prawn_caption_match_quality', 'yc2', g), ([], None))
            self.assertFalse(any('/evaluation/' in x for x in drive.downloaded))

    def test_corruption_is_not_accepted(self):
        drive = FakeDrive(); snap = fixture(drive)
        name = next(x for x in snap['ready']['files'] if x.endswith('/info.json'))
        drive.files[snap['remote'] + '/' + name] = b'{}'
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                v.Store(drive, tmp).get(snap, name)

    def test_incomplete_generator_rejected(self):
        drive = FakeDrive(); snap = fixture(drive)
        with tempfile.TemporaryDirectory() as tmp:
            store = v.Store(drive, tmp)
            real_get = store.get
            def get(s, name):
                data = real_get(s, name)
                if name.endswith('/info.json'):
                    data['last']['epoch'] = 0
                return data
            store.get = get
            with self.assertRaisesRegex(ValueError, 'incomplete'):
                v.generator(store, snap)

    def test_both_datasets_one_run_zip_no_private_files(self):
        drive = FakeDrive(); fixture(drive, 'yc2'); fixture(drive, 'anet')
        auth = SimpleNamespace(restore_auth=lambda *a: None, Drive=lambda *a: drive)
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(v, 'fetch_helper', return_value=auth), patch.object(v.shutil, 'which', return_value='rclone'), patch.object(sys, 'argv', ['viewer', '--output', tmp]):
                v.main()
            data = json.loads((Path(tmp)/'viewer_data.json').read_text(encoding='utf8'))
            self.assertEqual(set(data['datasets']), {'yc2', 'anet'})
            html = (Path(tmp)/'index.html').read_text(encoding='utf8')
            self.assertNotIn('atob(', html)
            self.assertIn('test \\u003ccaption>', html)
            self.assertNotIn('__PAYLOAD__', html)
            with zipfile.ZipFile(Path(tmp)/'PRAWN_YC2_ANET_HTML.zip') as z:
                self.assertIn('PRAWN_HTML/index.html', z.namelist())
                self.assertFalse(any(p.endswith(('.env', '.conf', '.pth', '.npz', '.pt')) for p in z.namelist()))

    def test_missing_rclone_installs_only_checked_binary(self):
        stream = io.BytesIO()
        name = 'rclone-v1.68.2-linux-amd64'
        with zipfile.ZipFile(stream, 'w') as archive:
            archive.writestr(name + '/rclone', b'fixture binary not executed')
        data = stream.getvalue()
        signed_sums = ('-----BEGIN PGP SIGNED MESSAGE-----\n\n' + v.sha(data) + '  ' + name + '.zip\n').encode()
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(v.shutil, 'which', return_value=None), patch.object(v.platform, 'machine', return_value='x86_64'), patch.object(v.sys, 'platform', 'linux'), patch.object(v, 'read_url', side_effect=[signed_sums, data]), patch.dict(v.os.environ):
                v.ensure_rclone(Path(tmp))
                self.assertEqual((Path(tmp)/'tools/bin/rclone').read_bytes(), b'fixture binary not executed')

    def test_quality_link_rejects_each_identity_difference(self):
        g = dict(checkpoint='a', raw_sha='b', config_sha='c', epoch=4)
        m = dict(base_checkpoint_sha256='a', raw_sha256='b', base_config_sha256='c', base_epoch=4)
        self.assertTrue(v.quality_matches(g, m))
        for key in m:
            changed = dict(m); changed[key] = 'wrong'
            self.assertFalse(v.quality_matches(g, changed))


if __name__ == '__main__':
    unittest.main()

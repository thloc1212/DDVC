"""Offline tests for whitelist-only downloads, cache checksums and atomic writes."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('prepare_tokens',Path(__file__).with_name('prepare_gpt2_tokenizer.py'))
helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)


class TokenizerAssetsTests(unittest.TestCase):
    def setUp(self):
        vocab={f'token{i}':i for i in range(50256)};vocab['<|endoftext|>']=50256
        self.data={'vocab.json':json.dumps(vocab).encode(),'merges.txt':b'#version: 0.2\n',
                   'tokenizer_config.json':b'{"model_max_length":1024}'}
        self.expected={name:hashlib.sha256(value).hexdigest() for name,value in self.data.items()}

    def test_three_tokenizer_files_only_and_verified_cache_needs_no_network(self):
        requested=[]
        def fetch(url):
            name=url.rsplit('/',1)[-1];requested.append(name)
            self.assertIn(helper.REVISION,url)
            return self.data[name]
        with tempfile.TemporaryDirectory() as folder,patch.object(helper,'FILES',self.expected):
            root=Path(folder)
            manifest=helper.prepare_assets(root,fetch)
            self.assertEqual(set(requested),set(self.data))
            self.assertEqual(set(p.name for p in root.iterdir()),set(self.data))
            self.assertTrue(manifest['tokenizer_only'])
            self.assertFalse(manifest['pretrained_model_weights'])
            with patch.object(helper.urllib.request,'urlopen',side_effect=AssertionError('unexpected network')):
                helper.prepare_assets(root)

    def test_bad_response_does_not_replace_existing_file_or_publish_partial(self):
        with tempfile.TemporaryDirectory() as folder,patch.object(helper,'FILES',self.expected),patch.object(helper.time,'sleep'):
            root=Path(folder);(root/'vocab.json').write_bytes(b'old incomplete file')
            with self.assertRaises(RuntimeError):helper.prepare_assets(root,lambda url:b'not a tokenizer')
            self.assertEqual((root/'vocab.json').read_bytes(),b'old incomplete file')
            self.assertFalse(list(root.glob('*.download')))

    def test_corrupt_cache_file_is_repaired_without_redownloading_other_files(self):
        with tempfile.TemporaryDirectory() as folder,patch.object(helper,'FILES',self.expected):
            root=Path(folder)
            helper.prepare_assets(root,lambda url:self.data[url.rsplit('/',1)[-1]])
            (root/'merges.txt').write_bytes(b'broken')
            requested=[]
            def fetch(url):
                name=url.rsplit('/',1)[-1];requested.append(name);return self.data[name]
            helper.prepare_assets(root,fetch)
            self.assertEqual(requested,['merges.txt'])


if __name__=='__main__':unittest.main()

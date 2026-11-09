"""Stage only pinned GPT-2 tokenizer files; never download language-model weights.

The legacy loader uses from_pretrained('gpt2') from the source working directory.
A verified local source/gpt2 directory bypasses Hub HEAD/cache resolution while
preserving the exact vocabulary/BPE and the random GPT-2 weight initialization.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
REVISION = '607a30d783dfa663caf39e06633721c8d4cfcd7e'
REPOSITORY = 'openai-community/gpt2'
FILES = {
    'vocab.json': '196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783',
    'merges.txt': '1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5',
    'tokenizer_config.json': '5e04eb606e3a1583530a42e36c2a6b6615c86f34fe77e44d9ddeb43ff940931f',
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def prepare_assets(target, fetch=None):
    target = Path(target).resolve()
    target.mkdir(parents=True, exist_ok=True)
    manifest = dict(repository=REPOSITORY, revision=REVISION, tokenizer_only=True,
                    pretrained_model_weights=False, files={})
    for name, expected in FILES.items():
        path = target / name
        if path.is_symlink():
            raise ValueError('Refusing a symlink tokenizer file: ' + name)
        data = path.read_bytes() if path.is_file() else b''
        if sha(data) != expected:
            url = f'https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{name}'
            print('TOKENIZER_DOWNLOAD', name, flush=True)
            for attempt in range(3):
                try:
                    if fetch is None:
                        request = urllib.request.Request(url, headers={'User-Agent': 'DDVC-tokenizer-staging/1.0'})
                        with urllib.request.urlopen(request, timeout=40) as response:
                            data = response.read(4 * 1024**2)
                    else:
                        data = fetch(url)
                    if sha(data) != expected:
                        raise ValueError('Tokenizer checksum mismatch for ' + name)
                    break
                except Exception as exc:
                    if attempt == 2:
                        raise RuntimeError('Cannot download verified tokenizer ' + name + ': ' + type(exc).__name__) from None
                    time.sleep(attempt + 1)
            fd, partial = tempfile.mkstemp(prefix=name + '.', suffix='.download', dir=target)
            try:
                with os.fdopen(fd, 'wb') as stream:
                    stream.write(data)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(partial, path)
            finally:
                if os.path.exists(partial):
                    os.unlink(partial)
        manifest['files'][name] = dict(bytes=len(data), sha256=expected)
    vocabulary = json.loads((target / 'vocab.json').read_text(encoding='utf8'))
    if len(vocabulary) != 50257 or vocabulary.get('<|endoftext|>') != 50256:
        raise ValueError('Unexpected GPT-2 vocabulary')
    return manifest


def validate_tokenizer(target):
    # Load the project's vendored implementation, not a different pip version.
    sys.path.insert(0, str(Path(target).resolve().parent))
    from transformers import GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(str(Path(target).resolve()), local_files_only=True)
    if len(tokenizer) != 50257 or tokenizer.eos_token_id != 50256:
        raise ValueError('Tokenizer IDs changed')
    if tokenizer.encode('hello world', add_special_tokens=False) != [31373, 995]:
        raise ValueError('GPT-2 BPE encoding changed')
    print('TOKENIZER_LOCAL_LOAD_PASS vocab=50257 eos=50256; decoder weights unchanged', flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--target', type=Path, default=ROOT / 'gpt2')
    parser.add_argument('--provenance', type=Path)
    args = parser.parse_args()
    target = args.target.resolve()
    manifest = prepare_assets(target)
    validate_tokenizer(target)
    provenance = args.provenance or target.parent.parent / 'provenance/gpt2_tokenizer.json'
    provenance.parent.mkdir(parents=True, exist_ok=True)
    temporary = provenance.with_name(provenance.name + '.tmp')
    temporary.write_text(json.dumps(manifest, indent=2), encoding='utf8')
    os.replace(temporary, provenance)
    print('TOKENIZER_READY', target, flush=True)


if __name__ == '__main__':
    main()

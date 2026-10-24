"""Offline checks for CLI defaults and idempotent runtime compatibility patches."""
import ast
import importlib.util
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace


def main():
    root=Path(__file__).parent
    spec=importlib.util.spec_from_file_location('molab_pipeline',root/'tools/molab_pipeline.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    files=['data/video_dataset.py','ddvc/CaptioningHead/gpt2.py',
           'transformers/dependency_versions_table.py','ddvc/ops/src/cuda/ms_deform_attn_cuda.cu','eval_utils_clip.py']
    with tempfile.TemporaryDirectory() as tmp:
        target=Path(tmp)/'source';target.mkdir()
        for rel in files:
            dest=target/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(root/rel,dest)
        module.ROOT=target;module.patch_runtime(SimpleNamespace())
        first={rel:(target/rel).read_bytes() for rel in files}
        module.patch_runtime(SimpleNamespace())
        assert first=={rel:(target/rel).read_bytes() for rel in files}
        assert '/DATA_DISK/plm_cache/gpt2' not in (target/'ddvc/CaptioningHead/gpt2.py').read_text()
        assert "getattr(opt, 'eval_enable_grounding', True)" in (target/'eval_utils_clip.py').read_text()
        assert (target.parent/'provenance/runtime_sources/eval_utils_clip.py').exists()
        for rel in files:
            if rel.endswith('.py'):ast.parse((target/rel).read_text())
    print('PASS: Molab CLI explicit compatibility patches, idempotence and preserved patched-source provenance')


if __name__=='__main__':main()

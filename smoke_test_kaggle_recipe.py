"""Exercise the actual prepare config expression and its recipe drift guard."""
import ast
import copy
import json
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent / 'tools'))
from kaggle_recipe import validate


def main():
    root = Path(__file__).parent
    tree = ast.parse((root / 'tools/molab_pipeline.py').read_text())
    prepare = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'prepare')
    updates = [n for n in prepare.body if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
               and isinstance(n.value.func, ast.Attribute) and isinstance(n.value.func.value, ast.Name)
               and n.value.func.value.id == 'opts' and n.value.func.attr == 'update']
    # Exercise the exact prepare validator loading block without tools on sys.path.
    guard = next(n for n in prepare.body if isinstance(n, ast.If)
                 and 'recipe_spec' in ast.unparse(n))
    saved_path = sys.path[:]
    sys.path = [entry for entry in sys.path if Path(entry or '.').resolve() != root.resolve()/'tools']
    class Checked:
        def __init__(self): self.called = False
    checked = Checked()
    import importlib.util
    original_spec = importlib.util.spec_from_file_location
    def tracked_spec(name, path):
        assert Path(path) == root.resolve()/'tools'/'kaggle_recipe.py'
        checked.called = True
        return original_spec(name, path)
    importlib.util.spec_from_file_location = tracked_spec
    try:
        config = json.loads((root/'tools/kaggle_mask_recipe.json').read_text())
        config['experiment_recipe'] = 'kaggle_cvpr2018_mask'
        with tempfile.TemporaryDirectory() as check_tmp:
            for key,count in [('train_caption_file',371),('val_caption_file',131)]:
                path=Path(check_tmp)/key;path.write_text(json.dumps(dict.fromkeys(map(str,range(count)),{})))
                config[key]=str(path)
            exec(compile(ast.Module(body=[guard], type_ignores=[]), '<actual validator import>', 'exec'),
                 dict(ROOT=root.resolve(), RUN_MODE='subset', opts=config))
        assert checked.called
    finally:
        importlib.util.spec_from_file_location = original_spec
        sys.path = saved_path
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp)
        for mode, counts in [('subset', (371, 131)), ('full', (1237, 436))]:
            train_json, val_json = p / 'train.json', p / 'val.json'
            train_json.write_text(json.dumps(dict.fromkeys(map(str, range(counts[0])), {})))
            val_json.write_text(json.dumps(dict.fromkeys(map(str, range(counts[1])), {})))
            opts = json.loads((root / 'opts.json').read_text())
            opts['experiment_recipe'] = 'kaggle_cvpr2018_mask'
            namespace = dict(opts=opts, args=SimpleNamespace(workers=0, pin_memory=False, precision='fp32',
                allow_tf32=False, hardware_profile='legacy-cvpr2018', accumulation=1, log_every=10, caption_chunk=16),
                train_json=train_json, val_json=val_json, para_json=p/'para.json', features_path=p/'clip.pth',
                vocab_path=p/'vocab.json', word_to_ix={'word':1}, PREPARED=p, SAVE_DIR=str(p/'runs'),
                RUN_ID='test', RUN_MODE=mode, EPOCHS=15, SEED=777, RESUME_RUN_ID='', MASK_MODE='learned_gated',
                MASK_BINS=64, MASK_ENCODING_DIM=128, MASK_LOSS_COEF=1., QUERY_CONDITIONING=False,
                MASK_TARGET='predicted', BALANCED_MASK_BCE=False, PRUNE_THRESHOLD=-1., GATE_MIN=0., GATE_MAX=1., MASK_CONTEXT_RATIO=0., MASK_MIN_FRAMES=1,
                CORRECTED_QUERY_CONTRASTIVE=False, LOCAL_CONTEXT_RATIO=.1, LOCAL_MIN_FRAMES=4,
                data_signature='test', train=range(counts[0]), time=time)
            for n in updates:
                exec(compile(ast.Module(body=[n], type_ignores=[]), '<actual prepare config>', 'exec'), namespace)
            assert validate(opts, mode) == [8, 11]
            for key, value in [('epoch',25),('lr',1e-4),('gradient_accumulation_steps',8),
                               ('mask_prune_threshold',.05),('learned_proposal_mask',False)]:
                bad=copy.deepcopy(opts);bad[key]=value
                try: validate(bad,mode)
                except ValueError: pass
                else: raise AssertionError('Uncaught drift: '+key)
    print('PASS actual subset/full prepare config matches pinned Kaggle recipe; rejects epoch/LR/batch/mask drift')


if __name__ == '__main__': main()

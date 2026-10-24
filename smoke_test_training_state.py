"""Run the actual train() loop with tiny CPU fixtures; interrupt validation and resume."""
import ast
import collections
from collections import OrderedDict
import contextlib
import io
import json
import logging
import os
import random
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from misc.training_state import (atomic_save, capture_rng, restore_rng, snapshot,
                                 schedule_contrastive, validate_resume, plain_metadata)
from misc.runtime_perf import amp_context,configure_runtime,vram_metrics,append_performance


def main():
    torch.set_num_threads(2)
    keys={'contrastive_loss':.5,'loss_contrastive':.5,'loss_contrastive_0':.5}
    schedule_contrastive(keys,0,[0,2],[0.,.1],corrected=True)
    assert set(keys.values())=={0.}
    schedule_contrastive(keys,2,[0,2],[0.,.1],corrected=True)
    assert set(keys.values())=={.1}
    tree=ast.parse((Path(__file__).parent/'train.py').read_text(encoding='utf8'))
    train_fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='train')
    class Dataset:
        length=2
        translator=None
        def __init__(self,*a):pass
        def __len__(self):return self.length
        def __getitem__(self,i):return dict(x=torch.tensor(float(i+1)),video_target=[])
    class Writer:
        def __init__(self,*a):pass
        def add_scalar(self,*a):pass
        def close(self):pass
    steps=[]; evaluations=[]; fail=[False]; records=[]
    class Model(nn.Module):
        def __init__(self):super().__init__();self.p=nn.Parameter(torch.randn(()))
        def forward(self,dt,*a):
            steps.append(1)
            x,y=dt['x'].mean(),torch.rand(())
            records.append((x.detach(),y.detach()))
            return {},{'loss_ce':(self.p*x-y)**2}
    def build(opt):
        return Model(),SimpleNamespace(weight_dict={'loss_ce':1.},matcher=SimpleNamespace(cost_cl=0)),None,None
    def evaluate(*a,**kw):
        value=float(torch.rand(()))+random.random()+float(np.random.rand())
        evaluations.append(value)
        if fail[0] and len(evaluations)==2:raise RuntimeError('simulated validation interruption')
        # Real eval_metrics returns a local-lambda defaultdict, not a plain dict.
        return collections.defaultdict(lambda:-1, METEOR=value,soda_c=value,CIDEr=value,
                                       Precision=.3,Recall=.2,avg_proposal_number=2),{}
    def seed(n):torch.manual_seed(n);random.seed(n);np.random.seed(n)
    def folder(opt):
        p=Path(opt.save_dir)/(opt.start_from or opt.id);p.mkdir(parents=True,exist_ok=True);return str(p)
    logger=logging.getLogger('state-test');logger.addHandler(logging.NullHandler())
    namespace=globals().copy()
    namespace.update(set_seed=seed,build_floder=folder,create_logger=lambda *a:logger,
        SummaryWriter=Writer,backup_envir=lambda *a:None,PropSeqDataset=Dataset,
        collate_fn=lambda rows:dict(x=torch.stack([r['x'] for r in rows]),video_target=[]),
        build=build,evaluate=evaluate,print_opt=lambda *a:None,
        print_alert_message=lambda *a:None,tqdm=lambda data,**kw:data)
    exec(compile(ast.Module(body=[train_fn],type_ignores=[]),'actual_train_loop','exec'),namespace)
    defaults=dict(seed=777,random_seed=True,id='run',exp_name='',start_from='',start_from_mode='last',
        pretrain=None,train_caption_file='',visual_feature_folder=[],dict_file='',val_caption_file='',
        batch_size=1,batch_size_for_eval=1,nthreads=0,lr=.01,device='cpu',optimizer_type='adam',
        weight_decay=0.,learning_rate_decay_start=1,learning_rate_decay_every=1,learning_rate_decay_rate=.5,
        epoch=3,cl_schedule_time=[0,2],cl_schedule_val=[0.,.1],corrected_query_contrastive=True,
        able_ret=False,scheduled_sampling_start=-1,set_cost_cl=0,save_mode=False,debug=False,
        disable_tqdm=True,transformer_input_type='queries',grad_clip=10,save_checkpoint_every=1,
        min_epoch_when_save=0,save_all_checkpoint=False,caption_decoder_type='gpt2',criteria_for_best_ckpt='dvc',ec_alpha=1.)
    with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
        full=SimpleNamespace(**defaults,save_dir=str(Path(tmp)/'full'))
        namespace['train'](full)
        expected=torch.load(Path(full.save_dir)/'run/model-last.pth',weights_only=False)
        assert expected['next_epoch']==3 and expected['phase']=='epoch_complete' and len(steps)==6
        score=expected['info']['history']['val_result_history'][0]['eval_score']
        assert type(score) is dict and score['CIDEr']==score['METEOR']
        nested=collections.defaultdict(lambda:None, a=collections.defaultdict(lambda:-1,b=np.float32(.5)))
        assert plain_metadata(nested)=={'a':{'b':.5}}
        # A failed serializer must preserve the last usable checkpoint and remove its temp file.
        protected=Path(full.save_dir)/'run/model-last.pth'
        original=protected.read_bytes()
        try:atomic_save({'bad':lambda:0},protected)
        except (AttributeError,TypeError):pass
        else:raise AssertionError('Non-pickleable fixture should fail')
        assert protected.read_bytes()==original and not protected.with_name(protected.name+'.tmp').exists()
        steps.clear();evaluations.clear();fail[0]=True
        interrupted=SimpleNamespace(**defaults,save_dir=str(Path(tmp)/'interrupted'))
        try:namespace['train'](interrupted)
        except RuntimeError as e:assert 'simulated' in str(e)
        else:raise AssertionError('Interruption fixture did not run')
        checkpoint=Path(interrupted.save_dir)/'run/model-last.pth'
        pending=torch.load(checkpoint,weights_only=False)
        assert pending['phase']=='validation_pending' and pending['next_epoch']==1 and len(steps)==4
        fail[0]=False;interrupted.start_from='run'
        namespace['train'](interrupted)
        resumed=torch.load(checkpoint,weights_only=False)
        assert len(steps)==6, 'Completed epoch was trained twice'
        torch.testing.assert_close(resumed['model']['p'],expected['model']['p'],rtol=0,atol=0)
        assert resumed['scheduler']==expected['scheduler'] and resumed['next_epoch']==3
        # The final checkpoint may be loaded again without taking an optimizer step.
        namespace['train'](interrupted)
        assert len(steps)==6
        state=capture_rng();wanted=(random.random(),np.random.rand(),torch.rand(()))
        restore_rng(state);got=(random.random(),np.random.rand(),torch.rand(()))
        assert wanted[0]==got[0] and wanted[1]==got[1] and wanted[2]==got[2]
        # The actual loop must average both full windows and the final partial window.
        Dataset.length=5;records.clear();steps.clear();evaluations.clear()
        original_adam=optim.Adam
        class CheckedAdam(original_adam):
            def step(self,*args,**kwargs):
                param=self.param_groups[0]['params'][0]
                expected=torch.stack([2*(param.detach()*x-y)*x for x,y in records]).mean()
                expected=expected*min(1.,defaults['grad_clip']/(expected.abs().item()+1e-6))
                torch.testing.assert_close(param.grad,expected,rtol=1e-5,atol=1e-6)
                records.clear()
                return super().step(*args,**kwargs)
        optim.Adam=CheckedAdam
        try:
            accumulated=SimpleNamespace(**defaults,gradient_accumulation_steps=3,save_dir=str(Path(tmp)/'accumulated'))
            namespace['train'](accumulated)
            state=torch.load(Path(accumulated.save_dir)/'run/model-last.pth',weights_only=False)
            assert state['optimizer_steps']==6 and len(steps)==15 and not records
        finally:
            optim.Adam=original_adam
    print('PASS: actual train loop with evaluator lambda-defaultdict metrics, nested metadata, failed-save protection, validation interruption/resume, no repeated epoch, exact model/scheduler/RNG continuity')


if __name__=='__main__':main()

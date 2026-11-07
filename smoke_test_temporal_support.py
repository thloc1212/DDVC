"""Offline checks of actual support, count loss, dataset and experiment preparation."""
import ast
import copy
import hashlib
import importlib.util
from itertools import chain
import json
from pathlib import Path
import tempfile
import subprocess
import sys
from types import SimpleNamespace
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import yaml
from ddvc.temporal_support import QueryTemporalSupport, temporal_support_objective, fractional_interval_targets

ROOT=Path(__file__).resolve().parent


def support_test():
    torch.set_num_threads(2);torch.manual_seed(37)
    module=QueryTemporalSupport(8,6,16)
    q=torch.randn(1,3,8,requires_grad=True);f=torch.randn(1,12,6,requires_grad=True)
    valid=torch.tensor([[True]*10+[False]*2]);boxes=torch.tensor([[[.5,.7],[.4,.08],[.8,.2]]],requires_grad=True)
    ctx=module(q,f,boxes,valid)
    torch.testing.assert_close(ctx['pred_boxes'],boxes,rtol=1e-5,atol=1e-6)
    assert not ctx['weights'][...,10:].any()
    changed=f.detach().clone();changed[:,10:]=1e9
    torch.testing.assert_close(module(q,changed,boxes,valid)['pred_boxes'],ctx['pred_boxes'])
    torch.testing.assert_close(module(q,changed,boxes,valid)['weights'],ctx['weights'])
    gt=torch.tensor([[.5,.8],[.4,.01],[.75,.3]])
    target=fractional_interval_targets(gt,valid.expand(3,-1))
    assert target[1].sum()>0 and bool(((target[0]>0)&(target[1]>0)).any())
    pairs=[(torch.tensor([2,0,1]),torch.tensor([0,2,1]))]
    losses=temporal_support_objective(ctx,[{'boxes':gt}],pairs)
    total=losses['loss_temporal_support']+losses['loss_temporal_refine'];total.backward()
    assert f.grad is None and q.grad.abs().sum()>0
    assert module.frame_proj[0].weight.grad.abs().sum()>0 and module.refiner[-1].weight.grad.abs().sum()>0
    assert boxes.grad.abs().sum()>0 and all(torch.isfinite(p.grad).all() for p in module.parameters() if p.grad is not None)
    empty=temporal_support_objective(ctx,[{'boxes':gt}],[(torch.tensor([],dtype=torch.long),)*2])
    assert empty['loss_temporal_support']==0
    # A fixed toy video must be learnable from real fractional-GT supervision.
    q=q.detach();f=f.detach();boxes=boxes.detach();optim=torch.optim.Adam(module.parameters(),lr=.015)
    first=float(total.detach())
    for _ in range(35):
        optim.zero_grad();ctx=module(q,f,boxes,valid)
        ls=temporal_support_objective(ctx,[{'boxes':gt}],pairs)
        loss=ls['loss_temporal_support']+ls['loss_temporal_refine'];loss.backward();optim.step()
    assert float(loss.detach())<first*.7,(float(loss),first)
    print('PASS support: identity init, actual visual/query gradients, padding isolation, fractional short/nested GT, reordered matching, toy optimization')


def count_test():
    tree=ast.parse((ROOT/'ddvc/criterion.py').read_text(encoding='utf8'))
    nodes=[n for n in tree.body if (isinstance(n,ast.FunctionDef) and n.name in ['sigmoid_focal_loss','cross_entropy_with_gaussian_mask'])
           or (isinstance(n,ast.ClassDef) and n.name=='SetCriterion_cl')]
    ns=dict(torch=torch,nn=nn,F=F,is_dist_avail_and_initialized=lambda:False,get_world_size=lambda:1)
    box_tree=ast.parse((ROOT/'misc/detr_utils/box_ops.py').read_text())
    box_ns={'torch':torch}
    exec(compile(ast.Module(body=[n for n in box_tree.body if isinstance(n,ast.FunctionDef)],type_ignores=[]),'actual_1d_box_ops','exec'),box_ns)
    ns['box_ops']=SimpleNamespace(**{k:v for k,v in box_ns.items() if not k.startswith('__')})
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'actual_criterion','exec'),ns)
    rates=[0.]*31;rates[3]=.8;rates[27]=.2
    opt=SimpleNamespace(max_eseq_length=30,counter_class_rates=rates,lloss_gau_mask=1,lloss_beta=1)
    criterion=ns['SetCriterion_cl'](1,None,{},[],opt=opt)
    count=torch.zeros(1,31,requires_grad=True)
    output={'pred_logits':torch.randn(1,30,1,requires_grad=True),'pred_count':count}
    targets=[{'boxes':torch.ones(27,2)*.1,'labels':torch.zeros(27,dtype=torch.long)}]
    indices=[(torch.arange(27),torch.arange(27))]
    loss=criterion.loss_labels(output,targets,(indices,indices),27)['loss_counter'];loss.backward()
    assert torch.isfinite(loss) and count.grad[0,27]<0 and count.grad[0,3]>0
    # Execute the real criterion forward: final-layer support losses use its one-to-one matching.
    module=QueryTemporalSupport(8,6,16)
    context=module(torch.randn(1,30,8),torch.randn(1,20,6),torch.full((1,30,2),.2),torch.ones(1,20,dtype=torch.bool))
    criterion.matcher=lambda outputs,targets:(indices,indices)
    criterion.losses=['labels','boxes','cardinality']
    actual=dict(output,pred_boxes=context['pred_boxes'],temporal_support=context)
    actual_losses,_=criterion(actual,targets)
    assert {'loss_temporal_support','loss_temporal_refine','loss_counter'}<=set(actual_losses)
    sum(actual_losses[k] for k in ['loss_temporal_support','loss_temporal_refine']).backward()
    assert module.frame_proj[0].weight.grad.abs().sum()>0
    try:ns['SetCriterion_cl'](1,None,{},[],opt=SimpleNamespace(**dict(vars(opt),counter_class_rates=[0.]*21)))
    except ValueError:pass
    else:raise AssertionError('Wrong class prior size accepted')
    print('PASS count: actual expanded 31-logit criterion learns count 27; prior/head mismatch rejected')


def dataset_test():
    tree=ast.parse((ROOT/'data/video_dataset.py').read_text(encoding='utf8'))
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='PropSeqDataset')
    item=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__getitem__')
    collate=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='collate_fn')
    ns=dict(torch=torch,np=np,chain=chain)
    exec(compile(ast.Module(body=[item,collate],type_ignores=[]),'actual_dataset','exec'),ns)
    record=dict(duration=30.,timestamps=[[i,i+.5] for i in range(27)],sentences=[f'event {i}.' for i in range(27)])
    opt=SimpleNamespace(visual_feature_type=['clip'],full_event_supervision=True,num_queries=30,max_eseq_length=30,num_classes=1)
    obj=SimpleNamespace(keys=['video'],opt=opt,viz_feats={'video':torch.randn(32,768)},anno={'video':record},
        gt_proposal_sample_num=30,num_classes=1,max_caption_len=4,
        tokenizer=lambda captions,**kw:{'input_ids':torch.ones(len(captions),4,dtype=torch.long),
                                      'attention_mask':torch.ones(len(captions),4,dtype=torch.long)},
        process_time_step=lambda duration,stamps,length:[[0,1] for _ in stamps])
    for training in (True,False):
        obj.is_training=training;row=ns['__getitem__'](obj,0);data=ns['collate_fn']([row])
        assert len(data['video_target'][0]['boxes'])==27 and len(data['cap_tensor'])==27
        assert data['video_length'][0,2]==27
    obj.gt_proposal_sample_num=10
    try:ns['__getitem__'](obj,0)
    except ValueError:pass
    else:raise AssertionError('Silent truncation under full supervision')
    print('PASS actual loader/collate: 27 GT/captions/count stay aligned in train and val; truncation fails loudly')


def preparation_test():
    spec=importlib.util.spec_from_file_location('qts_cli',ROOT/'tools/qts_pipeline.py')
    cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)
    for ds,sizes in [('yc2',(1237,436)),('anet',(10009,4917))]:
        with tempfile.TemporaryDirectory() as temp:
            work=Path(temp);assets=work/'assets_fixture';assets.mkdir()
            paths={}
            for split,size in zip(['train','val'],sizes):
                n=16 if ds=='yc2' else (27 if split=='train' else 25)
                rec=dict(duration=30.,timestamps=[[i,i+.5] for i in range(n)],sentences=['wash.']*n)
                rows={f'{split}{i}':(rec if i==0 else dict(duration=30.,timestamps=[[1,2]],sentences=['wash.'])) for i in range(size)}
                p=assets/(split+'.json');p.write_text(json.dumps(rows),encoding='utf8');paths[split]=p
            words=assets/'vocab.json';words.write_text('{"word_to_ix":{"wash":1}}')
            feature=assets/'clip.pth';feature.write_bytes(b'fixture-only-not-torch-loaded')
            bank=assets/'bank'/ds/'clip';bank.mkdir(parents=True)
            for suffix in ['scene_sentences','clip_token_embeds','scene_videoID']:(bank/f'{ds}_{suffix}.npy').write_bytes(b'fixture-bank')
            base=json.loads((ROOT/'tools/kaggle_mask_recipe.json').read_text())
            base.update(dataset=ds,bank_type=[ds],bank_path=str(assets/'bank'),vocab_size=1,
                visual_feature_folder=[str(feature)],train_caption_file=str(paths['train']),val_caption_file=str(paths['val']),
                dict_file=str(words),gt_file_for_eval=[str(paths['val'])],gt_file_for_auc=[str(paths['val'])],
                gt_file_for_para_eval=[str(paths['val'])])
            base['data_signature']=hashlib.sha256(b''.join(Path(base[k]).read_bytes() for k in ['train_caption_file','val_caption_file','dict_file'])).hexdigest()
            source=assets/'config.yml';source.write_text(yaml.safe_dump(base))
            for variant in ['baseline','refine','support']:
                cli.prepare(SimpleNamespace(work=work,dataset=ds,source_config=source,features=None,variant=variant))
                path=cli.config_path(work,ds,variant);cfg=cli.check(path)
                assert cfg['epoch']==(15 if ds=='yc2' else 6) and cfg['num_queries']==(50 if ds=='yc2' else 30)
                assert len(cfg['counter_class_rates'])==(21 if ds=='yc2' else 31)
                assert cfg['counter_class_rates'][16 if ds=='yc2' else 27]==1/sizes[0]
                assert cfg['proposal_mask_loss_coef']==0 and cfg['count_loss_coef']==1
                assert cfg['query_temporal_support']==(variant!='baseline')
                smoke=yaml.safe_load((path.parent/'smoke/config.yml').read_text())
                assert max(len(r['timestamps']) for r in json.loads(Path(smoke['train_caption_file']).read_text()).values())==(16 if ds=='yc2' else 27)
                assert Path(cfg['train_caption_file']).read_bytes()==paths['train'].read_bytes()
                from misc.training_state import validate_resume
                validate_resume(cfg,copy.deepcopy(cfg))
                drift=copy.deepcopy(cfg);drift['num_queries']+=1
                try:validate_resume(drift,cfg)
                except ValueError:pass
                else:raise AssertionError('Resume with another query budget accepted')
                original=path.read_text();bad=dict(cfg);bad['counter_class_rates'][0]+=.1
                path.write_text(yaml.safe_dump(bad))
                try:cli.check(path)
                except ValueError:pass
                else:raise AssertionError('Modified prior accepted')
                path.write_text(original)
                subprocess.run([sys.executable,'tools/qts_pipeline.py','check','--config',str(path)],cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
                subprocess.run([sys.executable,'tools/qts_pipeline.py','prepare','--work',str(work),
                    '--dataset',ds,'--variant',variant,'--source-config',str(source)],cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
    print('PASS preparation: both full splits, 3 ablations, train-only prior, expanded budgets, fresh runs, exact input bytes')


if __name__=='__main__':
    support_test();count_test();dataset_test();preparation_test()

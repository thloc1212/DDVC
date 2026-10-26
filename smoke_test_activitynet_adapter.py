"""Offline tests: feature aliases, reference separation, leakage and recipe drift."""
import importlib.util
import json
from pathlib import Path
import tempfile
import hashlib
import ast

ROOT=Path(__file__).parent
spec=importlib.util.spec_from_file_location('anet_adapter',ROOT/'tools/activitynet_pipeline.py')
adapter=importlib.util.module_from_spec(spec);spec.loader.exec_module(adapter)


def main():
    assert adapter.canonical_video_id('abcdefghijk')=='v_abcdefghijk'
    assert adapter.canonical_video_id('v_abcdefghijk.mp4')=='v_abcdefghijk'
    class Feature:
        ndim=2;shape=(4,768)
    assert set(adapter.canonical_features({'features':{'abcdefghijk':Feature()}}))=={'v_abcdefghijk'}
    try:adapter.canonical_features({'abcdefghijk':Feature(),'v_abcdefghijk':Feature()})
    except ValueError:pass
    else:raise AssertionError('Ambiguous alias accepted')
    rec=dict(duration=10.,timestamps=[[1.,3.]],sentences=['A person runs.'])
    train={'v_abcdefghijk':rec};r1={'v_12345678901':rec};r2={'v_12345678901':dict(rec,sentences=['A different reference.']),'v_12345678902':rec}
    tr,val,refs,coverage=adapter.select_splits(train,r1,r2,set(train)|set(r1)|set(r2),'full',777)
    assert len(val)==2 and len(refs[0])==1 and len(refs[1])==2
    assert val['v_12345678901']['sentences']==r1['v_12345678901']['sentences']
    assert refs[1]['v_12345678901']['sentences']!=refs[0]['v_12345678901']['sentences']
    try:adapter.select_splits(train,train,r2,set(train)|set(r2),'full',777)
    except ValueError:pass
    else:raise AssertionError('Leakage accepted')
    clean,dropped=adapter.validate_annotations({'v_abcdefghijk':dict(rec,timestamps=[[2,2]])},train=True)
    assert not clean and len(dropped)==1
    vocab=adapter.vocabulary(tr);assert 'different' not in vocab['word_to_ix']
    with tempfile.TemporaryDirectory() as temp:
        root=Path(temp);data=root/'data/anet';caption=data/'captiondata'
        adapter.dump(caption/'train.json',tr);adapter.dump(caption/'val.json',val)
        adapter.dump(data/'vocabulary_activitynet.json',vocab)
        for i,ref in enumerate(refs,1):adapter.dump(caption/f'val_{i}.json',ref)
        signature=hashlib.sha256(b''.join(p.read_bytes() for p in [caption/'train.json',caption/'val.json',data/'vocabulary_activitynet.json'])).hexdigest()
        cfg=adapter.make_config(root,'run',root/'runs','full',len(vocab['word_to_ix']),signature)
        adapter.validate_config(cfg)
        for k,v in dict(num_queries=50,learned_proposal_mask=True,dedup_loss_coef=0.,
            eval_merge_consecutive=True,eval_merge_duplicates=True,epoch=15,lr=5e-5,
            gradient_accumulation_steps=1,precision='fp32',feature_dim=768,able_ret=True).items():assert cfg[k]==v,(k,cfg[k])
        for key,value in [('num_queries',10),('dedup_loss_coef',.1),('learned_proposal_mask',False),('lr',1e-4)]:
            broken=dict(cfg);broken[key]=value
            try:adapter.validate_config(broken)
            except ValueError:pass
            else:raise AssertionError('Recipe drift accepted: '+key)
        adapter.dump(caption/'val_2.json',{})
        try:adapter.validate_config(cfg)
        except ValueError:pass
        else:raise AssertionError('Missing reference coverage accepted')
    import torch
    tree=ast.parse((ROOT/'data/video_dataset.py').read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='EDVCdataset')
    init=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
    load=next(n for n in init.body if isinstance(n,ast.If) and 'self.viz_feats' in ast.unparse(n))
    from types import SimpleNamespace
    with tempfile.TemporaryDirectory() as temp:
        path=Path(temp)/'features.pth';tensor=torch.arange(12).reshape(3,4).float();torch.save({'v_test':tensor},path)
        obj=SimpleNamespace(opt=SimpleNamespace(visual_feature_type=['clip'],feature_mmap=True))
        exec(compile(ast.Module(body=[load],type_ignores=[]),'<actual dataset feature load>','exec'),dict(self=obj,opt=obj.opt,torch=torch,feature_folder=[str(path)]))
        assert torch.equal(obj.viz_feats['v_test'],tensor)
    print('PASS ActivityNet adapter: IDs, dual references, leakage guard, exact model recipe, invalid-event reporting, mmap tensor values')


if __name__=='__main__':main()

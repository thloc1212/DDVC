"""Offline CPU checks: finalized recipe, actual GPT-2/mask gradients and legacy parity."""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
from types import SimpleNamespace

import torch
import yaml

ROOT=Path(__file__).resolve().parent
from misc.c1_recipe import BASE_COMMIT, METHOD, validate_method


def pipeline():
    spec=importlib.util.spec_from_file_location('c1_smoke_pipeline',ROOT/'tools/c1_pipeline.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def check_preparation():
    cli=pipeline()
    for ds,sizes in [('yc2',(1237,436)),('anet',(10009,4917))]:
        with tempfile.TemporaryDirectory() as directory:
            work=Path(directory);assets=work/'source_assets';assets.mkdir()
            record=dict(duration=10.,timestamps=[[1.,5.]],sentences=['wash the spoon'])
            paths={}
            for name,count in zip(('train','val'),sizes):
                paths[name]=assets/(name+'.json')
                paths[name].write_text(json.dumps({f'{name}{i}':record for i in range(count)}),encoding='utf8')
            words=assets/'vocabulary.json';words.write_text(json.dumps({'word_to_ix':{f'w{i}':i for i in range(json.loads((ROOT/'tools/c1_frozen_options.json').read_text())[ds]['vocab_size'])}}),encoding='utf8')
            para=assets/'para.json';para.write_text(paths['val'].read_text(encoding='utf8'),encoding='utf8')
            feature=assets/'clipvitl14.pth';feature.write_bytes(b'not-loaded-in-config-validation')
            bank=assets/'bank'/ds/'clip';bank.mkdir(parents=True)
            for suffix in ('scene_sentences','clip_token_embeds','scene_videoID'):
                (bank/f'{ds}_{suffix}.npy').write_bytes(b'fixture-input-bytes')
            cfg=json.loads((ROOT/'tools/kaggle_mask_recipe.json').read_text())
            cfg.update(METHOD)
            cfg.update(dataset=ds,bank_type=[ds],bank_folder='clip',bank_path=str(assets/'bank'),
                epoch=15 if ds=='yc2' else 6,num_queries=50 if ds=='yc2' else 10,vocab_size=json.loads((ROOT/'tools/c1_frozen_options.json').read_text())[ds]['vocab_size'],
                seed=777,batch_size=1,gradient_accumulation_steps=1,precision='fp32',allow_tf32=False,
                lr=5e-5,weight_decay=1e-4,aux_loss=True,able_ret=True,
                train_caption_file=str(paths['train']),val_caption_file=str(paths['val']),
                dict_file=str(words),visual_feature_folder=[str(feature)],gt_file_for_eval=[str(paths['val'])],
                gt_file_for_para_eval=[str(para)],gt_file_for_auc=[str(paths['val'])])
            cfg['data_signature']=hashlib.sha256(b''.join(Path(cfg[k]).read_bytes() for k in
                                  ('train_caption_file','val_caption_file','dict_file'))).hexdigest()
            source=assets/'config.yml';source.write_text(yaml.safe_dump(cfg),encoding='utf8')
            cli.prepare(SimpleNamespace(work=work,dataset=ds,source_config=source,features=None))
            path=cli.config_path(work,ds);actual=cli.check(path)
            assert actual['seed']==2016
            assert actual['proposal_mask_loss_coef']==0 and 'concept_vocab_path' not in actual
            assert not list(path.parent.glob('*concept*'))
            assert Path(actual['train_caption_file']).read_bytes()==paths['train'].read_bytes()
            before=path.read_text(encoding='utf8')
            for key,value in [('proposal_mask_loss_coef',1.),('event_concepts',True),('eval_merge_duplicates',True)]:
                bad=dict(actual);bad[key]=value;path.write_text(yaml.safe_dump(bad),encoding='utf8')
                try:cli.check(path)
                except ValueError:pass
                else:raise AssertionError('Retired method or recipe drift accepted: '+key)
            path.write_text(before,encoding='utf8')
    print('PASS C1 preparation: both full splits, exact staged bytes, smoke configs, no concept assets, drift rejection')


def check_caption(legacy_ref=None):
    from ddvc.CaptioningHead import gpt2
    torch.set_num_threads(2)
    original_config,original_tokenizer=gpt2.GPT2Config,gpt2.GPT2Tokenizer
    class OfflineTokenizer:
        eos_token='<eos>';pad_token_id=63
        @classmethod
        def from_pretrained(cls,*args,**kwargs):return cls()
        def __call__(self,*args,**kwargs):return {'input_ids':torch.tensor([[63]])}
        def encode(self,text):return [13]
    def tiny_config(**kwargs):
        return original_config(vocab_size=64,n_embd=768,n_head=12,n_layer=1,n_positions=32,
            bos_token_id=63,eos_token_id=63,pad_token_id=63,add_cross_attention=True,
            resid_pdrop=0.,embd_pdrop=0.,attn_pdrop=0.)
    gpt2.GPT2Config,gpt2.GPT2Tokenizer=tiny_config,OfflineTokenizer
    try:
        opt=SimpleNamespace(hidden_dim=512,device='cpu',gpt_layer=1,add_cross_attention=True,
                            learned_proposal_mask=True,proposal_mask_bins=64,proposal_mask_encoding_dim=128,
                            local_visual_attention=False,event_concepts=False,max_caption_len=6,gpt_beam_size=2)
        torch.manual_seed(777);head=gpt2.GPT2CaptionModel(opt)
        assert not any('event_concepts' in k for k in head.state_dict())
        torch.manual_seed(18)
        queries=torch.randn(2,1,512);features=torch.randn(1,7,768,requires_grad=True)
        valid=torch.tensor([[True]*6+[False]])
        labels=torch.tensor([[63,2,3,4,13],[63,5,6,7,13]])
        boxes=torch.tensor([[.3,.4],[.7,.2]],requires_grad=True)
        confidence=torch.tensor([.3,.4],requires_grad=True)
        kwargs=dict(event_boxes=boxes,video_mask=valid,event_scores=confidence,anchor_boxes=boxes.detach())
        args=(queries,labels,torch.ones_like(labels),{},labels,features)
        output=head(*args,**kwargs)
        mask_loss=head.proposal_mask.objective(boxes,valid,confidence,boxes.detach())['loss_proposal_mask']
        total=output.loss+0*mask_loss
        total.backward()
        assert torch.isfinite(total)
        assert features.grad is None and boxes.grad.abs().sum()>0 and confidence.grad.abs().sum()>0
        assert any(p.grad is not None and p.grad.abs().sum()>0 for p in head.proposal_mask.parameters())
        assert all(torch.isfinite(p.grad).all() for p in head.parameters() if p.grad is not None)
        head.eval()
        generated=head.generate(queries,None,{},features,**kwargs)
        assert generated.sequences.shape[0]==2 and torch.isfinite(generated.sequences_scores).all()
        print('PASS actual tiny GPT-2: CE only (mask BCE coefficient zero), finite mask/box/confidence gradients, frozen CLIP, beam generation')
        if legacy_ref:
            source=subprocess.check_output(['git','show',legacy_ref+':ddvc/CaptioningHead/gpt2.py'],cwd=ROOT,text=True)
            # Only replace imports for inactive methods that no longer ship.
            source=source.replace('from .local_visual import event_visual_mask','event_visual_mask = None')
            source=source.replace('from .event_concepts import EventConceptConditioner','EventConceptConditioner = None')
            module=types.ModuleType('ddvc.CaptioningHead._legacy_c2');module.__package__='ddvc.CaptioningHead'
            sys.modules[module.__name__]=module
            exec(compile(source,'legacy_c1_gpt2.py','exec'),module.__dict__)
            module.GPT2Config,module.GPT2Tokenizer=tiny_config,OfflineTokenizer
            torch.manual_seed(777);old=module.GPT2CaptionModel(opt)
            assert old.state_dict().keys()==head.state_dict().keys()
            for key,value in old.state_dict().items():assert torch.equal(value,head.state_dict()[key]),key
            head.load_state_dict(old.state_dict(),strict=True)
            old.eval();head.eval();old.zero_grad(set_to_none=True);head.zero_grad(set_to_none=True)
            reference=old(*args,**kwargs);current=head(*args,**kwargs)
            assert torch.equal(reference.logits,current.logits) and torch.equal(reference.loss,current.loss)
            reference.loss.backward();current.loss.backward()
            for (name,a),(name2,b) in zip(old.named_parameters(),head.named_parameters()):
                assert name==name2
                assert (a.grad is None)==(b.grad is None),name
                if a.grad is not None:assert torch.equal(a.grad,b.grad),name
            torch.optim.SGD(old.parameters(),lr=.01).step();torch.optim.SGD(head.parameters(),lr=.01).step()
            for name,value in old.state_dict().items():assert torch.equal(value,head.state_dict()[name]),name
            with torch.no_grad():
                a=old.generate(queries,None,{},features,**kwargs);b=head.generate(queries,None,{},features,**kwargs)
            assert torch.equal(a.sequences,b.sequences) and torch.equal(a.sequences_scores,b.sequences_scores)
            print('PASS legacy C1 parity:',legacy_ref,'same initialization, strict checkpoint keys, logits/loss, gradients, optimizer step and generation')
    finally:gpt2.GPT2Config,gpt2.GPT2Tokenizer=original_config,original_tokenizer


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--legacy-ref')
    args=parser.parse_args();check_preparation();check_caption(args.legacy_ref)


if __name__=='__main__':main()

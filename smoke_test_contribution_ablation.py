"""CPU checks for the C0-C5 recipe and real random GPT-2 caption-head paths."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import torch

ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('contribution_test_helper',ROOT/'tools/contribution_ablation.py')
matrix=importlib.util.module_from_spec(spec);spec.loader.exec_module(matrix)


def check_preparation():
    import yaml
    real_helper=matrix.helper
    concept=real_helper('concept_pipeline')
    for dataset,sizes in [('yc2',(1237,436)),('anet',(10009,4917))]:
        with tempfile.TemporaryDirectory() as directory:
            work=Path(directory)
            assets=work/'assets';assets.mkdir()
            record={'duration':10,'timestamps':[[1,5]],'sentences':['wash the spoon']}
            paths={}
            for name,count in zip(('train','val'),sizes):
                paths[name]=assets/(name+'.json')
                paths[name].write_text(json.dumps({f'{name}{i}':record for i in range(count)}))
            vocabulary=assets/'vocabulary.json';vocabulary.write_text('{"word_to_ix":{"wash":1}}')
            concepts=assets/'concepts.json';concepts.write_text('{"concepts":["wash"]}')
            para=assets/'para.json';para.write_text(paths['val'].read_text())
            feature=assets/'clipvitl14.pth';feature.write_bytes(b'fixture')
            bank=assets/'bank'/dataset/'clip';bank.mkdir(parents=True)
            for suffix in ('scene_sentences','clip_token_embeds','scene_videoID'):
                (bank/f'{dataset}_{suffix}.npy').write_bytes(b'fixture')
            source=work/'prepared'/concept.run_name(dataset,'control',matrix.epochs(dataset))/'config.yml'
            source.parent.mkdir(parents=True)
            baseline=dict(dataset=dataset,bank_type=[dataset],bank_folder='clip',bank_path=str(assets/'bank'),
                lr=5e-5,weight_decay=1e-4,seed=777,batch_size=1,gradient_accumulation_steps=1,
                precision='fp32',allow_tf32=False,able_ret=True,epoch=matrix.epochs(dataset),
                num_queries=50 if dataset=='yc2' else 10,train_caption_file=str(paths['train']),
                val_caption_file=str(paths['val']),dict_file=str(vocabulary),concept_vocab_path=str(concepts),
                gt_file_for_eval=[str(paths['val'])],gt_file_for_para_eval=[str(para)],
                visual_feature_folder=[str(feature)],data_signature='fixture')
            source.write_text(yaml.safe_dump(baseline))
            fake=SimpleNamespace(run_name=concept.run_name,prepare=lambda args:None,
                check=lambda path:copy.deepcopy(baseline),make_smoke=concept.make_smoke,
                load_helper=lambda name:SimpleNamespace(revision=lambda:'fixture'))
            matrix.helper=lambda name:fake if name=='concept_pipeline' else real_helper(name)
            try:
                matrix.prepare(SimpleNamespace(work=work,dataset=dataset,source_config=None,cases=list(matrix.CASES)))
                for case in matrix.CASES:
                    cfg_path=matrix.config_path(work,dataset,case)
                    cfg=matrix.check(cfg_path)
                    before=cfg_path.read_text()
                    cfg['eval_merge_duplicates']=True
                    cfg_path.write_text(yaml.safe_dump(cfg))
                    try:matrix.check(cfg_path)
                    except ValueError:pass
                    else:raise AssertionError('Merge drift accepted')
                    cfg_path.write_text(before)
                # A failed training command must not silently advance the queue.
                try:matrix.logged([sys.executable,'-c','raise SystemExit(3)'],work/'logs/failure.log',work/'logs/test.pid')
                except subprocess.CalledProcessError:pass
                else:raise AssertionError('Failed child command was accepted')
            finally:matrix.helper=real_helper


def main():
    base=dict(lr=5e-5,seed=777,num_queries=50,batch_size=1,gradient_accumulation_steps=1,
              precision='fp32',allow_tf32=False,weight_decay=1e-4,train_caption_file='train',
              val_caption_file='val',bank_path='bank',data_signature='same_data',gpt_layer=2)
    expected={
        'C0':(False,0.,False,False),'C1':(True,0.,False,False),'C2':(True,1.,False,False),
        'C3':(True,1.,True,False),'C4':(True,1.,True,True),'C5':(True,0.,True,True)}
    for dataset in ('yc2','anet'):
        for case in matrix.CASES:
            cfg=matrix.make_config(base,Path('/temporary/work'),dataset,case)
            assert tuple(cfg[k] for k in ('learned_proposal_mask','proposal_mask_loss_coef',
                         'event_concepts','concept_condition_prefix'))==expected[case]
            assert cfg['epoch']==(15 if dataset=='yc2' else 8)
            assert cfg['eval_selection_method']=='softnms' and not cfg['eval_merge_duplicates'] and not cfg['eval_merge_consecutive']
            assert not cfg['start_from'] and cfg['pretrain'] is None
            for key in base:assert cfg[key]==base[key],key
    assert base['gpt_layer']==2 and 'event_concepts' not in base
    subprocess.run([sys.executable,'-P',str(ROOT/'tools/contribution_ablation.py'),'--help'],
                   stdout=subprocess.DEVNULL,check=True)
    check_preparation()

    from misc.event_concepts import make_vocabulary
    from ddvc.CaptioningHead import gpt2
    torch.set_num_threads(2)
    original_config,original_tokenizer=gpt2.GPT2Config,gpt2.GPT2Tokenizer
    class OfflineTokenizer:
        eos_token='<eos>'
        pad_token_id=63
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
        with tempfile.TemporaryDirectory() as directory:
            vocab=make_vocabulary({'train':{'sentences':['woman washes spoon','man cuts bread']}},min_frequency=1)
            path=Path(directory)/'concepts.json';path.write_text(json.dumps(vocab))
            opt=SimpleNamespace(hidden_dim=512,feature_dim=768,device='cpu',gpt_layer=1,
                add_cross_attention=True,proposal_mask_bins=8,proposal_mask_encoding_dim=16,
                concept_vocab_path=str(path),concept_vocab_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                concept_hidden_dim=16,concept_embedding_dim=8,max_caption_len=6,gpt_beam_size=2)
            reference=None
            for case in matrix.CASES:
                cfg=matrix.make_config({},Path(directory),'yc2',case)
                options=copy.copy(opt)
                for key in ('event_concepts','concept_condition_prefix','learned_proposal_mask','local_visual_attention'):
                    setattr(options,key,cfg[key])
                torch.manual_seed(777)
                head=gpt2.GPT2CaptionModel(options)
                common={k:v.detach().clone() for k,v in head.state_dict().items()
                        if not k.startswith(('event_concepts.','proposal_mask.'))}
                if reference is None:reference=common
                else:
                    for key,value in common.items():assert torch.equal(value,reference[key]),key
                q=torch.randn(2,1,512,requires_grad=True)
                features=torch.randn(1,7,768)
                boxes=torch.tensor([[.3,.4],[.7,.2]])
                valid=torch.tensor([[1,1,1,1,1,1,0]],dtype=torch.bool)
                scores=torch.tensor([.3,.4]);labels=torch.tensor([[63,2,3,4,13],[63,5,6,7,13]])
                head.train()
                output=head(q,labels,torch.ones_like(labels),{},labels,features,
                            event_boxes=boxes,video_mask=valid,event_scores=scores,anchor_boxes=boxes)
                loss=output.loss
                if cfg['learned_proposal_mask']:
                    objective=head.proposal_mask.objective(boxes,valid,scores,boxes)
                    loss=loss+cfg['proposal_mask_loss_coef']*objective['loss_proposal_mask']
                if cfg['event_concepts']:
                    objective=head.concept_objective(q,features,boxes,valid,['woman washes spoon','man cuts bread'])
                    loss=loss+.1*objective['loss_event_concept']
                loss.backward()
                assert torch.isfinite(loss) and q.grad.abs().sum()>0
                assert all(torch.isfinite(p.grad).all() for p in head.parameters() if p.grad is not None)
                if cfg['learned_proposal_mask']:
                    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in head.proposal_mask.parameters())
                if cfg['event_concepts']:
                    assert head.event_concepts.classifier[0].weight.grad.abs().sum()>0
                    delta_grad=head.event_concepts.to_prefix.weight.grad
                    if cfg['concept_condition_prefix']:assert delta_grad is not None and delta_grad.abs().sum()>0
                    else:assert delta_grad is None or delta_grad.abs().sum()==0
                print('CONTRIBUTION_CAPTION_FORWARD_BACKWARD_PASS',case,flush=True)
                del head,output,loss,common
    finally:gpt2.GPT2Config,gpt2.GPT2Tokenizer=original_config,original_tokenizer
    print('CONTRIBUTION_MATRIX_PASS: 2 datasets x 6 recipes; Soft-NMS-only; shared initialization; actual caption gradients')


if __name__=='__main__':main()

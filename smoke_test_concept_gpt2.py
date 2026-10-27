"""Real vendored GPT-2 forward/generation with a tiny offline test vocabulary."""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import torch
from misc.event_concepts import make_vocabulary


def main():
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
                add_cross_attention=True,local_visual_attention=False,learned_proposal_mask=True,
                proposal_mask_bins=8,proposal_mask_encoding_dim=16,event_concepts=False,
                concept_vocab_path=str(path),concept_vocab_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                concept_condition_prefix=True,concept_hidden_dim=16,concept_embedding_dim=8,
                max_caption_len=6,gpt_beam_size=2)
            torch.manual_seed(777);baseline=gpt2.GPT2CaptionModel(opt)
            enabled=copy.copy(opt);enabled.event_concepts=True
            torch.manual_seed(777);head=gpt2.GPT2CaptionModel(enabled)
            for key,value in baseline.state_dict().items():
                assert torch.equal(value,head.state_dict()[key]),key
            features=torch.randn(1,7,768);queries=torch.randn(2,1,512)
            boxes=torch.tensor([[.3,.4],[.7,.2]]);valid=torch.tensor([[1,1,1,1,1,1,0]],dtype=torch.bool)
            scores=torch.tensor([.3,.4]);labels=torch.tensor([[63,2,3,4,13],[63,5,6,7,13]])
            kwargs=dict(event_boxes=boxes,video_mask=valid,event_scores=scores,anchor_boxes=boxes)
            args=(queries,labels,torch.ones_like(labels),{},labels,features)
            baseline.eval();head.eval()
            original=baseline(*args,**kwargs);initial=head(*args,**kwargs)
            assert torch.equal(original.logits,initial.logits),'Zero-init changed baseline logits'
            head.train()
            output=head(*args,**kwargs)
            objective=head.concept_objective(queries,features,boxes,valid,['woman washes spoon','man cuts bread'])
            loss=output.loss+.1*objective['loss_event_concept'];loss.backward()
            conditioner=head.event_concepts
            assert conditioner.classifier[0].weight.grad.abs().sum()>0
            assert conditioner.to_prefix.weight.grad.abs().sum()>0
            assert all(torch.isfinite(p.grad).all() for p in head.parameters() if p.grad is not None)
            torch.optim.SGD(conditioner.parameters(),lr=.1).step()
            head.eval()
            changed=head(*args,**kwargs)
            assert not torch.equal(original.logits,changed.logits),'Concept prefix never reached real GPT-2'
            # A caption's conditioning is independent of what other captions share its chunk.
            chunked=torch.cat([head(queries[i:i+1],labels[i:i+1],torch.ones_like(labels[i:i+1]),{},
                labels[i:i+1],features,event_boxes=boxes[i:i+1],video_mask=valid,
                event_scores=scores[i:i+1],anchor_boxes=boxes[i:i+1]).logits for i in range(2)])
            assert torch.allclose(changed.logits,chunked,atol=1e-5,rtol=1e-5)
            generated=head.generate(queries,None,{},features,**kwargs)
            assert len(generated.sequences)==2 and torch.isfinite(generated.sequences_scores).all()
            clone=gpt2.GPT2CaptionModel(enabled);clone.load_state_dict(head.state_dict(),strict=True);clone.eval()
            assert torch.allclose(changed.logits,clone(*args,**kwargs).logits)
        print('PASS actual GPT-2 concept integration: common initialization/logits unchanged at zero-init, '
              'finite CE+concept gradients, prefix changes decoder logits, chunk parity, generation, strict state reload')
    finally:gpt2.GPT2Config,gpt2.GPT2Tokenizer=original_config,original_tokenizer


if __name__=='__main__':main()

"""CPU regression: labels, predicted pooling, gradients, initialization and resume."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import torch
from misc.event_concepts import make_vocabulary, load_vocabulary

ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('event_concept_head',ROOT/'ddvc/CaptioningHead/event_concepts.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def main():
    records={'train':{'sentences':['A woman washes a spoon in water.', 'A man cuts bread.',
                                  'The woman washes washes a spoon.']}}
    vocab=make_vocabulary(records,max_concepts=20,min_frequency=1)
    assert 'heldout' not in vocab['concepts'] and 'the' not in vocab['concepts']
    assert vocab['document_counts'][vocab['concepts'].index('washes')]==2
    with tempfile.TemporaryDirectory() as folder:
        path=Path(folder)/'vocab.json';path.write_text(json.dumps(vocab))
        sha=hashlib.sha256(path.read_bytes()).hexdigest()
        opt=SimpleNamespace(concept_vocab_path=str(path),concept_vocab_sha256=sha,
            concept_condition_prefix=True,concept_hidden_dim=16,concept_embedding_dim=8,
            hidden_dim=12,feature_dim=6)
        head=module.EventConceptConditioner(opt)
        hs=torch.randn(2,1,12,requires_grad=True)
        features=torch.randn(1,7,6,requires_grad=True)
        boxes=torch.tensor([[.3,.4],[.99,0.]],requires_grad=True)
        valid=torch.tensor([[1,1,1,1,1,0,0]],dtype=torch.bool)
        pooled=head.pool_visual(features,boxes,valid)
        assert torch.allclose(pooled[1],features[0,4].detach())
        modified=features.detach().clone();modified[:,5:]=99999
        assert torch.allclose(pooled,head.pool_visual(modified,boxes,valid))
        logits,delta=head(hs,features,boxes,valid)
        assert torch.equal(delta,torch.zeros_like(delta))
        objective=head.objective(hs,features,boxes,valid,['A woman washes a spoon.', 'A man cuts bread.'])
        loss=objective['loss_event_concept']+delta.sum()
        loss.backward()
        assert torch.isfinite(loss) and head.classifier[0].weight.grad.abs().sum()>0
        assert head.to_prefix.weight.grad.abs().sum()>0 and hs.grad.abs().sum()>0
        assert features.grad is None and boxes.grad is None
        assert all(torch.isfinite(p.grad).all() for p in head.parameters() if p.grad is not None)
        optimizer=torch.optim.Adam(head.parameters(),lr=1e-3);optimizer.step()
        assert head(hs,features,boxes,valid)[1].abs().sum()>0
        out=io.BytesIO();torch.save(head.state_dict(),out);out.seek(0)
        clone=module.EventConceptConditioner(opt);clone.load_state_dict(torch.load(out,weights_only=True),strict=True)
        assert torch.allclose(head(hs,features,boxes,valid)[1],clone(hs,features,boxes,valid)[1])
        opt.concept_condition_prefix=False
        aux=module.EventConceptConditioner(opt)
        assert not aux(hs,features,boxes,valid)[1].any()
        unsupported=head.objective(hs,features,boxes,valid,['zzzzzz','zzzzzz'])
        assert unsupported['loss_event_concept']==0 and unsupported['concept_target_coverage']==0
        try:load_vocabulary(path,'0'*64)
        except ValueError:pass
        else:raise AssertionError('Changed vocabulary was accepted')
        from misc.training_state import validate_resume
        recipe=dict(event_concepts=True,concept_condition_prefix=True,concept_vocab_sha256=sha,
                    concept_loss_coef=.1,concept_hidden_dim=16,concept_embedding_dim=8)
        validate_resume(recipe,recipe)
        try:validate_resume(dict(recipe,concept_vocab_sha256='0'*64),recipe)
        except ValueError:pass
        else:raise AssertionError('Resume changed concept meaning without rejection')
    print('PASS event concepts: train-only vocabulary, padding/sub-frame pooling, zero-init prefix, '
          'finite nonzero classifier/prefix/query gradients, frozen visual/boxes, aux mode, checkpoint roundtrip')


if __name__=='__main__':main()

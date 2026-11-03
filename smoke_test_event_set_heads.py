"""CPU regression tests; runtime smoke additionally exercises the real DDVC model."""
import ast
import sys
import types
import tempfile
import subprocess
from pathlib import Path
from types import SimpleNamespace
import torch
from ddvc.event_set_heads import AttentionCounter, DuplicateHead, duplicate_objective, ordinal_cdf_loss
from misc.proposal_selection import select_proposals


def main():
    torch.manual_seed(7)
    q = torch.randn(2, 4, 512, requires_grad=True)
    boxes = torch.tensor([[[.3,.2],[.31,.2],[.6,.6],[.3,.1]]]*2, requires_grad=True)
    confidence = torch.randn(2,4,1, requires_grad=True)
    counter, duplicate = AttentionCounter(max_count=8), DuplicateHead()
    permutation = torch.tensor([2,0,3,1])
    count = counter(q, boxes, confidence)
    torch.testing.assert_close(count, counter(q[:,permutation], boxes[:,permutation], confidence[:,permutation]))
    logits = duplicate(q, boxes, confidence)
    torch.testing.assert_close(logits, logits.transpose(1,2))
    torch.testing.assert_close(logits[:,permutation][:,:,permutation], duplicate(q[:,permutation],boxes[:,permutation],confidence[:,permutation]))
    targets = [{'boxes':torch.tensor([[.3,.2],[.6,.6],[.3,.1]])}]*2
    objective = duplicate_objective(logits, boxes, targets)
    assert objective['duplicate_positive_pairs'] > 0 and objective['duplicate_negative_pairs'] > 0
    total = ordinal_cdf_loss(count, torch.tensor([3,3])) + objective['loss_duplicate_pair']
    total.backward()
    assert q.grad.abs().sum() > 0
    assert all(p.grad is None or torch.isfinite(p.grad).all() for m in (counter,duplicate) for p in m.parameters())
    empty = duplicate_objective(logits.detach(), boxes, [{'boxes':torch.empty(0,2)}]*2)
    assert empty['loss_duplicate_pair'] == 0
    # Nested GT ambiguity must be ignored, not forced into a positive pair.
    ambiguous = duplicate_objective(logits.detach(), boxes,
                    [{'boxes':torch.tensor([[.3,.2],[.3,.2]])}]*2)
    assert ambiguous['duplicate_positive_pairs'] == 0
    def row(i, interval):
        return dict(query_id=i,timestamp=interval,sentence='the men continue to play',
                    proposal_score=1-i*.1,sentence_score=0,duplicate_selection=True,
                    duplicate_probabilities=[[1,.99,.01],[.99,1,.01],[.01,.01,1]][i])
    rows=[row(0,[0,20]),row(1,[1,21]),row(2,[2,19])]
    selected=select_proposals(rows,3,method='softnms',merge=True,merge_consecutive=True)
    assert [p['query_id'] for p in selected] == [0,2], selected
    assert selected[0]['timestamp']==[0,20]  # no boundary union
    assert selected[1]['selection_score']==selected[1]['joint_score']  # valid overlap preserved
    for r in rows:r['duplicate_selection']=False
    assert len(select_proposals(rows,3))==3  # legacy policy unchanged
    # Test actual model integration method without downloading CLIP/GPT2.
    source=Path('ddvc/ddvc_ret_encdec_clip_capQ.py').read_text(encoding='utf8')
    tree=ast.parse(source)
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='CM2')
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='event_set_prediction')
    # Use the actual criterion module and the method's relative import context.
    namespace={'__name__':'ddvc._integration_test','__package__':'ddvc','torch':torch,
               'F':torch.nn.functional,'duplicate_objective':duplicate_objective,'ordinal_cdf_loss':ordinal_cdf_loss}
    criterion_tree=ast.parse(Path('ddvc/criterion.py').read_text(encoding='utf8'))
    original_loss=next(n for n in criterion_tree.body if isinstance(n,ast.FunctionDef) and n.name=='cross_entropy_with_gaussian_mask')
    criterion_module=types.ModuleType('ddvc.criterion')
    exec(compile(ast.Module(body=[original_loss],type_ignores=[]),'<actual V1 count loss>','exec'),
         {'torch':torch,'F':torch.nn.functional},criterion_module.__dict__)
    sys.modules['ddvc.criterion']=criterion_module
    exec(compile(ast.Module(body=[method],type_ignores=[]),'<actual model method>','exec'),namespace)
    opt=SimpleNamespace(lloss_gau_mask=1,lloss_beta=1,duplicate_assign_iou=.5,duplicate_assign_margin=.1)
    holder=SimpleNamespace(attention_counter=counter,duplicate_head=duplicate,opt=opt)
    out={'pred_boxes':boxes.detach(),'pred_logits':confidence.detach(),
         'aux_outputs':[{'pred_boxes':boxes.detach(),'pred_logits':confidence.detach()}]}
    loss={}
    namespace['event_set_prediction'](holder,out,loss,torch.stack([q.detach(),q.detach()]),
                {'video_target':targets},SimpleNamespace(counter_class_rate=torch.zeros(9)))
    assert all(k in loss for k in ['loss_counter','loss_counter_0','loss_count_ordinal','loss_duplicate_pair'])
    assert out['pred_count'].shape==(2,9) and out['duplicate_logits'].shape==(2,4,4)
    sum(v for k,v in loss.items() if k.startswith('loss_')).backward()
    # Preparation must keep the inherited recipe/data while resetting run state.
    import yaml
    with tempfile.TemporaryDirectory() as directory:
        work=Path(directory)
        annotation=work/'train.json';annotation.write_text('{}')
        original=dict(bank_type=['yc2'],learned_proposal_mask=True,dedup_loss_coef=0.,
                      train_caption_file=str(annotation),val_caption_file=str(annotation),dict_file=str(annotation),
                      lr=5e-5,seed=777,batch_size=1,num_queries=50,precision='fp32',proposal_mask_loss_coef=1.,
                      start_from='old_run',pretrain='old_weights')
        source_config=work/'base.yml';source_config.write_text(yaml.safe_dump(original))
        subprocess.run([sys.executable,'tools/event_set_pipeline.py','--work',str(work),
                        '--dataset','yc2','--variant','both','--source-config',str(source_config)],check=True)
        config=yaml.safe_load((work/'prepared/yc2_full_event_set_both_e15_b1/config.yml').read_text())
        for key in ['lr','seed','batch_size','num_queries','precision','proposal_mask_loss_coef','train_caption_file']:
            assert config[key]==original[key],key
        assert config['attention_counter'] and config['duplicate_head'] and config['duplicate_selection']
        assert not config['start_from'] and config['pretrain'] is None and config['epoch']==15
    print('EVENT_SET_HEADS_PASS: gradients, symmetry, permutation, ambiguity, nested events, actual model hook, legacy policy')


if __name__=='__main__':main()

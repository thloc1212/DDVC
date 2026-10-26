"""CPU regression for the real ANET truncated-colon failure; no downloads."""
import importlib.util
from pathlib import Path
import argparse
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('caption_targets_test',ROOT/'transformers/caption_targets.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
prepare=module.prepare_legacy_caption_targets


def legacy(labels):
    target=labels[:,1:].contiguous().clone()
    if target[0,0]!=50256 and 25 in target[0]:
        for row in target:
            for j in range(len(row)):
                end=bool(row[j]==25);row[j]=50256
                if end:break
    return target


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--gpt2',action='store_true');args=parser.parse_args()
    # Exact token IDs of the first caption, max_length=30, v_7TgvbmIn03A.
    first=[50256,257,2042,3159,3568,351,2330,2456,326,1100,366,70,404,305,4293,18,10,13424,1642,10908,8861,1257,705,76,7855,262,21133,6,6323,25]
    other_rows=[
        [50256,257,1570,422,262,1353,286,257,21133,285,789,2523,257,2415,10427,4291,262,2834,5412,286,262,21133,285,789,13,50256,50256,50256,50256,50256],
        [50256,220,262,1570,1487,284,262,1570,286,262,21133,285,789,290,262,285,789,318,3867,1863,257,12699,5901,351,4077,8701,13,50256,50256,50256],
        [50256,262,1570,2458,736,290,6071,422,21133,285,789,1570,290,262,1570,286,262,2187,2415,285,7855,262,21133,13,50256,50256,50256,50256,50256,50256]]
    labels=torch.tensor([first]+other_rows);before=labels.clone()
    old=legacy(labels)
    assert old.eq(50256).all()
    logits=torch.randn(len(labels),29,50257,requires_grad=True)
    old_loss=F.cross_entropy(logits.reshape(-1,50257),old.reshape(-1),ignore_index=50256)
    assert torch.isnan(old_loss)
    targets=prepare(labels)
    assert torch.equal(labels,before) and torch.equal(targets,labels[:,1:])
    loss=F.cross_entropy(logits.reshape(-1,50257),targets.reshape(-1),ignore_index=50256)
    loss.backward();assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()
    # Ordinary inputs and valid historical prefixes preserve their old targets.
    for ordinary in [torch.tensor([[50256,100,200,13,50256]]),torch.tensor([[50256,100,25,200,13],[50256,300,25,400,13]])]:
        assert torch.equal(prepare(ordinary),legacy(ordinary))
    # Single-row contiguous input must not be modified through an alias.
    single=labels[:1].clone();original=single.clone();prepare(single)
    assert torch.equal(single,original)
    # Chunking uses the global legacy flag, not each chunk's first caption.
    all_pairs=labels.repeat(10,1)
    whole=prepare(all_pairs,truncate_prefix=True)
    chunked=torch.cat([prepare(c,truncate_prefix=True) for c in all_pairs.split(3)])
    assert torch.equal(whole,chunked)
    try:prepare(torch.tensor([[50256,50256,50256]]))
    except ValueError:pass
    else:raise AssertionError('All-padding labels must fail explicitly')
    source=(ROOT/'transformers/models/gpt2/modeling_gpt2.py').read_text(encoding='utf8')
    model=(ROOT/'ddvc/ddvc_ret_encdec_clip_capQ.py').read_text(encoding='utf8')
    assert 'shift_labels = prepare_legacy_caption_targets(labels)' in source
    assert 'labels_chunk, truncate_prefix=truncate_prefix)' in model
    if args.gpt2:
        from transformers import GPT2Config,GPT2LMHeadModel
        tiny=GPT2LMHeadModel(GPT2Config(n_layer=1,n_head=2,n_embd=16,n_positions=32))
        output=tiny(input_ids=labels,labels=labels)
        assert torch.isfinite(output.loss);output.loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in tiny.parameters() if p.grad is not None)
        assert torch.equal(labels,before)
        print('PASS actual vendored GPT-2 forward/backward; labels unchanged')
    print('PASS ANET truncated-colon NaN reproduced; finite supervised targets/gradient; valid legacy identity; no mutation; chunk parity; invalid-label guard')


if __name__=='__main__':main()

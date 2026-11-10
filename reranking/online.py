"""Online joint-quality ablation. Same 2312D scorer; no cached/GT captions as inputs.

Hard temporal pooling and generated token IDs are discrete. Quality gradients reach
the DDVC confidence and duration inputs and its trained lexical embedding; no claim
of differentiating through beam search or hard bin membership is made.
"""
import contextlib
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .core import fit_idf, targets, words
from .head import JointQualityHead, objective


def partition(annotations):
    ids = sorted(annotations)
    dev = {ids[i] for i in np.random.default_rng(777).permutation(len(ids))[:max(1, round(.1*len(ids))) ]}
    fit = [v for v in ids if v not in dev]
    if not fit:
        raise ValueError('Online quality needs at least two TRAIN videos')
    return fit, sorted(dev)


def initialize(model, opt):
    if not getattr(opt, 'online_quality', False):
        return
    if opt.batch_size != 1 or opt.quality_gradient_mode not in ('joint', 'detached'):
        raise ValueError('Online quality preserves DDVC batch1; mode must be joint/detached')
    if opt.quality_loss_coef <= 0 or opt.quality_generation_chunk < 1:
        raise ValueError('Invalid quality weight/generation chunk')
    # Do not perturb baseline initialization or its RNG stream.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(777013)
        model.quality_head = JointQualityHead()
    annotation = json.loads(Path(opt.quality_annotation_file).read_text(encoding='utf8'))
    fit, dev = partition(annotation)
    model._quality_annotations = annotation
    model._quality_fit = set(fit)
    model._quality_dev = set(dev)
    model._quality_idf = fit_idf(annotation, fit)
    model._quality_gradient_verified = False


@contextlib.contextmanager
def generation_mode(head):
    states = [(m, m.training) for m in head.modules()]
    cuda = sorted({p.device.index for p in head.parameters() if p.is_cuda})
    try:
        # Generation must not consume RNG or change dropout mode for DDVC loss.
        with torch.random.fork_rng(devices=cuda), torch.no_grad():
            head.eval()
            yield
    finally:
        for m, state in states:
            m.training = state


def decode(tokenizer, sequence):
    return tokenizer.decode(sequence, skip_special_tokens=True).split(':')[-1].split('.')[0].lower().strip()+'.'


def generate_rows(model, dt, out, hs_cap, reference, others, eval_mode):
    from ddvc.CaptioningHead.proposal_mask import mask_layer_context, proposal_confidence, proposal_anchors
    head = model.caption_head[-1]
    q = out['pred_boxes'].shape[1]
    if out['pred_boxes'].shape[0] != 1:
        raise ValueError('Online quality requires one video')
    if model.training and not eval_mode:
        context = mask_layer_context(others, out['pred_logits'], reference, out['pred_boxes'])
        confidence, anchors = proposal_confidence(context), proposal_anchors(context)
        texts, scores = [], []
        with generation_mode(head):
            for start in range(0, q, model.opt.quality_generation_chunk):
                ids = torch.arange(start, min(q, start+model.opt.quality_generation_chunk), device=hs_cap.device)
                generated = head.generate(hs_cap[:, ids].permute(1, 0, 2), reference, context,
                    dt['video_tensor'], event_boxes=out['pred_boxes'][0, ids], video_mask=dt['video_mask'],
                    event_scores=confidence[ids], anchor_boxes=anchors[ids])
                if 'sequences_scores' not in generated:
                    raise ValueError('Online ablation needs original beam decoder sequence scores')
                texts.extend(decode(head.tokenizer, s) for s in generated['sequences'].cpu().tolist())
                scores.extend(generated['sequences_scores'].float().cpu().tolist())
    else:
        texts = [decode(head.tokenizer, s) for s in out['seq'][0].detach().cpu().tolist()]
        scores = out['caption_probs']['cap_prob_eval_score'][0].detach().float().cpu().tolist()
    boxes = out['pred_boxes'][0]
    center, duration = boxes.unbind(-1)
    edges = torch.stack((center-duration/2, center+duration/2), -1).clamp(0, 1)
    seconds = float(dt['video_length'][0, 1])
    confidence = out['pred_logits'][0].sigmoid().amax(-1)
    if len(texts) != q or len(scores) != q or bool((edges[:, 1] <= edges[:, 0]).any()):
        raise ValueError('Invalid online generated candidates; do not silently skip queries')
    rows = [dict(query_id=i, timestamp=(edges[i].detach()*seconds).cpu().tolist(),
                 sentence=texts[i], proposal_score=float(confidence[i].detach()),
                 sentence_score=float(scores[i]), pred_event_count=int(out['pred_count'][0].argmax().clamp(min=1)))
            for i in range(q)]
    return rows, edges, confidence


def differentiable_features(features, rows, edges, confidence, wte, tokenizer):
    """Same feature_pool forward definitions; only existing continuous inputs keep grad."""
    f = features.detach().float()
    if f.ndim != 2 or f.shape[-1] != 768 or not len(f):
        raise ValueError('Expected valid CLIP steps [T,768]')
    position = (torch.arange(len(f), device=f.device)+.5)/len(f)
    outputs = []
    for i, row in enumerate(rows):
        s, e = edges[i].unbind()
        sd, ed = float(s.detach()), float(e.detach())
        inside = (position >= sd) & (position <= ed)
        if not bool(inside.any()):
            inside[(position-(sd+ed)/2).abs().argmin()] = True
        margin = max((ed-sd)/2, 1/len(f))
        outside = ((position >= sd-margin) & (position <= ed+margin)) & ~inside
        if not bool(outside.any()): outside = ~inside
        if not bool(outside.any()): outside = torch.ones_like(inside)
        a, b = f[inside].mean(0), f[outside].mean(0)
        an, bn = F.normalize(a, dim=0), F.normalize(b, dim=0)
        ids = tokenizer.encode(row['sentence'], add_special_tokens=False)[:64] or [tokenizer.eos_token_id]
        lexical = F.normalize(wte(torch.tensor(ids, device=f.device)).float().mean(0), dim=0)
        sentence = f.new_tensor(float(row['sentence_score']))
        joint = confidence[i]+sentence/(len(row['sentence'].split())**2+1e-5)
        variance = ((f[inside]-a).square().mean()/(f[inside].square().mean()+1e-8)).clamp(max=2)
        extra = torch.stack((confidence[i], sentence.clamp(-10, 0)/10, joint.clamp(-2, 2)/2,
            f.new_tensor(math.log1p(len(words(row['sentence'])))/4), (e-s).clamp(max=1), variance,
            (an*bn).sum(), inside.float().mean()))
        outputs.append(torch.cat((an, bn, lexical, extra)))
    result = torch.stack(outputs)
    if result.shape != (len(rows), 2312) or not bool(torch.isfinite(result).all()):
        raise ValueError('Invalid online quality features')
    return result


def augment(model, dt, out, hs_cap, reference, others, eval_mode):
    rows, edges, confidence = generate_rows(model, dt, out, hs_cap, reference, others, eval_mode)
    valid = dt['video_mask'][0].bool()
    x = differentiable_features(dt['video_tensor'][0, valid], rows, edges, confidence,
                                model.caption_head[-1].model.transformer.wte, model.caption_head[-1].tokenizer)
    if model.opt.quality_gradient_mode == 'detached':
        x = x.detach()
    # Extra scorer dropout must not advance DDVC's RNG stream in either control.
    devices=sorted({p.device.index for p in model.quality_head.parameters() if p.is_cuda})
    with torch.random.fork_rng(devices=devices):
        logits = model.quality_head(x)
    out['pred_quality'] = logits.sigmoid().unsqueeze(0)
    if eval_mode or not model.training:
        return {}
    vid = str(dt['video_key'][0])
    if vid not in model._quality_annotations:
        raise ValueError('Quality training video is outside TRAIN annotation')
    statistics = dict(quality_mean=logits.sigmoid().detach().mean(),
                      quality_labeled_queries=logits.new_tensor(len(rows) if vid in model._quality_fit else 0))
    if vid not in model._quality_fit:
        # No head/GT loss on head-dev. DDVC still receives its original TRAIN losses.
        return dict(loss_quality=out['pred_logits'].sum()*0, **statistics)
    y, _ = targets(rows, model._quality_annotations[vid], model._quality_idf)
    target = torch.as_tensor(y, device=logits.device).detach()
    loss = objective(logits[None], target[None], torch.ones_like(logits[None], dtype=torch.bool))
    if not bool(torch.isfinite(loss)):
        raise RuntimeError('Nonfinite online quality loss')
    if not model._quality_gradient_verified and torch.is_grad_enabled():
        gradients = torch.autograd.grad(loss, (out['pred_logits'], out['pred_boxes'], model.quality_head.score[-1].weight),
                                        retain_graph=True, allow_unused=True)
        evidence = {name: bool(g is not None and torch.isfinite(g).all() and g.abs().sum()>0)
                    for name, g in zip(('confidence', 'boxes', 'quality_head'), gradients)}
        expected = model.opt.quality_gradient_mode == 'joint'
        if not evidence['quality_head'] or evidence['confidence'] != expected or evidence['boxes'] != expected:
            raise RuntimeError('Online quality gradient route failed: '+str(evidence))
        model._quality_gradient_verified = True
        print('ONLINE_QUALITY_GRADIENT_PASS', model.opt.quality_gradient_mode, json.dumps(evidence), flush=True)
    return dict(loss_quality=loss, quality_target_mean=target.mean(), **statistics)


def optimizer_groups(model, opt):
    if not getattr(opt, 'online_quality', False):
        return model.parameters()
    head = list(model.quality_head.parameters()); ids = {id(p) for p in head}
    base = [p for p in model.parameters() if id(p) not in ids]
    if not all(p.requires_grad for p in base+head):
        raise ValueError('Online joint experiment must not freeze C1 parameters')
    print('ONLINE_QUALITY_OPTIMIZER', json.dumps(dict(mode=opt.quality_gradient_mode,
          DDVC_lr=opt.lr, quality_lr=opt.quality_head_lr, loss_weight=opt.quality_loss_coef)), flush=True)
    return [dict(params=base, lr=opt.lr, name='DDVC'), dict(params=head, lr=opt.quality_head_lr, name='quality')]

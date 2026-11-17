"""GT-free caption generation followed by a detached joint matching reward."""
import contextlib
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from reranking.core import caption_f1, iou_matrix


def matching_weight(opt, epoch):
    if not getattr(opt, 'caption_match_enabled', False):
        return 0.
    maximum = float(getattr(opt, 'caption_match_weight', 0.))
    warmup = int(getattr(opt, 'caption_match_warmup_epochs', 2))
    ramp = int(getattr(opt, 'caption_match_ramp_epochs', 2))
    if not math.isfinite(maximum) or maximum < 0 or warmup < 0 or ramp < 1:
        raise ValueError('Invalid caption matching schedule')
    if int(epoch) < warmup:
        return 0.
    return maximum * min(1., (int(epoch) - warmup + 1) / ramp)


def load_idf(opt):
    path = Path(opt.caption_match_idf_path)
    blob = path.read_bytes()
    if hashlib.sha256(blob).hexdigest() != opt.caption_match_idf_sha256:
        raise ValueError('Matching TRAIN IDF fingerprint changed')
    result = json.loads(blob)
    if result.get('source_split') != 'train' or result.get('documents', 0) <= 0:
        raise ValueError('Matching requires TRAIN-only IDF')
    return result


@contextlib.contextmanager
def isolated_decoder(head, device):
    flags = [(module, module.training) for module in head.modules()]
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    # Auxiliary decoding must not consume the training RNG or alter dropout modes.
    with torch.random.fork_rng(devices=devices), torch.no_grad():
        head.eval()
        try:
            yield
        finally:
            for module, training in flags:
                module.training = training


def free_running_captions(head, queries, boxes, features, valid, confidence, anchors, chunk_size=4):
    """One caption per query; no GT caption/tokens/matching indices are accepted."""
    if queries.shape[0] != 1 or boxes.shape[:2] != queries.shape[:2]:
        raise ValueError('Caption matching currently requires batch 1 and aligned queries/boxes')
    if chunk_size < 1:
        raise ValueError('Decode chunk must be positive')
    captions = []
    with isolated_decoder(head, queries.device):
        for start in range(0, queries.shape[1], chunk_size):
            end = min(start + chunk_size, queries.shape[1])
            sequences = head.generate(
                queries[:, start:end].permute(1, 0, 2).detach(), None, {}, features,
                event_boxes=boxes[0, start:end].detach(), video_mask=valid,
                event_scores=confidence[start:end].detach(), anchor_boxes=anchors[start:end].detach(),
                num_beams=head.opt.gpt_beam_size, output_scores=False, return_dict_in_generate=False)
            texts = head.tokenizer.batch_decode(sequences.detach().cpu(), skip_special_tokens=True)
            if len(texts) != end - start:
                raise ValueError('One free caption per query is required')
            captions.extend(texts)
    return captions


def joint_reward(boxes, gt_boxes, captions, gt_captions, idf):
    """Reward for the SAME GT, not independent maxima of time and caption."""
    if boxes.ndim != 3 or boxes.shape[0] != 1 or boxes.shape[-1] != 2:
        raise ValueError('Expected normalized [1,Q,2] (center,width) boxes')
    if len(captions) != boxes.shape[1] or len(gt_captions) != len(gt_boxes):
        raise ValueError('Caption/query/GT ordering mismatch')
    if not gt_captions:
        raise ValueError('Matching requires at least one GT event')
    def edges(value):
        value = value.detach().float().cpu().numpy()
        return np.clip(np.stack((value[:, 0] - value[:, 1]/2, value[:, 0] + value[:, 1]/2), -1), 0, 1)
    temporal = iou_matrix(edges(boxes[0]), edges(gt_boxes))
    lexical = np.array([[caption_f1(a, b, idf) for b in gt_captions] for a in captions])
    result = torch.as_tensor(temporal * lexical, device=boxes.device, dtype=torch.float32).unsqueeze(0)
    if not bool(torch.isfinite(result).all()) or bool(((result < 0) | (result > 1)).any()):
        raise ValueError('Invalid caption matching reward')
    return result.detach()


def add_joint_cost(cost, reward, weight):
    if reward.requires_grad or tuple(cost.shape) != tuple(reward.shape):
        raise ValueError('Matching reward must be detached and match the full cost matrix')
    if not math.isfinite(weight) or weight < 0:
        raise ValueError('Invalid matching weight')
    if not bool(torch.isfinite(reward).all()) or bool(((reward < 0) | (reward > 1)).any()):
        raise ValueError('Invalid matching reward range')
    return cost - float(weight) * reward.to(device=cost.device, dtype=cost.dtype)


def prepare_reward(head, opt, dt, queries, boxes, confidence, anchors, idf):
    started = time.perf_counter()
    captions = free_running_captions(head, queries, boxes, dt['video_tensor'], dt['video_mask'],
                                    confidence, anchors, int(opt.caption_match_decode_chunk))
    # Reference strings enter only AFTER generation has finished.
    reward = joint_reward(boxes, dt['video_target'][0]['boxes'], captions, dt['cap_raw'][0], idf)
    return reward, time.perf_counter() - started

"""Atomic epoch-boundary checkpoints and explicit query-contrastive scheduling."""
import json
import os
import random
from collections.abc import Mapping
from pathlib import Path
import numpy as np
import torch


def plain_metadata(value):
    """Remove evaluator defaultdict factories from metadata, not model/optimizer state."""
    if isinstance(value, Mapping):
        return {key: plain_metadata(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain_metadata(item) for item in value]
    if isinstance(value, np.generic):
        return plain_metadata(value.item())
    if isinstance(value, np.ndarray):
        return plain_metadata(value.tolist())
    if isinstance(value, torch.Tensor):
        return plain_metadata(value.detach().cpu().tolist())
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError('Unsupported checkpoint metadata type: ' + type(value).__name__)


def atomic_save(payload, path, json_file=False):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    try:
        if json_file:
            temporary.write_text(json.dumps(plain_metadata(payload)), encoding='utf-8')
        else:
            torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def capture_rng():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def restore_rng(state):
    if not state:
        return
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda']])


def schedule_contrastive(weight_dict, epoch, times, values, corrected=False):
    if len(times) != len(values) or not times or sorted(times) != list(times):
        raise ValueError('Contrastive schedule needs aligned ordered times/values')
    value = next((v for t, v in reversed(list(zip(times, values))) if epoch >= t), 0.)
    for key in weight_dict:
        legacy = 'contrastive_loss' in key
        matched = key == 'loss_contrastive' or key.startswith('loss_contrastive_')
        if legacy or (corrected and matched):
            weight_dict[key] = value
    return value


def snapshot(model, optimizer, scheduler, epoch, iteration, info, config, phase, optimizer_steps=0):
    return dict(format_version=2, epoch=epoch,
                next_epoch=epoch + (phase == 'epoch_complete'), phase=phase,
                iteration=iteration, optimizer_steps=optimizer_steps,
                model=model.state_dict(), optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(), rng=capture_rng(),
                info=plain_metadata(info), config=plain_metadata(config))


def validate_resume(config, previous):
    if config.get('gradient_accumulation_steps', 1) != previous.get('gradient_accumulation_steps', 1):
        raise ValueError('Resume effective batch changed; start a new run for a batch ablation.')
    for key,default in [('precision','fp32'),('allow_tf32',False),('nthreads',0),('pin_memory',False)]:
        if config.get(key,default)!=previous.get(key,default):
            raise ValueError('Resume runtime precision/loader recipe changed: '+key)
    # Paths can change when restoring Kaggle outputs, but architecture/data recipe must not.
    keys = ('hidden_dim', 'num_queries', 'gpt_layer', 'dec_layers', 'enc_layers',
            'learned_proposal_mask', 'local_visual_attention', 'proposal_mask_bins',
            'proposal_mask_encoding_dim', 'mask_query_conditioning', 'mask_target',
            'mask_gate_min', 'mask_gate_max', 'mask_context_ratio', 'mask_min_frames',
            'mask_prune_threshold', 'mask_balanced_bce', 'proposal_mask_loss_coef',
            'corrected_query_contrastive', 'seed', 'epoch', 'lr', 'optimizer_type',
            'weight_decay', 'contrastive_loss_temperature', 'data_signature',
            'learning_rate_decay_start', 'learning_rate_decay_every', 'learning_rate_decay_rate',
            'cl_schedule_time', 'cl_schedule_val', 'cls_loss_coef', 'giou_loss_coef',
            'bbox_loss_coef', 'count_loss_coef', 'caption_loss_coef', 'loss_caption_cl',
            'contrastive_loss_coef', 'caption_cost_chunk_size')
    changed = [k for k in keys if k in previous and config.get(k) != previous[k]]
    if changed:
        raise ValueError('Resume recipe differs for: ' + ', '.join(changed) + '. Start a new run instead.')

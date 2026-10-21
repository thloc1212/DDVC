"""Fast offline CPU checks with the real vendored GPT-2 and a tiny random LM.

No tokenizer/model downloads, video data or CUDA extension are required.
"""
import importlib.util
import ast
import copy
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from transformers import GPT2Config, GPT2LMHeadModel


def main():
    torch.set_num_threads(2)
    root = Path(__file__).parent / 'ddvc' / 'CaptioningHead'
    package = types.ModuleType('_caption_test')
    package.__path__ = [str(root)]
    sys.modules[package.__name__] = package
    def read_module(name):
        spec = importlib.util.spec_from_file_location(package.__name__ + '.' + name, root / (name + '.py'))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    masks = read_module('local_visual')
    learned_masks = read_module('proposal_mask')
    caption_type = read_module('gpt2').GPT2CaptionModel
    torch.manual_seed(777)
    boxes = torch.tensor([[[0.2, 0.2], [0.8, 0.2]]])
    valid = torch.ones(1, 10, dtype=torch.bool)
    mask = masks.event_visual_mask(boxes[0], valid, context_ratio=0.1, min_frames=1)
    assert not (mask[0] & mask[1]).any()
    paired = masks.query_event_boxes(boxes, torch.tensor([0, 0, 1, 1]))
    torch.testing.assert_close(masks.event_visual_mask(paired, valid, 0.1, 1), mask.repeat_interleave(2, 0))
    matched = masks.query_event_boxes(boxes, torch.tensor([1, 0]))
    torch.testing.assert_close(masks.event_visual_mask(matched, valid, 0.1, 1), mask.flip(0))
    padded = torch.tensor([[True] * 8 + [False] * 2])
    short = masks.event_visual_mask(torch.tensor([[1.0, 0.0], [0.0, 0.0]]), padded, 0.1, 4)
    assert (short.sum(-1) == 4).all() and not short[:, 8:].any()
    singleton = masks.event_visual_mask(torch.tensor([[0.5, 0.0]]), torch.tensor([[True, False]]), 0.1, 4)
    assert singleton.tolist() == [[True, False]]

    head = caption_type.__new__(caption_type)
    nn.Module.__init__(head)
    head.opt = SimpleNamespace(add_cross_attention=True, local_visual_attention=True,
        local_visual_context_ratio=0.1, local_visual_min_frames=1, max_caption_len=6, gpt_beam_size=2)
    head.fc1, head.fc2, head.relu = nn.Linear(8, 16), nn.Linear(16, 768), nn.ReLU()
    head.model = GPT2LMHeadModel(GPT2Config(vocab_size=32, n_embd=768, n_layer=1,
        n_head=12, n_inner=32, n_positions=32, add_cross_attention=True,
        resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0,
        bos_token_id=0, eos_token_id=1, pad_token_id=0))
    head.prefix_token = torch.zeros(1, 1, dtype=torch.long)
    head.tokenizer = SimpleNamespace(encode=lambda _: [1], pad_token_id=0)
    head.eval()
    hs = torch.randn(2, 1, 8)
    visual = torch.randn(1, 10, 768, requires_grad=True)
    labels = torch.tensor([[0, 3, 4, 1], [0, 5, 6, 1]])
    tokens = torch.ones_like(labels)
    def forward(features):
        return head(hs, labels, tokens, {}, labels, features,
                    event_boxes=boxes[0], video_mask=valid).logits
    logits = forward(visual)
    changed = visual.detach().clone()
    changed[:, 5] += torch.randn_like(changed[:, 5]) * 100
    torch.testing.assert_close(logits, forward(changed))
    head.opt.local_visual_attention = False
    assert not torch.allclose(forward(visual), forward(changed)), 'Global control must see frame 5'
    head.opt.local_visual_attention = True
    logits.sum().backward()
    assert visual.grad is None, 'Keep the original frozen-feature behavior'
    assert head.fc1.weight.grad is not None and torch.isfinite(head.fc1.weight.grad).all()
    seen = []
    def capture(module, args, kwargs):
        seen.append((kwargs.get('encoder_hidden_states'), kwargs.get('encoder_attention_mask')))
    hook = head.model.register_forward_pre_hook(capture, with_kwargs=True)
    output = head.generate(hs, None, {}, visual, event_boxes=boxes[0], video_mask=valid)
    hook.remove()
    assert output.sequences.shape[0] == 2 and seen
    for features, attention in seen:
        assert features is not None and attention is not None, 'Generation lost visual context'
        assert features.shape[0] == 4, 'Beam expansion must preserve per-event contexts'
        torch.testing.assert_close(attention.bool(), mask.repeat_interleave(2, 0))
    assert len(seen) > 1, 'Test the autoregressive cached decoding steps, not only the prefix'

    # Exercise the actual matching-cost implementation, including pair chunking.
    model_file = Path(__file__).parent / 'ddvc' / 'ddvc_ret_encdec_clip_capQ.py'
    model_ast = ast.parse(model_file.read_text(encoding='utf-8'))
    model_class = next(node for node in model_ast.body if isinstance(node, ast.ClassDef) and node.name == 'CM2')
    methods = []
    for method_name in ('decode_event_boxes', 'caption_prediction_no_grad'):
        method = copy.deepcopy(next(node for node in model_class.body if isinstance(node, ast.FunctionDef) and node.name == method_name))
        method.decorator_list = []
        methods.append(method)
    def inverse_sigmoid(value):
        value = value.clamp(0, 1)
        return torch.log(value.clamp_min(1e-5) / (1-value).clamp_min(1e-5))
    scope = dict(torch=torch, inverse_sigmoid=inverse_sigmoid,
                 CrossEntropyLoss=nn.CrossEntropyLoss, query_event_boxes=masks.query_event_boxes,
                 proposal_confidence=learned_masks.proposal_confidence,
                 proposal_anchors=learned_masks.proposal_anchors)
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(model_file), 'exec'), scope)
    center_reference = torch.tensor([[[0.2], [0.8]]])
    decoded = scope['decode_event_boxes'](torch.zeros(1, 2, 2), center_reference)
    torch.testing.assert_close(decoded[..., 0], center_reference[..., 0])
    torch.testing.assert_close(decoded[..., 1], torch.full((1, 2), 0.5))
    torch.testing.assert_close(scope['decode_event_boxes'](torch.zeros_like(boxes), boxes), boxes)
    owner = SimpleNamespace(training=True, opt=SimpleNamespace(caption_decoder_type='gpt2', caption_cost_type='loss', caption_cost_chunk_size=1))
    dt = dict(cap_tensor=labels, cap_mask=tokens, gt_boxes_mask=torch.ones(1, 2, dtype=torch.bool),
              video_tensor=visual, video_mask=valid)
    other = dict(proposals_mask=torch.ones(1, 2, dtype=torch.bool))
    pairs = [(torch.tensor([0, 0, 1, 1]), torch.tensor([0, 1, 0, 1]))]
    contexts = []
    def capture_head(module, args, kwargs):
        contexts.append(kwargs['event_boxes'].detach().clone())
    hook = head.register_forward_pre_hook(capture_head, with_kwargs=True)
    with torch.no_grad():
        chunked = scope['caption_prediction_no_grad'](owner, head, dt, hs.transpose(0, 1), None,
            other, 'gpt2', pairs, event_boxes=boxes)[0]
        torch.testing.assert_close(torch.cat(contexts), paired)
        contexts.clear()
        owner.opt.caption_cost_chunk_size = 4
        unchunked = scope['caption_prediction_no_grad'](owner, head, dt, hs.transpose(0, 1), None,
            other, 'gpt2', pairs, event_boxes=boxes)[0]
    hook.remove()
    torch.testing.assert_close(chunked, unchunked)
    print('PASS: pair/match query mapping; padding and short events; local visual isolation; gradients; beam/cached generation. Decoder initialized randomly; no downloads.')
    print('PASS: actual all-pair caption costs preserve per-query intervals across chunks; center-only references preserve duration.')


if __name__ == '__main__':
    main()

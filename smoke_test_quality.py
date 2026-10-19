"""CPU checks of quality supervision, gradient isolation and real ranking/export."""
import ast
import importlib.util
import json
import math
import tempfile
from pathlib import Path
from types import SimpleNamespace
import torch
from torch import nn
import numpy as np
import copy

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('quality_module', ROOT/'ddvc/localization_quality.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def actual_definition(path, name, namespace):
    tree = ast.parse(path.read_text(encoding='utf-8'))
    definition = next(node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name == name)
    exec(compile(ast.Module(body=[definition], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[name]


def main():
    torch.set_num_threads(1)
    torch.manual_seed(7)
    boxes = torch.tensor([[[.5, .4], [.1, .1], [.5, .8]]], requires_grad=True)
    gt = [{'boxes': torch.tensor([[.5, .4]])}]
    targets = module.quality_targets(boxes, gt)
    assert torch.allclose(targets, torch.tensor([[1., 0., .5]]), atol=1e-6)
    assert not module.quality_targets(boxes, [{'boxes': torch.empty(0, 2)}]).any()
    head = module.LocalizationQualityHead(16, 12)
    query = torch.randn(1, 3, 16, requires_grad=True)
    video = torch.randn(1, 10, 12, requires_grad=True)
    valid = torch.tensor([[True]*7+[False]*3])
    scores = head(query, boxes, video, valid)
    poisoned_video = video.detach().clone()
    poisoned_video[:, 7:] = 1e6
    assert torch.allclose(scores, head(query, boxes, poisoned_video, valid))
    losses = module.quality_loss(scores, boxes, gt)
    losses['loss_quality'].backward()
    assert query.grad is None and boxes.grad is None and video.grad is None
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in head.parameters())
    optimizer = torch.optim.Adam(head.parameters(), lr=.005)
    initial = losses['loss_quality'].item()
    for _ in range(60):
        optimizer.zero_grad()
        loss = module.quality_loss(head(query, boxes, video, valid), boxes, gt)['loss_quality']
        loss.backward()
        optimizer.step()
    assert loss.item() < initial / 5, (initial, loss.item())
    restored = module.LocalizationQualityHead(16, 12)
    restored.load_state_dict(head.state_dict())
    assert torch.equal(restored(query, boxes, video, valid), head(query, boxes, video, valid))
    print('PASS: continuous IoU targets, padding isolation, detached base gradients, head learning, checkpoint round-trip')

    # Execute actual PostProcess class without importing the CUDA extension.
    box_ops = SimpleNamespace(box_cl_to_xy=module.interval_edges)
    PostProcess = actual_definition(ROOT/'ddvc/ddvc_ret_encdec_clip_capQ.py', 'PostProcess',
                                   {'torch': torch, 'nn': nn, 'copy': copy, 'box_ops': box_ops})
    post = PostProcess(SimpleNamespace(enable_contrastive=False))
    output = {'pred_logits': torch.tensor([[[0.], [3.], [1.]]]), 'pred_boxes': boxes.detach(),
              'seq': torch.tensor([]), 'caption_probs': {'cap_prob_eval': torch.empty(1, 3)},
              'pred_count': torch.tensor([[0., 0., 1.]]), 'pred_quality': torch.tensor([[.8, .1, .5]])}
    result = post(output, torch.tensor([100.]), None)[0]
    assert result['query_id'].tolist() == [1, 2, 0]
    assert torch.equal(result['quality_scores'], torch.tensor([.1, .5, .8]))
    print('PASS: actual PostProcess preserves query/quality alignment after score sorting')

    namespace = {'json': json, 'np': np, 'math': math}
    actual_definition(ROOT/'eval_utils_clip.py', 'save_dvc_json', namespace)
    rerank = actual_definition(ROOT/'eval_utils_clip.py', 'reranking', namespace)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory)/'raw.json'
        items = [dict(query_id=i, proposal_score=p, quality_score=q, sentence_score=-1.,
                      sentence='a b', cl_score=0., timestamp=[i, i+1], pred_event_count=1)
                 for i, (p, q) in enumerate([(.9, .1), (.6, .9)])]
        path.write_text(json.dumps({'results': {'v': items, 'empty': []}}))
        original = json.loads(Path(rerank(str(path), 1., 0., 2., quality_weight=0.)).read_text())
        ranked = json.loads(Path(rerank(str(path), 1., 0., 2., quality_weight=1.)).read_text())
        assert original['results']['v'][0]['query_id'] == 0
        assert ranked['results']['v'][0]['query_id'] == 1
        assert original['results']['v'][0]['joint_score'] == .9 - 1/(4+1e-5)
        assert ranked['results']['v'][0]['proposal_score'] == .6
        assert json.loads(path.read_text())['results']['v'] == items
        del items[0]['quality_score']
        path.write_text(json.dumps({'results': {'v': items}}))
        rerank(str(path), 1., 0., 2., quality_weight=0.)
        try:
            rerank(str(path), 1., 0., 2., quality_weight=1.)
        except KeyError:
            pass
        else:
            raise AssertionError('Missing quality silently accepted')
    print('PASS: actual reranking, weight-zero original formula, changed selection, empty videos, immutable raw file, strict quality validation')


if __name__ == '__main__':
    main()

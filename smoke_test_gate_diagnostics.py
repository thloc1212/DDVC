"""CPU check of gate logging and unchanged fusion outputs/gradients.

Extract the decoder class to avoid importing the optional CUDA extension.
"""
import ast
import copy
import math
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F


def main():
    source = Path(__file__).parent / 'ddvc' / 'deformable_transformer.py'
    tree = ast.parse(source.read_text(encoding='utf-8'))
    nodes = [node for node in tree.body
             if isinstance(node, (ast.ClassDef, ast.FunctionDef))
             and node.name in ('DeformableTransformerDecoder', '_get_clones')]
    namespace = dict(torch=torch, nn=nn, F=F, copy=copy, math=math)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), namespace)
    decoder_type = namespace['DeformableTransformerDecoder']

    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = nn.MultiheadAttention(8, 2)

        def forward(self, tgt, *args):
            return tgt * 1.25

    torch.manual_seed(777)
    decoder = decoder_type(Layer(), 2, return_intermediate=True, args=SimpleNamespace(
        enable_temporal_query_embedding=True, enable_gated_temporal_fusion=True))
    query = torch.randn(1, 3, 8, requires_grad=True)
    references = torch.full((1, 3, 1), 0.5)
    valid = torch.tensor([[True, True, False]])
    ratios = torch.ones(1, 1)
    actual, _ = decoder(query, references, None, None, None, ratios,
                        query_padding_mask=valid, txt_valid_ratios=ratios)
    expected, intermediates = query, []
    for _ in range(2):
        encoding = decoder._temporal_query_position(
            references[..., 0], torch.full((1, 3), 0.5), 8)
        temporal = decoder.temporal_projection(encoding)
        gate = torch.sigmoid(decoder.temporal_gate(torch.cat((expected, temporal), -1)))
        expected = (expected + gate * temporal) * 1.25
        intermediates.append(expected)
    expected = torch.stack(intermediates)
    torch.testing.assert_close(actual, expected)
    parameters = (query,) + tuple(decoder.temporal_gate.parameters()) + tuple(decoder.temporal_projection.parameters())
    actual_grads = torch.autograd.grad(actual.sum(), parameters)
    expected_grads = torch.autograd.grad(expected.sum(), parameters)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad)
    assert len(decoder.temporal_diagnostics) == 28
    for value in decoder.temporal_diagnostics.values():
        assert not value.requires_grad and value.ndim == 0 and torch.isfinite(value)
    torch.testing.assert_close(decoder.temporal_diagnostics['layer0_gate_mean'],
                               torch.sigmoid(torch.tensor(-2.0)))
    # Padded queries cannot skew the gate statistics.
    padded_gate = torch.ones_like(query) * 0.25
    padded_gate[:, 2] = 0.99
    stats = decoder._fusion_diagnostics(query, query, query, padded_gate, valid)
    torch.testing.assert_close(stats['gate_mean'], torch.tensor(0.25))
    torch.testing.assert_close(stats['gate_effective_scale'], torch.tensor(0.25))
    assert decoder._fusion_diagnostics(query, query, query, padded_gate, ~torch.ones_like(valid)) == {}
    zeros = torch.zeros_like(query)
    assert all(torch.isfinite(value) for value in decoder._fusion_diagnostics(zeros, zeros, zeros, padded_gate).values())
    fixed = decoder._fusion_diagnostics(query, query, query, None, valid)
    assert 'gate_mean' not in fixed
    torch.testing.assert_close(fixed['gate_effective_scale'], torch.tensor(1.0))
    decoder.enable_temporal_query_embedding = False
    decoder(query, references, None, None, None, ratios, txt_valid_ratios=ratios)
    assert decoder.temporal_diagnostics == {}, 'Stale statistics leaked across calls'
    print('PASS: gate statistics, padding, zero norms, cache reset, fusion outputs and gradients (CPU).')


if __name__ == '__main__':
    main()

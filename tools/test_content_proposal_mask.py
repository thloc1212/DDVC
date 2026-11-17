"""Offline behavioural tests for content masks and their real GPT-2 integration."""
from pathlib import Path
import csv
import json
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from ddvc.CaptioningHead.proposal_mask import LearnedGatedProposalMask

torch.set_num_threads(2)


def fixture(events=3, frames=9):
    torch.manual_seed(721)
    boxes = torch.tensor([[.3, .25], [.5, .6], [.7, .1]])[:events].clone().requires_grad_()
    valid = torch.ones(1, frames, dtype=torch.bool)
    scores = torch.full((events,), .35, requires_grad=True)
    queries = torch.randn(events, 512, requires_grad=True)
    visual = torch.randn(1, frames, 768, requires_grad=True)
    return boxes, valid, scores, boxes.detach(), queries, visual


def mask():
    torch.manual_seed(1731)
    return LearnedGatedProposalMask(content_conditioned=True)


def activate(module):
    # Represent a learned residual to test conditioning beyond its neutral start.
    with torch.no_grad():
        module.content_output.weight.copy_(torch.linspace(-.2, .2, 128)[None])


class MaskTests(unittest.TestCase):
    def test_audit_distinguishes_temporal_capacity_from_final_selection(self):
        from reranking.temporal_audit import write_audit
        raw = {'v': [dict(timestamp=[0., 100.]), dict(timestamp=[10., 12.])]}
        final = {'v': [raw['v'][0]]}
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            ref = folder / 'reference.json'
            ref.write_text(json.dumps({'v': dict(duration=100., timestamps=[[0., 100.], [10., 12.]])}))
            path = write_audit(raw, dict(reranker_softnms=final), [ref], folder)
            with path.open(encoding='utf-8-sig', newline='') as stream:
                rows = list(csv.DictReader(stream, delimiter='\t'))
            def percent(stage, group, measure):
                return float(next(row['percent'] for row in rows if row['stage'] == stage
                    and row['tiou'] == '0.5' and row['group'] == group and row['measure'] == measure))
            self.assertEqual(percent('raw', 'all', 'one_to_one_recall'), 100.)
            self.assertEqual(percent('reranker_softnms', 'all', 'one_to_one_recall'), 50.)
            self.assertEqual(percent('raw', 'short_le5pct', 'independent_gt_coverage'), 100.)
            self.assertEqual(percent('reranker_softnms', 'short_le5pct', 'independent_gt_coverage'), 0.)

    def test_initial_mask_exactly_matches_geometry_baseline(self):
        args = fixture()
        torch.manual_seed(1731)
        old = LearnedGatedProposalMask()
        new = mask()
        self.assertEqual(sum(p.numel() for p in old.parameters()), 33088)
        self.assertEqual(sum(p.numel() for p in new.parameters()) - 33088, 164993)
        for name, value in old.state_dict().items():
            self.assertTrue(torch.equal(value, new.state_dict()[name]), name)
        for key, value in old(*args[:4]).items():
            self.assertTrue(torch.equal(value, new(*args)[key]), key)
        with self.assertRaises(ValueError):
            new(*args[:4])

    def test_content_and_query_change_mask_but_not_hard_window(self):
        module = mask()
        activate(module)
        args = list(fixture())
        initial = module(*args)
        for index in (4, 5):
            changed = list(args)
            changed[index] = -args[index]
            actual = module(*changed)
            self.assertGreater((actual['weights'] - initial['weights']).abs().max().item(), 1e-5)
            self.assertTrue(torch.equal(actual['hard'], initial['hard']))
        self.assertTrue(((initial['weights'] >= 0) & (initial['weights'] <= 1)).all())

    def test_padding_and_repeated_pair_alignment(self):
        module = mask()
        activate(module)
        args = fixture()
        before = module(*args)
        padded = list(args)
        padded[1] = torch.cat((args[1], torch.zeros(1, 4, dtype=torch.bool)), 1)
        padded[5] = torch.cat((args[5], torch.full((1, 4, 768), float('nan'))), 1)
        actual = module(*padded)
        for key in ('logits', 'hard', 'weights'):
            # FP32 GEMM/sigmoid kernels can round differently when event/padded
            # dimensions change across PyTorch builds. Keep exact padding zeros,
            # but allow normal FP32 rounding on the valid continuous outputs.
            torch.testing.assert_close(before[key], actual[key][:, :9], rtol=1e-5, atol=1e-6)
            self.assertTrue(torch.equal(actual[key][:, 9:], torch.zeros(3, 4)), key)
        ids = torch.tensor([2, 0, 2, 1, 0, 1])
        repeated = [args[0][ids], args[1], args[2][ids], args[3][ids], args[4][ids], args[5]]
        repeated_weights = module(*repeated)['weights']
        torch.testing.assert_close(repeated_weights, before['weights'][ids], rtol=1e-5, atol=1e-6)
        # The tolerance must still reject an actual event/query alignment error.
        with self.assertRaises(AssertionError):
            torch.testing.assert_close(repeated_weights, before['weights'][ids.roll(1)], rtol=1e-5, atol=1e-6)
        # Matched E can be one, raw E is the full query budget.
        one = [args[0][:1], args[1], args[2][:1], args[3][:1], args[4][:1], args[5]]
        self.assertEqual(module(*one)['weights'].shape, (1, 9))
        many = [args[0][:1].expand(50, -1), args[1], args[2][:1].expand(50),
                args[3][:1].expand(50, -1), args[4][:1].expand(50, -1), args[5]]
        self.assertEqual(module(*many)['weights'].shape, (50, 9))

    def test_caption_gradient_updates_content_branch_and_freezes_clip(self):
        from ddvc.CaptioningHead import gpt2
        original_config = gpt2.GPT2Config

        class Tokenizer:
            eos_token = '<eos>'
            pad_token_id = 63
            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                return cls()
            def __call__(self, *args, **kwargs):
                return {'input_ids': torch.tensor([[63]])}
            def encode(self, text):
                return [13]

        def tiny_config(**kwargs):
            return original_config(vocab_size=64, n_embd=768, n_head=12, n_layer=1,
                n_positions=16, add_cross_attention=True, bos_token_id=63, eos_token_id=63,
                pad_token_id=63, resid_pdrop=0., embd_pdrop=0., attn_pdrop=0.)

        opt = SimpleNamespace(hidden_dim=512, device='cpu', gpt_layer=1, add_cross_attention=True,
            learned_proposal_mask=True, proposal_mask_bins=64, proposal_mask_encoding_dim=128,
            proposal_mask_content_conditioned=True, proposal_mask_content_dim=128,
            max_caption_len=6, gpt_beam_size=2)
        with patch.object(gpt2, 'GPT2Config', tiny_config), patch.object(gpt2, 'GPT2Tokenizer', Tokenizer):
            torch.manual_seed(777)
            head = gpt2.GPT2CaptionModel(opt)
        args = fixture(events=2)
        boxes, valid, scores, anchors, queries, visual = args
        labels = torch.tensor([[63, 2, 3, 13], [63, 5, 6, 13]])
        kwargs = dict(event_boxes=boxes, video_mask=valid, event_scores=scores, anchor_boxes=anchors)
        # Keep the generator fixed for this local update to isolate the mask path.
        optimizer = torch.optim.SGD(head.proposal_mask.parameters(), lr=.5)
        for step in range(2):
            head.zero_grad(set_to_none=True)
            output = head(queries[:, None], labels, torch.ones_like(labels), {}, labels, visual, **kwargs)
            output.loss.backward()
            self.assertTrue(torch.isfinite(output.loss))
            self.assertIsNone(visual.grad)
            self.assertGreater(head.proposal_mask.content_output.weight.grad.abs().sum().item(), 0)
            if step == 1:
                for name in ('query_project', 'visual_project', 'relative_project'):
                    self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                                        for p in getattr(head.proposal_mask, name).parameters()), name)
            optimizer.step()
        self.assertGreater(boxes.grad.abs().sum().item(), 0)
        self.assertGreater(queries.grad.abs().sum().item(), 0)
        self.assertGreater(scores.grad.abs().sum().item(), 0)
        head.eval()
        generated = head.generate(queries[:, None], None, {}, visual, **kwargs)
        self.assertEqual(generated.sequences.shape[0], 2)
        self.assertTrue(torch.isfinite(generated.sequences_scores).all())
        # Diagnostic mask BCE is not an additional training objective.
        report = head.proposal_mask.objective(*args)
        self.assertTrue(torch.isfinite(report['loss_proposal_mask']))


if __name__ == '__main__':
    unittest.main()

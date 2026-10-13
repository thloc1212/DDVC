import unittest
from types import SimpleNamespace

import torch

from ddvc.criterion import SetCriterion_cl
from ddvc.matcher import HungarianMatcher_cl


class TemporalAwareExperimentTest(unittest.TestCase):
    def test_temporal_matching_cost_prefers_correct_center(self):
        opt = SimpleNamespace(
            set_cost_caption=0.0,
            semantic_caption_matching=False,
            boundary_nll_loss_coef=0.0,
        )
        matcher = HungarianMatcher_cl(
            cost_class=0.0,
            cost_bbox=0.0,
            cost_giou=0.0,
            cost_caption=0.0,
            cost_cl=0.0,
            cost_center=1.0,
            cost_duration=1.0,
            opt=opt,
        )
        outputs = {
            'pred_logits': torch.zeros(1, 2, 1),
            'pred_boxes': torch.tensor([[[0.2, 0.1], [0.8, 0.1]]]),
            'cl_match_mats': 0.0,
        }
        targets = [{
            'labels': torch.zeros(1, dtype=torch.long),
            'boxes': torch.tensor([[0.8, 0.1]]),
        }]

        matched, _ = matcher(outputs, targets)
        self.assertEqual(matched[0][0].tolist(), [1])
        self.assertEqual(matched[0][1].tolist(), [0])

    def test_temporal_order_loss_penalizes_reversed_events_and_backprops(self):
        opt = SimpleNamespace(temporal_order_loss_coef=1.0, temporal_order_margin=0.0)
        criterion = SetCriterion_cl(1, matcher=None, weight_dict={}, losses=[], opt=opt)
        pred_boxes = torch.tensor([[[0.75, 0.1], [0.25, 0.1]]], requires_grad=True)
        outputs = {'pred_boxes': pred_boxes}
        targets = [{
            'boxes': torch.tensor([[0.25, 0.1], [0.75, 0.1]]),
        }]
        one_to_one = [(torch.tensor([0, 1]), torch.tensor([0, 1]))]
        many_to_one = [(torch.tensor([0, 1]), torch.tensor([0, 1]))]

        losses = criterion.loss_boxes(outputs, targets, (one_to_one, many_to_one), num_boxes=2)
        self.assertGreater(losses['loss_temporal_order'].item(), 0.0)
        losses['loss_temporal_order'].backward()
        self.assertIsNotNone(pred_boxes.grad)
        self.assertGreater(pred_boxes.grad.abs().sum().item(), 0.0)

    def test_temporal_order_loss_allows_ground_truth_gaps(self):
        opt = SimpleNamespace(temporal_order_loss_coef=1.0, temporal_order_margin=0.0)
        criterion = SetCriterion_cl(1, matcher=None, weight_dict={}, losses=[], opt=opt)
        pred_boxes = torch.tensor([[[0.15, 0.1], [0.85, 0.1]]], requires_grad=True)
        outputs = {'pred_boxes': pred_boxes}
        targets = [{'boxes': torch.tensor([[0.15, 0.1], [0.85, 0.1]])}]
        matches = [(torch.tensor([0, 1]), torch.tensor([0, 1]))]

        losses = criterion.loss_boxes(outputs, targets, (matches, matches), num_boxes=2)
        self.assertEqual(losses['loss_temporal_order'].item(), 0.0)


if __name__ == '__main__':
    unittest.main()

"""Mathematical checks for oracle bounds and temporal-intervention bookkeeping."""
import json
from pathlib import Path
import unittest
import numpy as np
import torch
from c1_diagnostics import matrix,matching,oracle_rank
from c1_visual_probe import shift_intervals,paired_items


class DiagnosticsTests(unittest.TestCase):
    def test_nested_events_are_not_duplicates(self):
        gt=[[0,100],[20,30]]
        rows=[dict(query_id=i,timestamp=ts,sentence=f'caption {i}') for i,ts in enumerate([[0,100],[20,30],[1,99]])]
        chosen=oracle_rank(rows,gt,2)
        self.assertEqual({r['query_id'] for r in chosen},{0,1})
        self.assertEqual(len(matching(matrix([r['timestamp'] for r in chosen],gt))),2)
        self.assertEqual(chosen[1]['sentence'],'caption 1')

    def test_matching_maximizes_cardinality_before_iou(self):
        m=np.array([[.99,.51],[.51,.49]])
        self.assertEqual(set(matching(m)),{(0,1),(1,0)})

    def test_oracle_respects_budget_and_available_queries(self):
        rows=[dict(query_id=0,timestamp=[0,10],sentence='one')]
        self.assertEqual(len(oracle_rank(rows,[[0,10],[20,30]],5)),1)
        self.assertEqual(oracle_rank(rows,[[0,10]],0),[])

    def test_shift_drops_only_join_crossing_events(self):
        self.assertEqual(shift_intervals([[0,10],[70,90],[80,100]],100,.25),
                         [(0,[25,35]),(2,[5,25])])
        self.assertEqual(shift_intervals([[0,75],[75,100]],100,.25),
                         [(0,[25,100]),(1,[0,25])])

    def test_paired_features_labels_masks_and_gt_are_aligned(self):
        feat=np.arange(8,dtype=np.float32).reshape(4,2)
        item=(feat,[[0,0],[2,3],[3,3]],[0,1,2],torch.arange(6).reshape(3,2),
              torch.ones(3,2),[[0,10],[70,90],[80,100]],100.,['a','b','c'],'video')
        normal,shifted=paired_items(item,1)
        np.testing.assert_equal(normal[0],feat)
        np.testing.assert_equal(shifted[0],np.roll(feat,1,axis=0))
        self.assertEqual(normal[5],[[0,10],[80,100]])
        self.assertEqual(shifted[5],[[25,35],[5,25]])
        self.assertEqual(normal[2],[0,2]);self.assertEqual(shifted[7],['a','c'])
        self.assertTrue(torch.equal(normal[3],item[3][[0,2]]))
        np.testing.assert_equal(item[0],feat)


if __name__=='__main__':unittest.main()

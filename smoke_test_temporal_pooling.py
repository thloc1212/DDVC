"""Numeric regression: empty GT means NaN; repaired pooling stays differentiable."""
import importlib.util
import json
from pathlib import Path
import tempfile
import torch


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def main():
    root=Path(__file__).parent
    pool=load('temporal_pool',root/'ddvc/temporal_pooling.py').pool_temporal_segments
    features=torch.arange(20,dtype=torch.float32).reshape(5,4).requires_grad_()
    assert not torch.isfinite(features[2:2].mean(dim=0)).all()
    assert torch.equal(pool(features,[[1,4]])[0],features[1:4].mean(dim=0))
    fixed=pool(features,[[2,2],[4,4],[0,0],[-2,-1],[8,8]])
    assert torch.isfinite(fixed).all()
    assert torch.equal(fixed[0],features[2]) and torch.equal(fixed[1],features[4])
    fixed.sum().backward();assert torch.isfinite(features.grad).all()
    assert pool(features,[]).shape==(0,4)
    report=load('nonfinite_report',root/'misc/nonfinite_diagnostics.py').report_nonfinite_loss
    with tempfile.TemporaryDirectory() as temp:
        path,bad=report(dict(loss_contrastive=torch.tensor(float('nan')),loss_caption=torch.tensor(2.)),
            dict(loss_contrastive=.5),dict(video_key=['v_test'],video_tensor=features,
            gt_featstamps=[[2,2]],cap_raw=[['caption']]),temp,0,30,31)
        data=json.loads(path.read_text());assert bad==['loss_contrastive'] and data['empty_gt_spans']==[0]
    print('PASS empty-slice NaN reproduced; finite nearest-frame fallback; valid-slice identity; gradients; failure diagnostics')


if __name__=='__main__':main()

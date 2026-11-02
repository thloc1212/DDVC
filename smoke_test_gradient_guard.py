"""CPU regression: a finite scalar loss does not guarantee finite gradients."""
import copy
import json
from pathlib import Path
import tempfile
import torch
from misc.gradient_guard import clip_gradients_or_fail


def main():
    with tempfile.TemporaryDirectory() as folder:
        context = dict(epoch=4, batch_index=3994, iteration=44031,
                       video_ids=['synthetic_failure'])
        torch.manual_seed(7)
        model = torch.nn.Linear(3, 2)
        original = copy.deepcopy(model)
        x = torch.tensor([[.2, .3, -.5]])
        model(x).square().sum().backward()
        original(x).square().sum().backward()
        expected = torch.nn.utils.clip_grad_norm_(original.parameters(), .1)
        actual = clip_gradients_or_fail(model, .1, folder, **context)
        assert torch.equal(actual, expected)
        assert all(torch.equal(a.grad,b.grad) for a,b in zip(model.parameters(),original.parameters()))
        assert not (Path(folder)/'nonfinite_gradients.json').exists()
        # Zero gradients are valid and should not trigger a false alarm.
        model.zero_grad(set_to_none=True)
        (model(x).sum()*0).backward()
        assert clip_gradients_or_fail(model, .1, folder, **context).item()==0
        # sqrt(0) has finite forward loss and infinite derivative.
        bad = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():bad.weight.zero_()
        optimizer = torch.optim.Adam(bad.parameters(), lr=.01)
        loss = bad(torch.ones(1,1)).sqrt().sum()
        assert torch.isfinite(loss)
        before = bad.weight.detach().clone()
        loss.backward()
        try:
            clip_gradients_or_fail(bad, .1, folder, **context)
            optimizer.step()
        except RuntimeError as error:
            assert 'before optimizer update' in str(error)
        else:raise AssertionError('Invalid gradient reached optimizer')
        assert torch.equal(before,bad.weight) and not optimizer.state
        report = json.loads((Path(folder)/'nonfinite_gradients.json').read_text())
        assert report['bad_gradients'][0]['name']=='weight'
        assert report['optimizer_update_executed'] is False
        assert report['iteration']==44031
        # Finite elements whose aggregate FP32 norm overflows must also stop.
        bad.weight.grad = torch.full_like(bad.weight, 3e38)
        # Use two elements so the norm reduction overflows rather than abs(x).
        wide = torch.nn.Linear(2, 1, bias=False)
        wide.weight.grad = torch.full_like(wide.weight, 3e38)
        try:clip_gradients_or_fail(wide, .1, folder, **context)
        except RuntimeError:pass
        else:raise AssertionError('Overflowed total gradient norm accepted')
        report=json.loads((Path(folder)/'nonfinite_gradients.json').read_text())
        assert report['bad_gradients']==[] and report['bad_parameters']==[]
    print('PASS gradient guard: original finite clipping preserved; finite-loss/Inf-gradient '
          'and aggregate overflow stop before optimizer; diagnostics recorded')


if __name__=='__main__':main()

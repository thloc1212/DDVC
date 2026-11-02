"""Preserve ordinary clipping; stop before an invalid optimizer update."""
import json
from pathlib import Path
import torch


def clip_gradients_or_fail(model, max_norm, save_folder, *, epoch, batch_index,
                           iteration, video_ids=None):
    parameters = [p for p in model.parameters() if p.grad is not None]
    if not parameters:
        raise RuntimeError('No model gradients before optimizer update')
    try:
        # Same clipping operation as the original recipe on finite gradients.
        # The error check runs before multiplying gradients by a clipping factor.
        return torch.nn.utils.clip_grad_norm_(parameters, max_norm,
                                             error_if_nonfinite=True)
    except RuntimeError as error:
        bad_gradients = []
        bad_parameters = []
        for name, parameter in model.named_parameters():
            if not bool(torch.isfinite(parameter.detach()).all()):
                bad_parameters.append(name)
            if parameter.grad is None:
                continue
            value = parameter.grad.detach()
            if value.is_sparse:
                value = value.coalesce().values()
            finite = torch.isfinite(value)
            if not bool(finite.all()):
                bad_gradients.append(dict(name=name, shape=list(parameter.shape),
                                          nonfinite=int((~finite).sum().item())))
        # A norm can overflow even if every individual gradient is finite.
        # Keep the original exception rather than pretending an empty list is safe.
        report = dict(epoch=epoch, batch_index=batch_index, iteration=iteration,
                      video_ids=video_ids, max_norm=float(max_norm),
                      error=str(error), bad_gradients=bad_gradients,
                      bad_parameters=bad_parameters,
                      optimizer_update_executed=False)
        folder = Path(save_folder)
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / 'nonfinite_gradients.json'
        temporary = path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(report, indent=2, default=str), encoding='utf8')
        temporary.replace(path)
        raise RuntimeError('Gradient clipping failed; stopped before optimizer update. '
                           'See ' + str(path)) from error

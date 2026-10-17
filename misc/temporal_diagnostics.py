"""Persist the same temporal diagnostics shown in the training log."""
import json
from pathlib import Path


def write_temporal_diagnostics(save_folder, phase, epoch, iteration, values, scores=None):
    diagnostics = {key: float(value) for key, value in values.items()
                   if key.startswith('diag_temporal_')}
    if not diagnostics:
        return
    record = {'phase': phase, 'epoch': int(epoch), 'iteration': int(iteration),
              'diagnostics': diagnostics}
    if scores is not None:
        record['metrics'] = scores
    with (Path(save_folder) / 'gate_diagnostics.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(record) + '\n')

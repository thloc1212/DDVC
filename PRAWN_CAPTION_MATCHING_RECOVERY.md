# One-command recovery after session loss

This standalone helper runs outside the training checkout. It clones the fixed
`prawn-caption-matching-20261117-v1` release if source is missing. An existing
checkout must still be commit `31f1eac41841f3a870bb56d9c74a52fc55057466`;
no pull/checkout changes an existing training source.

Select a GPU session and upload `/marimo/KLTN_DRIVE_PRIVATE.env`. The Molab
terminal must provide native Python `/tmp/uv-venv/bin/python`, git, curl, rclone.

Account A (YouCook2):

```bash
curl -fL --retry 3 -o /tmp/prawn_resume_drive.py https://raw.githubusercontent.com/thloc1212/DDVC/prawn-caption-matching-recovery-20261118-v1/tools/resume_caption_matching_drive.py && LC_ALL=C LANG=C /tmp/uv-venv/bin/python -u /tmp/prawn_resume_drive.py yc2
```

Account B (ActivityNet Captions):

```bash
curl -fL --retry 3 -o /tmp/prawn_resume_drive.py https://raw.githubusercontent.com/thloc1212/DDVC/prawn-caption-matching-recovery-20261118-v1/tools/resume_caption_matching_drive.py && LC_ALL=C LANG=C /tmp/uv-venv/bin/python -u /tmp/prawn_resume_drive.py anet
```

The helper:

1. Restores private rclone auth without printing tokens; checks GPU and live
   workers before touching experiment artifacts.
2. Lists matching `KLTNRUNS/runs/prawn_caption_match_generator/<dataset>` backups,
   requiring a committed READY manifest with a resume checkpoint, the same
   source/work/dataset/variant. Picks the newest eligible run; optional
   `--run-id ID` overrides automatic selection.
3. Finds a quality backup only if its saved generator run ID matches that run.
4. Preserves conflicting local run/prepared/log/provenance artifacts under
   `.before_drive_restore_<dataset>_<timestamp>_<suffix>`. Source is retained;
   nothing is deleted. Stops if another experiment is active.
5. Uses the original hash-verifying restore with two-slot corruption fallback.
   Restores the exact training source's environment using its setup command.
6. If the generator checkpoint is epoch-complete at its full budget, preserves
   its YAML bytes and re-stages shared feature data before resuming only quality.
   It does not rewrite `start_from` on a finished generator, which would change
   the quality cache's base-config fingerprint. Otherwise resumes the generator
   through its original optimizer/scheduler/RNG checkpoint mechanism.
7. Restores the linked quality state if present, resumes the original two-stage
   chain, then follows its logs. It never silently starts a fresh generator when
   no checkpoint exists. A new quality head is started only if no linked quality
   stage had begun.

Recovery downloads generator weights and any needed dataset/quality caches;
it is not a logs-only operation. Budget remains YC2 15 epochs / ANET 6 epochs,
video batch 1, the saved matching schedule and fixed quality beta unchanged.

Validation: six offline tests passed for fixed-tag bootstrap, no pull on existing
source, linked run selection, missing-checkpoint refusal, completion boundaries,
source-preserving conflict moves and metadata filtering. Live Google Drive/CUDA
recovery has not been executed from the local CPU workstation.

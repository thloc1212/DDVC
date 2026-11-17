# Validation of the caption-matching experiment

Local environment: Windows, CPU PyTorch 2.8.0, matching torchvision 0.23.0 CPU.
These packages were installed only into local test dependency directories;
the Molab launcher preserves its native GPU torch/torchvision.

Passed checks:

- Actual HungarianMatcher_cl: known assignment changes under joint reward;
  zero added weight exactly preserves query/GT assignments; uniqueness retained.
- Same-GT product: correct timestamp against one GT and correct caption against
  another GT cannot combine into a falsely high reward.
- Warm-up/ramp and epoch-derived resume schedule; separate lambda=0 control.
- Actual tiny vendored GPT-2 free-running generation and subsequent supervised
  forward/backward. No weight or RNG mutation during auxiliary decoding;
  mixed train/eval modes restored, including on an injected decoding failure.
- Full YC2/ANET preparation fixtures, exact staged annotation bytes, TRAIN-only
  IDF provenance/hash guard, config drift rejection and actual GPT-2 mask/box/
  confidence gradients through existing caption loss.
- Existing Drive backup/restore (alternating slots and corruption fallback),
  quality training guards/fixtures, stage ordering, and CLI safe-path imports.
- Existing content-mask tests and supervised caption-target regression checks.
- Bash syntax for the experiment and stage launchers.

Commands run with the local dependency paths configured:

```text
python tools/test_caption_matching.py
python smoke_test_c1.py
python -m unittest discover -s tools -p test_c1*.py
python tools/test_prawn_content_mask.py
python tools/test_content_proposal_mask.py
python smoke_test_caption_targets.py --gpt2
```

The fixture scorer outputs in unit tests are mocked; they are not benchmark
results. No new full-training accuracy is available from these checks.

Full native CUDA DDVC forward/backward, checkpoint smoke and inference cannot be
claimed from this CPU machine. The Molab pipeline invokes
tools/caption_matching_smoke.py before each training/resume operation. It uses
the existing actual full-model smoke with a deliberately active matching epoch,
checks the new reward/cost was executed, checks finite gradients, and runs
generation/postprocessing. A failure stops the chain before training.

The three shared-data preparation identity files are unchanged from the parent
release. No existing source checkout, trained checkpoint or Drive credential is
modified by the experiment preparation.

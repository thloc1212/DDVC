# ANET caption NaN at v_7TgvbmIn03A

The first reference, truncated to 30 GPT-2 tokens, ends with token 25 (`:`).
The legacy GPT-2 loss removes labels through token 25 when the first reference
contains it; rows without this delimiter are also completely ignored. Here all
four references become ignore_index=50256, so mean CrossEntropy is NaN.
The defect can depend on Hungarian pair order, explaining why the same video
can pass earlier epochs and fail later. Finite features/boxes and other losses
do not prevent this empty-target reduction.

The repair shares target preparation between supervised GPT-2 CE and all-pair
caption matching. It clones targets to avoid caller mutation and restores the
original next-token labels for each row whose legacy prefix rule erased every
target. Valid nonempty legacy rows are unchanged; genuinely all-padding input
raises an explicit error. The all-pair caller retains its global prefix flag
across chunks. Thus this is a target-preparation correction on degenerate rows,
not a new architecture or merely a NaN-to-zero replacement.

No model layers, weight shapes, LR, queries, mask or selection recipe change.
Checkpoint compatibility remains intact. Resume is from model-last.pth's saved
state, not from the failing in-memory minibatch. Record the code change when
reporting the run. The finite-loss-before-update guard remains enabled.

## Molab commands (same session, existing source and data)

```bash
git -C /marimo/DDVC_ActivityNet/source pull --ff-only https://github.com/thloc1212/DDVC.git exp/activitynet-current-model

cd /marimo/DDVC_ActivityNet/source
/tmp/uv-venv/bin/python smoke_test_caption_targets.py --gpt2

test -s /marimo/DDVC_ActivityNet/runs/anet_full_current_model_q10_e15_b1/model-last.pth
LC_ALL=C bash /marimo/DDVC_ActivityNet/source/tools/activitynet_molab.sh smoke-resume full
LC_ALL=C bash /marimo/DDVC_ActivityNet/source/tools/activitynet_molab.sh resume full

tail -n 80 -f /marimo/DDVC_ActivityNet/logs/anet_full_current_model_q10_e15_b1.log
```

Do not delete checkpoints/PIDs or start a second trainer to bypass the PID guard.
If pull reports local modifications, inspect the named files rather than reset
or overwrite a running checkout. No feature download, CUDA rebuild or package
upgrade is required by this target fix. Restore the environment/toolchain first
only if the Molab session itself was replaced.

The regression includes all four tokenized captions from the real failed video,
old NaN reproduction, finite repaired CE/backward, unchanged nondegenerate rows,
no input-label mutation, chunk parity, and an actual tiny vendored GPT-2 forward.

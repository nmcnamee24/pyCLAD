# `pyclad.video`

`pyclad.video` is pyCLAD's video-anomaly namespace. Its supported workflow is
UCF-Crime with a clean-room recreation of COMMAND and its ContTrain++ training
procedure.

Install the optional model dependency and inspect an archive before training:

```shell
pip install -e '.[video]'
pyclad-video ucf-audit \
  --data-root /path/to/UCF-Crime \
  --check-feature-arrays
```

Run the recreated three-task continual stream:

```shell
pyclad-video ucf-command \
  --data-root /path/to/UCF-Crime \
  --tasks T1,T2,T3 \
  --epochs 100 \
  --device cuda
```

The UCF-Crime benchmark uses 1,610 unique training videos and 290 test
videos. The 4/4/5 continual grouping is taken from COMMAND Section IV-C and is
identified everywhere as a paper recreation, not an official UCF-Crime split
or author code.

## Package boundary

The generic video layer provides:

- temporal-window and complete-video-bag metadata;
- feature stores and window-to-frame score aggregation;
- weak-label and bag-identity schema columns;
- frame ROC-AUC and average precision;
- adapters for ordinary callable and PyTorch models.

`CommandVideoModel` implements a compact, strategy-compatible COMMAND model
for comparisons with pyCLAD strategies. The primary `ucf-command` workflow
uses `PaperCommandVideoModel` and `ContTrainPlusPlusTrainer`, which retain
complete `(video, time, feature)` bags through training and replay.

See the [UCF-Crime and COMMAND guide](../../../docs/video_ucf_command.md) and
the [recreation contract](../../../docs/command_recreation.md).

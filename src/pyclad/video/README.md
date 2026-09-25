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

The video layer follows the same pattern as `pyclad.vision`:

- `VideoConcept` extends the core `Concept` with aligned temporal windows;
- `CommandVideoModel` directly implements the core `TorchBackbone` contract;
- core strategies and `TorchModelAdapter` are reused without video-specific
  copies;
- video-only code is limited to UCF-Crime records, complete-video bags,
  frame-score aggregation, and frame metrics.

`CommandVideoModel` implements a compact, strategy-compatible COMMAND model
for comparisons with pyCLAD's regular PyTorch strategies. Wrap it in the
existing `TorchModelAdapter` for strategies that consume the ordinary
`Model.fit()` interface. The primary `ucf-command` workflow runs a core `ConceptIncrementalScenario`
with `ContTrainPlusPlusStrategy` and `VideoFrameEvaluationCallback`. The
strategy delegates whole-bag optimization to `ContTrainPlusPlusTrainer`;
`PaperCommandVideoModel` is its PyTorch network. It retains complete
`(video, time, feature)` bags through training and replay. The compact model
is an experimental baseline, not the paper recreation.

```python
from pyclad.models.adapters.torch_adapter import TorchModelAdapter
from pyclad.models.training.runners.standard import StandardRunner
from pyclad.strategies.baselines.naive import NaiveStrategy
from pyclad.video.models.command.model import CommandVideoModel

backbone = CommandVideoModel(feature_dim=2048)
model = TorchModelAdapter(backbone, runner=StandardRunner(max_epochs=10), batch_size=32)
strategy = NaiveStrategy(model)
```

See the [UCF-Crime and COMMAND guide](../../../docs/video_ucf_command.md) and
the [recreation contract](../../../docs/command_recreation.md).

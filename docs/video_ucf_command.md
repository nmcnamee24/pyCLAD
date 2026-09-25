# UCF-Crime and COMMAND

`pyclad.video` provides a focused workflow for continual video anomaly
detection:

- UCF-Crime defines the benchmark and evaluation annotations.
- COMMAND defines the recreated model family and continual task stream.
- ContTrain++ provides complete-video replay and optimization.
- Frame ROC-AUC and average precision provide detector evaluation.

## Benchmark split and continual stream

The package keeps two sources separate.

The official UCF-Crime anomaly-detection benchmark contains 1,610 unique
training videos—800 normal and 810 anomalous—and 290 test videos—150 normal
and 140 anomalous. Training labels are weak video-level labels; temporal
annotations are used during test evaluation.

PyCLAD recreates the three-task 4/4/5 anomaly grouping reported in COMMAND
Section IV-C:

| Task | Anomaly classes |
| --- | --- |
| T1 | Abuse, Arrest, Arson, Assault |
| T2 | Burglary, Explosion, Fighting, RoadAccidents |
| T3 | Robbery, Shooting, Shoplifting, Stealing, Vandalism |

This stream is named `command-paper-recreation-4-4-5`. It is not presented
as a universal UCF-Crime continual split, author code, or an exact numerical
reproduction.

## Relationship to pyCLAD core

The video package follows the modality conventions used by `pyclad.vision`.
`CommandUcfCrimeDataset` implements the core `Dataset` metadata contract and
its `read_dataset()` returns a standard `ConceptsDataset`. Each
`VideoBagConcept` stores complete video bags as object-array rows and keeps
frame annotations outside the model inputs.

The primary workflow composes `ConceptIncrementalScenario`,
`ContTrainPlusPlusStrategy`, `VideoPredictionResults`, and
`VideoFrameEvaluationCallback`. The core scenario owns iteration and callback
ordering. Frame ROC-AUC and AP implement `BaseMetric`. The dedicated trainer
owns COMMAND's bag losses, dual-memory updates, calibration, and replay.

The compact `CommandVideoModel(TorchBackbone)` is an experimental baseline for
ordinary strategies using `StandardRunner` and `TorchModelAdapter`. It is a
different architecture from the full-bag paper recreation and must not be used
to claim a COMMAND reproduction.

## Install and audit

```shell
pip install -e '.[video]'

pyclad-video ucf-audit \
  --data-root /path/to/UCF-Crime \
  --check-feature-arrays \
  --output-json output/ucf-audit.json
```

The adapter expects the COMMAND archive layout:

```text
UCF-Crime/
├── all_rgbs/
├── all_flows/
├── train_normal.txt
├── train_anomaly.txt
├── test_normalv2.txt
└── test_anomalyv2.txt
```

The audit verifies manifests, train/test overlap, RGB/flow pairing, feature
shapes, and benchmark counts. It reports the official 1,610 unique training
videos separately from the 1,620 COMMAND manifest rows. Ten normal paths are
repeated in that manifest; the recreation retains them as distinct records
without describing them as additional UCF-Crime videos.

## Run the recreation

```shell
pyclad-video ucf-command \
  --data-root /path/to/UCF-Crime \
  --tasks T1,T2,T3 \
  --epochs 100 \
  --device cuda \
  --checkpoint-root output/checkpoints \
  --output-json output/command-ucf.json
```

Every video remains a complete `(time, 2048)` RGB/flow bag through AugFuseNet,
TempMamba, MemDualNet, and ContTrain++. Replay is complete-bag replay rather
than generic row replay. Dual-memory deviation is the primary anomaly score;
the classifier score is reported only as a diagnostic.

Each task result records training diagnostics, frame-level evaluation,
checkpoint location, archive audit, scenario provenance, architecture and
loss configuration, runtime information, and the source commit SHA.

The historical `command-paper` command remains callable as an unadvertised
compatibility alias.

## Programmatic scenario access

```python
from pyclad.scenarios.concept_incremental import ConceptIncrementalScenario
from pyclad.video.callbacks.frame_evaluation import VideoFrameEvaluationCallback
from pyclad.video.datasets.command_ucf_crime import CommandUcfCrimeDataset
from pyclad.video.models.command.paper_training import ContTrainPlusPlusTrainer
from pyclad.video.strategies.cont_train import ContTrainPlusPlusStrategy

reader = CommandUcfCrimeDataset("/path/to/UCF-Crime")
dataset = reader.read_dataset()
strategy = ContTrainPlusPlusStrategy(ContTrainPlusPlusTrainer())
callback = VideoFrameEvaluationCallback(strategy)
ConceptIncrementalScenario(dataset, strategy, [callback]).run()
results = callback.info()
```

The supported stream is T1, T2, T3. Ordered subsets are accepted for smoke
runs; duplicates and reordered tasks are rejected. The serializable protocol
metadata in `ucf_crime.scenarios` describes task composition; execution uses
the core scenario above. Historical research-variant metadata remains
available for compatibility but is not an executable COMMAND workflow.

## Evidence boundary

Unit tests establish package behavior. A successful archive audit establishes
the local data contract. A smoke run establishes execution mechanics. None of
those results alone reproduces COMMAND's published numbers.

A research reproduction claim requires full training under the declared
configuration, repeated seeds, retained checkpoints and result files, and a
comparison against the paper under an equivalent evaluator.

See [COMMAND recreation contract](command_recreation.md) for the exact source
precedence and implementation assumptions.

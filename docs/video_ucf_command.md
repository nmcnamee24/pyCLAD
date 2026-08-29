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

`pyclad.video` is a sibling of `pyclad.vision`, not a second framework. Its
`VideoConcept` extends the regular `Concept` with aligned temporal windows,
and `CommandVideoModel` directly implements the existing `TorchBackbone`
interface. Ordinary strategies reuse `TorchModelAdapter`; no video-specific
strategy or model adapter is required.

The paper recreation remains a small deliberate exception because generic
pyCLAD strategies batch matrix rows, while COMMAND and ContTrain++ train and
replay variable-length, complete-video bags.

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
from pyclad.video.ucf_crime.scenarios import build_command_ucf_crime_scenario

primary = build_command_ucf_crime_scenario()
classwise = build_command_ucf_crime_scenario("classwise")
shooting_last = build_command_ucf_crime_scenario(
    "12-1",
    held_out_class="Shooting",
)
```

`classwise` and `12-1` are controlled PyCLAD research variants. They are
not COMMAND protocols and do not replace the official benchmark split.

## Evidence boundary

Unit tests establish package behavior. A successful archive audit establishes
the local data contract. A smoke run establishes execution mechanics. None of
those results alone reproduces COMMAND's published numbers.

A research reproduction claim requires full training under the declared
configuration, repeated seeds, retained checkpoints and result files, and a
comparison against the paper under an equivalent evaluator.

See [COMMAND recreation contract](command_recreation.md) for the exact source
precedence and implementation assumptions.

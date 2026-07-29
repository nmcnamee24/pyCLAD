# `pyclad.video`

This package adds video anomaly detection without modifying pyCLAD's existing
interfaces.

The boundary is intentionally simple:

1. Decode a video and create frame or window embeddings inside
   `pyclad.video`.
2. Store window metadata in `VideoWindow` sidecars.
3. Pass only a two-dimensional `float32` feature matrix into an existing
   pyCLAD strategy.
4. Map the returned window anomaly scores back to frames inside
   `pyclad.video`.

```python
from pyclad.strategies.baselines.cumulative import CumulativeStrategy
from pyclad.video import CallableVideoAnomalyModel

model = CallableVideoAnomalyModel(
    fit_fn=fit_video_scorer,
    score_fn=score_video_windows,
)
strategy = CumulativeStrategy(model)

strategy.learn(train_video_concept.features)
prediction = strategy.predict(test_video_concept.features)
```

Because strategy inputs are ordinary NumPy matrices, Naive, Cumulative, MSTE,
Replay Only, and Replay Enhanced retain their existing behavior. A
`TorchVideoBackbone` implementation can also be used by EWC, LwF, A-GEM, and
DER++.

Labels, video identifiers, frame ranges, anomaly classes, timestamps, and
payloads remain on `VideoFeatureConcept` and `VideoWindow`. Core pyCLAD never
needs to understand or preserve those fields.

## Weak supervision

When a model needs numeric training targets, `VideoStrategySchema` can reserve
columns in the same matrix:

```python
schema = VideoStrategySchema(
    feature_dim=2048,
    target_names=("weak_label",),
)

concept = VideoFeatureConcept(
    name="camera-1",
    features=window_embeddings,
    windows=window_metadata,
    strategy_schema=schema,
    strategy_targets={"weak_label": video_labels},
)
```

Existing cumulative and replay strategies safely concatenate and select rows
from this matrix. `CallableWeaklySupervisedVideoModel` separates the reserved
targets before calling its fit function, so labels never become model input
features. Test concepts may omit targets; their reserved columns are filled
with `NaN` and ignored during prediction.

## COMMAND

`CommandVideoModel` is a clean-room PyTorch implementation of COMMAND's public
architecture: augmented feature fusion, a selective temporal block, trainable
normal/anomalous memories, and weak-label losses. It implements
`TorchVideoBackbone`, so it works with ordinary pyCLAD strategies and the
differentiable EWC, LwF, A-GEM, and DER++ strategies.

```python
from pyclad.strategies.regularization.ewc import EWCStrategy
from pyclad.video import CommandVideoModel, VideoStrategySchema

schema = VideoStrategySchema(2048, target_names=("weak_label", "bag_id"))
model = CommandVideoModel(
    feature_dim=2048,
    strategy_schema=schema,
    epochs=10,
)
strategy = EWCStrategy(model=model, epochs=10)

strategy.learn(
    schema.pack(
        train_features,
        {
            "weak_label": video_labels,
            "bag_id": globally_unique_numeric_video_ids,
        },
    )
)
prediction = strategy.predict(test_features)
```

Passing feature-only matrices to `predict` is supported. Missing weak labels
during training use the model's differentiable memory-compactness objective.
When `bag_id` is present, the weak classification objective performs
multiple-instance max pooling within each video represented in a mini-batch.
Bag IDs must be numeric and globally unique across concepts.
Replay is deliberately managed by the selected pyCLAD strategy rather than by
a second, hidden buffer inside COMMAND.

## NOLA

`NolaVideoModel` adapts NOLA's nominal spatial/temporal k-nearest-neighbor
memories and ODIT/CUSUM statistic. The standard matrix is five spatial object
features, three count/time features, and an optional trajectory prediction
error. Custom column layouts are supported.

```python
from pyclad.strategies.baselines.cumulative import CumulativeStrategy
from pyclad.video import NolaVideoModel, pack_nola_features

features, layout = pack_nola_features(
    spatial=object_box_and_class_features,
    temporal=vehicle_person_and_time_features,
    trajectory_error=next_box_errors,
)
strategy = CumulativeStrategy(NolaVideoModel(layout=layout))
strategy.learn(features)
prediction = strategy.predict(features)
```

The optional `NolaTrajectoryPredictor` is a modern PyTorch replacement for the
original TensorFlow 1.x three-layer LSTM. `compute_average_precision_delay`
provides NOLA's threshold-swept APD evaluation.

NOLA is adapted under its MIT license. COMMAND is independently implemented
from its public paper and architecture because its reference repository does
not provide a source-code license. Exact provenance is recorded in
`third_party/PROVENANCE.md`.

## Real datasets and CLI

The COMMAND adapter reads the reference archive directly: `all_rgbs/`,
`all_flows/`, and its four train/test list files. RGB and flow windows are
concatenated into 2048-dimensional rows, while weak labels and globally unique
video bag IDs remain reserved schema columns. Each anomaly class is paired with
a disjoint, equally sized normal-video shard to form a continual experience.

```shell
PYTHONPATH=src python -m pyclad.video command \
  --data-root /path/to/UCF-Crime \
  --strategy cumulative \
  --concepts Abuse,Arrest \
  --videos-per-class 1
```

Set a numeric limit to `0` to remove it. COMMAND supports `naive`,
`cumulative`, `mste`, `replay-only`, `replay-enhanced`, `ewc`, `lwf`, `agem`,
and `der++`.

NOLA's archive includes prepared JSON and tracks for the eleven training
stages, but its test split contains only MP4s. If a released training cache is
incomplete, regenerate that video into a separate stage/video tree with
`nola-preprocess --source-root` and pass it with `--processed-train-root`; the
source archive is never modified. The lightweight compatibility
path uses torchvision SSDLite and the built-in IoU tracker:

```shell
PYTHONPATH=src python -m pyclad.video nola-preprocess \
  --data-root /path/to/NOLA \
  --output-root /path/to/nola-processed \
  --video-ids mon_14_1 \
  --frame-stride 30 \
  --max-frames 12
```

Use `--video-ids all` and remove the smoke limits to prepare the complete test
split.

For the paper protocol, preprocess every test frame with the paper's
YOLOv4-CSP detector family and DeepSORT tracker. The maintained DeepSORT
adapter requires the optional `deep-sort-realtime` package:

```shell
python -m pip install deep-sort-realtime
PYTHONPATH=src python -m pyclad.video nola-preprocess \
  --data-root /path/to/NOLA \
  --output-root /path/to/nola-processed-paper \
  --video-ids all \
  --frame-stride 1 \
  --confidence-threshold 0.25 \
  --detector darknet \
  --darknet-binary /path/to/darknet \
  --darknet-data /path/to/coco.data \
  --darknet-config /path/to/yolov4-csp.cfg \
  --darknet-weights /path/to/yolov4-csp.weights \
  --darknet-names /path/to/coco.names \
  --tracker deepsort
```

For a benchmark run, point the stage loader at the extracted NOLA root and the
test loader at that reusable preprocessing cache:

```shell
PYTHONPATH=src python -m pyclad.video nola \
  --data-root /path/to/NOLA \
  --processed-train-root /path/to/nola-processed-train \
  --processed-test-root /path/to/nola-processed \
  --ground-truth /path/to/gt.txt \
  --implementation paper \
  --strategy replay-enhanced \
  --evaluate-each-stage \
  --stages M-Train,Train0,Train1,Train2,Train3,Train4,Train5,Train6,Train7,Train8,Train9
```

NOLA evaluates each test video separately so its ODIT/CUSUM state resets at
video boundaries in legacy mode. Paper mode uses the documented three-layer
k-DNN, two-step decision LSTM, three-layer trajectory LSTM, contextual object
features, and replay without applying the legacy fixed-drift ODIT transform.
It reports the initial model and CL-1 through CL-10 and fails if final anomaly
scores collapse to one value. NOLA supports `naive`, `cumulative`, `mste`,
`replay-only`, and `replay-enhanced`; gradient-regularization methods require
a trainable `TorchVideoBackbone` such as COMMAND.

The publication does not release its k-DNN/replay source, learned detector or
tracker checkpoints, exact time-bin boundaries, replay schedule, or
stage-specific thresholds/epoch counts. This implementation records all such
choices in result metadata and follows the architecture stated in the paper;
exact reproduction of Table 2 therefore remains contingent on unavailable
author artifacts.

## Reproducible HPC runs

All three CLI workflows accept `--seed` (default `42`) and `--output-json`.
Structured result files include the complete arguments, Git commit, UTC
timestamp, runtime versions, dataset counts, and a validation section that
lists any non-finite metric paths. COMMAND exposes the research-sized
`--hidden-dim 128`, `--embedding-dim 128`, and `--memory-size 64` defaults.

AU Lovelace LSF wrappers for the complete COMMAND run, the throttled NOLA
preprocessing array, and NOLA scoring live in `hpc/`. They keep all generated
data and logs beneath `~/pyvad_hpc`; see `hpc/README.md` for the required
remote layout and submission order.

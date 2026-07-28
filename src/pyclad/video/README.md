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

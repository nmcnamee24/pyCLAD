# Video model provenance

## COMMAND

- Reference repository: <https://github.com/khansobuz/video-anomaly-detection>
- Inspected commit: `bfc986067aa9352da4d1598d225184a2481a4d9c`
- License status at inspection: no license file was present.

The `pyclad.video.models.command` code is a clean-room implementation based on
the published architecture and observable behavior. No source code was copied
from the reference repository. Its implementation deliberately uses pyCLAD's
existing strategy-owned replay and regularization instead of embedding a
second continual-learning controller in the model.

## NOLA

- Reference repository: <https://github.com/Secure-and-Intelligent-Systems-Lab/NOLA>
- Inspected commit: `d62096d8fb681899d22bf3cbfc9d01bd58dcf138`
- License: MIT, Copyright (c) 2021 SIS Lab.

The `pyclad.video.models.nola` and `pyclad.video.metrics.nola` modules adapt
the reference implementation into reusable NumPy/scikit-learn components. The
legacy TensorFlow 1.x trajectory network was reimplemented with optional
PyTorch. The upstream license notice is included in `NOLA_LICENSE.txt`.

The paper's Table 2 k-DNN/experience-replay implementation is not present in
the reference repository. `NolaPaperModel` is therefore a clean-room
implementation of the architecture and training protocol documented in the
paper: three 20-unit k-DNN hidden layers, a single-layer two-step decision
LSTM, synthetic distance anomalies, and pyCLAD-owned replay. Paper
preprocessing invokes a separately installed native AlexeyAB Darknet binary,
matching the public NOLA preprocessing script, and uses the separately
installed `deep-sort-realtime` package. An OpenCV-DNN adapter remains available
for compatibility. No third-party detector code, weights, or DeepSORT source
are vendored here.

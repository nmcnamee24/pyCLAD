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

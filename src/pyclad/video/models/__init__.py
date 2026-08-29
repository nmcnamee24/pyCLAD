"""Video model interfaces compatible with existing pyCLAD strategies."""

from pyclad.video.models.base import VideoAnomalyModel
from pyclad.video.models.callable import (
    CallableVideoAnomalyModel,
    CallableWeaklySupervisedVideoModel,
)

__all__ = [
    "CallableVideoAnomalyModel",
    "CallableWeaklySupervisedVideoModel",
    "CommandNormalOnlyModel",
    "CommandVideoModel",
    "PaperCommandVideoModel",
    "TorchVideoBackbone",
    "VideoAnomalyModel",
]


def __getattr__(name):
    if name == "TorchVideoBackbone":
        from pyclad.video.models.torch import TorchVideoBackbone

        return TorchVideoBackbone
    if name in {"CommandNormalOnlyModel", "CommandVideoModel", "PaperCommandVideoModel"}:
        from pyclad.video.models import command

        value = getattr(command, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

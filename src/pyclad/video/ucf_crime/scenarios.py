"""Named UCF-Crime continual scenarios supported by the COMMAND package.

The scenario contract is deliberately independent from model code and data
loading.  A result can therefore serialize the exact task composition before
an expensive training job starts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from pyclad.video.datasets.command_ucf_crime import (
    COMMAND_UCF_CRIME_CONCEPT_ORDER,
    COMMAND_UCF_CRIME_PAPER_TASKS,
)

COMMAND_PAPER_RECREATION_PROTOCOL = "command-paper-recreation-4-4-5"
# Backward-compatible symbol for callers that imported the original constant.
PAPER_PROTOCOL = COMMAND_PAPER_RECREATION_PROTOCOL
CLASSWISE_PROTOCOL = "classwise-13"
HELD_OUT_LAST_PROTOCOL = "held-out-last-12-1"
COMMAND_UCF_CRIME_PROTOCOLS = (
    PAPER_PROTOCOL,
    CLASSWISE_PROTOCOL,
    HELD_OUT_LAST_PROTOCOL,
)

_PROTOCOL_ALIASES = {
    "paper": PAPER_PROTOCOL,
    "4-4-5": PAPER_PROTOCOL,
    "paper-4-4-5": PAPER_PROTOCOL,
    "command-paper-recreation": PAPER_PROTOCOL,
    PAPER_PROTOCOL: PAPER_PROTOCOL,
    "classwise": CLASSWISE_PROTOCOL,
    "13-task": CLASSWISE_PROTOCOL,
    CLASSWISE_PROTOCOL: CLASSWISE_PROTOCOL,
    "12-1": HELD_OUT_LAST_PROTOCOL,
    "held-out-last": HELD_OUT_LAST_PROTOCOL,
    HELD_OUT_LAST_PROTOCOL: HELD_OUT_LAST_PROTOCOL,
}


@dataclass(frozen=True)
class CommandUcfCrimeTask:
    """One ordered COMMAND experience over one or more anomaly classes."""

    name: str
    anomaly_classes: Tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("task name must be non-empty")
        if not self.anomaly_classes:
            raise ValueError("a UCF-Crime task must contain at least one anomaly class")
        unknown = sorted(set(self.anomaly_classes) - set(COMMAND_UCF_CRIME_CONCEPT_ORDER))
        if unknown:
            raise ValueError(f"unknown UCF-Crime anomaly classes: {unknown}")
        if len(set(self.anomaly_classes)) != len(self.anomaly_classes):
            raise ValueError("a UCF-Crime task must not repeat an anomaly class")

    def as_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "anomaly_classes": list(self.anomaly_classes),
        }


@dataclass(frozen=True)
class CommandUcfCrimeScenario:
    """Serializable task contract for one UCF-Crime COMMAND run."""

    protocol: str
    tasks: Tuple[CommandUcfCrimeTask, ...]
    held_out_class: Optional[str] = None

    def __post_init__(self) -> None:
        if self.protocol not in COMMAND_UCF_CRIME_PROTOCOLS:
            raise ValueError(f"unsupported UCF-Crime protocol: {self.protocol!r}")
        flattened = tuple(anomaly_class for task in self.tasks for anomaly_class in task.anomaly_classes)
        expected = set(COMMAND_UCF_CRIME_CONCEPT_ORDER)
        if set(flattened) != expected or len(flattened) != len(expected):
            raise ValueError("a UCF-Crime scenario must cover every anomaly class exactly once")

    @property
    def task_sizes(self) -> Tuple[int, ...]:
        return tuple(len(task.anomaly_classes) for task in self.tasks)

    @property
    def display_name(self) -> str:
        if self.protocol == PAPER_PROTOCOL:
            return "COMMAND paper recreation - UCF-Crime 4-4-5 continual stream"
        if self.protocol == CLASSWISE_PROTOCOL:
            return "PyCLAD UCF-Crime classwise 13-task research variant"
        return "PyCLAD UCF-Crime held-out-last 12+1 research variant"

    @property
    def provenance(self) -> Dict[str, object]:
        if self.protocol == PAPER_PROTOCOL:
            return {
                "relationship": "clean-room recreation",
                "source_title": (
                    "COMMANDing anomalies: Continual video anomaly detection via "
                    "dual-memory and temporal mamba modeling"
                ),
                "source_section": "Section IV-C",
                "source_doi": "10.1016/j.neucom.2026.132943",
                "scope": "three-task UCF-Crime anomaly-class grouping (4, 4, 5)",
                "not_claimed": "author code, author environment, or exact numerical reproduction",
            }
        return {
            "relationship": "PyCLAD controlled research variant",
            "source_title": None,
            "source_section": None,
            "source_doi": None,
            "scope": "not a task stream reported by the COMMAND paper",
        }

    def as_dict(self) -> Dict[str, object]:
        return {
            "protocol": self.protocol,
            "display_name": self.display_name,
            "provenance": self.provenance,
            "held_out_class": self.held_out_class,
            "task_sizes": list(self.task_sizes),
            "tasks": [task.as_dict() for task in self.tasks],
        }


def build_command_ucf_crime_scenario(
    protocol: str = PAPER_PROTOCOL,
    *,
    held_out_class: Optional[str] = None,
) -> CommandUcfCrimeScenario:
    """Build one of the three accepted UCF-Crime continual task contracts.

    ``command-paper-recreation-4-4-5`` is the primary package workflow and is
    explicitly labeled as PyCLAD's clean-room recreation of the task grouping
    reported in COMMAND Section IV-C. ``classwise-13`` and
    ``held-out-last-12-1`` are PyCLAD research variants, not COMMAND protocols.
    The held-out protocol requires the final anomaly class to be named
    explicitly. Historical ``paper-4-4-5`` input remains an accepted alias.
    """

    normalized = _PROTOCOL_ALIASES.get(protocol.strip().lower())
    if normalized is None:
        raise ValueError(f"unknown UCF-Crime protocol {protocol!r}; choose from {COMMAND_UCF_CRIME_PROTOCOLS}")
    if normalized != HELD_OUT_LAST_PROTOCOL and held_out_class is not None:
        raise ValueError("held_out_class is valid only for the held-out-last-12-1 protocol")

    if normalized == PAPER_PROTOCOL:
        tasks = tuple(
            CommandUcfCrimeTask(name=f"T{index}", anomaly_classes=tuple(classes))
            for index, classes in enumerate(COMMAND_UCF_CRIME_PAPER_TASKS, start=1)
        )
    elif normalized == CLASSWISE_PROTOCOL:
        tasks = tuple(
            CommandUcfCrimeTask(name=f"T{index:02d}-{anomaly_class}", anomaly_classes=(anomaly_class,))
            for index, anomaly_class in enumerate(COMMAND_UCF_CRIME_CONCEPT_ORDER, start=1)
        )
    else:
        if held_out_class not in COMMAND_UCF_CRIME_CONCEPT_ORDER:
            raise ValueError(
                "held-out-last-12-1 requires held_out_class to be one of " f"{COMMAND_UCF_CRIME_CONCEPT_ORDER}"
            )
        retained = tuple(
            anomaly_class for anomaly_class in COMMAND_UCF_CRIME_CONCEPT_ORDER if anomaly_class != held_out_class
        )
        tasks = (
            CommandUcfCrimeTask(name="T1-retained-12", anomaly_classes=retained),
            CommandUcfCrimeTask(name=f"T2-new-{held_out_class}", anomaly_classes=(held_out_class,)),
        )

    return CommandUcfCrimeScenario(
        protocol=normalized,
        tasks=tasks,
        held_out_class=held_out_class,
    )

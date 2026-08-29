"""Primary pyCLAD video package: UCF-Crime with COMMAND.

Dataset contracts and auditing stay importable without PyTorch. Model and
trainer symbols load lazily when the optional video dependency is installed.
"""

from pyclad.video.datasets.command_ucf_crime import (
    COMMAND_UCF_CRIME_CONCEPT_ORDER,
    COMMAND_UCF_CRIME_PAPER_TASKS,
    CommandUcfCrimeDataset,
    CommandUcfCrimeRecord,
)
from pyclad.video.ucf_crime.audit import (
    COMMAND_RELEASE_TRAIN_ANOMALY_RECORDS,
    COMMAND_RELEASE_TRAIN_NORMAL_RECORDS,
    COMMAND_RELEASE_TRAIN_RECORDS,
    OFFICIAL_TEST_ANOMALY_VIDEOS,
    OFFICIAL_TEST_NORMAL_VIDEOS,
    OFFICIAL_TEST_VIDEOS,
    OFFICIAL_TRAIN_ANOMALY_VIDEOS,
    OFFICIAL_TRAIN_NORMAL_VIDEOS,
    OFFICIAL_TRAIN_RECORDS,
    OFFICIAL_TRAIN_VIDEOS,
    OFFICIAL_UNIQUE_TRAIN_VIDEOS,
    CommandUcfCrimeAudit,
    audit_command_ucf_crime,
)
from pyclad.video.ucf_crime.scenarios import (
    CLASSWISE_PROTOCOL,
    COMMAND_PAPER_RECREATION_PROTOCOL,
    COMMAND_UCF_CRIME_PROTOCOLS,
    HELD_OUT_LAST_PROTOCOL,
    PAPER_PROTOCOL,
    CommandUcfCrimeScenario,
    CommandUcfCrimeTask,
    build_command_ucf_crime_scenario,
)

__all__ = [
    "CLASSWISE_PROTOCOL",
    "COMMAND_PAPER_RECREATION_PROTOCOL",
    "COMMAND_RELEASE_TRAIN_ANOMALY_RECORDS",
    "COMMAND_RELEASE_TRAIN_NORMAL_RECORDS",
    "COMMAND_RELEASE_TRAIN_RECORDS",
    "COMMAND_UCF_CRIME_CONCEPT_ORDER",
    "COMMAND_UCF_CRIME_PAPER_TASKS",
    "COMMAND_UCF_CRIME_PROTOCOLS",
    "HELD_OUT_LAST_PROTOCOL",
    "OFFICIAL_TEST_ANOMALY_VIDEOS",
    "OFFICIAL_TEST_NORMAL_VIDEOS",
    "OFFICIAL_TEST_VIDEOS",
    "OFFICIAL_TRAIN_ANOMALY_VIDEOS",
    "OFFICIAL_TRAIN_NORMAL_VIDEOS",
    "OFFICIAL_TRAIN_RECORDS",
    "OFFICIAL_TRAIN_VIDEOS",
    "OFFICIAL_UNIQUE_TRAIN_VIDEOS",
    "PAPER_PROTOCOL",
    "CommandUcfCrimeAudit",
    "CommandUcfCrimeDataset",
    "CommandUcfCrimeRecord",
    "CommandUcfCrimeScenario",
    "CommandUcfCrimeTask",
    "CommandVideoModel",
    "ContTrainPlusPlusTrainer",
    "PaperCommandTrainerConfig",
    "PaperCommandVideoModel",
    "audit_command_ucf_crime",
    "build_command_ucf_crime_scenario",
]


def __getattr__(name):
    if name in {
        "CommandVideoModel",
        "ContTrainPlusPlusTrainer",
        "PaperCommandTrainerConfig",
        "PaperCommandVideoModel",
    }:
        from pyclad.video.models import command

        value = getattr(command, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

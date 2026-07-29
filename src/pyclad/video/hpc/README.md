# AU Lovelace LSF wrappers

These wrappers keep COMMAND and NOLA runs inside the `pyclad.video` boundary.
Submit them from the remote account's home directory after creating
`~/pyvad_hpc` and exporting one shared run ID:

```shell
export PYCLAD_RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
bsub < pyvad_hpc/code/src/pyclad/video/hpc/command.lsf.sh
bsub < pyvad_hpc/code/src/pyclad/video/hpc/nola_preprocess.lsf.sh
```

Submit `nola_score.lsf.sh` only after every NOLA array index has completed and
all 50 cache validation JSON files report `"valid": true`.

The wrappers expect:

- an editable installation in `~/pyvad_hpc/env`;
- the exact archived source in `~/pyvad_hpc/code`;
- its Git commit in `~/pyvad_hpc/code/PYCLAD_COMMIT_SHA`;
- COMMAND data under `~/pyvad_hpc/data/command_ucf_crime/UCF-Crime`;
- NOLA data and ground truth under `~/pyvad_hpc/data/nola`;
- 50 sorted NOLA test IDs in `~/pyvad_hpc/jobs/nola_test_ids.txt`.

Result JSON, environment snapshots, cache validations, scheduler logs, and GPU
diagnostics are written below `~/pyvad_hpc/results` and `~/pyvad_hpc/logs`.

# COMMAND provenance

- Paper: *COMMANDing anomalies: Continual video anomaly detection via
  dual-memory and temporal mamba modeling*
- DOI: <https://doi.org/10.1016/j.neucom.2026.132943>
- Public reference repository:
  <https://github.com/khansobuz/video-anomaly-detection>
- Reference commit inspected: `bfc986067aa9352da4d1598d225184a2481a4d9c`
- License status when inspected: no license file was present.

The `pyclad.video.models.command` implementation is independently written from
the published architecture and documented experimental behavior. No source
code from the unlicensed reference repository is copied into pyCLAD.

The three-task 4/4/5 class grouping is recreated from COMMAND Section IV-C.
It is not an official UCF-Crime continual split. Architecture and training
details not fixed by the paper are identified as assumptions in
`docs/command_recreation.md` and serialized with experiment output.

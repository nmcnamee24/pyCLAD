# COMMAND recreation contract

This implementation is PyCLAD's clean-room recreation of COMMAND. It is based
on the published paper and public experimental material; it is not the
authors' source code and does not claim exact numerical equivalence.

## Source precedence

1. Published equations and implementation tables define the graph and stated
   dimensions.
2. The public experiment script is consulted only where the paper is silent.
3. Remaining unknowns are exposed as serialized PyCLAD assumptions.

The recreated continual grouping comes from Section IV-C of *COMMANDing
anomalies: Continual video anomaly detection via dual-memory and temporal
mamba modeling*, DOI
[`10.1016/j.neucom.2026.132943`](https://doi.org/10.1016/j.neucom.2026.132943).

## Architecture contract

| Component | Recreation contract |
| --- | --- |
| AugFuseNet | Concatenate 1,024-dimensional RGB and 1,024-dimensional flow features while retaining 32 temporal snippets |
| TempMamba | Project the 2,048-dimensional stream into gated temporal branches |
| Local operator | Causal depthwise one-dimensional convolution followed by SiLU and dropout |
| State-space operator | Diagonal recurrent state with learned input, readout, and skip parameters |
| Attention | Sequential channel and temporal one-dimensional CBAM |
| MemDualNet | Learnable primary and secondary `256 x 2048` memories with Euclidean nearest-slot distances |
| Heads | Per-snippet anomaly classifier and 128-dimensional contrastive projection |

The default architecture has 15,036,943 trainable parameters. Tests lock that
count and verify that the complete temporal dimension reaches the temporal
blocks.

## Training contract

`ContTrainPlusPlusTrainer` owns the complete-bag training loop and replay
buffer. The objective combines weak MIL ranking, supervised contrastive,
focal classification, anomaly separation, temporal sparsity, temporal
smoothness, and score regularization.

Replay selection is deterministic for a fixed seed and balances weak label,
previous task, and anomaly-peak quartile. Primary and secondary memory
parameters use separate optimizer groups. Optimizer, scheduler, replay,
calibration, and model state are checkpointed.

The published full-training defaults exposed by the CLI are 100 epochs, batch
size 32, Adam at `1e-4`, a secondary-memory learning rate of `1e-5`,
learning-rate decay by 0.1 every 10 epochs, and a 1,000-video replay buffer.

## Explicit assumptions and non-claims

The paper does not completely specify the number of temporal blocks, CBAM
reduction ratio, CBAM temporal kernel, state width, projection width, dropout,
differentiated memory learning rates, novelty threshold, or exact memory-bank
replacement decision. Defaults are recorded in every result and checkpoint.

Results from this implementation must be labeled **COMMAND paper recreation**
or **clean-room recreation of COMMAND**. They must not be labeled author code,
bitwise equivalent, or an exact reproduction without separate experimental
evidence.

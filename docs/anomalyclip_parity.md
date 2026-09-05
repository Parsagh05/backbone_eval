# AnomalyCLIP parity audit

Alireza's instruction for the learnable-prompt setting was: CoOp-style shallow
prompts, object-agnostic like AnomalyCLIP, **without** deep token tuning inside
the text transformer, trained with **AnomalyCLIP's loss**. Everything he did
not specify now follows the official `zqhang/AnomalyCLIP` implementation.

## Protocol now implemented

| Aspect | Official AnomalyCLIP | Here |
| --- | --- | --- |
| prompt form | `[X × 12] object.` / `[X × 12] damaged object.` | same |
| object-agnostic prompts | yes | same |
| deep text-prompt tuning | yes | **deliberately absent per Alireza** |
| trainable shallow tensors | normal and abnormal contexts | same; encoders frozen and asserted |
| visual stages | 6, 12, 18, 24 | same fractions for CLIP and SigLIP2 |
| CLIP dense visual path | DPAM V-V attention, starting at layer 6 | same functional dual path |
| SigLIP2 dense visual path | not applicable | native `map_token` projection at the same four stages |
| training objective | image CE + `4 × Σ_layers(focal + dice_abnormal + dice_normal)` | same |
| similarity temperature | 0.07 | same |
| inference anomaly map | softmax per layer, resize, sum layers | same |
| image anomaly score | global abnormal-text probability only | same by default |
| Gaussian smoothing | sigma 4 | same |
| evaluation-map resolution | 518×518 | same by default |
| optimiser | Adam, lr 1e-3, betas (0.5, 0.999), 15 epochs | same |

The one intentional method difference is the one Alireza explicitly requested:
there are no compound/deep prompt tokens inside the text transformer. The only
trainable parameters are two shallow context tensors, one normal and one
abnormal.

## Cross-backbone adaptation

DPAM is specific to CLIP's transformer implementation. SigLIP2 has no literal
DPAM module, so its patch tokens are projected through its own attention-pool
head (`map_token`). Both backbones expose four proportional stages. This keeps
the number of per-layer focal/Dice terms equal without forcing SigLIP2 through
a CLIP-only projection.

The configuration field `use_value_attention` retains its historical name so
old JSON configs still load. In version 0.7.0, `true` means the official
AnomalyCLIP DPAM-style accumulating V-V dense branch. It no longer means the
old independent one-block value projection.

## Image-score correction

Runs through version 0.6.0 used:

```text
global abnormal probability + max(pixel anomaly map)
```

That is not AnomalyCLIP's image metric and caused the learned CLIP VisA score
to appear better than the published AnomalyCLIP number. Version 0.7.0 defaults
to the global abnormal probability alone. `add_local_evidence=true` remains an
explicit experimental ablation, but its results must be labelled *fused* and
must not be compared directly with AnomalyCLIP image AUROC/AP.

## Map correction

Runs through version 0.6.0 averaged layer logits and then applied one softmax.
Official AnomalyCLIP instead computes a two-class softmax for every layer and
sums the resized abnormal-probability maps. These operations are not
equivalent. Version 0.7.0 follows the official ordering and does not clamp the
sum to one.

The old 64×64 map setting also made AUPRO more forgiving than the published
518×518 evaluation. The default is now 518. A smaller `map_res` remains an
explicit storage/runtime ablation and must be labelled accordingly.

## Reproducibility

The package version is part of the configuration fingerprint. Version 0.7.0
therefore cannot resume incompatible version-0.6.0 artifacts or prompt
checkpoints. Existing result directories remain historical records rather than
being overwritten.

Regression coverage lives in `tests/test_scoring.py`, `tests/test_losses.py`,
`tests/test_clip_backbone.py`, `tests/test_config.py`, and
`tests/test_pipeline_smoke.py`.

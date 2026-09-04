# AnomalyCLIP parity audit

Alireza's instruction for the learnable-prompt setting was: CoOp-style shallow
prompts, object-agnostic like AnomalyCLIP, **without** deep token tuning inside
the text transformer, trained with **AnomalyCLIP's loss**.

This is a line-by-line audit against two independent references, not against
the notebook this repository started from:

1. **AnomalyCLIP** — `train.py`, `loss.py`, `prompt_ensemble.py`,
   `AnomalyCLIP_lib/model_load.py` (zqhang/AnomalyCLIP).
2. **The group's own** `Parsagh05/object-agnostic-prompt-training`, which was
   built to match AnomalyCLIP for the adversarial track.

Eight deviations were found. All eight are fixed: the learnable-prompt setting
now matches AnomalyCLIP on every point that the protocol does not deliberately
remove.

## Matched

| Aspect | AnomalyCLIP | Here |
| --- | --- | --- |
| prompt form | `[X × n] object.` / `[X × n] damaged object.`, `classnames = ["object"]` | same |
| `n_ctx` | 12 | 12 |
| object-agnostic | no class name in the learnable prompt | same |
| deep text-prompt tuning | present (`compound_prompts_text`, depth 9) | **deliberately absent** |
| DPAM visual surgery, learnable visual tokens, feature adapters | present | **deliberately absent** |
| trainable parameters | shallow contexts (+ deep tokens and projections) | two shallow contexts only, asserted every run |
| dense layers | 6, 12, 18, 24 | same for CLIP |
| dice | `1 - mean(2·inter + 1)/(sum + sum + 1)` | same |
| focal | `-(1-pt)^2 · log(pt)` on softmax probabilities, `gamma = 2`, `smooth = 1e-5` | same |
| optimiser | Adam, lr 1e-3 | same constant learning rate; no default gradient clipping |

## Fixed

| # | Aspect | AnomalyCLIP / own repo | Was | Now |
| --- | --- | --- | --- | --- |
| 1 | image cross-entropy | always added | absent (`loss_mode="local"`) | `loss_mode="both"` |
| 2 | pixel terms | per layer, summed | applied to the layer-*averaged* map | per layer, summed |
| 3 | pixel weight | `lam = 4` | 1.0 | 4.0 |
| 4 | epochs | 15 | 2 | 15 |
| 5 | similarity temperature | `1/0.07` for image and patch alike | backbone's learned scale (CLIP 100, SigLIP2 108) | `map_temperature = 0.07` |
| 6 | loss resolution | prediction upsampled to the mask | mask area-downsampled to the patch grid | prediction upsampled to the mask |
| 7 | context init | `N(0, 0.02)` | `"X"` token embeddings + 0.02·noise | `N(0, init_std)` |

### Why each mattered

**1 — the missing image term.** With pixel terms only, nothing opposes
"predict normal everywhere": for an all-zero mask the abnormal dice reduces to
`1 - 1/(sum(p_abnormal) + 1)`, minimised by driving that channel to zero. This
is the leading explanation for the collapse in run `1f2fb45b459e`, and it
matches the direction exactly — the normal-heavy source (VisA) collapsed, the
anomaly-heavy one (MVTec) did not.

**2 — averaging before the loss** lets one layer compensate for another instead
of forcing every layer that is read to be discriminative. CLIP reads four
layers; SigLIP2 reads one, so only CLIP was affected.

**5 — temperature.** CLIP's learned scale is 100 and SigLIP2's ~108, against
AnomalyCLIP's 14.3. At 100 the two-class softmax saturates and the focal
gradient flattens; the fixed-prompt SigLIP2 maps in run `1f2fb45b459e` spanned
[0.004, 0.994], which is what saturation looks like. Ranking metrics are
invariant to the scale, but the training signal and the score fusion
`global + max(map)` are not. A shared value also puts both backbones' maps on
the same sharpness, which is what makes adding `max(map)` to a probability
comparable across them. `map_temperature = None` restores each backbone's own
learned scale.

**6 — loss resolution.** The mask was downsampled from 64×64 to the patch grid
(37 for CLIP, 24 for SigLIP2) with a `> 0.5` area threshold and no small-defect
guard, so small defects vanished from the target entirely. Measured on a 64×64
mask against a 7×7 grid: **a 100-pixel defect survives as 1 pixel.** It also
meant training optimised a different resolution than the metrics are computed
at. AnomalyCLIP upsamples the prediction instead; so does the group's own repo
(`output_size=tuple(masks.shape[-2:])`). Bilinear interpolation of a softmax
output keeps the channels summing to one, so the upsampled map is still a
distribution.

**7 — context init.** Seeding from the `"X"` placeholder embeddings gives the
context real token-scale magnitude, which can swamp the fixed suffix and start
`object` and `damaged object` nearly on top of each other. At `N(0, 0.02)` the
suffix dominates early, so the two prompts start apart.

## A note on `n_ctx`

Runs up to `8ffd5f816d58` used `n_ctx = 8`, following Tipsomaly's ablation
(Figure 3), which finds 8 best and reports longer prompts overfitting the source
domain — the failure this cross-dataset protocol is most exposed to.

Now 12, AnomalyCLIP's default, because that is the reference Alireza named and
an unexplained deviation costs more than it buys. `n_ctx` is part of the config
fingerprint, so 8 and 12 produce separate artefacts and can sit side by side as
an ablation rather than a choice made in advance.

Verified that 12 tokenises cleanly for SigLIP2: the placeholder maps to exactly
12 tokens and the full prompt occupies 15 of the 64 positions.

## Not applicable

AnomalyCLIP's `text_probs` are computed from DPAM-modified features and its
`compute_similarity` folds in the class-token similarity. Both belong to the
internal adaptation that this protocol removes by design, so they have no
counterpart here.

## Verification

`tests/test_pipeline_smoke.py` and `tests/test_clip_backbone.py` pin items
1, 2, 5, 6 and 7, plus the frozen-encoder constraint. The CLIP tests skip
unless a checkpoint is already cached; nothing downloads.

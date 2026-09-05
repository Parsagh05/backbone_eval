# Run findings

| Run | Version | What it was |
| --- | --- | --- |
| `1f2fb45b459e` | 0.1.0 | first measured run; three defects found in it |
| `cae0b9678540` | 0.5.0 | training-loss parity under legacy inference |
| `25497c2775b0` | 0.6.0 | exact prompts; legacy fused image score and 64px maps |

> Versions through 0.6.0 are historical. Their image score added the maximum
> pixel-map response to the global probability, their inference map softmaxed
> averaged logits, and metrics used 64×64 maps. Version 0.7.0 corrected those
> inference choices. Version 0.8.0 restores SigLIP2's standard final-layer
> readout and uses only the final layer in both backbones' learned pixel loss.
> Do not compare older image/AUPRO numbers directly with published AnomalyCLIP.

---

# Run `1f2fb45b459e` — the first measured run

Seed 111, clean only, both protocol directions, MVTec + VisA. These were the
first *measured* numbers in this track; the CSVs in `vendor/` were transcribed
from Tipsomaly (see [provenance_note.md](provenance_note.md)).

## 1. SigLIP2 localisation is not at chance — the published failure was a readout bug

| SigLIP2, fixed prompts | Tipsomaly Table 9 | This run |
| --- | --- | --- |
| MVTec pixel AUROC | 47.3 | **82.1** |
| VisA pixel AUROC | 47.0 | **78.9** |
| MVTec AUPRO | 00.1 | **74.1** |
| VisA AUPRO | 00.0 | **68.0** |

**0 of 27 categories fall below chance.** The weakest is transistor at 59.7; the
strongest are candle 97.5, carpet 96.5, zipper 96.4, macaroni1 94.9.

The only change is the dense readout: pooling each patch through the model's own
`trunk.attn_pool` instead of comparing unprojected trunk tokens to text. This is
the result the repository exists to establish, and it confirms the diagnosis in
[siglip2_defects.md](siglip2_defects.md) (D2).

Image level corroborates rather than contradicts: 93.2 MVTec AUROC here against
88.7 reported, so the image path is sane and the pixel gain is not an artefact
of a differently-scaled score.

Not a strict reproduction — Tipsomaly scores through the TIPS framework at 518px
and this harness runs SigLIP2 at its native 384px — but the direction is not
subtle.

## 2. The CLIP baseline was broken, not merely weaker

CLIP fixed-prompt pixel AUROC was **below chance in 14 of 27 categories**:
leather 3.1, grid 6.5, bottle 14.0, carpet 15.1, wood 19.0, pcb4 19.8.

Below chance means the map is *anti-correlated* with the defect — it reliably
highlights the wrong pixels. That is the "opposite visualisation" of raw CLIP
patch-text similarity that CLIP-Surgery describes, and the reason WinCLIP and
AnomalyCLIP apply attention surgery at all.

**Cause.** `use_value_attention` patched only `max(layers)` — layer 24 — while
layers 6, 12 and 18 were read raw and averaged in alongside it by
`dense_logits`, dragging the result under chance.

**Fixed** in 0.2.0: the value path is taken at *every* layer that is read. The
main forward still uses the ordinary block output, so the global embedding is
bit-identical and image-level numbers stay comparable
(`tests/test_clip_backbone.py` pins both properties).

Until this is re-run, no SigLIP2-vs-CLIP comparison from run `1f2fb45b459e`
should be quoted: it compares a fixed SigLIP2 against a broken CLIP.

## 3. Learned prompts collapsed onto "normal everywhere"

CLIP's learned prompts evaluated on MVTec produced a literally constant map:

```
clip learned mvtec hazelnut   maps mean 0.0000 std 0.00000 min 0.0000 max 0.0000
```

Hence pixel AUROC of exactly 50.0. SigLIP2's were nearly as degenerate
(mean 0.0004, std 0.0037). Fixed prompts beat learned ones everywhere:
82.1 → 71.2 on MVTec, 78.9 → 62.8 on VisA.

**Leading hypothesis: the class balance of the training split.** For a normal
image the mask is all zero, so `dice(prob[:, 1], target)` reduces to
`1 - 1/(sum(prob_abnormal) + 1)`, which is minimised by driving the abnormal
channel to zero *everywhere*. The direction of the failure matches the split
composition exactly:

| Prompts fitted on | Composition | Evaluated on | Pixel AUROC |
| --- | --- | --- | --- |
| VisA test (1,200 normal / 962 anomalous — mostly normal) | normal-heavy | MVTec | **50.0**, constant map |
| MVTec test (467 normal / 1,258 anomalous — mostly anomalous) | anomaly-heavy | VisA | 94.7 |

The normal-heavy source collapses; the anomaly-heavy one does not.

**Root cause: the objective was not AnomalyCLIP's.** Run `1f2fb45b459e` used
`loss_mode="local"` — the pixel terms only, with **no image-level
cross-entropy**. Nothing in that objective opposes "predict normal everywhere":
the abnormal dice term is the sole counterweight, and on a normal-heavy split it
loses. AnomalyCLIP's image cross-entropy directly penalises calling an anomalous
image normal, which is exactly the missing pressure.

AnomalyCLIP's `train.py`:

```python
image_loss = F.cross_entropy(text_probs, label)      # always included
loss = 0
for i in range(len(similarity_map_list)):            # per layer, summed
    loss += loss_focal(similarity_map_list[i], gt)
    loss += loss_dice(similarity_map_list[i][:, 1], gt)
    loss += loss_dice(similarity_map_list[i][:, 0], 1 - gt)
loss = lam * loss                                    # lam = 4
(loss + image_loss).backward()
```

Three deviations, all fixed in 0.3.0:

| | run `1f2fb45b459e` | AnomalyCLIP / 0.3.0 |
| --- | --- | --- |
| image cross-entropy | absent (`loss_mode="local"`) | included (`loss_mode="both"`) |
| pixel terms | applied to the layer-*averaged* map | applied per layer and summed |
| pixel weight | 1.0 | 4.0 (`lam`) |
| epochs | 2 | 15 |

Averaging the layers before the loss also mattered: it lets one layer compensate
for another instead of forcing every layer that is read to be discriminative.
With four CLIP layers this is a real difference; SigLIP2 reads one layer, so
only CLIP is affected.

**Also added:** `train_prompts` reports the mean and peak of the abnormal
channel each epoch and warns when the peak never exceeds `COLLAPSE_PEAK` (0.01).
A collapse is now visible during training instead of being discovered hours
later as a table of 50.0.

If it still collapses, the next things to try, in order: a lower learning rate;
class-balanced sampling in `make_train_loader`; reweighting the normal-region
dice term.

## Consequences for reproducibility

Both fixes change results without changing any setting that used to enter the
config fingerprint, so run `1f2fb45b459e` would have been silently resumed on
top of. Fixed in 0.2.0:

- the package version is part of the fingerprint, so a behaviour change
  invalidates earlier artefacts;
- `epochs`, `learning_rate` and `max_train_images_per_category` are part of the
  fingerprint;
- the prompt checkpoint filename carries `epochs`, so a 15-epoch fit cannot load
  a 2-epoch checkpoint.

Delete nothing: the previous run stays valid under its own id, and remains the
"raw readout / 2 epochs" control.


---

# Run `cae0b9678540` — after the fixes

Same protocol, seed 111, clean only. Version 0.5.0: `loss_mode="both"`,
15 epochs, `n_ctx=12`, `lam=4`, `map_temperature=0.07`, `init_std=0.02`,
value path on every CLIP dense layer.

## The collapse is gone

| | `1f2fb45b459e` | `cae0b9678540` |
| --- | --- | --- |
| CLIP learned, MVTec pixel AUROC | **50.0** (constant map) | **87.7** |
| CLIP learned, MVTec AUPRO | 14.8 | **76.2** |
| CLIP learned, VisA AUPRO | 76.9 | **91.6** |
| SigLIP2 learned, MVTec pixel AUROC | 71.2 | **84.4** |
| SigLIP2 learned, VisA pixel AUROC | 62.8 | **88.4** |
| SigLIP2 learned, VisA AUPRO | 40.0 | **80.7** |

Sanity against AnomalyCLIP, which this setting deliberately under-powers by
removing DPAM, deep prompts and adapters:

| Direction | AnomalyCLIP | Here (CLIP, learned) |
| --- | --- | --- |
| MVTec → VisA | 95.5 pixel AUROC | 95.0 |
| VisA → MVTec | 91.1 pixel AUROC | 87.7 |

Landing just under it while training only 12 context vectors is the expected
shape of the result.

## Dataset level

Pixel AUROC / AUPRO, then image AUROC:

| | MVTec fixed | MVTec learned | VisA fixed | VisA learned |
| --- | --- | --- | --- | --- |
| CLIP | 53.0 / 15.2 — 86.8 | 87.7 / 76.2 — 77.4 | 59.1 / 16.1 — 76.7 | 95.0 / 91.6 — 84.8 |
| SigLIP2 | **82.8 / 74.2** — **92.7** | 84.4 / 73.5 — **93.6** | **80.2 / 69.4** — **84.0** | 88.4 / 80.7 — 84.4 |

`decoupled` shares the learned map by construction, so its pixel columns equal
`learned` and only its image score differs. Seeing that identity holds is a
check that the wiring is right.

## The finding

**Untrained, SigLIP2 localises and CLIP does not** — 82.8 / 80.2 against
53.0 / 59.1. CLIP's fixed maps span [0.21, 0.50], never crossing 0.5 and barely
varying; SigLIP2's span [0.15, 0.75]. This is not a CLIP bug: raw patch-text
similarity with handwritten prompts is known to be weak, which is why WinCLIP
needs windowing and AnomalyCLIP needs training. SigLIP2's attention-pooled
readout gets there with neither.

**Trained, CLIP overtakes it** at pixel level (87.7 / 95.0 against 84.4 / 88.4),
while SigLIP2 keeps the image level nearly everywhere. That trade-off —
strong without adaptation, overtaken with it — is the counter-trend the deck
asks for on slide 9.

## Two things to know when reading the table

**1. CLIP's VisA → MVTec maps live at ~5e-5**, in all 15 categories. The ranking
metrics are unaffected and were checked directly: **0 % zeros, 542–1126 distinct
values per category**, well clear of float16's smallest subnormal (5.96e-8). So
87.7 is real signal.

But the image score is `global_prob + max(map)`, and at `max(map) ≈ 1e-4` the
local-evidence term contributes nothing. That is why CLIP's learned image AUROC
on MVTec is 77.4 against 86.8 for fixed, and why `decoupled` (86.9) is CLIP's
best mode there.

The prompts themselves are healthy — `cos(normal, abnormal)` is −0.024, +0.004,
−0.032, −0.074 across the four checkpoints, with comparable norms. This is a
CLIP-specific domain-transfer effect, not degenerate training: SigLIP2's
VisA → MVTec maps peak at 0.82.

**2. Pixel metrics are computed at `map_res = 64`, not the full 518.** Our VisA
AUPRO (91.6) exceeds AnomalyCLIP's published 87.0 despite a lower AUROC, which
is consistent with coarser maps and coarser ground-truth regions being more
forgiving. Do not place these numbers in a table beside published ones without
saying so, or re-score at full resolution.

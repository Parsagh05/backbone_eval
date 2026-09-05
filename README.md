# Backbone Evaluation

Backbone benchmarking for zero-shot anomaly detection: how much does a newer
contrastive vision-language encoder actually help, compared to CLIP, under an
identical prompt-learning protocol?

The current target is **SigLIP2**, the last backbone from slide 23 of
*Contrastive VLMs in Anomaly Detection*. CLIP is the reference baseline.

Logic lives in the `bbeval` package. Notebooks are thin runners.

## Why this repository exists

Tipsomaly ([arXiv:2602.03594](https://arxiv.org/abs/2602.03594), Table 9)
reports SigLIP2 at **47.3 / 47.0 pixel AUROC** on MVTec / VisA — chance level —
with AUPRO near zero, while its *image*-level score on the same model with the
same prompts is a healthy 88.7. A backbone whose dense features top the
linear-probe tables cannot be at chance on defect localisation.

The gap is not the sigmoid training objective. It is that a CLIP-shaped harness
makes assumptions SigLIP2 does not satisfy, and every one of them fails
silently. Six such defects are documented, with evidence, in
[docs/siglip2_defects.md](docs/siglip2_defects.md) — the load-bearing one being
that CLIP's joint space is a per-token linear map that can be reused on patch
tokens, whereas SigLIP2's is a set-to-vector attention-pooling head with no
per-token equivalent.

## What "fair" means here

Two definitions are available, and they give opposite answers:

1. **Identical code path** — run every backbone through CLIP's pipeline
   unchanged. This measures how CLIP-shaped a model is, and ranks CLIP first by
   construction.
2. **Identical question, asked in each model's own terms** — freeze everything
   that defines the question, and let each model receive it the way it was
   built to.

This repository takes the second. Concretely:

| Frozen across backbones | Adapted per backbone |
| --- | --- |
| prompt text and ensemble structure | image normalisation |
| datasets, splits, seed | tokenizer, context length, text pooling |
| map post-processing, aggregation | resolution and patch grid |
| metrics and how they are computed | temperature (and bias) |
| stored artefact format | patch → joint-space readout |

Every adapted value comes from the model's own config or paper — never from
tuning — and each is written into `run_manifest_<config_id>.json` next to the
results, so a reader can audit the choices.

## Dense visual readout

**CLIP's dense features use AnomalyCLIP's DPAM V-V attention branch. SigLIP2
uses its native attention-pool projection. This architecture-specific readout
is recorded in every run manifest.**

To draw a heatmap, every patch has to be comparable against text. CLIP's raw
patch similarities can point at the wrong regions. AnomalyCLIP addresses this
with DPAM: from layer 6 onward, a second residual stream accumulates V-V
attention outputs and skips its FFNs while the ordinary CLIP path remains
intact for the global image embedding. This is implemented functionally without
adding or training parameters. The historical configuration name
`use_value_attention=True` enables it.

SigLIP2 has no literal DPAM module. Its patch tokens instead use the model's own
attention-pool projection. Both backbones expose four proportional stages.
`use_value_attention=False` remains the raw-CLIP control; the global embedding
is identical and only dense localization features differ.

## The variable under test

`siglip2_dense_readout` selects how a patch token reaches the joint space:

- `"map_token"` — pool each token through `trunk.attn_pool`, the function that
  defines SigLIP2's joint space. Verified to reproduce `encode_image`
  bit-for-bit when given the full token set.
- `"raw"` — leave trunk tokens unprojected. The CLIP-shaped assumption, kept as
  the **control** that reproduces the published failure. Both produce
  identically shaped tensors, which is exactly why the defect never raised.

## Install

```bash
python -m pip install -e ".[siglip2]"
python -m pip install "git+https://github.com/openai/CLIP.git"   # for the clip backbone
python -m pip install -e ".[corruptions]"                        # only if corruptions are enabled
```

A missing optional dependency disables one backbone rather than breaking the
run; `bbeval --list-backbones` reports what is available and why.

## Use

```bash
bbeval --config configs/kaggle_siglip2.example.json
bbeval --config configs/kaggle_siglip2.example.json --set limit=4
bbeval --list-backbones
```

Or from Python:

```python
from bbeval import BackboneEvalConfig, run_evaluation

result = run_evaluation(BackboneEvalConfig(
    mvtec_root=..., visa_root=..., output_root=...,
    backbones=("clip", "siglip2"), siglip2_dense_readout="map_token",
))
```

## Kaggle

Create a notebook, paste in `notebooks/kaggle_backbone_eval.ipynb`, attach the
MVTec AD and VisA datasets, enable a **GPU**, switch **Internet on**, and run
it. The notebook fetches this repository itself, so nothing needs uploading and
re-running the cell is safe.

**Internet must be enabled.** Without it, `git clone` receives an intercepted
response, decides it needs credentials, and reports
`could not read Username for 'https://github.com'` — which says nothing about
the real cause. The cell therefore sets `GIT_TERMINAL_PROMPT=0`, and its
failure message names the Internet setting first.

The fetch degrades in three stages, so an awkward network costs a warning
rather than the run:

| Situation | Behaviour |
| --- | --- |
| no checkout yet | `git clone --depth 1` |
| clone fails | plain-HTTPS tarball from `codeload.github.com` |
| checkout present | `git pull --ff-only`; local edits fail loudly |
| pull fails | warn and use the checkout already on disk |

It prints the short commit when git metadata is available, so a run ties back
to its exact source. Cloning an *empty* repository also succeeds and leaves
nothing behind, so the cell checks for `pyproject.toml` rather than failing
later inside pip.

**Editable installs are invisible to a running kernel.** A PEP 660 editable
install writes a `.pth` file, and `.pth` files are read only when the
interpreter starts, so `pip install -e` followed by `import bbeval` in the same
session raises `ModuleNotFoundError`. The notebook adds `<root>/src` to
`sys.path` and calls `importlib.invalidate_caches()` instead of asking for a
kernel restart. The `pip install` still matters — it resolves the dependencies
declared in `pyproject.toml`.

The same cell prints the loaded backbones and any that are unavailable. Worth
reading before a long run: a missing optional dependency disables a backbone
rather than raising, so a silent `clip` failure would otherwise surface only as
a missing column in the results.

MVTec and VisA are published as separate Kaggle datasets under different
parents, so both roots are given explicitly rather than discovered under one
directory.

## Output

Everything lands under `output_root`:

```text
results/
  run_manifest_<config_id>.json    resolved config + every per-backbone choice
  artifacts/<backbone>/<mode>/<dataset>/<category>/<corruption>_s<severity>.npz
  artifacts/_ground_truth/<dataset>/<category>/*.npz
  prompts/<backbone>_<source>_<loss>_ctx<n>_ep<epochs>_seed<seed>_cfg<id>.pt
  tables/{category,dataset,robustness}_<config_id>.csv
```

`run_evaluation` then writes **one ZIP** of all of it, reported as
`result["archive"]` — placed *beside* `output_root` rather than inside it, since
an archive created within its own source tree tries to add itself while it is
still being written. On Kaggle that is
`/kaggle/working/backbone_eval_<config_id>.zip`, which shows up in the output
panel as a single file to download. Set `archive_results=False` to skip it.

The AnomalyCLIP-compatible default stores 518×518 float16 maps. Budget several
gigabytes for a clean two-backbone/four-mode run; exact compression depends on
map smoothness. `map_res=64` is still available when storage is constrained,
but it is a labelled low-resolution ablation and is not directly comparable to
published AnomalyCLIP pixel metrics.

## Runtime

Measured in visual forward passes, which dominate everything else. Per backbone,
one clean pass over both protocol directions:

| Work | Forward passes per backbone |
| --- | --- |
| prompt fitting (15 epochs x both source splits) | ~58,300 |
| evaluation sweep (MVTec 1,725 + VisA 2,162) | ~3,900 |

All configured prompt modes come out of one visual forward pass, so additional
frozen modes add little inference time. Every additional learned mode still
requires its own prompt-fitting passes.

Calibrated against a measured run: **40 minutes** on a Kaggle T4 for both
backbones at 2 epochs, of which roughly 5-7 was fixed cost (4.4 GB of weight
downloads, then the AUPRO pass over 162 category x mode cells). Fifteen epochs
is 5.3x the compute, so:

The historical 64×64 run took about 3 hours on a T4. The 518×518 reference
default leaves backbone-forward time similar but makes artifact I/O and the
200-threshold AUPRO sweep substantially heavier; benchmark one category before
assuming the old wall-clock estimate.

Prompt fitting dominates: at 15 epochs it is about 58,000 forward passes per
backbone against 3,900 for the evaluation sweep. Two levers if that does not fit
a session:

- `resume=True` (the default) — prompt checkpoints are written per
  (backbone, source) and shards per cell, so a run continues across sessions.
  A session needs to finish one source's fit to checkpoint it.
- `max_train_images_per_category` — two 12-token shallow contexts do not need
  1,700 images. Capping at 50 cuts fitting roughly fivefold. It is part of the
  config fingerprint, so a capped run cannot be confused with a full one.

Epochs was 2 through run `1f2fb45b459e`, which was not enough: the prompts
collapsed onto "normal everywhere" and pixel AUROC came out at exactly 50.0.
See [docs/run_findings.md](docs/run_findings.md). Training now
reports the abnormal channel's mean and peak each epoch and warns when it never
rises above `COLLAPSE_PEAK`, so this is visible while it happens.

CLIP runs at 518px (37x37 = 1,369 tokens) and is the slower of the two;
SigLIP2 runs at its native 384px (24x24 = 576 tokens) but currently encodes
twice per batch — once for the dense layers, once via `encode_image` for the
global vector — so the two end up roughly comparable. Collapsing that second
pass is the obvious optimisation and is noted in `siglip2.py`.

**Corruptions do not fit in one session.** Enabling them turns 1 setting per
category into 34 (11 corruptions x 3 severities + clean), so the evaluation
sweep grows ~34x, and `imagecorruptions` is itself expensive at 518px. Budget
several sessions, raise `num_workers`, and lean on `resume=True`.

## Protocol

Only the **test** splits are read, and the two roles are kept disjoint
(pptx slide 21):

| Prompts fitted on | Evaluated on | Category overlap |
| --- | --- | --- |
| MVTec test split (15 categories) | VisA test split (12 categories) | none |
| VisA test split (12 categories) | MVTec test split (15 categories) | none |

The only trainable tensors anywhere are the normal and abnormal shallow prompt
contexts, each `[n_ctx, D]`. The backbone is held in a plain list inside
`LearnablePrompts` so it is never registered as a submodule: the checkpoint then
contains those two contexts and nothing else, which is the audit trail for the
claim that no encoder parameter is touched. `assert_prompt_learning_only`
re-checks this before every fit.

Three prompt modes come out of one visual forward pass:

| Mode | Ensemble | Category name | Prompts/category |
| --- | --- | --- | --- |
| `fixed` | WinCLIP: 22 templates x (7 normal, 4 anomalous) | **yes** — `"a cropped photo of the hazelnut."` | 154 + 88 |
| `fixed_agnostic` | the same | **no** — `"a cropped photo of the object."` | 154 + 88 |
| `fixed_compact` | 7 templates x (6, 6) | **yes** — `"a photo of a hazelnut."` | 42 + 42 |
| `learned` | 12 trained context vectors + fixed suffix | **no** | 2 |

The three frozen modes vary one thing each, so each pair isolates a factor:
`fixed` vs `fixed_agnostic` measures what the **category name** is worth, and
`fixed` vs `fixed_compact` measures what the **ensemble size** is worth.
`fixed_compact` reproduces run `cae0b9678540` exactly — verified prompt by
prompt across all 27 categories of both datasets. That commit changed two
things at once: the ensemble, and a category-name mapping where `pcb1`-`pcb4`
were `"printed circuit board"` and `macaroni1`/`macaroni2` were `"macaroni"`.
Both belong to `fixed_compact`, so the mode is a faithful control rather than a
partial one, and the jump that followed (CLIP +8.7 pixel AUROC, SigLIP2 +0.8)
can be attributed to the frozen vocabulary as a whole.

Select which modes to run with `prompt_modes`. Shards are stored per mode and
`resume` merges them, so a run may cover part of the protocol and a later run
into the same `output_root` completes it. A frozen-only run therefore skips
prompt fitting entirely — minutes instead of hours — and covering only one half
warns rather than failing.

The exact frozen vocabulary is pinned by `tests/test_prompts.py`.

`fixed` and `fixed_agnostic` differ *only* in the label that fills the template,
which makes the pair a direct measurement of **how much knowing the category is
worth**. It also matters for reading `fixed` against `learned`: the learned
prompts are object-agnostic by construction, so comparing them with class-aware
`fixed` changes two things at once. `fixed_agnostic` is the like-for-like
comparison, isolating the effect of training alone.

Being category-independent, the agnostic ensemble is encoded once per backbone
rather than once per category.

**Learned prompts follow the group's shallow AnomalyCLIP-style reference** —
`[v1…v12] object.` and `[u1…u12] damaged object.`, no class name — trained with
AnomalyCLIP's objective:

```
image_cross_entropy + lam * sum_over_layers(focal + dice_abnormal + dice_normal)
```

with `lam = 4`, 15 epochs, and constant Adam lr 1e-3. The pixel terms are
applied to **each of four proportional layers** and summed for both backbones.
At inference, each layer is softmaxed and resized separately, then the four
abnormal-probability maps are summed, matching official AnomalyCLIP ordering.

The explicit exception is deep prompt tuning inside the text transformer. DPAM
is used for CLIP's frozen dense branch; SigLIP2 uses its own frozen
attention-pool readout. Only the two shallow context tensors train.

Audited line by line against AnomalyCLIP's own `train.py`, `loss.py` and
`prompt_ensemble.py`, and against the group's
`object-agnostic-prompt-training` — see
[docs/anomalyclip_parity.md](docs/anomalyclip_parity.md).

`loss_mode` selects halves of that objective — `"both"` is AnomalyCLIP,
`"local"` drops the image term (Tipsomaly's localisation-only ablation) and
`"global"` keeps only it. `"local"` was the default through run
`1f2fb45b459e` and left nothing opposing a collapse onto "normal everywhere";
see [docs/run_findings.md](docs/run_findings.md).

Seed 111. Metrics: pixel AUROC / F1-max / AUPRO / threshold, image AUROC /
F1-max / AP / threshold, plus ECE for the slide-24 calibration track.
Maps and raw scores are stored so metrics can be recomputed without GPU time.

## Layout

```text
src/bbeval/
  config.py       one dataclass; a run is fully described by one JSON file
  datasets.py     test-split indexing, dataset object, loaders
  corruptions.py  deterministic slide-18/19 corruptions (optional dependency)
  backbones/      registry + the frozen-encoder interface
    base.py       the contract, and what each backbone must declare
    clip.py       OpenAI CLIP
    siglip2.py    SigLIP2 via OpenCLIP/timm
  prompts.py      fixed ensembles + the learnable context
  scoring.py      patch/text logits, anomaly maps, image scores
  losses.py       focal + dice + cross-entropy
  training.py     prompt fitting, with the frozen-encoder assertion
  artifacts.py    map/score/ground-truth storage + run manifest
  metrics.py      AUROC, F1-max, AP, AUPRO, ECE
  aggregate.py    category-level and dataset-level tables
  engine.py       the sweep
configs/          example JSON configurations
scripts/          Kaggle entry point
notebooks/        thin runner
tests/            correctness gates (CPU, random weights, no downloads)
docs/             defect analysis and provenance notes
vendor/           the upstream notebook this work started from, unmodified
```

## Tests

```bash
python -m pytest -q
```

The SigLIP2 gates run on **random weights**: each one is a forward-path
equivalence, which does not depend on weight values, so the suite needs no
checkpoint download and runs on CPU in seconds. Anything weight-dependent
belongs on the GPU. They assert, among other things, that the text forward
matches OpenCLIP's own `encode_text` exactly, and that attention-pooling over
all tokens reproduces `encode_image` exactly.

## Status and scope

Ported from the upstream `TIPS_vs_CLIP_Benchmark.ipynb` (kept in `vendor/`):
configuration, data, corruptions, CLIP and SigLIP2 backbones, prompts, scoring,
losses, prompt fitting, artefacts, metrics, aggregation, sweep.

**Not yet ported:** the TIPS, TIPS-v2 and DINOv2.txt backbones, and the
post-hoc reporting, published-comparison and qualitative-figure cells. The
first is a straightforward addition behind the same registry; the second
operates on stored artefacts and can be written against the tables.

## Results so far

The first measured run is written up in
[docs/run_findings.md](docs/run_findings.md). In short:

Two runs are recorded in [docs/run_findings.md](docs/run_findings.md).
The latest recorded numbers are historical run `25497c2775b0` (v0.6.0).
They predate the v0.7.0 inference-parity corrections described above:

- **SigLIP2 localises without any training.** Fixed-prompt pixel AUROC 82.8
  (MVTec) and 80.2 (VisA), against 47.3 / 47.0 reported in Tipsomaly Table 9 —
  and against **53.0 / 59.1 for CLIP** under the same prompts. The published
  failure was the readout, not the backbone.
- **Trained, CLIP overtakes it at pixel level** (87.7 / 95.0 against
  84.4 / 88.4) while SigLIP2 keeps the image level. Strong without adaptation,
  overtaken with it — the counter-trend slide 9 asks for.
- **Sanity:** CLIP learned reaches 95.0 pixel AUROC on VisA against
  AnomalyCLIP's published 95.5, while training only 12 context vectors and
  none of its internal adaptation.

Those recorded numbers are historical version-0.6.0 results at `map_res=64`
with a fused image score. Version 0.7.0 changes the fingerprint and requires a
new run before quoting AnomalyCLIP-compatible comparisons. The CSVs in
`vendor/` remain transcriptions, not measurements — see
[docs/provenance_note.md](docs/provenance_note.md).

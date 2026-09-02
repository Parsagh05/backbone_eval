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

On Kaggle, run `notebooks/kaggle_backbone_eval.ipynb` or
`scripts/kaggle_run_backbone.py`. MVTec and VisA are published as separate
Kaggle datasets under different parents, so both roots are given explicitly
rather than discovered under one directory.

## Protocol

Only the **test** splits are read, and the two roles are kept disjoint
(pptx slide 21):

| Prompts fitted on | Evaluated on | Category overlap |
| --- | --- | --- |
| MVTec test split (15 categories) | VisA test split (12 categories) | none |
| VisA test split (12 categories) | MVTec test split (15 categories) | none |

The only trainable tensor anywhere is the prompt context, `[2, n_ctx, D]`. The
backbone is held in a plain list inside `LearnablePrompts` so it is never
registered as a submodule: the checkpoint then contains the context vectors and
nothing else, which is the audit trail for the claim that no encoder parameter
is touched. `assert_prompt_learning_only` re-checks this before every fit.

Three prompt modes come out of one visual forward pass: `fixed` (template ×
state ensemble, no training), `learned`, and `decoupled` (fixed score, learned
map).

Seed 111. Metrics: pixel AUROC / F1-max / AUPRO / threshold, image AUROC /
F1-max / AP / threshold, plus ECE for the slide-24 calibration track.
Low-resolution maps and raw scores are stored so any metric can be recomputed
without GPU time.

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

**No benchmark numbers have been produced yet** — everything verified so far is
correctness, not performance. See [docs/provenance_note.md](docs/provenance_note.md)
for why the CSVs in `vendor/` should not be treated as measurements.

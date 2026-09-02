# Note on `Experiments/backbone` committed results

**Status:** factual observation, raised before building on these files.
**Files:** `AnomalyDetectionVLMSurvey/Experiments/backbone/*.csv`, added in commit
`a1192a3e` ("Add backbone benchmark experiment results").

## What the files show

**1. The notebook in that commit has never been executed.**

`TIPS_vs_CLIP_Benchmark.ipynb` contains 21 code cells with **0 execution counts
and 0 stored outputs**. The same commit adds no results, artifacts or log
directories.

```
code cells: 21   with execution_count: 0   with outputs: 0
```

**2. The SigLIP2 and TIPS rows match Tipsomaly's published tables exactly.**

Tipsomaly (arXiv:2602.03594) reports triplets as `(AUROC, AP, F1-max)` for
image level and `(AUROC, AUPRO, F1-max)` for pixel level. The CSVs use a
different column order. Reordered, the values are identical:

| CSV row | Tipsomaly |
| --- | --- |
| SigLIP 2, MVTec, image — 88.7 / 90.1 / 94.4 | Table 9, *No Learning* — (88.7, 94.4, 90.1) |
| SigLIP 2, VisA, image — 74.9 / 78.1 / 80.5 | Table 9, *No Learning* — (74.9, 80.5, 78.1) |
| SigLIP 2, MVTec, pixel — 61.3 / 10.3 / 31.0 | Table 9, *Local Loss* — (61.3, 31.0, 10.3) |
| SigLIP 2, VisA, pixel — 56.9 / 2.2 / 27.9 | Table 9, *Local Loss* — (56.9, 27.9, 02.2) |
| TIPS (L/14 HR), MVTec, image — 93.4 / 92.9 / 96.1 | TIPS-L/14 HR — (93.4, 96.1, 92.9) |
| TIPS (L/14 HR), MVTec, pixel — 90.9 / 43.8 / 84.0 | TIPS-L/14 HR — (90.9, 84.0, 43.8) |

**3. The TIPS-v2 and DINOv2.txt rows have no published source.**

Tipsomaly does not report either backbone. The notebook's own markdown says so:

> "Tipsomaly reports TIPS and a SigLIP2 ablation. It does **not** report TIPS-v2
> or DINOv2.txt — never invent published numbers for those."

Those four rows (TIPS-v2 and dinov2.txt, MVTec and VisA) are therefore neither
measured nor cited.

## Why it matters here

The backbone track's headline claim is a comparison between backbones. If the
SigLIP2 and TIPS rows are transcriptions of Tipsomaly and the TIPS-v2 and
DINOv2.txt rows are unsourced, the comparison table currently contains no
independent measurement.

There are ordinary explanations — most likely placeholder rows written to fix
the schema, intended to be overwritten once the notebook ran. The point of this
note is only that they are still in the shared repository under a results
filename, so anything built on top of them inherits the problem.

## What is being done

The SigLIP2 backbone work continues in `backbone-eval/`, forked from this
notebook. `vendor/` holds the pristine copies for diffing. No file in
`AnomalyDetectionVLMSurvey` has been modified.

Independently of provenance, Cell 8b (SigLIP2) has defects that would produce
chance-level localization regardless of who runs it — see
`docs/siglip2_defects.md`.

Suggested resolution: replace the four CSVs with measured values once the
notebook runs, or mark them explicitly as `source: Tipsomaly Table 9
(transcribed)` / `source: placeholder, not measured` until then.

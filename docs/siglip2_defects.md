# SigLIP2 backbone (Cell 8b) — defects and proposed fixes

Reference: `vendor/TIPS_vs_CLIP_Benchmark.orig.ipynb`, Cell 8b (`SigLip2Backbone`).
Model: `SIGLIP2_MODEL = "hf-hub:timm/ViT-L-16-SigLIP2-384"`.

Ground truth for every claim below is the official OpenCLIP config
`open_clip/model_configs/ViT-L-16-SigLIP2-384.json`:

```json
{ "embed_dim": 1024, "init_logit_bias": -10,
  "vision_cfg": { "image_size": 384, "timm_model_name": "vit_large_patch16_siglip_384",
                  "timm_pool": "map", "timm_proj": "none" },
  "text_cfg":   { "context_length": 64, "vocab_size": 256000, "width": 1024,
                  "layers": 24, "no_causal_mask": true, "pool_type": "last" } }
```

Defects 1 and 2 are each independently sufficient to produce chance-level
pixel AUROC, which is what Tipsomaly Table 9 reports (47.3 / 47.0).

---

## D1 — Text pooling uses CLIP's rule, not SigLIP's

**Code.**

```python
aux = { ..., "eot_index": tokenized.argmax(dim=-1) }
...
pooled = x[torch.arange(x.shape[0], device=x.device), eot_index]
```

**Why it's wrong.** `argmax` over token ids is CLIP's EOT-finding trick: in
CLIP's 49408-token vocab the EOT id (49407) is the largest id present, so
`argmax` lands on it. SigLIP2 uses a 256000-token Gemma vocab padded to 64, and
its config specifies `"pool_type": "last"` — pool at position −1, unconditionally.
OpenCLIP implements this in `text_global_pool`:

```python
elif pool_type == 'last':
    pooled = x[:, -1]
```

With `argmax`, the pooled position is whichever token happens to carry the
numerically largest id — effectively arbitrary, and different for every prompt.
Affects both `encode_text` (learned) and `encode_fixed_text` (fixed).

**Fix.** Pool at `x[:, -1]`. Applies to both text paths. `eot_index` becomes
unused for this backbone.

**Verify.** Encode a batch of plain strings through the fixed path and compare
against `open_clip`'s own `model.encode_text(tokenizer(texts))`. Must match to
fp32 numerical precision. This single test covers pooling, positional
embeddings, the absence of a causal mask, and the projection.

---

## D2 — Patch tokens never reach the joint embedding space

**Code.**

```python
head = getattr(self.visual, "head", None)
for layer, patches in dense.items():
    if isinstance(head, torch.nn.Linear):
        projected[layer] = head(patches)
    elif hasattr(self.visual, "proj") and self.visual.proj is not None:
        ...
    else:
        projected[layer] = patches          # <-- taken
```

**Why it's wrong.** `"timm_proj": "none"` means OpenCLIP's `TimmModel` builds
`head = nn.Identity()` and defines no `.proj`, so both branches fail and raw
trunk tokens are used. Meanwhile the global embedding is produced by
`self.model.encode_image(x)`, which runs the trunk's `forward_head` — including
`attn_pool`, the MAP head, because `"timm_pool": "map"`.

So the global vector is in the joint image–text space and the patch vectors are
in the trunk's hidden space. They have the same width (1024), so the cosine
similarity computes without error and returns noise.

This is a CLIP-specific assumption leaking through: CLIP's joint space is
`visual.proj @ ln_post(token)`, one linear map applied identically to every
token, so patch tokens transfer for free. SigLIP2's joint space is defined by a
**set-to-vector** attention-pooling head, so there is no per-token map to reuse.

Secondary: dense tokens are captured inside the block loop, i.e. **before**
`trunk.norm` is applied, so even the hidden-space representation is
inconsistent with the one the MAP head consumes.

**Fix — genuine design choice, to be measured rather than assumed:**

- **(a) MAP-head per token.** Run each patch token through `trunk.attn_pool` as
  a length-1 sequence. Uses exactly the function that defines the joint space.
  Training-free. Expected primary candidate.
- **(b) Window-restricted MAP head.** Restrict the probe's keys/values to a
  window's tokens — the WinCLIP masked-CLS trick expressed in SigLIP's
  architecture. More faithful, more expensive.
- **(c) Value-projection surgery.** SigLIP analogue of DPAM / CLIP-Surgery.
  Note `USE_VALUE_ATTENTION = True` already exists in the config for other
  backbones.
- **(d) Keep raw tokens.** The current behaviour. Retain as the **control** —
  it reproduces the published failure and demonstrates the mismatch empirically.

Apply `trunk.norm` before whichever path is chosen.

**Verify.** A window covering the whole image under (b) must reproduce
`model.encode_image(x)` exactly. Under (a), the mean of per-token outputs should
correlate strongly with the true pooled embedding; it will not be identical, and
that gap is itself worth reporting.

### Verified against the built architecture

Built with `open_clip.create_model("ViT-L-16-SigLIP2-384", pretrained=None)`:

```
TEXT    pool_type = last      attn_mask is None = True      context_length = 64
VISION  visual.head = Sequential()   (empty)   visual has .proj = False
        trunk.attn_pool = AttentionPoolLatent(latent_len=1, pool='token', heads=16)
        trunk.num_prefix_tokens = 0            trunk.fc_norm / trunk.head = Identity
```

`visual.head` being an *empty* `Sequential` (not `Linear`) confirms the
fall-through in D2: no projection is skipped, because none exists. The entire
image→joint-space map is `trunk.attn_pool`, applied after `trunk.norm`.

The exact chain was confirmed to reproduce `encode_image` bit-for-bit:

```python
feats  = trunk.forward_features(x)          # [B, 576, 1024], trunk.norm already applied
manual = visual.head(trunk.head(trunk.head_drop(trunk.fc_norm(trunk.attn_pool(feats)))))
# max |encode_image(x) - manual| == 0.0
```

Since `fc_norm`, `head`, `head_drop` and `visual.head` are all identities, this
reduces to **`embedding = trunk.attn_pool(trunk_norm_tokens)`**.

Two properties make fixes (a) and (b) straightforward, with no monkey-patching:

- `attn_pool` accepts **any token count** — verified at L = 576, 4 and 1, each
  returning `[B, 1024]`. Fix (a) is `attn_pool` over a length-1 sequence.
- `attn_pool.forward(x, attn_mask=None)` takes an **optional attention mask**.
  Fix (b) is `attn_pool(tokens, attn_mask=window_mask)`.

---

## D3 — Off-native resolution

`INPUT_SIZE = 518` is applied to every backbone. SigLIP2 ViT-L/16 is trained at
384 (24×24 grid). 518/16 = 32.375 — not an integer patch fit; the code sets
`self.grid = 518 // 16 = 32` and relies on positional-embedding resampling.

CLIP ViT-L/14@336 at 518 is also off-native, but that is the configuration
AnomalyCLIP and Tipsomaly use, so it is the established baseline. SigLIP2 has no
such precedent at 518.

**Fix.** Run SigLIP2 at 384 (24×24) and record the grid size per backbone in the
results table. Alternatively evaluate at both 384 and 518 and report the gap —
this is a fairness axis worth quantifying rather than hiding.

**Open question for Salehi:** hold input resolution constant across backbones,
or hold patch-grid size constant? They cannot both be constant across patch-14
and patch-16 models.

---

## D4 — `logit_bias` is discarded

```python
scale = self.model.logit_scale.exp().item()
self.temperature = float(1.0 / scale)
```

SigLIP2 scores pairs as `t · cos + b`, with `b` learned (`init_logit_bias: -10`;
the released `siglip2-base-patch16-224` checkpoint carries `logit_scale = 4.7245`
→ `t = 112.67`, `logit_bias = −16.77`).

**This is not a bug for the current metrics.** In a two-class softmax over
`{normal, anomalous}`, `b` is a single scalar added to both logits and cancels
exactly:

```
softmax([t·cos_n + b, t·cos_a + b])_a  =  σ( t·(cos_a − cos_n) )
```

Since AUROC, AP, F1-max and AUPRO are invariant to strictly monotone transforms,
the softmax score is well-defined and the choice of `b` cannot affect them.

**It does matter for:** the training loss (softmax CE supervises only the margin;
SigLIP-consistent BCE supervises absolute alignment), prompt-ensemble
aggregation (mean-of-embeddings vs mean-of-probabilities), any additive fusion of
image and pixel scores at different scales, and the calibration/ECE track
(slide 24), where absolute probabilities are the measurement.

**Fix.** Store `logit_bias` alongside `logit_scale` so sigmoid-native scoring is
available as a comparison arm. Keep the two-class softmax as the primary,
CLIP-parity score.

---

## D5 — Learnable prompt slots assume a `[SOS]` token that does not exist

**Code.**

```python
seed_source = embeds[:, 1:1 + N_CTX]
aux = {"prefix": embeds[:, :1], "tail": embeds[:, 1 + N_CTX:], ...}
```

**Why it's wrong.** That layout is CLIP's: `[SOS] [ctx x N] [suffix] [EOS]`.
SigLIP's SentencePiece tokenizer prepends no BOS. Measured on the real
tokenizer with `N_CTX = 8`:

```
"X X X X X X X X object"
  ids: [235297, 1141, 1141, 1141, 1141, 1141, 1141, 1141, 4018, 1, 0, 0, ...]
        └────────── 8 placeholder tokens, positions 0-7 ──────────┘ │  │
                                                       "object" ────┘  └── EOS
```

So the placeholder occupies `[0, N_CTX)`, not `[1, 1+N_CTX)`. The original
slicing takes placeholder tokens 2–8 **plus `"object"`** as the learnable
context, and starts the "fixed suffix" at EOS. The suffix that is supposed to
stay fixed — `object` / `damaged object` — was inside the trainable region and
overwritten, so the two prompts had no fixed semantic difference at all.

**Fix.** No prefix; `ctx = embeds[:, :N_CTX]`, `tail = embeds[:, N_CTX:]`. The
placeholder length is now *measured* at init and raises if it is not `N_CTX`,
so a tokenizer change cannot silently slide the slots again.

---

## D6 — The text sequence is transposed into a batch-first transformer

**Code.**

```python
x = x.permute(1, 0, 2)
x = self._text_tower(x, ...)
x = x.permute(1, 0, 2)
```

**Why it's wrong.** That transpose is correct for OpenAI-CLIP's `[L, B, D]`
convention. OpenCLIP's `TextTransformer` for SigLIP runs **batch-first** —
confirmed on the instance:

```
transformer.batch_first : True
```

and its own `forward` passes `x` straight through, commenting
`x.shape = [batch_size, n_ctx, transformer.width]`. Transposing means the
transformer treats the 2 prompts as the sequence and the 64 positions as the
batch: attention is computed across prompts instead of across tokens.

**Fix.** Remove both permutes.

---

## Verification

`tests/test_siglip2_backbone.py` executes **cell 18 of the working notebook
directly**, so the code under test is the artefact that runs on Kaggle. Weights
are random: every gate is a forward-path equivalence, which does not depend on
weight values.

```
=== D3: native resolution ===
  [PASS] image_size is the checkpoint's native size, not INPUT_SIZE=518  -- image_size=224
  [PASS] grid is a whole number of patches  -- grid=14, patch=16
=== D4: logit_bias retained ===
  [PASS] logit_bias stored  -- logit_bias=-10.0000, temperature=0.07000
=== D1 + D6: text forward matches open_clip exactly ===
  [PASS] encode_fixed_text == open_clip encode_text  -- max|diff|=0.000e+00
=== the old code path, for contrast ===
  [PASS] old path differs from open_clip  -- max|diff|=5.614e+00, cos=[0.0523, 0.0522, 0.0503]
  [PASS] argmax lands on position 0 for every prompt  -- argmax indices=[0, 0, 0], length=64
=== D5: learnable prompt slots align with the placeholder ===
  [PASS] ctx shape [2, N_CTX, D]  -- ctx=(2, 8, 768)
  [PASS] ctx + tail restores full context length  -- 8 + 56 = 64
  [PASS] tail preserves the fixed suffix  -- max|diff|=0.000e+00
=== D2: patch tokens reach the joint space ===
  [PASS] attn_pool over all tokens == encode_image  -- max|diff|=0.000e+00
  [PASS] dense is [B, grid, grid, D]  -- (2, 14, 14, 768)
  [PASS] control mode 'raw' reproduces the unprojected tokens  -- max|diff|=5.170e+00
  [PASS] raw and projected have identical shape (why the bug was silent)  -- (2, 14, 14, 768)
```

Two results are worth carrying into the write-up:

- **The old text embedding is near-orthogonal to the correct one — cosine
  0.05.** It was not a small misalignment; the text tower was returning a
  vector unrelated to the prompt.
- **`argmax` returns position 0 for every prompt tested.** SentencePiece gives
  rare and capitalised pieces high ids, so the first token usually holds the
  largest id in the row. CLIP-style pooling read the *first* token, not the last.

`raw` and `map_token` produce identically shaped tensors — hidden width and
joint width are both 768 (base) / 1024 (large). That is precisely why D2 never
raised an error.

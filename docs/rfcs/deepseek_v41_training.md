# RFC: DeepSeek-V4.1 Training Support in TorchTitan-NPU

- **Status**: Draft
- **Target package**: `torchtitan_npu/models/deepseek_v41`
- **Base implementation**: `torchtitan_npu/models/deepseek_v4`
- **Reference checkpoint**: `deepseek-ai/DeepSeek-V4.1-Flash`
- **Scope**: V4.1 backbone training, Single-Pass mHC, CED/CSA2, MoE, multimodal vision path and distributed-training integration
- **Out of scope**: DSpark and Engram

## 1. Summary

DeepSeek-V4.1 retains a large part of DeepSeek-V4's model vocabulary and training infrastructure, but changes several graph-level semantics: CSA becomes cross-layer shared CSA2, the 40-layer backbone forms a causal-encoder/decoder topology, mHC changes from same-sublayer consumption to Single-Pass mHC, and the released model adds a vision path and a separate VL routing bias.

This RFC proposes that V4.1 is implemented as an **incremental specialization of the existing V4 implementation**, not as a second copied model stack.

The governing rule is:

> Reuse/inherit DeepSeek-V4 whenever the mathematical and distributed semantics are unchanged. Override only the V4.1-specific behavior. If an override would copy a substantial V4 forward or sharding path, first add a small semantic extension seam to V4 and keep V4's default behavior unchanged.

Concretely:

- `DeepSeekV41Model` subclasses `DeepSeekV4Model`;
- `DeepSeekV41TransformerBlock` subclasses `DeepSeekV4TransformerBlock` and overrides the block forward for Single-Pass mHC and cross-layer CSA2 state;
- `DeepSeekV41Attention` subclasses the V4 attention and reuses Q/SWA/output projections after those are factored into protected helpers;
- the V4 compressor is generalized by semantic configuration so V4.1 can reuse it for ratio-2 pooling and ratio-1 projection;
- the V4 sparse-attention wrapper/core is generalized to accept externally supplied Top-K indices instead of assuming ratio-4 owns the indexer;
- the V4 metadata/CP plan code is generalized only where the block geometry is truly common; V4.1 keeps its own layer-policy/state orchestration;
- the V4 state-dict adapter is refactored into common mapping hooks and V4-specific long-range mapping hooks, allowing `DeepSeekV41StateDictAdapter` to inherit the common mappings without inheriting V4's ratio-based assumptions;
- the vision module and V4.1 indexer are new V4.1 code because no sufficiently similar V4 implementation exists.

The package name is deliberately `deepseek_v41` to match the released Hugging Face architecture name `deepseek_v41`.

## 2. Source of truth and known limits

The structural source of truth for this RFC is the released V4.1 inference repository and configuration:

- https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash
- https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/inference/config.json
- https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/inference/model.py
- https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/inference/vision.py

V4 reference:

- https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash

The public V4.1 inference implementation is sufficient to recover the backbone forward topology and checkpoint structure, but it does **not** publish every training-time algorithm. In particular, the sparse indexer's discrete Top-K selection is non-differentiable and the inference repository does not define the training-time V4.1 indexer supervision/loss. This RFC therefore separates:

1. **model-architecture training support**: differentiable backbone, FSDP/TP/EP integration, checkpoint load/save, AC and optimizer integration;
2. **DeepSeek-private training recipe parity**: exact indexer supervision, correction-bias update coefficients, quantized-training recipe, curriculum and optimizer settings.

The first is the implementation target of this RFC. Unknown private training details must remain explicit extension points rather than being inferred and presented as official behavior.

## 3. V4 vs V4.1: changes relevant to this RFC

| Area | DeepSeek-V4-Flash | DeepSeek-V4.1-Flash | Design implication |
|---|---|---|---|
| Backbone | 43 layers, hidden 4096 | 40 layers, hidden 5120 | config-only for most dense/MoE components |
| MoE | 256 routed experts + 1 shared, Top-6; initial hash routing | 384 routed experts + 1 shared, Top-6 | reuse common MoE; V4.1 disables hash bootstrap |
| Norm epsilon | `1e-6` | `1e-20` | config change; NPU numerical test required |
| Sparse attention | V4 CSA/HCA, layer-local compressor/indexer; ratios 4/128 | CSA2, cross-layer shared compressed KV/index K/Top-K; ratios 2/1 | attention state becomes explicit cross-layer data flow |
| SWA-only encoding | local implementation uses ratio 1 as no-compression sentinel | released config uses ratio 0 for SWA-only; ratio 1 is real global KV | ratio value can no longer imply ownership/attention mode |
| KV sources | layer-owned | `[2, 8, 14, 20]` | source layer publishes shared KV state |
| Index sources | CSA layer owns its indexer | `[2, 8, 14, 20, 24, 28, 32, 36]` | reindex layers reuse index K but recompute query-side ranking |
| Candidate source | none | layer 20, 2048 blocks, block size 8 | hierarchical sparse selection |
| CED | one decoder-like stack | layers 0-19 causal encoder, 20-39 decoder | no physical encoder module required; policy captures topology |
| mHC | current sublayer computes and consumes current pre-mix | current sublayer computes pre-mix for the next sublayer | block forward carries pre-mix state |
| Vision | none | 32-layer ViT, 2D RoPE, 3x3 downsample, aligner | new vision module and input contract |
| MoE VL routing | one correction bias | separate text and image correction biases | specialized router + load-balance hook |
| FP8 block | 128x128 in V4 release | 32x32 in V4.1 release | quantization config, not backbone class semantics |

Released V4.1 backbone constants relevant here:

```text
vocab_size              = 129280
dim                     = 5120
moe_inter_dim           = 2304
n_layers                = 40
n_heads                 = 64
head_dim                = 512
rope_head_dim           = 64
q_lora_rank             = 1280
o_groups                = 8
o_lora_rank             = 1024
n_routed_experts        = 384
n_shared_experts        = 1
n_activated_experts     = 6
score_func              = sqrtsoftplus
route_scale             = 1.5
swiglu_limit            = 10.0
norm_eps                = 1e-20
window_size             = 128
compress_ratios         = [0, 0] + [2] * 18 + [1] * 20
kv_source_layers        = [2, 8, 14, 20]
index_source_layers     = [2, 8, 14, 20, 24, 28, 32, 36]
index_n_heads           = 32
index_head_dim          = 128
index_topk              = 512
candidate_source_layer  = 20
candidate_topk_blocks   = 2048
candidate_block_size    = 8
hc_mult                 = 4
hc_sinkhorn_iters       = 20
hc_eps                  = 1e-6
original_seq_len        = 65536
compress_rope_theta     = 160000
rope_theta              = 10000
rope_factor             = 16
```

Vision constants:

```text
vision_n_layers         = 32
vision_dim              = 1024
vision_n_heads          = 16
vision_inter_dim        = 2816
vision_patch_size       = 14
vision_downsample_ratio = 3
vision_max_n_token      = 1024
vision_rope_theta       = 10000
image_token_id          = 129264
```

## 4. Goals and non-goals

### 4.1 Goals

1. Preserve V4 behavior and checkpoint compatibility while adding only narrow extension seams to V4.
2. Keep V4.1 implementation small enough that a reviewer can identify every semantic delta from V4.
3. Express CSA2 sharing as explicit tensors passed between layers; no process-global or module-global mutable attention cache in training.
4. Reuse V4's Q projection, sliding-window KV projection, output projection, MoE experts, RoPE, token dispatcher, TP/EP/FSDP conventions and model registration patterns.
5. Support text-only V4.1 training first, while making the multimodal path a first-class part of the same model family rather than a separate model.
6. Preserve TorchTitan's `_skip_lm_head` / chunked-loss contract.
7. Make activation checkpointing and future graph compilation possible by keeping cross-layer state as registered pytrees/ordinary tensors.
8. Make CP correctness and production scalability separate milestones: do not hide a full-KV all-gather behind a claim of scalable V4.1 CP support.
9. Keep unknown training-only algorithms, especially indexer supervision, behind explicit interfaces.

### 4.2 Non-goals

This RFC does not implement or design:

- DSpark;
- Engram;
- inference KV-cache lifetime / Bounded Replay;
- FP4 inference kernels;
- exact DeepSeek pretraining hyperparameters;
- exact private indexer supervision until a source of truth is available;
- first-PR Pipeline Parallel support;
- first-PR GraphTrainer support.

## 5. Package and dependency structure

New package:

```text
torchtitan_npu/models/deepseek_v41/
├── __init__.py
├── model.py
├── attention.py
├── csa2.py
├── indexer.py
├── moe.py
├── metadata.py
├── vision.py
├── sharding.py
├── parallelize.py
├── state_dict_adapter.py
├── config_registry.py
└── reference.py
```

No V4.1 copies of `mhc.py`, `compressor.py` or `token_dispatcher.py` should be created unless implementation proves that a semantic split is unavoidable. The initial design reuses/generalizes those V4 modules.

Expected dependency direction:

```text
deepseek_v41
    |
    +--> deepseek_v4.model / attention / compressor / mhc / metadata / token_dispatcher
    |
    +--> torchtitan.models.common.*
```

V4 must not import V4.1.

## 6. Inheritance/override matrix

| V4 component | V4.1 strategy | Reason |
|---|---|---|
| `DeepSeekV4Model` | subclass | same decoder infrastructure, attention metadata lifecycle, embedding/norm/lm-head conventions |
| `DeepSeekV4TransformerBlock` | subclass; reuse `__init__`, override `forward` | parameters/modules are mostly identical; mHC sequencing differs |
| V4 `Attention` | subclass; reuse projection helpers, override long-range-context construction | Q/SWA/output are same; CSA2 ownership/state differs |
| `CompressedSparseAttention` | reuse after small extension | V4.1 needs externally supplied Top-K |
| `CompressedSparseInnerAttention` | reuse after selection-mode generalization | same sparse core topology; ratio value must not imply selection mode |
| V4 `Compressor` | reuse after semantic configuration/refactor | both are projection + optional softmax pooling + norm + RoPE |
| V4 `Indexer` | do not subclass as primary implementation | V4.1 derives index K from compressor latent with new `wk/k_norm`, supports cross-layer K reuse and candidates |
| `HcPre/HcPost` | reuse after splitting computation/application helpers | exact same mHC math; consumption timing differs |
| V4 MoE experts | reuse | expert math and dispatch remain compatible |
| `TokenChoiceTopKRouter` | V4.1 subclass | separate VL correction bias is a narrow routing delta |
| `CPTokenDispatcher` | reuse + later extend selected-row fetch | window/block exchange infrastructure remains useful |
| V4 metadata dataclasses | reuse | represent packed/compressed plans; builders need ratio-1/2 support |
| V4 state-dict adapter | subclass after mapping-hook refactor | most base weights map identically; long-range ownership differs |
| V4 eager parallelize path | reuse where possible | common TP/EP/FSDP sequencing unchanged |
| V4 GraphTrainer path | not inherited initially | V4 GraphTrainer already has sparse-attention CP limitations |
| V4 MTP | not used | DSpark is out of scope |
| Vision | new | no V4 equivalent |

## 7. Minimal changes to `deepseek_v4`

The following changes are proposed specifically to avoid V4.1 copying large V4 bodies. All defaults must preserve current V4 behavior.

### 7.1 `mhc.py`: split mix computation from pre-mix application

Today `HcPre.forward()` both computes `pre/post/comb` and immediately applies the newly computed `pre` to the same residual stream. Keep this public behavior, but factor it into reusable helpers:

```python
class HcPre(Module):
    def compute_mixes(self, x):
        # existing fp32 normalization, linear projection and Sinkhorn
        return pre, post, comb

    @staticmethod
    def apply_pre(x, pre):
        return torch.sum(pre.unsqueeze(-1) * x, dim=2).type_as(x)

    def forward(self, x):
        pre, post, comb = self.compute_mixes(x)
        return self.apply_pre(x, pre), post, comb
```

V4 remains unchanged. V4.1 calls `compute_mixes()` but feeds an incoming previous-sublayer pre-mix to `apply_pre()`.

`HcPost` is reused unchanged.

### 7.2 `compressor.py`: make pooling semantics explicit and expose pre-RoPE latent

V4.1's index K is derived from the **pre-RoPE compressed latent**, while the V4 compressor currently returns its final RoPE-applied key stream.

Refactor the existing compressor into:

```python
class Compressor(Module):
    class Config(Module.Config):
        compress_ratio: int
        pooling: Literal["identity", "softmax"] = "softmax"
        overlap: bool = False
        use_ape: bool = False
        wkv: Linear.Config
        wgate: Linear.Config | None
        ...

    def pool(self, x, attention_masks):
        """Return normalized pre-RoPE latent and latent positions."""
        ...
        return latent, positions

    def apply_rope(self, latent, positions):
        ...

    def forward(self, x, attention_masks):
        latent, positions = self.pool(x, attention_masks)
        return self.apply_rope(latent, positions)
```

V4 registry preserves current semantics:

```text
ratio 4:   pooling=softmax, overlap=True,  use_ape=True
ratio 128: pooling=softmax, overlap=False, use_ape=True
```

V4.1 registry uses:

```text
ratio 2: pooling=softmax, overlap=False, use_ape=False
ratio 1: pooling=identity, overlap=False, use_ape=False, wgate=None
```

The purpose of these fields is semantic, not version gating. There must be no `if deepseek_v41` inside V4 compressor code.

### 7.3 `metadata.py`: separate layer attention mode from materialized compression ratio

Current V4 code embeds a V4-specific convention: ratio 1 means no compressed region. V4.1 makes ratio 1 a real global-KV projection and uses ratio 0 for SWA-only layers.

Do not add a model-version conditional. Add an explicit model-level concept:

```python
materialized_compression_ratios: tuple[int, ...]
```

V4 computes this as `(4, 128)` for the released model. V4.1 uses `(1, 2)`.

Refactor the non-CP plan builder so the generic per-document block-layout helper supports every positive ratio. The V4 caller still validates its own supported ratios; V4.1 performs its own validation.

For ratio 1 the layout is a real identity block layout:

```text
one token -> one compressed slot
block_remainder = 0
block_positions = token positions
gather_indices = token rows
```

No implicit sentinel behavior should remain in the generic builder.

### 7.4 `token_dispatcher.py`: generalize positive-ratio plan geometry without changing V4 output

The existing CP block planner is largely ratio-generic for ratio > 1. V4.1 ratio 2 should reuse it. Ratio 1 requires a simple identity-slot plan rather than the multi-token straddle/borrow geometry.

Refactor only the block-range assembly boundary:

```python
def build_block_plan(..., ratio: int):
    if ratio == 1:
        return build_identity_block_plan(...)
    return build_multitoken_block_plan(...)
```

V4 tests must prove bitwise/equivalent plan output for ratios 4/128 after this refactor.

This is only the **correctness plan**. Production-scale CSA2 CP is discussed separately in section 15.

### 7.5 `attention.py`: factor common projections and long-range context

Today `Attention.forward()` combines several concerns. V4.1 would otherwise copy most of the function.

Factor three helpers whose math is shared:

```python
class Attention(BaseAttention):
    def _project_q(self, x, positions):
        # wq_a -> q_norm -> wq_b -> q normalization -> RoPE
        return qr, q

    def _project_window_kv(self, x, positions, attention_masks):
        # wkv -> kv_norm -> RoPE -> CP window gather
        return swa_k

    def _project_output(self, o, positions):
        # inverse RoPE -> grouped wo_a -> wo_b
        return output
```

Represent long-range inputs in a small internal object:

```python
@dataclass(slots=True)
class LongRangeContext:
    compressed_kv: torch.Tensor | None = None
    index_q: torch.Tensor | None = None
    index_k: torch.Tensor | None = None
    index_weight: torch.Tensor | None = None
    sparse_indices: torch.Tensor | None = None
```

Add an overridable method:

```python
def _build_long_range_context(
    self,
    x,
    qr,
    attention_masks,
    positions,
    state=None,
) -> tuple[LongRangeContext, object | None]:
    # current V4 compressor/indexer behavior
```

Expose a state-aware internal forward while keeping V4's public forward unchanged:

```python
def forward_with_state(self, x, attention_masks, positions, state=None):
    qr, q = self._project_q(...)
    swa_k = self._project_window_kv(...)
    ctx, state = self._build_long_range_context(..., state)
    o = self.compressed_sparse_attention(
        q,
        swa_k,
        ctx.compressed_kv,
        idx_q=ctx.index_q,
        idx_k=ctx.index_k,
        idx_w=ctx.index_weight,
        sparse_indices=ctx.sparse_indices,
        attn_sink=self.attn_sink,
        attention_masks=attention_masks,
    )
    return self._project_output(o, positions), state


def forward(self, x, attention_masks, positions):
    y, _ = self.forward_with_state(x, attention_masks, positions)
    return y
```

Thus V4 call sites do not change.

### 7.6 Sparse-attention wrapper/core: selection mode must be explicit

Current V4 code implicitly equates `compress_ratio == 4` with Top-K CSA. That is not valid for V4.1, where both ratio 2 and ratio 1 use selected compressed/global KV.

Extend `CompressedSparseAttention.forward()`:

```python
def forward(..., sparse_indices=None):
    if sparse_indices is None and self.lightning_indexer is not None:
        sparse_indices = self.lightning_indexer(...)
    return self.inner_attention(..., sparse_indices=sparse_indices)
```

Extend inner-attention config with semantics instead of ratio checks:

```python
compressed_selection: Literal["none", "all", "topk"]
```

V4 config:

```text
SWA-only  -> none
CSA-4     -> topk
HCA-128   -> all
```

V4.1 config:

```text
ratio 0 -> none
ratio 2 -> topk
ratio 1 -> topk
```

The causal visibility rule remains ratio-driven and works for ratio 1:

```text
compressed block j is visible to query position p iff
j < floor((p + 1) / ratio)
```

Packed-document boundaries remain enforced by metadata.

### 7.7 `mtp.py`: permit V4-derived models without a parameterized `HcHead`

V4.1 does not collapse the final mHC streams with V4's learned `HcHead`; it applies the final propagated pre-mix. To allow `DeepSeekV41Model` to inherit the V4 model hierarchy without an unused checkpoint parameter, make the decoder config's main `hc_head` optional:

```python
hc_head: HcHead.Config | None
```

V4 always supplies it, so V4 behavior and state dict are unchanged. V4.1 sets it to `None` and overrides the main-backbone forward/final collapse. Existing V4 MTP paths continue to require `hc_head` and should raise a configuration error if MTP is enabled while it is absent.

This is an inheritance seam only; DSpark is not implemented by this RFC.

### 7.8 `sharding.py`: split common attention sharding from V4 long-range sharding

Current `set_deepseek_v4_attention_sharding()` configures both shared Q/K/output modules and V4 compressor/indexer internals. Split it into:

```python
set_deepseek_v4_attention_common_sharding(...)
set_deepseek_v4_long_range_sharding(...)
```

The public V4 helper calls both. V4.1 calls the common helper and its own CSA2/indexer sharding helper.

Likewise keep mHC, norms and MoE common sharding reusable from `set_deepseek_v4_layer_sharding()` through small helpers rather than copying the entire function.

### 7.9 `state_dict_adapter.py`: factor mapping registration

Refactor V4 adapter initialization into protected mapping methods:

```python
_register_global_common_mappings()
_register_layer_common_mappings(layer_id, layer_cfg)
_register_v4_long_range_mappings(layer_id, layer_cfg)
_register_mhc_mappings(layer_id, layer_cfg)
```

`DeepSeekV4StateDictAdapter` calls all of them.

`DeepSeekV41StateDictAdapter` inherits expert stacking/splitting and common mappings but replaces `_register_v4_long_range_mappings()` with source-policy-based V4.1 mappings.

## 8. V4.1 static CSA2 layer policy

Do not scatter layer-id checks through forward code. Build one immutable policy per layer in the registry:

```python
@dataclass(frozen=True, slots=True)
class CSA2LayerPolicy:
    mode: Literal["swa", "full", "reindex", "reuse"]
    compress_ratio: int | None
    is_kv_source: bool
    is_index_source: bool
    is_candidate_source: bool
    uses_candidates: bool
```

Released 40-layer policy:

```text
0-1   SWA

2     FULL    ratio=2
3-7   REUSE   ratio=2
8     FULL    ratio=2
9-13  REUSE   ratio=2
14    FULL    ratio=2
15-19 REUSE   ratio=2

20    FULL    ratio=1, candidate source
21-23 REUSE   ratio=1
24    REINDEX ratio=1, candidate-filtered
25-27 REUSE   ratio=1
28    REINDEX ratio=1, candidate-filtered
29-31 REUSE   ratio=1
32    REINDEX ratio=1, candidate-filtered
33-35 REUSE   ratio=1
36    REINDEX ratio=1, candidate-filtered
37-39 REUSE   ratio=1
```

This policy makes CED emerge naturally: layer 20's attention input is the final causal-encoder representation and layer 20 publishes the ratio-1 global KV consumed by the decoder layers. No physical `Encoder`/`Decoder` module split or cross-attention class is needed.

## 9. Explicit cross-layer CSA2 state

### 9.1 Training state, not inference cache

The released inference implementation uses a global mutable `SharedAttentionRuntime`. That is appropriate for sequential inference but unsuitable for training because it complicates:

- autograd ownership;
- activation-checkpoint recomputation;
- multiple microbatches;
- reentrant forwards;
- pipeline stage boundaries;
- graph capture.

Training uses explicit forward state:

```python
@dataclass(slots=True)
class SharedAttentionState:
    compressed_kv: torch.Tensor | None = None
    index_k: torch.Tensor | None = None
    sparse_indices: torch.Tensor | None = None
    candidates: torch.Tensor | None = None
```

Register it as a pytree.

One slot per field is sufficient because source and consumer layers are strictly ordered and a new Full source replaces the previous source before any later consumer needs it.

### 9.2 Full

A Full layer owns both compressor and index source:

```text
attention input x
    |
    +-- Compressor.pool() ------------------+
    |      -> pre-RoPE latent               |
    |                                       |
    +-- V41Indexer.publish_k(latent)         |
    |      -> shared index_k                 |
    |                                       |
    +-- Compressor.apply_rope(latent)        |
           -> shared compressed_kv           |
                                            |
current qr/x + shared index_k               |
    -> index scores                         |
    -> optional candidate generation/filter |
    -> top-k                                |
                                            v
state = {compressed_kv, index_k, topk, candidates}
```

Gradient from later Reuse/Reindex layers' attention can flow through `compressed_kv` back to the source compressor because the state is an ordinary tensor edge in the autograd graph.

### 9.3 Reindex

A Reindex layer owns query-side index projections but not KV/index-K publication:

```text
reuse state.compressed_kv
reuse state.index_k
current x/qr -> new index q + weights
             -> score shared index_k
             -> apply state.candidates
             -> new top-k
update state.sparse_indices only
```

### 9.4 Reuse

A Reuse layer owns neither compressor nor indexer:

```text
reuse state.compressed_kv
reuse state.sparse_indices
```

No duplicate compressor/indexer parameters are built for these layers.

## 10. `DeepSeekV41Attention`

`DeepSeekV41Attention(V4Attention)` reuses:

- `wq_a`, `q_norm`, `wq_b`;
- q normalization and partial RoPE;
- `wkv`, `kv_norm` for the local sliding window;
- attention sink;
- inverse RoPE;
- grouped `wo_a` and `wo_b`.

It only replaces `_build_long_range_context()`.

Conceptual config:

```python
class DeepSeekV41Attention(V4Attention):
    @dataclass(kw_only=True, slots=True)
    class Config(V4Attention.Config):
        policy: CSA2LayerPolicy
        compressor: Compressor.Config | None
        indexer: DeepSeekV41Indexer.Config | None
```

Construction rules:

```text
SWA      compressor=None, indexer=None
FULL     compressor=yes,  indexer=yes
REINDEX  compressor=None, indexer=yes
REUSE    compressor=None, indexer=None
```

The `compress_ratio` describes the geometry of the currently shared KV. It does **not** imply that this layer owns a compressor.

This distinction is the central architectural difference between V4 and V4.1 and must be visible in the config model.

## 11. V4.1 Indexer and hierarchical candidates

V4.1 indexer is implemented in `deepseek_v41/indexer.py`; it should not inherit V4 Indexer's module layout because V4 owns an internal compressor while V4.1 derives index K from the CSA2 source compressor's latent.

It may reuse small pure scoring helpers from V4 after refactoring, but its parameter ownership is V4.1-specific.

### 11.1 Parameter ownership

Full source layer:

```text
wq_b
weights_proj
wk
k_norm
```

Reindex layer:

```text
wq_b
weights_proj
```

No `wk/k_norm` on reindex-only layers.

### 11.2 K publication

At a Full source:

```text
pre-RoPE compressor latent
  -> wk
  -> k_norm
  -> compressed-position RoPE
  -> shared index_k
```

This happens before the same latent is RoPE-transformed into main compressed KV.

### 11.3 Scoring

For every index source:

```text
qr -> wq_b -> index q -> RoPE
x  -> weights_proj

scores(q, index_k)
 -> ReLU
 -> head weighting
 -> sum over index heads
 -> causal mask
 -> optional candidate mask
 -> top-k 512
```

The Top-K result is position-sorted after selection to match the released inference semantics.

### 11.4 Hierarchical candidate source

Layer 20 produces the candidate mask from its full index-score tensor:

```text
compressed positions
 -> group into blocks of 8
 -> block score
 -> top 2048 blocks
 -> expand selected blocks back to position mask
```

Layers 24/28/32/36 score the same shared `index_k` with their own q/weights but mask all positions outside layer-20 candidates before final Top-K.

Candidate construction is a pure helper and should be independently unit tested.

### 11.5 Training-time indexer loss seam

Because Top-K indices are discrete, the indexer projection parameters do not receive a useful training signal solely through the selected attention path. The released inference repository does not expose the V4.1 training objective for this component.

Add an explicit interface rather than hard-coding V4's objective:

```python
class IndexerTrainingObjective(Protocol):
    def __call__(
        self,
        *,
        index_scores,
        attention_context,
        selected_indices,
        layer_id,
    ) -> torch.Tensor | None: ...
```

`DeepSeekV41Indexer` may return both `selected_indices` and an optional auxiliary-loss carrier/metrics object. The default structural implementation can disable the objective for checkpoint parity tests. Pretraining from scratch must not be declared recipe-complete until this objective is grounded in an authoritative source.

## 12. Single-Pass mHC

V4.1 block parameters are still the same mHC parameter families, so `DeepSeekV41TransformerBlock` reuses the V4 block constructor and only overrides forward sequencing.

Initial state:

```python
pre_mix = make_identity_pre_mix(hidden, hc_mult)
```

Block forward:

```python
def forward(
    self,
    x,
    input_ids,
    attention_masks,
    positions,
    pre_mix,
    shared_attention_state,
    image_mask=None,
):
    # attention sublayer
    residual = x
    attn_pre, attn_post, attn_comb = self.hc_attn_pre.compute_mixes(x)
    attn_input = self.hc_attn_pre.apply_pre(x, pre_mix)
    attn_out, shared_attention_state = self.attention.forward_with_state(
        self.attention_norm(attn_input),
        attention_masks,
        positions,
        shared_attention_state,
    )
    x = self.hc_post(attn_out, residual, attn_post, attn_comb)

    # FFN sublayer
    residual = x
    ffn_pre, ffn_post, ffn_comb = self.hc_ffn_pre.compute_mixes(x)
    ffn_input = self.hc_ffn_pre.apply_pre(x, attn_pre)
    ffn_out = self.moe(
        self.ffn_norm(ffn_input),
        input_ids=input_ids,
        image_mask=image_mask,
    )
    x = self.hc_post(ffn_out, residual, ffn_post, ffn_comb)

    # ffn_pre is consumed by the next layer's attention
    return x, ffn_pre, shared_attention_state
```

After the last backbone block:

```text
hidden = apply_pre(hidden, final_pre_mix)
normalized = norm(hidden)
```

There is no V4 parameterized `HcHead` in the V4.1 backbone checkpoint.

## 13. `DeepSeekV41Model`

### 13.1 Inheritance

```python
class DeepSeekV41Model(DeepSeekV4Model):
    ...
```

Reuse from V4 model:

- model config/update conventions;
- tokenizer embedding and LM head configs;
- sparse attention mask/metadata hook shape;
- parameter initialization framework;
- `_skip_lm_head` contract;
- FSDP/TP/EP model-spec conventions.

Override the main-backbone forward because V4.1 must carry `pre_mix` and `SharedAttentionState`.

### 13.2 Main forward

Conceptually:

```text
input_ids
  -> token embedding
  -> optional image embedding replacement
  -> expand hc_mult streams
  -> identity pre_mix
  -> empty SharedAttentionState
  -> layer 0 ... layer 39
       carrying:
         hidden
         pre_mix
         shared CSA2 state
  -> final apply_pre
  -> norm
  -> lm_head / hidden for ChunkedLossWrapper
```

`input_ids` are preserved for MoE/router context. `image_mask` is an optional tensor and is propagated only to the MoE router; text-only training pays no vision/routing overhead beyond static branches.

### 13.3 CED is policy, not module topology

Do **not** define `DeepSeekV41Encoder` and `DeepSeekV41Decoder` modules.

The released CED is naturally represented by the source schedule:

- layers 0-19 execute causally;
- layer 20 attention receives layer-19 output and publishes ratio-1 global KV;
- layers 20-39 combine local sliding-window KV with selected positions from that global KV.

A physical encoder/decoder split would add pipeline/state-dict complexity without improving correctness.

## 14. Multimodal path

### 14.1 New vision implementation

`deepseek_v41/vision.py` contains:

```text
PatchEmbed
VisionAttention
VisionMLP
VisionBlock
DeepSeekViT
Aligner
```

Architecture from the released reference:

```text
flattened RGB patches
 -> Linear(3 * patch_size^2, 1024)
 -> 32 ViT blocks
      full bidirectional attention
      2D RoPE
      SwiGLU MLP
 -> RMSNorm
 -> 3x3 spatial downsample/rearrange
 -> 2-layer aligner
 -> LM hidden 5120
```

There is no V4 equivalent worth subclassing. Use TorchTitan common Linear/RMSNorm abstractions where practical so initialization/sharding remain consistent.

### 14.2 Training batch contract

Avoid Python object graphs in the model forward. Define a tensorized input structure, for example:

```python
@dataclass(slots=True)
class VisionBatch:
    patches: torch.Tensor
    image_offsets: torch.Tensor
    grid_hw: torch.Tensor
    span_indices: torch.Tensor
    token_types: torch.Tensor
```

Exact packing can evolve, but the contract must allow the model to:

1. run each image through ViT/aligner;
2. scatter aligned image rows into the matching LM token spans;
3. write learned start/end/newline embeddings;
4. derive `image_mask` for VL routing.

Text-only batches use `vision_batch=None`.

### 14.3 VL correction bias

TorchTitan's common `TokenChoiceTopKRouter` already separates selection bias from routing weights and accepts arbitrary `router_kwargs`. Reuse that behavior.

Implement:

```python
class DeepSeekV41TokenChoiceTopKRouter(TokenChoiceTopKRouter):
    # owns persistent expert_bias_vl_E and VL routing counts

    def _select_experts(
        self,
        scores_TE,
        expert_bias_E=None,
        *,
        image_mask=None,
        **kwargs,
    ):
        # choose text bias or VL bias per token only for expert selection
        ...
```

No custom routed-expert implementation is required.

The ordinary `MoE.forward(..., image_mask=image_mask)` already passes router kwargs through to the router.

For balancing, keep TorchTitan's existing total `tokens_per_expert_E` and add VL-specific counts in the V4.1 router. The V4.1 optimizer-step pre-hook can derive:

```text
text_count = total_count - vl_count
```

and update text and VL correction biases independently. The exact DeepSeek update coefficient is a training-recipe item and must be configurable rather than guessed.

## 15. Context Parallel design

CSA2 creates a new scalability problem that should not be hidden by reusing V4's existing `S(1) -> R` compressed-container all-gather.

### 15.1 Correctness path

For the first structural implementation:

- CP=1 is the required reference path;
- ratio-2/ratio-1 metadata is implemented and unit tested;
- an optional small-sequence CP correctness path may reuse a replicated compressed/index container if memory permits;
- production support documentation must state this is not a scalable 64K+ design.

This lets model correctness, checkpoint parity, TP, EP and FSDP land independently of the new distributed sparse-selection protocol.

### 15.2 Production CSA2 CP

For long context, `compressed_kv` and `index_k` remain sequence-sharded across CP rather than all-gathered.

The distributed algorithm should be:

```text
source layer:
  local source tokens
   -> local compressed/index K shard with global slot IDs

index source:
  local query shard x local index-K shard
   -> local scores
   -> local top-K candidates
   -> distributed top-K merge across CP
   -> global selected slot IDs

attention:
  selected global slot IDs
   -> request selected KV rows from owner ranks
   -> all-to-all selected-row fetch
   -> sparse attention

backward:
  selected-row gradient
   -> reverse owner routing / scatter-add
   -> source compressed-KV shard
```

Candidate selection follows the same pattern at block granularity: local block candidates followed by a global candidate merge.

Add a future dispatcher seam rather than embedding communication in V4.1 attention:

```python
class CPTokenDispatcher:
    ...
    def distributed_topk(...): ...
    def fetch_selected(...): ...
```

or introduce a sibling `CPSparseKVDispatcher` if implementation shows the ownership semantics differ too much from the current row/block dispatcher.

The final choice should be based on code cohesion, not forced inheritance.

### 15.3 Shared-state placements

Expected logical placements:

```text
compressed_kv: TP replicated; CP sequence-sharded in production
index_k:       TP replicated; CP sequence-sharded in production
sparse_indices: query-sequence sharded; invariant across TP after score reduction
candidates:     query-sequence sharded; invariant across TP
```

If index heads are TP-sharded, index scores are partial over index heads and must be reduced before Top-K.

## 16. Tensor Parallel / Expert Parallel / FSDP

### 16.1 TP

Reuse V4 placements for:

- `wq_a`, q norm;
- `wq_b` head sharding;
- shared `wkv`, kv norm;
- grouped `wo_a`;
- row-parallel `wo_b`;
- mHC parameters/norms;
- main MoE expert tensors.

V4.1 indexer policy:

- `wq_b`: column/head-sharded when enabled;
- `weights_proj`: sharded over index heads consistently with `wq_b`;
- source `wk` and `k_norm`: replicated across TP because they produce one shared index-K stream;
- score sum over sharded index heads: all-reduce/Partial->Invariant before candidate/Top-K selection.

A replicated-indexer correctness mode is acceptable initially if needed, but it should be explicit in the config and not confused with the final topology.

### 16.2 EP

Reuse V4/TorchTitan routed-expert dispatch and grouped expert layout. V4.1 changes expert count to 384 but not expert-dispatch semantics.

`image_mask` follows the token stream into the router only; it is not dispatched to experts because expert compute does not depend on token type after routing.

### 16.3 FSDP

Use existing Decoder/layer FSDP boundaries. `SharedAttentionState` is activation data and is not FSDP state.

CSA2 source tensors must remain alive across their consumer layers; activation-checkpoint/memory-policy tuning should account for that explicitly. They must not be silently recomputed through module-global caches.

## 17. Activation checkpointing and compile

### 17.1 AC

Register `SharedAttentionState` as a pytree and ensure every V4.1 block is a pure function of:

```text
(hidden, pre_mix, shared_state, static metadata, token context)
```

The same input state during recomputation must reproduce the same Top-K routing decisions. Top-K should be computed inside the checkpointed block from deterministic tensors, or marked as a non-recomputed routing decision using the same rematerialization conventions as TorchTitan MoE routing where appropriate.

### 17.2 GraphTrainer

Do not make GraphTrainer part of the first V4.1 acceptance bar. The current V4 GraphTrainer path already restricts sparse-attention CP. Once eager semantics are stable, add V4.1 graph annotations and state-pytree support as a separate PR.

No V4.1 design decision in this RFC should make GraphTrainer impossible; in particular, avoid hidden mutable cross-layer runtime state.

## 18. State-dict adapter

`DeepSeekV41StateDictAdapter` inherits common V4 adapter machinery but uses V4.1 ownership policy.

### 18.1 Common mappings reused

Reuse mappings/helpers for:

```text
embedding
lm head
attention wq_a/q_norm/wq_b
attention wkv/kv_norm
attention wo_a/wo_b
attention sink
attn_norm / ffn_norm
routed expert stack/split
shared experts
MoE gate weight
mHC attn/ffn parameters
```

### 18.2 V4.1-specific mappings

Compressor parameters exist only on `kv_source_layers`.

Indexer source mappings:

```text
FULL source:
  indexer.wq_b
  indexer.weights_proj
  indexer.wk
  indexer.k_norm

REINDEX source:
  indexer.wq_b
  indexer.weights_proj
```

MoE:

```text
gate.bias     -> text correction bias
gate.bias_vl  -> VL correction bias
```

Vision:

```text
vision.*
aligner.*
image_start
image_end
image_newline
```

V4's main `hc_head` mapping is not present in the V4.1 backbone adapter.

The adapter must validate unexpected/missing source-owned parameters against `CSA2LayerPolicy`; do not infer ownership from `compress_ratio` alone.

## 19. Model registry and configuration

`deepseek_v41/__init__.py` should follow the V4 registry pattern while keeping V4.1 config construction independent.

Suggested flavors:

```text
debugmodel
v41_flash
v41_flash_small_experts     # CI/smoke convenience, not a released model
```

The released flavor constructs 40 `DeepSeekV41TransformerBlock.Config` objects with precomputed CSA2 policies.

Do not pass `num_mtp_layers`; DSpark is out of scope. The model's `mtp_layers` is empty.

The model spec uses:

```text
name = "deepseek_v41"
parallelize_fn = parallelize_deepseek_v41
state_dict_adapter = DeepSeekV41StateDictAdapter
```

The eager `parallelize_deepseek_v41` should delegate as much sequencing as possible to V4/DeepSeek-V3 parallelization after V4.1 sharding configs are attached.

## 20. Optimizer and Muon policy

V4.1 should not copy the full `_dsv4_muon_profile` and edit FQNs by hand.

Factor from V4 a helper that returns common per-layer Muon layouts for:

- attention Q/output projections;
- shared experts;
- routed experts;
- router gate;
- mHC projection parameters.

V4 then appends its compressor/indexer/MTP-specific entries. V4.1 appends:

- source compressor parameters;
- Full/Reindex indexer parameters;
- vision/aligner policy if Muon is intended for those matrices.

The exact optimizer assignment for vision should be a recipe configuration, not hard-coded in the model.

AdamW/native optimizer remains a valid baseline correctness path.

## 21. File-level implementation plan

### 21.1 Existing V4 files modified

| File | Change | V4 behavior change? |
|---|---|---|
| `deepseek_v4/mhc.py` | factor `compute_mixes` / `apply_pre` | no |
| `deepseek_v4/compressor.py` | expose pre-RoPE pooling and semantic pooling config | no |
| `deepseek_v4/metadata.py` | generic positive-ratio layout helper | no |
| `deepseek_v4/token_dispatcher.py` | ratio-1 identity-plan seam; preserve ratio 4/128 output | no |
| `deepseek_v4/attention.py` | projection helpers, state-aware internal forward, external Top-K support | no |
| `deepseek_v4/mtp.py` | optional main `hc_head` seam; V4 still supplies it | no |
| `deepseek_v4/sharding.py` | split common vs V4 long-range sharding helpers | no |
| `deepseek_v4/state_dict_adapter.py` | mapping registration hooks | no |
| `deepseek_v4/config_registry.py` | optional common Muon-layout helper | no |

Every V4 modification must have a regression test proving the default V4 model/module contract is unchanged.

### 21.2 New V4.1 files

| File | Responsibility |
|---|---|
| `deepseek_v41/__init__.py` | config builders, registry, model spec |
| `deepseek_v41/model.py` | V4-derived model/block, Single-Pass mHC orchestration, vision merge |
| `deepseek_v41/attention.py` | V4-derived attention, CSA2 context construction |
| `deepseek_v41/csa2.py` | `CSA2LayerPolicy`, `SharedAttentionState`, candidate helpers |
| `deepseek_v41/indexer.py` | V4.1 Full/Reindex indexer |
| `deepseek_v41/moe.py` | VL-aware router and balancing hook |
| `deepseek_v41/metadata.py` | V4.1 ratio/policy validation, V41-specific metadata extension if needed |
| `deepseek_v41/vision.py` | ViT + aligner |
| `deepseek_v41/sharding.py` | V4 common sharding reuse + V41 state/indexer/vision layouts |
| `deepseek_v41/parallelize.py` | eager integration; GraphTrainer later |
| `deepseek_v41/state_dict_adapter.py` | released checkpoint mapping |
| `deepseek_v41/config_registry.py` | trainer recipes, optimizer policy |
| `deepseek_v41/reference.py` | pure/reference CSA2 path for golden testing |

## 22. Testing strategy

### 22.1 Mandatory V4 regression tests

Before V4.1 tests, all existing V4 tests must pass after the extension-seam refactors.

Add focused regression tests for:

1. `HcPre.forward()` equals `compute_mixes + apply_pre`;
2. V4 compressor ratio 4/128 outputs and gradients remain equivalent;
3. V4 metadata plans for 4/128 remain identical;
4. V4 attention output/gradients remain equivalent with refactored projection helpers;
5. V4 state-dict key set is unchanged;
6. V4 sharding configs are unchanged for released flavors.

### 22.2 V4.1 unit tests

Small deterministic shapes, BF16/FP32 reference where possible:

- ratio-1 compressor identity projection;
- ratio-2 softmax pooling;
- pre-RoPE latent -> index K;
- Full publishes all shared-state fields;
- Reuse does not own/call compressor/indexer;
- Reindex updates Top-K without changing shared KV/index K;
- layer-20 candidate generation;
- candidate-filtered reindex;
- ratio-1 causal visibility;
- packed documents never attend across boundaries;
- Single-Pass mHC sequencing, including final pre-mix collapse;
- VL router uses `bias_vl` for image tokens but routing weights remain based on unbiased scores;
- vision 2D RoPE / downsample / aligner shapes;
- text-only path creates no vision compute.

### 22.3 Golden parity

Create a tiny V4.1 configuration matching the official reference semantics and compare TorchTitan-NPU against the released inference implementation with quantization disabled/dequantized weights:

- compressor latent;
- index K/Q/scores;
- candidate mask;
- Top-K indices;
- attention output;
- one block output + propagated pre-mix;
- 40-layer text-only tiny-model logits;
- vision encoder/aligner output;
- multimodal merged logits.

Where the released inference path uses mutable caches, build an offline prefill-only reference harness so comparison is pure and deterministic.

### 22.4 Gradient tests

Verify:

- later Reuse attention gradients reach the Full source compressor via shared `compressed_kv`;
- Single-Pass mHC gradients cross layer boundaries through `pre_mix`;
- vision loss reaches ViT/aligner parameters;
- VL routing bias remains non-gradient correction state if implemented as load-balancing state;
- indexer objective tests are added only when the objective is defined.

### 22.5 Distributed tests

Incremental matrix:

```text
1 NPU:           reference / smoke
FSDP:            backbone state-dict and one-step parity
TP:              attention + indexer score reduction parity
EP:              384-expert small-flavor dispatch parity
TP + EP:         routing/sharding smoke
CP correctness:  small sequence only initially
```

Long-context production CP gets a separate acceptance gate after distributed Top-K and selected-row KV fetch are implemented.

## 23. Rollout / PR decomposition

### PR 1: V4 extension seams only

No V4.1 model yet.

- mHC helper split;
- compressor pre-RoPE/pooling refactor;
- attention projection/long-range helper split;
- external sparse-indices support;
- adapter/sharding helper split;
- generic metadata positive-ratio helper.

Acceptance: all V4 tests pass and V4 released state dict is unchanged.

### PR 2: V4.1 text-only reference backbone

- `deepseek_v41` package;
- 40-layer policy;
- ratio 1/2 compressor support;
- V4.1 indexer;
- SharedAttentionState;
- Single-Pass mHC;
- BF16 tiny golden tests;
- CP=1 only.

Acceptance: forward/backward + checkpoint structural parity.

### PR 3: V4.1 state dict and NPU eager training

- released-checkpoint adapter;
- V41 sharding;
- FSDP / TP / EP;
- NPU kernels or reference fallback where kernels are not ready;
- optimizer/Muon parameter policy.

Acceptance: one-step training smoke and checkpoint round trip.

### PR 4: Multimodal

- ViT/aligner;
- tensorized vision batch;
- embedding merge;
- VL router bias and balancing state;
- distributed sharding tests.

### PR 5: scalable CSA2 Context Parallel

- distributed index scoring/top-K;
- candidate distributed merge;
- selected-row KV fetch + backward;
- long-context memory/communication benchmarks.

### Later RFC/PRs

- DSpark;
- Engram;
- GraphTrainer;
- Pipeline Parallel.

## 24. Acceptance criteria

The feature may be described as **DeepSeek-V4.1 backbone training supported** when:

1. V4 regression suite passes with unchanged released-model parameter/state-dict contract;
2. `deepseek_v41` contains no copied V4 attention/block forward whose shared portion could have been inherited through the seams above;
3. text-only V4.1 tiny-model forward matches the official reference at agreed tolerance;
4. backward works through Single-Pass mHC and cross-layer shared compressed KV;
5. released V4.1 backbone checkpoint can be loaded with expected DSpark/Engram exclusions explicitly reported;
6. FSDP, TP and EP eager training smoke tests pass;
7. multimodal support, when enabled, matches the released vision/merge forward and VL routing semantics;
8. unsupported features are rejected by configuration rather than silently approximated;
9. CP support level is accurately labeled as correctness-only or scalable-production according to the implemented dispatcher path;
10. indexer pretraining is not claimed recipe-complete until its training objective is grounded.

## 25. Rejected alternatives

### 25.1 Copy `deepseek_v4` into `deepseek_v41` and edit

Rejected. It creates two copies of Q/SWA/output projection, mHC math, sharding, CP dispatch and state-dict expert handling. Future bug fixes would drift immediately.

### 25.2 Make V4.1 a V4 flavor only

Rejected. A flavor is appropriate for shape/config differences, not for cross-layer shared attention state and different mHC timing.

### 25.3 Put all V4/V4.1 branching inside V4 classes

Rejected. `if v41` flags inside every forward make V4 harder to reason about and compile. V4 owns stable extension seams; V4.1 owns the changed policy.

### 25.4 Physical Encoder/Decoder modules for CED

Rejected. CED in this model is represented by layer-20's ratio-1 source KV and downstream reuse/reindex schedule. A module split adds complexity without matching the released forward more closely.

### 25.5 Copy the inference `SharedAttentionRuntime`

Rejected for training. Hidden mutable cache state is incompatible with clean autograd, activation checkpointing and future graph/pipeline integration.

### 25.6 Treat V4.1 ratio 1 as V4's ratio-1 sentinel

Rejected. In V4.1 ratio 1 is a real one-token-per-global-KV projection. SWA-only is ratio 0.

## 26. Open questions

1. **Indexer training objective**: authoritative V4.1 training supervision/loss is not present in the public inference repository. Architecture provides a plug-in seam; pretraining recipe parity remains open.
2. **VL correction-bias update**: separate image bias semantics are public, but exact training update hyperparameters should be configurable until grounded.
3. **Quantized training**: first correctness implementation should use BF16/standard mixed precision; FP8/FP4 training policy belongs to a separate numerical/performance workstream.
4. **CSA2 CP kernel boundary**: decide whether selected-row fetch belongs as extensions to `CPTokenDispatcher` or a sibling `CPSparseKVDispatcher` after prototype complexity is measured.
5. **Vision TP**: start with FSDP/reduced parallelism if necessary; add TP only if model-scale profiling justifies the extra sharding complexity.
6. **Pipeline Parallel**: explicit SharedAttentionState allows PP in principle, but stage boundaries that split source/consumer groups need a dedicated state-transfer design.

## 27. Final recommendation

Implement V4.1 as a **thin specialization over a slightly more extensible V4**, with the following ownership boundary:

```text
V4 owns stable reusable mechanics:
  mHC math
  Q/SWA/output projections
  compressor mechanics
  sparse attention core
  packed metadata primitives
  CP token exchange primitives
  MoE experts
  TP/EP/FSDP conventions
  common checkpoint conversion helpers

V4.1 owns changed semantics:
  Single-Pass mHC sequencing
  CSA2 source/reindex/reuse policy
  explicit SharedAttentionState
  V4.1 indexer + candidates
  CED source schedule
  VL routing bias
  vision encoder/aligner
  V4.1 checkpoint ownership rules
```

This boundary minimizes duplicated code without turning the V4 implementation into a version-switching framework. It also keeps later DSpark and Engram work independent: those features can be added to `deepseek_v41` without changing the CSA2/mHC inheritance model defined here.

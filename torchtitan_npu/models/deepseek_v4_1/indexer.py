# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Lightning indexer for DeepSeek V4.1 (CSA2).

Shape legend for this file:
    B = batch, L = sequence length, D = model dimension,
    Hi = ``num_index_heads``, Di = ``index_head_dim``,
    R = ``compress_ratio``, N = L // R compressed entries, K = ``index_topk``.

Score of query ``t`` against compressed entry ``j``::

    S_{t,h,j} = <q^I_{t,h}, k^I_j>
    I_{t,j}   = sum_h w_{t,h} * relu(S_{t,h,j})

The top-``K`` entries of ``I_{t,.}`` are what the sparse attention reads.  Entry
``j`` covers tokens ``[j * R, (j + 1) * R)``, so it is rotated at its first
token's position: the packed positions reset per document, so the ratio stride
selects exactly that token.

Every layer owns an :class:`Indexer`, but only *source* layers carry parameters:
an index-source layer produces the top-k, a layer that owns the compressed KV
also produces the index keys, and a reuse layer returns what it was handed.
``is_source`` and ``owns_k`` encode that contract and are asserted in ``forward``,
so a misconfigured layer fails loudly instead of silently recomputing or silently
reusing.

The hierarchical candidate pool is declared here too: the layer that builds the
pool scores every visible entry, and the layers inside the pool window restrict
their own selection to it.  The pool tensor itself is threaded through the
forward, not stored on the module.

Selection is discrete, hence carries no gradient: the indexer is trained *only* by
:class:`IndexerKLLoss`, which distills each consumer layer's attention mass on those
entries -- a marginal weighted by that layer's own compressed share -- into
``softmax(I)``.

The queries and keys are rotated by the rope module with the site's un-rotated
prefix width, at the group's first token for the keys.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.nn_modules import RMSNorm
from torchtitan.models.common.rope import RoPE
from torchtitan.protocols.module import Module

from torchtitan_npu.patches.torchtitan.models.common.aux_loss import LoggedAuxLoss


def select_candidate_blocks(
    scores_BLN: torch.Tensor,
    newest_BL1: torch.Tensor,
    newest_valid_BL1: torch.Tensor,
    topk_blocks: int,
    block_size: int,
) -> torch.Tensor:
    """Level one of the hierarchical indexer: keep the best-scoring blocks.

    Args:
        scores_BLN: Index scores ``[B, L, N]``, already masked to ``-inf`` on entries the
            query cannot select (other documents and incomplete groups).
        newest_BL1: Index of each query's newest selectable entry, ``[B, L, 1]``.
        newest_valid_BL1: Whether that entry exists (the query's document has at least
            one complete group), ``[B, L, 1]``.
        topk_blocks: Maximum number of blocks to keep.
        block_size: Positions per block.

    Returns:
        Boolean mask ``[B, L, N]`` selecting every position of the kept blocks.
    """
    width = scores_BLN.size(-1)
    if width % block_size != 0:
        scores_BLN = F.pad(scores_BLN, (0, -width % block_size), value=-torch.inf)
    # A block is scored by its best position, which is what makes the pool
    # recall-oriented rather than a second token-level selection.
    block_scores_BLB = scores_BLN.unflatten(-1, (-1, block_size)).amax(dim=-1)
    num_blocks = block_scores_BLB.size(-1)

    # The block holding a query's newest selectable entry is only partly filled, and must
    # not be outscored by an older, full block.  A query whose document has no complete
    # group yet owns no such block and pins nothing.
    last_BL1 = newest_BL1 // block_size
    pin_BLB = torch.arange(num_blocks, device=scores_BLN.device).view(1, 1, -1) == last_BL1
    block_scores_BLB = block_scores_BLB.masked_fill(pin_BLB & newest_valid_BL1, torch.inf)

    top = block_scores_BLB.topk(min(topk_blocks, num_blocks), dim=-1)
    # Fewer reachable blocks than ``topk_blocks`` leaves -inf picks behind: drop them.
    keep_BLB = torch.zeros_like(block_scores_BLB, dtype=torch.bool).scatter_(-1, top.indices, top.values > -torch.inf)
    return keep_BLB.repeat_interleave(block_size, dim=-1)[..., :width]


class Indexer(Module):
    """Score compressed entries and keep the top ``index_topk`` per query.

    Key modes: a key-owning indexer owns ``wk``/``k_norm`` and consumes the
    KV-source compressor's latent; an external-key indexer owns no key
    projection and requires the shared ``idx_k``.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        num_index_heads: int
        index_head_dim: int
        index_topk: int
        compress_ratio: int
        is_source: bool
        # This layer projects the index keys from its own compressor latent.
        owns_k: bool
        # Hierarchical indexer: this layer builds the shared candidate pool, or
        # restricts its own selection to it.
        is_candidate_source: bool = False
        uses_candidates: bool = False
        candidate_topk_blocks: int = 0
        candidate_block_size: int = 0
        # Whether a distillation loss consumes the student logits at the selected
        # entries.  When no loss is attached (inference, or an LM-only run) the
        # gather-and-score recomputation is dead work and is skipped.
        needs_selection_scores: bool = False
        # Present on source layers:
        rope: RoPE.Config | None = None
        wq_b: Linear.Config | None = None
        weights_proj: Linear.Config | None = None
        # Present on key-owning layers:
        wk: Linear.Config | None = None
        k_norm: RMSNorm.Config | None = None

    def __init__(self, config: Config):
        super().__init__()
        self.num_index_heads = config.num_index_heads
        self.index_head_dim = config.index_head_dim
        self.index_topk = config.index_topk
        self.compress_ratio = config.compress_ratio
        self.softmax_scale = config.index_head_dim**-0.5
        self.is_source = config.is_source
        self.owns_k = config.owns_k
        self.is_candidate_source = config.is_candidate_source
        self.uses_candidates = config.uses_candidates
        self.candidate_topk_blocks = config.candidate_topk_blocks
        self.candidate_block_size = config.candidate_block_size
        self.needs_selection_scores = config.needs_selection_scores
        if not self.is_source:
            return
        if config.rope is None or config.wq_b is None or config.weights_proj is None:
            raise ValueError("An index-source layer requires rope, wq_b and weights_proj configs.")
        wk, k_norm = config.wk, config.k_norm
        if self.owns_k and (wk is None or k_norm is None):
            raise ValueError("A key-owning indexer requires wk and k_norm configs.")
        self.rope = config.rope.build()
        self.wq_b = config.wq_b.build()
        self.weights_proj = config.weights_proj.build()
        if wk is not None and k_norm is not None:
            self.wk = wk.build()
            self.k_norm = k_norm.build()

    def _project_qkw(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        positions: torch.Tensor,
        *,
        latent: torch.Tensor | None,
        idx_k: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """First half of the indexer: the projections that produce q, k and w.

        A key-owning layer projects its keys from the compressor latent; a re-indexing
        layer scores the shared ``idx_k`` instead.

        Returns:
            ``(idx_q, idx_k, idx_w)``.
        """
        bsz, seqlen, _ = qr.size()
        idx_q = self.wq_b(qr).view(bsz, seqlen, self.num_index_heads, self.index_head_dim)
        # The rope config carries the site's un-rotated prefix width.
        idx_q = self.rope(idx_q, positions=positions)
        if self.owns_k:
            # ``forward`` asserts both before calling: a key owner always has its own
            # weights and the compressor latent.
            assert latent is not None
            # The indexer's inputs are detached: the distillation loss must train the
            # indexer and nothing else, so its graph starts at the indexer's own
            # parameters.
            projected_k = self.k_norm(self.wk(latent.detach()))
            # Entry j is rotated at the first token of the group it stands for.  One
            # rank-2 head; the rope rotates rank-3 [B, N, 1, H].
            idx_k = self.rope(projected_k.unsqueeze(2), positions=positions[..., :: self.compress_ratio]).squeeze(2)
        else:
            # A re-indexing layer scores the shared keys ``forward`` asserted are present.
            assert idx_k is not None
        # ``weights_proj`` is scaled by the index softmax scale and the head count, as in
        # the reference: the per-head scores are averaged rather than summed.
        idx_w = self.weights_proj(x) * (self.softmax_scale * self.num_index_heads**-0.5)
        return idx_q, idx_k, idx_w

    def _selection_mask(
        self,
        doc_ids_BL: torch.Tensor,
        num_cmp: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Document isolation plus causal completeness over the entry axis.

        Returns ``(visible_BLN, newest_BL1, newest_valid_BL1)``.  Entry ``j`` belongs to
        ``doc_ids_BL[:, j * compress_ratio]`` and is complete for query ``t`` iff
        ``j < (t + 1) // compress_ratio``; the two conditions are exactly the
        ``selected_attention`` window rule one axis over.
        """
        num_tokens = doc_ids_BL.size(1)
        ratio = self.compress_ratio
        entry_BL1 = torch.arange(num_tokens, device=device).view(1, -1, 1)
        # Global number of complete groups up to and including each query.
        complete_BL1 = (entry_BL1 + 1) // ratio
        entry_BN1 = torch.arange(num_cmp, device=device).view(1, 1, -1)
        cmp_doc_ids_BN = doc_ids_BL[:, ::ratio]
        visible_BLN = (entry_BN1 < complete_BL1) & (cmp_doc_ids_BN.unsqueeze(1) == doc_ids_BL.unsqueeze(-1))

        newest_BL1 = complete_BL1 - 1
        in_range_BL1 = newest_BL1 >= 0
        newest_doc_BL = cmp_doc_ids_BN.gather(1, newest_BL1.clamp_min(0).squeeze(-1))
        newest_valid_BL1 = in_range_BL1 & (newest_doc_BL.unsqueeze(-1) == doc_ids_BL.unsqueeze(-1))
        return visible_BLN, newest_BL1, newest_valid_BL1

    def _selected_scores(
        self,
        idx_q: torch.Tensor,
        idx_k: torch.Tensor,
        idx_w: torch.Tensor,
        topk_indices_BLK: torch.Tensor,
    ) -> torch.Tensor:
        """Recompute the index scores at the selected entries, with gradient.

        The gradient this produces is what trains the indexer: it flows into ``wq_b``,
        ``weights_proj``, and into the shared index keys' owner through ``idx_k``.
        """
        # Invalid (-1) slots gather entry 0; the loss masks them out.
        batch_index = torch.arange(idx_k.size(0), device=idx_k.device)[:, None, None]
        selected_BLKD = idx_k[batch_index, topk_indices_BLK.clamp_min(0)]
        logits_BLHK = torch.einsum("blhd,blkd->blhk", idx_q, selected_BLKD)
        logits_BLHK = logits_BLHK.relu() * idx_w.unsqueeze(-1)
        return logits_BLHK.sum(dim=2)

    def _select_topk(
        self,
        idx_q: torch.Tensor,
        idx_k: torch.Tensor,
        idx_w: torch.Tensor,
        attention_masks,
        *,
        candidates: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Second half: score the visible entries, keep the top ``index_topk``.

        This is the half a fused kernel replaces: the relaxed score
        ``relu(q @ k^T) * w`` summed over heads, the visibility mask, the optional
        candidate-pool restriction, the top-k and the candidate pool itself.

        Returns ``(topk_indices, topk_scores, candidates)``.  ``topk_scores`` carries
        the student logits at the selected entries; it is produced with the indexer
        distillation loss and stays ``None`` until then.
        """
        visible_BLN, newest_BL1, newest_valid_BL1 = self._selection_mask(
            attention_masks.doc_ids_BL, idx_k.shape[1], idx_q.device
        )
        topk = min(self.index_topk, idx_k.shape[1])
        index_score = torch.einsum("bshd,btd->bsht", idx_q, idx_k)
        index_score = index_score.relu_() * idx_w.unsqueeze(-1)
        index_score = index_score.sum(dim=2)
        index_score = index_score.where(visible_BLN, float("-inf"))
        if self.uses_candidates:
            if candidates is None:
                raise ValueError("a layer inside the candidate window requires the shared pool")
            candidate_mask = candidates.squeeze(1) if candidates.ndim == 4 else candidates
            if candidate_mask.shape != index_score.shape:
                raise ValueError(
                    "candidate mask shape must match index score shape: "
                    f"{tuple(candidate_mask.shape)} vs {tuple(index_score.shape)}"
                )
            index_score = index_score.where(candidate_mask, float("-inf"))

        indices = index_score.topk(topk, dim=-1, sorted=False).indices.sort(dim=-1).values
        valid_BLK = visible_BLN.gather(-1, indices)
        topk_indices = torch.where(valid_BLK, indices, -1)

        if self.is_candidate_source:
            candidates = select_candidate_blocks(
                index_score,
                newest_BL1,
                newest_valid_BL1,
                self.candidate_topk_blocks,
                self.candidate_block_size,
            )

        # The student logits only exist to be distilled.  Producing them when no loss
        # consumes them (inference, or a run with the coefficient set to ``None``) would
        # add a gather plus an einsum over the selected entries for nothing.
        topk_scores = None
        if self.needs_selection_scores and self.training and torch.is_grad_enabled():
            topk_scores = self._selected_scores(idx_q, idx_k, idx_w, topk_indices)
        return topk_indices, topk_scores, candidates

    def forward(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        positions: torch.Tensor,
        attention_masks,
        *,
        latent: torch.Tensor | None = None,
        idx_k: torch.Tensor | None = None,
        topk_indices: torch.Tensor | None = None,
        topk_scores: torch.Tensor | None = None,
        candidates: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        """Args:
            x: Hidden states of shape ``[B, L, D]``.
            qr: Query LoRA latent of shape ``[B, L, Q]``.
            positions: Position ids of shape ``[B, L]``.
            attention_masks: The forward's varlen metadata; its document ids own the
                entry isolation and causal completeness the selection is masked with.
            latent: The compressor's pre-RoPE latent; a key-owning layer projects its
                keys from it.
            idx_k: The shared index keys when this layer does not own them.
            topk_indices: The shared top-k when this layer does not produce it.
            topk_scores: The shared student logits at those entries.
            candidates: The shared candidate pool mask.

        Returns:
            ``(idx_k, topk_indices, topk_scores, candidates)``.  Reuse layers pass their
            inputs through: they have no parameters of their own to train, but they must
            not drop the state their group depends on.
        """
        # A source supersedes whatever was in flight, so these are consumption
        # contracts: an index source that does not own the keys must be handed the
        # shared ones, and a reusing layer must be handed a selection to reuse.
        if self.owns_k:
            assert latent is not None, "A key-owning indexer projects its keys from the compressor latent."
        elif self.is_source:
            assert idx_k is not None, (
                "A re-indexing layer scores the shared index keys, which no preceding key-owning layer produced."
            )
        if self.compress_ratio > 0 and not self.is_source:
            assert topk_indices is not None, (
                "A layer that reuses the top-k must receive it: no index source "
                f"precedes this one (compress_ratio={self.compress_ratio})."
            )
        if not self.is_source:
            return idx_k, topk_indices, topk_scores, candidates

        idx_q, idx_k, idx_w = self._project_qkw(
            x.detach(),
            qr.detach(),
            positions,
            latent=latent,
            idx_k=idx_k,
        )
        topk_indices, topk_scores, candidates = self._select_topk(
            idx_q,
            idx_k,
            idx_w,
            attention_masks,
            candidates=candidates,
        )
        return idx_k, topk_indices, topk_scores, candidates


class IndexerKLLoss(LoggedAuxLoss):
    """Distill the attention's distribution over the top-k entries into the indexer.

    The teacher is the head-averaged attention mass on the selected entries, with the
    full softmax denominator (sliding window, selected compressed entries and sink
    alike).  That mass is a *marginal* ``p`` whose row sum ``Z <= 1``: the window and the
    sink hold the rest of the probability.  The loss scores the conditional teacher
    ``t = p / Z`` against the student ``softmax(I)`` and weights the row by ``Z``::

        L = sum_j p_j (log t_j - log Y_j),   dI = Z * Y - p

    Pre-normalising ``p`` to ``t`` would set ``Z = 1`` and quietly change the objective,
    so ``_teacher`` returns ``p`` unchanged.

    Like ``MicrobatchWiseLoadBalanceLoss``, the loss owns the whole computation: the
    attention hands over the tensors the teacher needs (queries, compressed keys, the
    selected entries and the operator's LSE) plus the student logits, and the forward
    builds the teacher, forms the weighted KL and injects the gradient on the carrier.

    One instance is attached per layer that consumes the selection, and they all score
    the *same* student logits, because the indexer's ``topk_scores`` tensor is shared
    across the group.  Each layer's backward therefore flows into that shared tensor, so
    the indexer accumulates the gradient of every consumer; ``dI`` being affine in the
    teacher is what makes this per-layer sum equal to the single pooled-teacher loss.

    The fused NPU counterpart is ``sparse_lightning_indexer_kl_loss_grad`` in
    ops-transformer (see ``deepseek_v41_indexer_distill_findings.md``): it reads the
    teacher verbatim and derives ``p_reduce = Z`` as its row sum, so the tensor handed
    over here (``p``, not ``p / Z``) is what that kernel expects, and ``q``/``k``/``w``
    must be exactly the post-RoPE projections the forward used.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(LoggedAuxLoss.Config):
        """The ``LoggedAuxLoss`` fields plus the teacher's temperature."""

        softmax_scale: float
        """Attention softmax scale; the teacher recomputes its logits with the same
        temperature as the sparse attention that produced the LSE."""

    def __init__(self, config: Config):
        super().__init__(config)
        self.softmax_scale = config.softmax_scale

    def _teacher(
        self,
        q_BLHD: torch.Tensor,
        cmp_k_BND: torch.Tensor,
        topk_indices_BLK: torch.Tensor,
        lse_BHL: torch.Tensor,
    ) -> torch.Tensor:
        """Raw head-averaged attention mass on the selected entries, ``[B, L, K]``.

        ``lse`` is the operator's per-head log-sum-exp over window + selected compressed
        + sink, so ``exp(logit - lse)`` is each head's probability on that entry with the
        full denominator.  Averaging over heads gives the *marginal* mass ``p``, whose
        row sum ``Z <= 1`` is the compressed slice's share of the full softmax.

        ``p`` is deliberately returned unnormalised: the loss weights each row's KL by
        ``Z``, so pre-normalising to ``p / Z`` would set ``Z = 1`` and change the
        objective.
        """
        valid_BLK = topk_indices_BLK >= 0
        row_valid_BL = valid_BLK.any(dim=-1)
        batch_index = torch.arange(cmp_k_BND.size(0), device=cmp_k_BND.device)[:, None, None]
        selected_BLKD = cmp_k_BND[batch_index, topk_indices_BLK.clamp_min(0)]
        logits_BLHK = torch.einsum("blhd,blkd->blhk", q_BLHD, selected_BLKD).float() * self.softmax_scale
        logits_BLHK = logits_BLHK.masked_fill(~valid_BLK.unsqueeze(2), -torch.inf)

        comp_lse_BLH = torch.logsumexp(logits_BLHK, dim=-1)
        # The operator returns the LSE as [B, H, L]; the compressed logits are [B, L, H].
        mass_BLH = torch.exp(comp_lse_BLH - lse_BHL.transpose(1, 2).float())
        # A row with no valid slot has an all -inf softmax row; its mass is zero anyway.
        conditional_BLHK = torch.softmax(logits_BLHK.masked_fill(~row_valid_BL[:, :, None, None], 0.0), dim=-1)
        return (mass_BLH.unsqueeze(-1) * conditional_BLHK).sum(dim=2) / q_BLHD.size(2)

    @staticmethod
    def _kl(
        p_BLK: torch.Tensor,
        t_BLK: torch.Tensor,
        topk_scores_BLK: torch.Tensor,
        topk_indices_BLK: torch.Tensor,
    ) -> torch.Tensor:
        """Marginal-weighted KL from the student to the teacher, summed over rows.

        Row ``(b, l)`` contributes ``sum_j p_j (log t_j - log Y_j)``: that is
        ``Z * KL(t || Y)``, whose gradient w.r.t. the student logits is ``Z * Y - p``.
        Both ``p`` and ``t`` are detached constants.
        """
        valid_BLK = topk_indices_BLK >= 0
        row_valid_BL = valid_BLK.any(dim=-1)
        logits_BLK = topk_scores_BLK.float().masked_fill(~valid_BLK, -torch.inf)
        # A row with no valid slot would produce NaN in log_softmax; it is zeroed below.
        logits_BLK = logits_BLK.masked_fill(~row_valid_BL.unsqueeze(-1), 0.0)
        log_student_BLK = F.log_softmax(logits_BLK, dim=-1)

        # xlogy keeps 0 * log(0) = 0: an entry with no teacher mass contributes nothing
        # even though its conditional is 0, and the invalid slots are dropped below.
        weighted_BLK = torch.special.xlogy(p_BLK, t_BLK) - p_BLK * log_student_BLK
        weighted_BLK = weighted_BLK.masked_fill(~valid_BLK, 0.0)
        return weighted_BLK.sum(dim=-1).masked_fill(~row_valid_BL, 0.0).sum()

    def forward(
        self,
        q_BLHD: torch.Tensor,
        cmp_k_BND: torch.Tensor,
        topk_indices_BLK: torch.Tensor,
        lse_BHL: torch.Tensor,
        topk_scores_BLK: torch.Tensor,
        *,
        carrier: torch.Tensor,
    ) -> torch.Tensor:
        """Build the teacher, score the student against it, inject the gradient.

        Args:
            q_BLHD: Attention queries ``[B, L, H, Dk]``.
            cmp_k_BND: Shared compressed KV ``[B, N, Dk]``.
            topk_indices_BLK: Selected compressed entries ``[B, L, K]``; ``-1`` unused.
            lse_BHL: Per-head log-sum-exp of the sparse softmax, ``[B, H, L]``.
            topk_scores_BLK: Student logits at the selected entries, ``[B, L, K]``.
            carrier: Tensor whose backward path carries the injected gradient (the
                attention output).

        Returns:
            ``carrier`` unchanged.
        """
        with torch.no_grad():
            p_BLK = self._teacher(q_BLHD, cmp_k_BND, topk_indices_BLK, lse_BHL)
            # The conditional teacher; a row whose mass underflowed keeps t = 0 and its
            # p = 0 makes the contribution vanish.
            eps = torch.finfo(torch.float32).tiny
            t_BLK = p_BLK / p_BLK.sum(dim=-1, keepdim=True).clamp_min(eps)
        return self.inject(carrier, self._kl(p_BLK, t_BLK, topk_scores_BLK, topk_indices_BLK))

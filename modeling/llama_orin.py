"""MCore/MindSpeed port of ``modeling_llama_orin_legacylegacy.py``."""

from __future__ import annotations

import math
import os
from typing import Any

import torch
from torch.distributed.nn import functional as dist_nn
from torch import Tensor

from megatron.core import parallel_state, tensor_parallel
from megatron.core.transformer.transformer_config import TransformerConfig

from modeling.llama_mcore_common import MCoreLlamaLMBase


class LlamaOrinModel(MCoreLlamaLMBase):
    """Legacy Orin recurrent logits-to-embedding interpolation path."""

    def __init__(
        self,
        config: TransformerConfig,
        model_config: Any,
        vocab_size: int,
        max_sequence_length: int,
        pre_process: bool = True,
        post_process: bool = True,
        parallel_output: bool = True,
        use_transformer_engine_spec: bool = False,
    ) -> None:
        needs_accelerated_spec = bool(getattr(config, "sequence_parallel", False)) or (
            int(getattr(config, "context_parallel_size", 1) or 1) > 1
        )
        if needs_accelerated_spec and not use_transformer_engine_spec:
            raise ValueError("LlamaOrinModel requires the TE/MindSpeed layer spec for SP/CP paths.")
        super().__init__(
            config=config,
            model_config=model_config,
            vocab_size=vocab_size,
            max_sequence_length=max_sequence_length,
            pre_process=pre_process,
            post_process=post_process,
            parallel_output=parallel_output,
            use_transformer_engine_spec=use_transformer_engine_spec,
        )

    def _num_interpolation_steps(self) -> int:
        return int(getattr(self.model_config, "more_iterations", 0) or 0)

    def _validate_legacy_recurrent_options(self) -> None:
        if bool(getattr(self.model_config, "output_hidden_states", False)) and getattr(
            self.model_config,
            "hidden_layer_num",
            None,
        ):
            raise NotImplementedError(
                "The legacy output_hidden_states/hidden_layer_num interpolation branch is not ported to "
                "MCore because TransformerBlock does not return all intermediate layer states."
            )

    def _output_logits_partition_sbh(self, hidden_states_sbh: Tensor) -> Tensor:
        logits, _ = self.backbone.output_layer(
            hidden_states_sbh,
            weight=None,
            runtime_gather_output=False,
        )
        return logits

    def _all_reduce_autograd(
        self,
        tensor: Tensor,
        op: torch.distributed.ReduceOp.RedOpType = torch.distributed.ReduceOp.SUM,
    ) -> Tensor:
        if parallel_state.get_tensor_model_parallel_world_size() == 1:
            return tensor
        return dist_nn.all_reduce(
            tensor,
            op=op,
            group=parallel_state.get_tensor_model_parallel_group(),
        )

    def _all_reduce_no_grad(
        self,
        tensor: Tensor,
        op: torch.distributed.ReduceOp.RedOpType,
    ) -> Tensor:
        if parallel_state.get_tensor_model_parallel_world_size() == 1:
            return tensor
        reduced = tensor.detach().clone()
        torch.distributed.all_reduce(
            reduced,
            op=op,
            group=parallel_state.get_tensor_model_parallel_group(),
        )
        return reduced

    def _topk_weighted_embedding_sbh(
        self,
        logits: Tensor,
        embedding_weight: Tensor,
        temperature: float,
        topk: int,
    ) -> Tensor:
        if parallel_state.get_tensor_model_parallel_world_size() != 1:
            raise RuntimeError("Top-k interpolation is not compatible with vocab-parallel TP; use full softmax.")
        actual_topk = min(int(topk), logits.shape[-1])
        if actual_topk <= 0:
            return torch.zeros(
                (*logits.shape[:-1], embedding_weight.shape[-1]),
                dtype=logits.dtype,
                device=logits.device,
            )
        topk_values, topk_indices = torch.topk(logits.float() / float(temperature), k=actual_topk, dim=-1)
        probs = torch.softmax(topk_values, dim=-1).to(embedding_weight.dtype)
        selected = embedding_weight[topk_indices]
        return torch.einsum("sbk,sbkh->sbh", probs, selected).to(logits.dtype)

    def _vocab_parallel_weighted_embedding_sbh(
        self,
        hidden_states_sbh: Tensor,
        *,
        temperature: float,
        use_topk: bool,
        topk: int,
    ) -> Tensor:
        if temperature <= 0.0:
            raise ValueError(f"softmax_temperature must be positive, got {temperature}.")
        logits = self._output_logits_partition_sbh(hidden_states_sbh)
        embedding_weight = self.backbone.embedding.word_embeddings.weight
        if use_topk:
            return self._topk_weighted_embedding_sbh(logits, embedding_weight, temperature, topk)

        logits = logits.float() / float(temperature)
        local_max = logits.max(dim=-1, keepdim=True).values
        global_max = self._all_reduce_no_grad(local_max, torch.distributed.ReduceOp.MAX)
        exp_logits = torch.exp(logits - global_max)
        local_denom = exp_logits.sum(dim=-1, keepdim=True)
        denom = self._all_reduce_autograd(local_denom).clamp_min(torch.finfo(exp_logits.dtype).tiny)
        probs = (exp_logits / denom).to(embedding_weight.dtype)
        local_embeds = torch.matmul(probs, embedding_weight)
        embeds = tensor_parallel.reduce_from_tensor_model_parallel_region(local_embeds)
        return embeds.to(hidden_states_sbh.dtype)

    def _compute_orin_lm_loss_sum(
        self,
        hidden_states_bsh: Tensor,
        labels: Tensor,
        loss_mask: Tensor,
    ) -> Tensor:
        logits = self._output_logits_partition_sbh(hidden_states_bsh.transpose(0, 1).contiguous())
        per_token_loss = self.compute_language_model_loss(labels, logits)
        return (per_token_loss * loss_mask.to(per_token_loss.dtype)).sum()

    def _compute_shifted_loss(self, hidden_states_bsh: Tensor, labels: Tensor) -> Tensor:
        if parallel_state.get_context_parallel_world_size() > 1:
            shifted_hidden = hidden_states_bsh.contiguous()
            shifted_labels = self._build_cp_next_token_labels(labels)
        elif self.config.sequence_parallel and parallel_state.get_tensor_model_parallel_world_size() > 1:
            shifted_hidden = hidden_states_bsh.contiguous()
            shifted_labels = torch.empty_like(labels)
            shifted_labels[:, :-1] = labels[:, 1:]
            shifted_labels[:, -1:] = -100
        else:
            shifted_hidden = hidden_states_bsh[:, :-1, :].contiguous()
            shifted_labels = labels[:, 1:].contiguous()

        loss_mask = shifted_labels.ne(-100)
        safe_labels = shifted_labels.masked_fill(~loss_mask, 0)
        token_count = self._loss_denominator(loss_mask)
        chunk_size = int(os.getenv("CHUNKED_LM_LOSS_TOKENS", "0") or "0")
        if chunk_size <= 0 or chunk_size >= shifted_hidden.shape[1]:
            return self._compute_orin_lm_loss_sum(shifted_hidden, safe_labels, loss_mask) / token_count

        loss = shifted_hidden.new_zeros((), dtype=torch.float32)
        sequence_parallel_labels = (
            self._uses_sequence_parallel()
            and safe_labels.shape[1]
            == shifted_hidden.shape[1] * parallel_state.get_tensor_model_parallel_world_size()
        )
        for start in range(0, shifted_hidden.shape[1], chunk_size):
            end = min(start + chunk_size, shifted_hidden.shape[1])
            if sequence_parallel_labels:
                chunk_labels = self._sequence_parallel_labels_for_local_chunk(
                    safe_labels,
                    local_sequence_length=shifted_hidden.shape[1],
                    start=start,
                    end=end,
                )
                chunk_loss_mask = self._sequence_parallel_labels_for_local_chunk(
                    loss_mask,
                    local_sequence_length=shifted_hidden.shape[1],
                    start=start,
                    end=end,
                )
            else:
                chunk_labels = safe_labels[:, start:end]
                chunk_loss_mask = loss_mask[:, start:end]
            loss = loss + self._compute_orin_lm_loss_sum(
                shifted_hidden[:, start:end, :],
                chunk_labels,
                chunk_loss_mask,
            )
        return loss / token_count

    def _interpolate_from_hidden(self, hidden_states_sbh: Tensor) -> Tensor:
        interpolated = self._vocab_parallel_weighted_embedding_sbh(
            hidden_states_sbh,
            temperature=float(getattr(self.model_config, "softmax_temperature", 1.0)),
            use_topk=bool(getattr(self.model_config, "interpolation_use_topk", False)),
            topk=int(getattr(self.model_config, "interpolation_topk", getattr(self.model_config, "top_k_num", 100))),
        )
        if getattr(self.model_config, "scale_embeds", False):
            interpolated = interpolated * math.sqrt(hidden_states_sbh.shape[-1])
        return interpolated

    def forward(
        self,
        tokens: Tensor,
        position_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        labels: Tensor | None = None,
        loss_mask: Tensor | None = None,
        global_step: int | None = None,
    ) -> Tensor:
        del loss_mask, global_step
        if labels is None:
            raise ValueError("LlamaOrinModel expects labels and returns a scalar loss.")

        self._validate_legacy_recurrent_options()
        token_attention_mask = attention_mask if attention_mask is not None and attention_mask.dim() == 2 else None
        use_sequence_parallel = self._uses_sequence_parallel()
        initial_embeds = self._embed_tokens_sbh(tokens, position_ids)
        initial_embeds = self._ensure_sequence_parallel_sbh(
            initial_embeds,
            full_sequence_length=labels.shape[1],
        )
        current_embeds = initial_embeds
        for _ in range(self._num_interpolation_steps()):
            hidden_states_sbh = self._decoder_forward_sbh(
                current_embeds,
                token_attention_mask,
                position_ids,
                input_is_sequence_parallel=use_sequence_parallel,
            )
            interpolated = self._interpolate_from_hidden(hidden_states_sbh)
            interpolated = self._ensure_sequence_parallel_sbh(
                interpolated,
                full_sequence_length=labels.shape[1],
            )
            if getattr(self.model_config, "residual_interpolated_embeds", False):
                current_embeds = current_embeds + interpolated
            else:
                current_embeds = initial_embeds + interpolated

        final_hidden_sbh = self._decoder_forward_sbh(
            current_embeds,
            token_attention_mask,
            position_ids,
            input_is_sequence_parallel=use_sequence_parallel,
        )
        return self._compute_shifted_loss(final_hidden_sbh.transpose(0, 1).contiguous(), labels)


def build_model(
    *,
    config: TransformerConfig,
    model_config: Any,
    vocab_size: int,
    max_sequence_length: int,
    pre_process: bool = True,
    post_process: bool = True,
    parallel_output: bool = True,
    use_transformer_engine_spec: bool = False,
) -> LlamaOrinModel:
    return LlamaOrinModel(
        config=config,
        model_config=model_config,
        vocab_size=vocab_size,
        max_sequence_length=max_sequence_length,
        pre_process=pre_process,
        post_process=post_process,
        parallel_output=parallel_output,
        use_transformer_engine_spec=use_transformer_engine_spec,
    )

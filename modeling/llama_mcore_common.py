"""Shared MCore LLaMA helpers for recurrent modeling variants."""

from __future__ import annotations

import math
import os
from typing import Any

import torch
from torch import Tensor
from torch.distributed.nn import functional as dist_nn
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from megatron.core import parallel_state, tensor_parallel
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.models.common.language_module.language_module import LanguageModule
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.transformer.enums import ModelType
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig


class MCoreLlamaLMBase(LanguageModule):
    """MCore GPT backbone plus the LLaMA-Factory-compatible LM helpers.

    The concrete recurrent variants use this class so the actual training path
    stays inside Megatron/MindSpeed tensor, sequence, and context parallel code.
    """

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
        super().__init__(config=config)
        self.config = config
        self.model_config = model_config
        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        self.pre_process = pre_process
        self.post_process = post_process
        self.parallel_output = parallel_output
        self.use_transformer_engine_spec = use_transformer_engine_spec
        self.share_embeddings_and_output_weights = False
        self.model_type = ModelType.encoder_or_decoder
        self.use_null_attention_mask = (
            os.getenv(
                "MODEL_USE_NULL_ATTENTION_MASK",
                "1" if use_transformer_engine_spec else "0",
            )
            == "1"
        )

        if use_transformer_engine_spec:
            layer_spec = get_gpt_layer_with_transformer_engine_spec(
                qk_layernorm=config.qk_layernorm,
                multi_latent_attention=config.multi_latent_attention,
                moe_use_legacy_grouped_gemm=config.moe_use_legacy_grouped_gemm,
            )
        else:
            layer_spec = get_gpt_layer_local_spec(normalization=config.normalization)

        self.backbone = GPTModel(
            config=config,
            transformer_layer_spec=layer_spec,
            vocab_size=vocab_size,
            max_sequence_length=max_sequence_length,
            pre_process=pre_process,
            post_process=post_process,
            fp16_lm_cross_entropy=False,
            parallel_output=parallel_output,
            share_embeddings_and_output_weights=False,
            position_embedding_type="rope",
            rotary_percent=1.0,
            rotary_base=getattr(model_config, "rope_theta", 10000),
            scatter_embedding_sequence_parallel=False,
        )

    def sharded_state_dict(
        self,
        prefix: str = "",
        sharded_offsets: tuple[tuple[int, int, int], ...] = (),
        metadata: dict | None = None,
    ) -> ShardedStateDict:
        return MegatronModule.sharded_state_dict(self, prefix, sharded_offsets, metadata)

    def set_input_tensor(self, input_tensor: Tensor) -> None:
        self.backbone.set_input_tensor(input_tensor)

    def _uses_sequence_parallel(self) -> bool:
        return self.config.sequence_parallel and parallel_state.get_tensor_model_parallel_world_size() > 1

    def _build_causal_mask(self, attention_mask: Tensor | None, seq_len: int, device: torch.device) -> Tensor:
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=device),
            diagonal=1,
        ).view(1, 1, seq_len, seq_len)
        if attention_mask is not None and attention_mask.dim() == 2:
            padding_mask = attention_mask[:, None, None, :].eq(0)
            causal_mask = causal_mask | padding_mask
        return causal_mask

    def _embed_tokens_sbh(self, tokens: Tensor, position_ids: Tensor | None) -> Tensor:
        if position_ids is None:
            position_ids = torch.arange(tokens.shape[1], device=tokens.device, dtype=torch.long).unsqueeze(0)
        embeddings = self.backbone.embedding(input_ids=tokens, position_ids=position_ids)
        if getattr(self.model_config, "scale_embeds", False):
            embeddings = embeddings * math.sqrt(embeddings.shape[-1])
        return embeddings

    def _embed_tokens_bsh(self, tokens: Tensor, position_ids: Tensor | None) -> Tensor:
        return self._embed_tokens_sbh(tokens, position_ids).transpose(0, 1).contiguous()

    def _ensure_sequence_parallel_sbh(self, tensor: Tensor, full_sequence_length: int) -> Tensor:
        if not self._uses_sequence_parallel():
            return tensor
        tp_size = parallel_state.get_tensor_model_parallel_world_size()
        if full_sequence_length % tp_size != 0:
            raise ValueError(
                f"Sequence length {full_sequence_length} must be divisible by TP={tp_size} for sequence parallel."
            )
        local_sequence_length = full_sequence_length // tp_size
        if tensor.shape[0] == local_sequence_length:
            return tensor
        if tensor.shape[0] != full_sequence_length:
            raise ValueError(
                "Unexpected sequence-parallel tensor length: "
                f"got {tensor.shape[0]}, expected {full_sequence_length} or {local_sequence_length}."
            )
        return tensor_parallel.scatter_to_sequence_parallel_region(tensor)

    def _gather_sequence_parallel_sbh(self, tensor: Tensor) -> Tensor:
        if not self._uses_sequence_parallel():
            return tensor
        return tensor_parallel.gather_from_sequence_parallel_region(
            tensor,
            tensor_parallel_output_grad=False,
        )

    def _scatter_sequence_parallel_sbh(self, tensor: Tensor) -> Tensor:
        if not self._uses_sequence_parallel():
            return tensor
        return tensor_parallel.scatter_to_sequence_parallel_region(tensor)

    def _slice_sequence_parallel_mask(self, attention_mask: Tensor | None) -> Tensor | None:
        if attention_mask is None or attention_mask.dim() != 2 or not self._uses_sequence_parallel():
            return attention_mask
        tp_size = parallel_state.get_tensor_model_parallel_world_size()
        tp_rank = parallel_state.get_tensor_model_parallel_rank()
        seq_len = attention_mask.shape[1]
        if seq_len % tp_size != 0:
            raise ValueError(f"Sequence length {seq_len} must be divisible by TP={tp_size} for sequence parallel.")
        local_seq_len = seq_len // tp_size
        start = tp_rank * local_seq_len
        return attention_mask[:, start : start + local_seq_len].contiguous()

    def _build_rotary_pos_emb(self, seq_len: int, position_ids: Tensor | None) -> Tensor:
        cp_size = parallel_state.get_context_parallel_world_size()
        cp_algo = getattr(self.config, "context_parallel_algo", "megatron_cp_algo")
        needs_explicit_positions = self._uses_sequence_parallel() or (
            cp_size > 1 and cp_algo == "mamba_cp_algo"
        )
        if not needs_explicit_positions:
            return self.backbone.rotary_pos_emb(seq_len)
        if position_ids is None:
            raise ValueError("Sequence/CP parallel RoPE generation requires local position_ids.")

        rotary_embedding = self.backbone.rotary_pos_emb
        inv_freq = rotary_embedding.inv_freq.to(device=position_ids.device)
        positions = position_ids[0].to(device=position_ids.device, dtype=inv_freq.dtype)
        if rotary_embedding.seq_len_interpolation_factor is not None:
            positions = positions * (1.0 / rotary_embedding.seq_len_interpolation_factor)
        freqs = torch.outer(positions, inv_freq)
        if rotary_embedding.rotary_interleaved:
            emb = torch.stack((freqs.view(-1, 1), freqs.view(-1, 1)), dim=-1).view(freqs.shape[0], -1)
        else:
            emb = torch.cat((freqs, freqs), dim=-1)
        return emb[:, None, None, :]

    def _decoder_forward_sbh(
        self,
        hidden_states_sbh: Tensor,
        attention_mask: Tensor | None,
        position_ids: Tensor | None,
        *,
        input_is_sequence_parallel: bool = False,
        attention_bias: Tensor | None = None,
    ) -> Tensor:
        decoder_input = hidden_states_sbh.contiguous()
        rotary_pos_emb = self._build_rotary_pos_emb(hidden_states_sbh.shape[0], position_ids)
        use_sequence_parallel = self._uses_sequence_parallel()
        if use_sequence_parallel and not input_is_sequence_parallel:
            decoder_input = tensor_parallel.scatter_to_sequence_parallel_region(decoder_input)
            if (
                parallel_state.get_context_parallel_world_size() > 1
                and getattr(self.config, "context_parallel_algo", "megatron_cp_algo") == "mamba_cp_algo"
            ):
                rotary_pos_emb = tensor_parallel.scatter_to_sequence_parallel_region(rotary_pos_emb)
        causal_mask = None
        if not self.use_null_attention_mask:
            causal_mask = self._build_causal_mask(
                attention_mask,
                decoder_input.shape[0],
                decoder_input.device,
            )
        decoder_kwargs = {
            "hidden_states": decoder_input,
            "attention_mask": causal_mask,
            "inference_context": None,
            "rotary_pos_emb": rotary_pos_emb,
        }
        if attention_bias is not None:
            decoder_kwargs["attention_bias"] = attention_bias
        hidden_states = self.backbone.decoder(**decoder_kwargs)
        if use_sequence_parallel and not input_is_sequence_parallel:
            hidden_states = tensor_parallel.gather_from_sequence_parallel_region(
                hidden_states,
                tensor_parallel_output_grad=False,
            )
        return hidden_states

    def _decoder_forward_bsh(
        self,
        hidden_states_bsh: Tensor,
        attention_mask: Tensor | None,
        position_ids: Tensor | None,
    ) -> Tensor:
        hidden_states_sbh = self._decoder_forward_sbh(
            hidden_states_bsh.transpose(0, 1).contiguous(),
            attention_mask,
            position_ids,
        )
        return hidden_states_sbh.transpose(0, 1).contiguous()

    def _build_cp_next_token_labels(self, labels: Tensor) -> Tensor:
        cp_rank = parallel_state.get_context_parallel_rank()
        cp_size = parallel_state.get_context_parallel_world_size()
        next_labels = torch.empty_like(labels)
        next_labels[:, :-1] = labels[:, 1:]

        first_labels = labels[:, :1].contiguous()
        gathered_first_labels = torch.empty(
            (cp_size, *first_labels.shape),
            dtype=first_labels.dtype,
            device=first_labels.device,
        )
        torch.distributed.all_gather_into_tensor(
            gathered_first_labels,
            first_labels,
            group=parallel_state.get_context_parallel_group(),
        )
        if cp_rank + 1 < cp_size:
            next_labels[:, -1:] = gathered_first_labels[cp_rank + 1]
        else:
            next_labels[:, -1:] = -100
        return next_labels

    def _compute_lm_loss_sum(
        self,
        hidden_states_bsh: Tensor,
        labels: Tensor,
        loss_mask: Tensor,
    ) -> Tensor:
        hidden_states_sbh = hidden_states_bsh.transpose(0, 1).contiguous()
        logits, _ = self.backbone.output_layer(
            hidden_states_sbh,
            weight=None,
            runtime_gather_output=False,
        )
        per_token_loss = self.compute_language_model_loss(labels, logits)
        return (per_token_loss * loss_mask.to(per_token_loss.dtype)).sum()

    def _sequence_parallel_labels_for_local_chunk(
        self,
        labels: Tensor,
        *,
        local_sequence_length: int,
        start: int,
        end: int,
    ) -> Tensor:
        tp_size = parallel_state.get_tensor_model_parallel_world_size()
        expected_sequence_length = local_sequence_length * tp_size
        if labels.shape[1] != expected_sequence_length:
            raise ValueError(
                "Unexpected labels length for sequence-parallel chunked loss: "
                f"got {labels.shape[1]}, expected {expected_sequence_length}."
            )
        chunks = [
            labels[:, rank * local_sequence_length + start : rank * local_sequence_length + end]
            for rank in range(tp_size)
        ]
        return torch.cat(chunks, dim=1).contiguous()

    def _compute_lm_loss_sum_maybe_checkpoint(
        self,
        hidden_states_bsh: Tensor,
        labels: Tensor,
        loss_mask: Tensor,
    ) -> Tensor:
        if os.getenv("CHUNKED_LM_LOSS_CHECKPOINT", "0") != "1":
            return self._compute_lm_loss_sum(hidden_states_bsh, labels, loss_mask)
        return torch_checkpoint(
            self._compute_lm_loss_sum,
            hidden_states_bsh,
            labels,
            loss_mask,
            use_reentrant=False,
        )

    def _loss_denominator(self, loss_mask: Tensor) -> Tensor:
        token_count = loss_mask.sum(dtype=torch.float32).clamp_min(1.0)
        if parallel_state.get_context_parallel_world_size() <= 1:
            return token_count
        global_token_count = token_count.detach().clone()
        torch.distributed.all_reduce(
            global_token_count,
            op=torch.distributed.ReduceOp.SUM,
            group=parallel_state.get_context_parallel_group(),
        )
        return global_token_count.clamp_min(1.0)

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
            return self._compute_lm_loss_sum_maybe_checkpoint(shifted_hidden, safe_labels, loss_mask) / token_count

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
            loss = loss + self._compute_lm_loss_sum_maybe_checkpoint(
                shifted_hidden[:, start:end, :],
                chunk_labels,
                chunk_loss_mask,
            )
        return loss / token_count

    def _vocab_weighted_embedding_sbh(
        self,
        hidden_states_sbh: Tensor,
        *,
        temperature: float = 1.0,
        use_topk: bool = False,
        topk: int = 100,
    ) -> Tensor:
        """Map hidden states through vocab-parallel logits back to embeddings."""

        logits, _ = self.backbone.output_layer(
            hidden_states_sbh,
            weight=None,
            runtime_gather_output=False,
        )
        logits = logits.float() / float(temperature)
        tp_group = parallel_state.get_tensor_model_parallel_group()
        tp_size = parallel_state.get_tensor_model_parallel_world_size()
        embedding_weight = self.backbone.embedding.word_embeddings.weight

        if use_topk:
            if tp_size != 1:
                raise RuntimeError("Top-k interpolation is not implemented for TP>1; use full TP softmax.")
            actual_topk = min(int(topk), logits.shape[-1])
            if actual_topk <= 0:
                return torch.zeros(
                    (*hidden_states_sbh.shape[:-1], embedding_weight.shape[-1]),
                    dtype=hidden_states_sbh.dtype,
                    device=hidden_states_sbh.device,
                )
            topk_values, topk_indices = torch.topk(logits, k=actual_topk, dim=-1)
            probs = torch.softmax(topk_values, dim=-1).to(embedding_weight.dtype)
            selected = embedding_weight[topk_indices]
            return torch.einsum("sbk,sbkh->sbh", probs, selected).to(hidden_states_sbh.dtype)

        local_max = logits.max(dim=-1, keepdim=True).values
        if tp_size > 1:
            local_max = local_max.detach().clone()
            torch.distributed.all_reduce(local_max, op=torch.distributed.ReduceOp.MAX, group=tp_group)
        probs = torch.exp(logits - local_max)
        denom = probs.sum(dim=-1, keepdim=True)
        if tp_size > 1:
            denom = dist_nn.all_reduce(denom, op=torch.distributed.ReduceOp.SUM, group=tp_group)
        probs = (probs / denom).to(embedding_weight.dtype)
        local_embeddings = torch.matmul(probs, embedding_weight)
        if tp_size > 1:
            local_embeddings = tensor_parallel.reduce_from_tensor_model_parallel_region(local_embeddings)
        return local_embeddings.to(hidden_states_sbh.dtype)

    def _interleave_stages_sbh(
        self,
        stages_sbh: list[Tensor],
        position_ids: Tensor | None,
        attention_mask: Tensor | None,
        stage_weights: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None, Tensor | None, Tensor]:
        if not stages_sbh:
            raise ValueError("At least one stage is required for interleaving.")
        seq_len, batch_size, hidden_size = stages_sbh[0].shape
        num_stages = len(stages_sbh)
        device = stages_sbh[0].device
        interleaved_bsh = torch.empty(
            batch_size,
            seq_len * num_stages,
            hidden_size,
            dtype=stages_sbh[0].dtype,
            device=device,
        )
        for idx, stage in enumerate(stages_sbh):
            interleaved_bsh[:, idx::num_stages, :] = stage.transpose(0, 1).contiguous()

        loop_position_ids = None
        if position_ids is not None:
            position_seq_len = position_ids.shape[1]
            if position_seq_len != seq_len:
                tp_size = parallel_state.get_tensor_model_parallel_world_size()
                if not self._uses_sequence_parallel() or position_seq_len != seq_len * tp_size:
                    raise ValueError(
                        "position_ids length must match the stage sequence length, or the full "
                        "sequence length when stages are sequence-parallel local chunks: "
                        f"got {position_seq_len}, stage length {seq_len}."
                    )
            loop_position_ids = torch.empty(
                batch_size,
                position_seq_len * num_stages,
                dtype=position_ids.dtype,
                device=position_ids.device,
            )
            for idx in range(num_stages):
                loop_position_ids[:, idx::num_stages] = position_ids

        loop_attention_mask = None
        if attention_mask is not None:
            mask_seq_len = attention_mask.shape[1]
            if mask_seq_len != seq_len:
                tp_size = parallel_state.get_tensor_model_parallel_world_size()
                if not self._uses_sequence_parallel() or mask_seq_len != seq_len * tp_size:
                    raise ValueError(
                        "attention_mask length must match the stage sequence length, or the full "
                        "sequence length when stages are sequence-parallel local chunks: "
                        f"got {mask_seq_len}, stage length {seq_len}."
                    )
            loop_attention_mask = torch.empty(
                batch_size,
                mask_seq_len * num_stages,
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            for idx in range(num_stages):
                loop_attention_mask[:, idx::num_stages] = attention_mask

        ponder_gate = interleaved_bsh.new_ones(batch_size, seq_len * num_stages)
        if stage_weights is not None:
            if stage_weights.shape != (batch_size, seq_len, num_stages):
                raise ValueError(
                    "stage_weights must have shape "
                    f"{(batch_size, seq_len, num_stages)}, got {tuple(stage_weights.shape)}."
                )
            for idx in range(num_stages):
                ponder_gate[:, idx::num_stages] = stage_weights[..., idx]
            if self._uses_sequence_parallel():
                ponder_gate = tensor_parallel.gather_from_sequence_parallel_region(
                    ponder_gate.transpose(0, 1).contiguous().unsqueeze(-1),
                    tensor_parallel_output_grad=False,
                ).squeeze(-1).transpose(0, 1).contiguous()
            if loop_attention_mask is not None:
                loop_attention_mask = loop_attention_mask * (ponder_gate > 0).to(loop_attention_mask.dtype)

        return interleaved_bsh.transpose(0, 1).contiguous(), loop_position_ids, loop_attention_mask, ponder_gate

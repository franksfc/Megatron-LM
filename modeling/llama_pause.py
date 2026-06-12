"""MCore/MindSpeed port of ``modeling_llama_pause.py``."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor

from megatron.core import parallel_state
from megatron.core.transformer.transformer_config import TransformerConfig

from modeling.llama_mcore_common import MCoreLlamaLMBase


class LlamaPauseModel(MCoreLlamaLMBase):
    """Insert pause tokens after each token and train on the last pause state."""

    def _num_pause_tokens(self) -> int:
        num_pause_tokens = int(getattr(self.model_config, "more_iterations", 0) or 0)
        if num_pause_tokens < 0:
            raise ValueError(f"more_iterations must be non-negative, got {num_pause_tokens}.")
        return num_pause_tokens

    def _normalize_position_ids(
        self,
        position_ids: Tensor | None,
        *,
        batch_size: int,
        seq_len: int,
        device: torch.device,
    ) -> Tensor:
        if position_ids is None:
            return torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0)
        if position_ids.dim() == 1:
            position_ids = position_ids.unsqueeze(0)
        if position_ids.dim() != 2:
            raise ValueError(
                "LlamaPauseModel expects 1D or 2D position_ids before pause expansion, "
                f"got shape {tuple(position_ids.shape)}."
            )
        if position_ids.shape[1] != seq_len:
            raise ValueError(
                "position_ids length must match tokens before pause expansion: "
                f"got {position_ids.shape[1]}, expected {seq_len}."
            )
        if position_ids.shape[0] not in (1, batch_size):
            raise ValueError(
                "position_ids batch dimension must be 1 or match tokens: "
                f"got {position_ids.shape[0]}, expected 1 or {batch_size}."
            )
        return position_ids.contiguous()

    def _normalize_attention_mask(
        self,
        attention_mask: Tensor | None,
        *,
        batch_size: int,
        seq_len: int,
    ) -> Tensor | None:
        if attention_mask is None:
            return None
        if attention_mask.dim() != 2:
            raise ValueError(
                "LlamaPauseModel only supports a 2D token attention_mask [batch, seq] before "
                f"pause expansion, got shape {tuple(attention_mask.shape)}."
            )
        if tuple(attention_mask.shape) != (batch_size, seq_len):
            raise ValueError(
                "attention_mask shape must match tokens before pause expansion: "
                f"got {tuple(attention_mask.shape)}, expected {(batch_size, seq_len)}."
            )
        return attention_mask.contiguous()

    def _expand_with_pause_tokens(
        self,
        tokens: Tensor,
        position_ids: Tensor | None,
        attention_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor | None, Tensor]:
        if tokens.dim() != 2:
            raise ValueError(f"LlamaPauseModel expects 2D tokens [batch, seq], got {tuple(tokens.shape)}.")

        batch_size, seq_len = tokens.shape
        position_ids = self._normalize_position_ids(
            position_ids,
            batch_size=batch_size,
            seq_len=seq_len,
            device=tokens.device,
        )
        attention_mask = self._normalize_attention_mask(
            attention_mask,
            batch_size=batch_size,
            seq_len=seq_len,
        )

        num_pause_tokens = self._num_pause_tokens()
        if num_pause_tokens <= 0:
            indices = torch.arange(seq_len, device=tokens.device)
            return tokens, position_ids, attention_mask, indices

        group = num_pause_tokens + 1
        new_seq_len = seq_len * group
        pause_token_id = int(getattr(self.model_config, "pause_token_id", 50288))
        expanded_tokens = torch.full(
            (batch_size, new_seq_len),
            pause_token_id,
            dtype=tokens.dtype,
            device=tokens.device,
        )
        original_positions = torch.arange(seq_len, device=tokens.device) * group
        expanded_tokens[:, original_positions] = tokens

        expanded_position_ids = position_ids.repeat_interleave(group, dim=1).contiguous()

        expanded_attention_mask = None
        if attention_mask is not None:
            expanded_attention_mask = attention_mask.repeat_interleave(group, dim=1).contiguous()

        selected_indices = original_positions + num_pause_tokens
        return expanded_tokens, expanded_position_ids, expanded_attention_mask, selected_indices

    def _embed_pause_tokens_sbh(self, tokens: Tensor, position_ids: Tensor) -> Tensor:
        embeddings = self.backbone.embedding(input_ids=tokens, position_ids=position_ids)
        return embeddings * math.sqrt(embeddings.shape[-1])

    def _select_last_pause_states(
        self,
        hidden_states_sbh: Tensor,
        selected_indices: Tensor,
        *,
        expanded_sequence_length: int,
    ) -> Tensor:
        if self._uses_sequence_parallel() and hidden_states_sbh.shape[0] != expanded_sequence_length:
            tp_size = parallel_state.get_tensor_model_parallel_world_size()
            tp_rank = parallel_state.get_tensor_model_parallel_rank()
            if expanded_sequence_length % tp_size != 0:
                raise ValueError(
                    f"Expanded pause sequence length {expanded_sequence_length} must be divisible by TP={tp_size}."
                )
            local_sequence_length = expanded_sequence_length // tp_size
            start = tp_rank * local_sequence_length
            end = start + local_sequence_length
            if hidden_states_sbh.shape[0] != local_sequence_length:
                raise RuntimeError(
                    "Unexpected local sequence-parallel pause hidden length: "
                    f"got {hidden_states_sbh.shape[0]}, expected {local_sequence_length}."
                )
            local_mask = (selected_indices >= start) & (selected_indices < end)
            local_selected = (selected_indices[local_mask] - start).contiguous()
            expected_local_selected = selected_indices.numel() // tp_size
            if local_selected.numel() != expected_local_selected:
                raise RuntimeError(
                    "Pause selected indices must shard evenly across sequence-parallel ranks: "
                    f"rank {tp_rank} has {local_selected.numel()}, expected {expected_local_selected}."
                )
            return hidden_states_sbh.index_select(0, local_selected)

        if hidden_states_sbh.shape[0] <= int(selected_indices[-1]):
            raise RuntimeError(
                "Decoder returned fewer sequence positions than the expanded pause sequence requires: "
                f"hidden length={hidden_states_sbh.shape[0]}, last selected index={int(selected_indices[-1])}."
            )
        return hidden_states_sbh.index_select(0, selected_indices)

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
            raise ValueError("LlamaPauseModel expects labels and returns a scalar loss.")
        num_pause_tokens = self._num_pause_tokens()
        group = num_pause_tokens + 1
        is_preexpanded = num_pause_tokens > 0 and tokens.shape[1] == labels.shape[1] * group
        if labels.shape != tokens.shape and not is_preexpanded:
            raise ValueError(
                "labels must either match tokens before pause expansion or remain in the original "
                "sequence space when tokens are pre-expanded for CP: "
                f"got labels {tuple(labels.shape)} and tokens {tuple(tokens.shape)}."
            )

        if is_preexpanded:
            expanded_tokens = tokens
            expanded_positions = self._normalize_position_ids(
                position_ids,
                batch_size=tokens.shape[0],
                seq_len=tokens.shape[1],
                device=tokens.device,
            )
            expanded_mask = self._normalize_attention_mask(
                attention_mask,
                batch_size=tokens.shape[0],
                seq_len=tokens.shape[1],
            )
            selected = torch.arange(labels.shape[1], device=tokens.device) * group + num_pause_tokens
        else:
            expanded_tokens, expanded_positions, expanded_mask, selected = self._expand_with_pause_tokens(
                tokens,
                position_ids,
                attention_mask,
            )
        use_sequence_parallel = self._uses_sequence_parallel()
        hidden_states_sbh = self._embed_pause_tokens_sbh(expanded_tokens, expanded_positions)
        hidden_states_sbh = self._ensure_sequence_parallel_sbh(
            hidden_states_sbh,
            full_sequence_length=expanded_tokens.shape[1],
        )
        hidden_states_sbh = self._decoder_forward_sbh(
            hidden_states_sbh,
            expanded_mask,
            expanded_positions,
            input_is_sequence_parallel=use_sequence_parallel,
        )
        selected_hidden_sbh = self._select_last_pause_states(
            hidden_states_sbh,
            selected,
            expanded_sequence_length=expanded_tokens.shape[1],
        )
        return self._compute_shifted_loss(selected_hidden_sbh.transpose(0, 1).contiguous(), labels)


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
) -> LlamaPauseModel:
    needs_accelerated_spec = bool(getattr(config, "sequence_parallel", False)) or (
        int(getattr(config, "context_parallel_size", 1) or 1) > 1
    )
    if needs_accelerated_spec and not use_transformer_engine_spec:
        raise ValueError("LlamaPauseModel requires the TE/MindSpeed layer spec for SP/CP paths.")
    return LlamaPauseModel(
        config=config,
        model_config=model_config,
        vocab_size=vocab_size,
        max_sequence_length=max_sequence_length,
        pre_process=pre_process,
        post_process=post_process,
        parallel_output=parallel_output,
        use_transformer_engine_spec=use_transformer_engine_spec,
    )

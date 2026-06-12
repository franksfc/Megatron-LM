"""MCore/MindSpeed port of ``modeling_llama_loop.py``."""

from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Any, Iterator

import torch
from torch import Tensor

from megatron.core.transformer.transformer_config import TransformerConfig

from modeling.llama_mcore_common import MCoreLlamaLMBase


class LlamaLoopModel(MCoreLlamaLMBase):
    """Repeat the same decoder stack ``more_iterations + 1`` times."""

    def _num_loops(self) -> int:
        more_iterations = int(getattr(self.model_config, "more_iterations", 0) or 0)
        if more_iterations < 0:
            raise ValueError(f"more_iterations must be non-negative, got {more_iterations}.")
        return more_iterations + 1

    def _embed_loop_tokens_sbh(self, tokens: Tensor, position_ids: Tensor | None) -> Tensor:
        if position_ids is None:
            position_ids = torch.arange(tokens.shape[1], device=tokens.device, dtype=torch.long).unsqueeze(0)
        embeddings = self.backbone.embedding(input_ids=tokens, position_ids=position_ids)
        return embeddings * math.sqrt(2.5 * embeddings.shape[-1])

    @contextmanager
    def _decoder_final_layernorm(self, enabled: bool) -> Iterator[None]:
        decoder = self.backbone.decoder
        final_layernorm = getattr(decoder, "final_layernorm", None)
        if enabled or final_layernorm is None:
            yield
            return

        decoder.final_layernorm = None
        try:
            yield
        finally:
            decoder.final_layernorm = final_layernorm

    def _decoder_forward_loop_sbh(
        self,
        hidden_states_sbh: Tensor,
        attention_mask: Tensor | None,
        position_ids: Tensor | None,
        *,
        input_is_sequence_parallel: bool,
        apply_final_layernorm: bool,
    ) -> Tensor:
        with self._decoder_final_layernorm(apply_final_layernorm):
            return self._decoder_forward_sbh(
                hidden_states_sbh,
                attention_mask,
                position_ids,
                input_is_sequence_parallel=input_is_sequence_parallel,
            )

    def _validate_forward_inputs(
        self,
        tokens: Tensor,
        position_ids: Tensor | None,
        attention_mask: Tensor | None,
        labels: Tensor,
    ) -> None:
        if tokens.dim() != 2:
            raise ValueError(f"tokens must have shape [batch, sequence], got {tuple(tokens.shape)}.")
        if labels.shape != tokens.shape:
            raise ValueError(f"labels shape {tuple(labels.shape)} must match tokens shape {tuple(tokens.shape)}.")
        if position_ids is not None and position_ids.shape != tokens.shape:
            raise ValueError(
                f"position_ids shape {tuple(position_ids.shape)} must match tokens shape {tuple(tokens.shape)}."
            )
        if attention_mask is not None and attention_mask.dim() == 2 and attention_mask.shape != tokens.shape:
            raise ValueError(
                f"2D attention_mask shape {tuple(attention_mask.shape)} must match tokens shape {tuple(tokens.shape)}."
            )
        if attention_mask is not None and attention_mask.dim() not in (2,):
            raise NotImplementedError(
                "LlamaLoopModel only supports the 2D token attention mask used by the migrated training path. "
                "Passing a 4D HF additive mask would require preserving the full mask through MCore attention."
            )

    def _default_position_ids(self, tokens: Tensor) -> Tensor:
        return torch.arange(tokens.shape[1], device=tokens.device, dtype=torch.long).unsqueeze(0).expand_as(tokens)

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
            raise ValueError("LlamaLoopModel expects labels and returns a scalar loss.")
        self._validate_forward_inputs(tokens, position_ids, attention_mask, labels)
        if position_ids is None:
            position_ids = self._default_position_ids(tokens)

        use_sequence_parallel = self._uses_sequence_parallel()
        token_attention_mask = attention_mask if attention_mask is not None and attention_mask.dim() == 2 else None

        hidden_states_sbh = self._embed_loop_tokens_sbh(tokens, position_ids)
        hidden_states_sbh = self._ensure_sequence_parallel_sbh(
            hidden_states_sbh,
            full_sequence_length=labels.shape[1],
        )

        num_loops = self._num_loops()
        for loop_idx in range(num_loops):
            hidden_states_sbh = self._decoder_forward_loop_sbh(
                hidden_states_sbh,
                token_attention_mask,
                position_ids,
                input_is_sequence_parallel=use_sequence_parallel,
                apply_final_layernorm=loop_idx + 1 == num_loops,
            )
        return self._compute_shifted_loss(hidden_states_sbh.transpose(0, 1).contiguous(), labels)


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
) -> LlamaLoopModel:
    needs_accelerated_spec = bool(getattr(config, "sequence_parallel", False)) or (
        int(getattr(config, "context_parallel_size", 1) or 1) > 1
    )
    if needs_accelerated_spec and not use_transformer_engine_spec:
        raise ValueError(
            "LlamaLoopModel requires transformer_engine/MindSpeed layer spec for sequence/context parallel paths."
        )
    return LlamaLoopModel(
        config=config,
        model_config=model_config,
        vocab_size=vocab_size,
        max_sequence_length=max_sequence_length,
        pre_process=pre_process,
        post_process=post_process,
        parallel_output=parallel_output,
        use_transformer_engine_spec=use_transformer_engine_spec,
    )

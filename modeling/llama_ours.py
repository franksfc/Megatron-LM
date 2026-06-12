"""MCore/MindSpeed port of ``modeling_llama_ours.py``."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from megatron.core import parallel_state
from megatron.core.transformer.transformer_config import TransformerConfig

from modeling.llama_mcore_common import MCoreLlamaLMBase


class LlamaOursModel(MCoreLlamaLMBase):
    """Staged interpolation/refinement recurrent path."""

    def _num_interpolation_steps(self) -> int:
        return int(getattr(self.model_config, "more_iterations", 0) or 0)

    def _num_refinement_steps(self, global_step: int | None) -> int:
        if bool(getattr(self.model_config, "vary_refine_steps", False)):
            low = self._num_interpolation_steps() + 1
            high = self._num_interpolation_steps() + 4
            if high <= low:
                return 0
            if self.training and global_step is None:
                return int(getattr(self.model_config, "training_refinement_steps", 5) or 0)
            device = next(self.parameters()).device
            return int(torch.randint(low, high, (1,), device=device).item())
        if self.training:
            return int(getattr(self.model_config, "training_refinement_steps", 5) or 0)
        return int(getattr(self.model_config, "eval_refinement_steps", 10) or 0)

    def _embed_tokens_sbh(self, tokens: Tensor, position_ids: Tensor | None) -> Tensor:
        embeddings = super()._embed_tokens_sbh(tokens, position_ids)
        if getattr(self.model_config, "scale_embeds", False) and not bool(
            getattr(self.model_config, "interpolation", False)
        ):
            embeddings = embeddings * math.sqrt(2.5)
        return embeddings

    def _attention_mask_2d(self, attention_mask: Tensor | None, seq_len: int) -> Tensor | None:
        if attention_mask is None:
            return None
        if attention_mask.dim() == 2:
            return attention_mask
        if attention_mask.dim() != 4:
            raise ValueError(
                f"Unsupported attention_mask dimension for LlamaOursModel: {attention_mask.dim()}; expected 2D or 4D."
            )
        if attention_mask.shape[1] == 1 and attention_mask.shape[2] == 1:
            return attention_mask[:, 0, 0, :seq_len].contiguous()
        if attention_mask.shape[1] == 1 and attention_mask.shape[2] == seq_len and attention_mask.shape[3] == seq_len:
            return torch.ones(
                attention_mask.shape[0],
                seq_len,
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
        reduced = attention_mask.squeeze(1).any(dim=1).to(attention_mask.dtype)
        if reduced.shape[1] == seq_len:
            return reduced.contiguous()
        return torch.ones(
            attention_mask.shape[0],
            seq_len,
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )

    def _stage_update_from_hidden(self, hidden_states_sbh: Tensor) -> Tensor:
        if not bool(getattr(self.model_config, "interpolation", False)):
            return hidden_states_sbh
        interpolated = self._vocab_weighted_embedding_sbh(
            hidden_states_sbh,
            temperature=float(getattr(self.model_config, "softmax_temperature", 1.0)),
            use_topk=bool(getattr(self.model_config, "interpolation_use_topk", False)),
            topk=int(getattr(self.model_config, "interpolation_topk", getattr(self.model_config, "top_k_num", 100))),
        )
        if self._uses_sequence_parallel():
            interpolated = self._ensure_sequence_parallel_sbh(
                interpolated,
                full_sequence_length=hidden_states_sbh.shape[0] * parallel_state.get_tensor_model_parallel_world_size(),
            )
        if getattr(self.model_config, "scale_embeds", False):
            interpolated = interpolated * math.sqrt(hidden_states_sbh.shape[-1])
        return interpolated

    def _stage_updates_from_interleaved_hidden(self, hidden_states_sbh: Tensor, num_stages: int) -> list[Tensor]:
        return [
            self._stage_update_from_hidden(hidden_states_sbh[stage_idx::num_stages].contiguous())
            for stage_idx in range(num_stages)
        ]

    def _source_interleaved_position_ids(
        self,
        position_ids: Tensor | None,
        *,
        batch_size: int,
        seq_len: int,
        num_stages: int,
        device: torch.device,
    ) -> Tensor:
        """Match HF LlamaModel default positions for interleaved ``inputs_embeds``.

        ``modeling_llama_ours.py`` prepares repeated position ids, but the
        initial interpolation, refinement, and final model calls leave
        ``position_ids`` commented out. HF LlamaModel therefore creates
        monotonic positions over the already-interleaved sequence and ignores
        any caller-provided reset/packed positions.
        """

        full_seq_len = seq_len
        if position_ids is not None and position_ids.dim() >= 2:
            full_seq_len = max(full_seq_len, position_ids.shape[1])

        if parallel_state.get_context_parallel_world_size() > 1:
            if position_ids is None:
                raise ValueError("LlamaOursModel CP requires global position_ids to build interleaved RoPE offsets.")
            base_positions = position_ids.to(device=device)
            if base_positions.dim() == 1:
                base_positions = base_positions.unsqueeze(0)
            if base_positions.shape[1] != seq_len:
                raise ValueError(
                    "position_ids length must match the local sequence before interleaving: "
                    f"got {base_positions.shape[1]}, expected {seq_len}."
                )
            if base_positions.shape[0] == 1 and batch_size != 1:
                base_positions = base_positions.expand(batch_size, seq_len)
            interleaved_cp = torch.empty(
                batch_size,
                seq_len * num_stages,
                dtype=base_positions.dtype,
                device=device,
            )
            for stage_idx in range(num_stages):
                interleaved_cp[:, stage_idx::num_stages] = base_positions * num_stages + stage_idx
            return interleaved_cp.contiguous()

        interleaved = torch.arange(full_seq_len * num_stages, dtype=torch.long, device=device).unsqueeze(0)
        if batch_size != 1:
            interleaved = interleaved.expand(batch_size, full_seq_len * num_stages)
        return interleaved.contiguous()

    def _run_interleaved_decoder(
        self,
        stages_sbh: list[Tensor],
        position_ids: Tensor | None,
        attention_mask: Tensor | None,
        stage_weights: Tensor | None = None,
    ) -> tuple[Tensor, int]:
        interleaved_sbh, _, loop_attention_mask, _ = self._interleave_stages_sbh(
            stages_sbh,
            position_ids,
            attention_mask,
            stage_weights,
        )
        seq_len, batch_size, _ = stages_sbh[0].shape
        num_stages = len(stages_sbh)
        loop_position_ids = self._source_interleaved_position_ids(
            position_ids,
            batch_size=batch_size,
            seq_len=seq_len,
            num_stages=num_stages,
            device=interleaved_sbh.device,
        )
        return (
            self._decoder_forward_sbh(
                interleaved_sbh,
                loop_attention_mask,
                loop_position_ids,
                input_is_sequence_parallel=self._uses_sequence_parallel(),
            ),
            num_stages,
        )

    def _build_stages(
        self,
        initial_embeds_sbh: Tensor,
        position_ids: Tensor | None,
        attention_mask: Tensor | None,
        global_step: int | None,
    ) -> list[Tensor]:
        stages = [initial_embeds_sbh]
        num_initial = self._num_interpolation_steps()
        for iter_idx in range(num_initial):
            final_sbh, num_stages = self._run_interleaved_decoder(stages, position_ids, attention_mask)
            updates = self._stage_updates_from_interleaved_hidden(final_sbh, num_stages)
            if iter_idx == 0:
                stages.append(updates[0])
            else:
                for stage_idx in range(iter_idx):
                    stages[stage_idx + 1] = updates[stage_idx]
                stages.append(updates[iter_idx])

        if num_initial > 0:
            for _ in range(self._num_refinement_steps(global_step)):
                final_sbh, num_stages = self._run_interleaved_decoder(stages, position_ids, attention_mask)
                updates = self._stage_updates_from_interleaved_hidden(final_sbh, num_stages)
                for stage_idx in range(num_initial):
                    stages[stage_idx + 1] = updates[stage_idx]
        return stages

    def _final_hidden_and_interleaved(
        self,
        stages: list[Tensor],
        position_ids: Tensor | None,
        attention_mask: Tensor | None,
        stage_weights: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, int, int]:
        final_sbh, num_stages = self._run_interleaved_decoder(stages, position_ids, attention_mask, stage_weights)
        target_idx = self._num_interpolation_steps()
        if target_idx >= num_stages:
            target_idx = 0
        target_sbh = final_sbh[target_idx::num_stages].contiguous()
        return target_sbh, final_sbh, num_stages, target_idx

    def _final_hidden_from_stages(
        self,
        stages: list[Tensor],
        position_ids: Tensor | None,
        attention_mask: Tensor | None,
        stage_weights: Tensor | None = None,
    ) -> Tensor:
        final_hidden_sbh, _, _, _ = self._final_hidden_and_interleaved(
            stages,
            position_ids,
            attention_mask,
            stage_weights,
        )
        return final_hidden_sbh

    def _consistency_loss(
        self,
        stages: list[Tensor],
        final_interleaved_sbh: Tensor,
        num_stages: int,
        target_idx: int,
    ) -> Tensor:
        weight = float(getattr(self.model_config, "consistency_weight", 0.0) or 0.0)
        if (not self.training) or weight <= 0.0 or len(stages) <= 1:
            self.last_consistency_loss = None
            return final_interleaved_sbh.new_zeros(())
        source_idx = max(0, min(target_idx - 1, num_stages - 1))
        hidden_for_check = final_interleaved_sbh[source_idx::num_stages].contiguous()
        recomputed = self._stage_update_from_hidden(hidden_for_check)
        target = stages[-1]
        target_fp32 = target.float()
        recomputed_fp32 = recomputed.float()
        consistency = 1.0 - F.cosine_similarity(target_fp32, recomputed_fp32, dim=-1).mean()
        scaled = final_interleaved_sbh.new_tensor(weight) * consistency
        self.last_consistency_loss = scaled.detach().float().cpu().item()
        return scaled

    def forward(
        self,
        tokens: Tensor,
        position_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        labels: Tensor | None = None,
        loss_mask: Tensor | None = None,
        global_step: int | None = None,
    ) -> Tensor:
        del loss_mask
        if labels is None:
            raise ValueError("LlamaOursModel expects labels and returns a scalar loss.")
        token_attention_mask = self._attention_mask_2d(attention_mask, tokens.shape[1])
        initial_embeds = self._embed_tokens_sbh(tokens, position_ids)
        initial_embeds = self._ensure_sequence_parallel_sbh(
            initial_embeds,
            full_sequence_length=labels.shape[1],
        )
        stages = self._build_stages(initial_embeds, position_ids, token_attention_mask, global_step)
        final_hidden_sbh, final_interleaved_sbh, num_stages, target_idx = self._final_hidden_and_interleaved(
            stages,
            position_ids,
            token_attention_mask,
        )
        loss = self._compute_shifted_loss(final_hidden_sbh.transpose(0, 1).contiguous(), labels)
        return loss + self._consistency_loss(stages, final_interleaved_sbh, num_stages, target_idx)


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
) -> LlamaOursModel:
    needs_accelerated_spec = bool(getattr(config, "sequence_parallel", False)) or (
        int(getattr(config, "context_parallel_size", 1) or 1) > 1
    )
    if needs_accelerated_spec and not use_transformer_engine_spec:
        raise ValueError("LlamaOursModel requires the TE/MindSpeed layer spec for SP/CP paths.")
    return LlamaOursModel(
        config=config,
        model_config=model_config,
        vocab_size=vocab_size,
        max_sequence_length=max_sequence_length,
        pre_process=pre_process,
        post_process=post_process,
        parallel_output=parallel_output,
        use_transformer_engine_spec=use_transformer_engine_spec,
    )

"""MCore/MindSpeed port of ``modeling_llama_new.py``."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from megatron.core import parallel_state
from megatron.core.transformer.transformer_config import TransformerConfig

from modeling.llama_ours import LlamaOursModel


class LlamaNewModel(LlamaOursModel):
    """Ponder-weighted staged recurrent path.

    This ports the main training path from ``modeling_llama_new.py``: a ponder
    head predicts per-token stage probabilities ``s``, cumulative stage gates
    ``w`` are fed into the interleaved MCore decoder passes, final hidden states
    are weighted by ``s``, and the min-weight penalty is added as an aux loss.

    The decoder itself stays on Megatron Core/MindSpeed specs. The HF
    FlashAttention log-gate augmentation is represented with TE
    ``attention_bias`` plus the matching expanded-head softmax scale; the
    ``adaptive_generate`` helper remains outside this training backend.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        hidden_size = int(getattr(self.model_config, "hidden_size", self.config.hidden_size))
        self.ponder_head = nn.Linear(hidden_size, self._num_ponder_steps(), bias=True)
        self._internal_global_step = 0

    def _num_ponder_steps(self) -> int:
        return self._num_interpolation_steps() + 1

    def _compute_ponder_weights(
        self,
        hidden_states_sbh: Tensor,
        *,
        return_logits: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor] | tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        logits = self.ponder_head(hidden_states_sbh.transpose(0, 1).contiguous())
        probs_float = torch.softmax(logits.float(), dim=-1)
        cumulative_float = torch.flip(
            torch.cumsum(torch.flip(probs_float, dims=[-1]), dim=-1),
            dims=[-1],
        )
        steps = torch.arange(
            probs_float.shape[-1],
            device=probs_float.device,
            dtype=probs_float.dtype,
        ).view(1, 1, -1)
        expected_steps = (steps * probs_float).sum(dim=-1)
        entropy = -(probs_float * torch.log(probs_float.clamp_min(1e-12))).sum(dim=-1)
        mean_probs = probs_float.mean(dim=(0, 1))
        mean_entropy = -(mean_probs * torch.log(mean_probs.clamp_min(1e-12))).sum()

        probs = probs_float.to(dtype=hidden_states_sbh.dtype)
        cumulative = cumulative_float.to(dtype=hidden_states_sbh.dtype)
        if return_logits:
            return probs, cumulative, logits, expected_steps, entropy, mean_entropy
        return probs, cumulative, expected_steps, entropy, mean_entropy

    def _stage_weights_for_count(self, weights: Tensor | None, num_stages: int) -> Tensor | None:
        if weights is None:
            return None
        if num_stages > weights.shape[-1]:
            raise RuntimeError(
                f"Ponder head produced {weights.shape[-1]} gates, but recurrent path needs {num_stages}."
            )
        selected = weights[..., :num_stages].contiguous()
        if (not self.training) and bool(getattr(self.model_config, "eval_prune_by_gate", True)):
            threshold = float(getattr(self.model_config, "ponder_gate_eval_thr", 1e-4))
            selected = selected.masked_fill(selected < threshold, 0.0)
        return selected

    def _compute_lambda_min_weight_penalty(self, global_step: int | None) -> float:
        lambda_start = float(getattr(self.model_config, "min_weight_penalty_lambda_start", 0.0) or 0.0)
        lambda_max = float(getattr(self.model_config, "min_weight_penalty_lambda_max", 0.0) or 0.0)
        warmup = int(getattr(self.model_config, "min_weight_penalty_warmup_steps", 1000) or 0)
        peak = int(getattr(self.model_config, "min_weight_penalty_peak_steps", 4000) or 0)
        step = int(global_step or 0)
        if step < warmup:
            return lambda_start
        if step >= peak or peak <= warmup:
            return lambda_max
        ratio = float(step - warmup) / float(max(1, peak - warmup))
        return lambda_start + ratio * (lambda_max - lambda_start)

    def _min_weight_penalty(
        self,
        stage_weights: Tensor | None,
        global_step: int | None,
        step_penalty_ratios: Tensor | None = None,
    ) -> Tensor:
        if (not self.training) or stage_weights is None:
            return self.ponder_head.weight.new_zeros(())
        lam = self._compute_lambda_min_weight_penalty(global_step)
        if lam <= 0.0 or stage_weights.shape[-1] <= 1:
            return stage_weights.new_zeros(())

        total_min_mean = stage_weights.new_zeros((), dtype=torch.float32)
        num_steps = stage_weights.shape[-1]
        previous_ratio: float | None = None
        for stage_idx in range(1, num_steps):
            if step_penalty_ratios is not None and step_penalty_ratios.numel() > stage_idx - 1:
                current_ratio = float(step_penalty_ratios[stage_idx - 1].clamp(0.0, 1.0).item())
            else:
                current_ratio = (stage_idx - 1) / num_steps
            if stage_idx == 1 or previous_ratio is None:
                penalty_ratio = current_ratio
            else:
                penalty_ratio = max(current_ratio - previous_ratio, 0.0)
            previous_ratio = current_ratio

            flat_weights = stage_weights[..., stage_idx].float().flatten()
            num_min = min(flat_weights.numel(), max(1, int(flat_weights.numel() * penalty_ratio)))
            min_values = torch.topk(flat_weights, k=num_min, largest=False).values
            total_min_mean = total_min_mean + min_values.mean()
        return stage_weights.new_tensor(lam) * total_min_mean.to(stage_weights.dtype)

    def _blend_ponder_state(
        self,
        old_state: tuple[Tensor, Tensor, Tensor, Tensor, Tensor],
        new_state: tuple[Tensor, Tensor, Tensor, Tensor, Tensor],
        refinement_idx: int,
        num_refinement_steps: int,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        alpha = float(getattr(self.model_config, "damping_alpha", 0.1) or 0.0)
        alpha = min(1.0, max(0.0, alpha))
        last_n_hard = int(getattr(self.model_config, "last_n_steps_update_w", 1) or 0)
        last_n_hard = max(0, last_n_hard)
        hard_tail_start = max(0, num_refinement_steps - last_n_hard)
        if refinement_idx >= hard_tail_start or alpha >= 1.0:
            return new_state
        if alpha <= 0.0:
            return old_state
        keep = 1.0 - alpha
        return tuple(keep * old + alpha * new for old, new in zip(old_state, new_state))  # type: ignore[return-value]

    def _build_stages_with_ponder(
        self,
        initial_embeds_sbh: Tensor,
        position_ids: Tensor | None,
        attention_mask: Tensor | None,
        global_step: int | None,
    ) -> tuple[list[Tensor], tuple[Tensor, Tensor, Tensor, Tensor, Tensor] | None]:
        stages = [initial_embeds_sbh]
        num_initial = self._num_interpolation_steps()
        if num_initial <= 0:
            return stages, None

        use_sequence_parallel = self._uses_sequence_parallel()
        pre_hidden_sbh = self._decoder_forward_sbh(
            initial_embeds_sbh,
            attention_mask,
            position_ids,
            input_is_sequence_parallel=use_sequence_parallel,
        )
        ponder_state = self._compute_ponder_weights(pre_hidden_sbh)

        for iter_idx in range(num_initial):
            stage_weights = self._stage_weights_for_count(ponder_state[1], len(stages))
            final_sbh, num_stages = self._run_interleaved_decoder(
                stages,
                position_ids,
                attention_mask,
                stage_weights,
            )
            updates = [
                self._stage_update_from_hidden(final_sbh[stage_idx::num_stages])
                for stage_idx in range(num_stages)
            ]
            if iter_idx == 0:
                stages.append(updates[0])
            else:
                for stage_idx in range(iter_idx):
                    stages[stage_idx + 1] = updates[stage_idx]
                stages.append(updates[iter_idx])

        num_refinement_steps = self._num_refinement_steps(global_step)
        if num_refinement_steps <= 0:
            return stages, ponder_state

        for ref_idx in range(num_refinement_steps):
            stage_weights = self._stage_weights_for_count(ponder_state[1], len(stages))
            final_sbh, num_stages = self._run_interleaved_decoder(
                stages,
                position_ids,
                attention_mask,
                stage_weights,
            )
            new_state = self._compute_ponder_weights(final_sbh[0::num_stages])
            ponder_state = self._blend_ponder_state(
                ponder_state,
                new_state,
                ref_idx,
                num_refinement_steps,
            )
            for stage_idx in range(num_initial):
                stages[stage_idx + 1] = self._stage_update_from_hidden(final_sbh[stage_idx::num_stages])
        return stages, ponder_state

    def _run_interleaved_decoder(
        self,
        stages_sbh: list[Tensor],
        position_ids: Tensor | None,
        attention_mask: Tensor | None,
        stage_weights: Tensor | None = None,
    ) -> tuple[Tensor, int]:
        interleaved_sbh, loop_position_ids, loop_attention_mask, ponder_gate = self._interleave_stages_sbh(
            stages_sbh,
            position_ids,
            attention_mask,
            stage_weights,
        )
        attention_bias = self._ponder_attention_bias(ponder_gate, dtype=interleaved_sbh.dtype)
        return (
            self._decoder_forward_sbh(
                interleaved_sbh,
                loop_attention_mask,
                loop_position_ids,
                attention_bias=attention_bias,
                input_is_sequence_parallel=self._uses_sequence_parallel(),
            ),
            len(stages_sbh),
        )

    def _ponder_attention_bias(self, ponder_gate: Tensor, *, dtype: torch.dtype) -> Tensor:
        gate = ponder_gate.float()
        log_gate = torch.log(gate.clamp_min(1e-12))
        log_gate = torch.where(gate <= 1e-4, torch.full_like(log_gate, -1e4), log_gate)
        return log_gate.to(dtype=dtype).unsqueeze(1).unsqueeze(1).contiguous()

    def _output_logits_partition_sbh(self, hidden_states_sbh: Tensor) -> Tensor:
        logits, _ = self.backbone.output_layer(
            hidden_states_sbh,
            weight=None,
            runtime_gather_output=False,
        )
        return logits

    def _global_predictions_and_probs_bsh(self, logits_sbh: Tensor) -> tuple[Tensor, Tensor]:
        logits = logits_sbh.float()
        local_values, local_indices = logits.max(dim=-1)
        tp_size = parallel_state.get_tensor_model_parallel_world_size()
        tp_rank = parallel_state.get_tensor_model_parallel_rank()
        partition_vocab_size = logits.shape[-1]

        if tp_size > 1:
            tp_group = parallel_state.get_tensor_model_parallel_group()
            gathered_values = torch.empty(
                (tp_size, *local_values.shape),
                dtype=local_values.dtype,
                device=local_values.device,
            )
            gathered_indices = torch.empty(
                (tp_size, *local_indices.shape),
                dtype=local_indices.dtype,
                device=local_indices.device,
            )
            torch.distributed.all_gather_into_tensor(gathered_values, local_values.contiguous(), group=tp_group)
            torch.distributed.all_gather_into_tensor(gathered_indices, local_indices.contiguous(), group=tp_group)
            winner_rank = gathered_values.argmax(dim=0)
            global_max = gathered_values.gather(0, winner_rank.unsqueeze(0)).squeeze(0)
            winner_local_indices = gathered_indices.gather(0, winner_rank.unsqueeze(0)).squeeze(0)
            global_indices = winner_local_indices + winner_rank.to(winner_local_indices.dtype) * partition_vocab_size
        else:
            winner_rank = None
            global_max = local_values
            global_indices = local_indices

        exp_logits = torch.exp(logits - global_max.unsqueeze(-1))
        denom = exp_logits.sum(dim=-1)
        selected_exp = exp_logits.gather(dim=-1, index=local_indices.unsqueeze(-1)).squeeze(-1)
        if tp_size > 1:
            tp_group = parallel_state.get_tensor_model_parallel_group()
            selected_exp = torch.where(
                winner_rank.eq(tp_rank),
                selected_exp,
                torch.zeros_like(selected_exp),
            )
            torch.distributed.all_reduce(denom, op=torch.distributed.ReduceOp.SUM, group=tp_group)
            torch.distributed.all_reduce(selected_exp, op=torch.distributed.ReduceOp.SUM, group=tp_group)

        probs = selected_exp / denom.clamp_min(torch.finfo(denom.dtype).tiny)
        return global_indices.transpose(0, 1).contiguous(), probs.transpose(0, 1).contiguous()

    def _weighted_step_hidden_sbh(
        self,
        final_interleaved_sbh: Tensor,
        stage_probs: Tensor,
        step_idx: int,
        num_stages: int,
    ) -> Tensor:
        seq_len = final_interleaved_sbh.shape[0] // num_stages
        hidden = final_interleaved_sbh.new_zeros(
            (seq_len, final_interleaved_sbh.shape[1], final_interleaved_sbh.shape[2])
        )
        for stage_idx in range(step_idx + 1):
            probs_sbh = stage_probs[..., stage_idx].transpose(0, 1).contiguous().unsqueeze(-1)
            hidden = hidden + final_interleaved_sbh[stage_idx::num_stages].contiguous() * probs_sbh
        return hidden

    def _per_token_ce_bsh(self, hidden_sbh: Tensor, labels: Tensor) -> Tensor:
        logits_sbh = self._output_logits_partition_sbh(hidden_sbh[:-1].contiguous())
        shift_labels = labels[:, 1:].contiguous()
        safe_labels = shift_labels.masked_fill(shift_labels.eq(-100), 0)
        return self.compute_language_model_loss(safe_labels, logits_sbh)

    def _step_penalty_ratios(
        self,
        final_interleaved_sbh: Tensor,
        stage_probs: Tensor | None,
        labels: Tensor,
        num_stages: int,
    ) -> Tensor | None:
        if stage_probs is None or num_stages <= 1 or final_interleaved_sbh.shape[0] % num_stages != 0:
            return None
        method = getattr(self.model_config, "min_weight_penalty_method", "accuracy")
        shift_labels = labels[:, 1:].contiguous()
        mask = shift_labels.ne(-100)
        token_count = mask.sum(dtype=torch.float32)
        if token_count <= 0:
            return final_interleaved_sbh.new_zeros((num_stages - 1,), dtype=torch.float32)

        ratios: list[Tensor] = []
        if method == "accuracy":
            for step_idx in range(num_stages):
                hidden_sbh = self._weighted_step_hidden_sbh(
                    final_interleaved_sbh,
                    stage_probs,
                    step_idx,
                    num_stages,
                )
                logits_sbh = self._output_logits_partition_sbh(hidden_sbh[:-1].contiguous())
                preds, pred_probs = self._global_predictions_and_probs_bsh(logits_sbh)
                correct = preds.eq(shift_labels) & mask
                ratio = (correct.to(pred_probs.dtype) * pred_probs).masked_select(mask).sum() / token_count
                ratios.append(ratio.detach().float())
            return torch.stack(ratios)

        if method == "delta_loss":
            losses: list[Tensor] = []
            for step_idx in range(num_stages):
                hidden_sbh = self._weighted_step_hidden_sbh(
                    final_interleaved_sbh,
                    stage_probs,
                    step_idx,
                    num_stages,
                )
                losses.append(self._per_token_ce_bsh(hidden_sbh, labels).detach())
            for step_idx in range(1, num_stages):
                delta = torch.clamp(losses[step_idx] - losses[step_idx - 1], max=0.0)
                ratio = torch.sigmoid(50.0 * delta).masked_select(mask).sum() / token_count
                ratios.append(ratio.detach().float())
            return torch.stack(ratios) if ratios else None

        if method == "ce_loss":
            for step_idx in range(1, num_stages):
                hidden_sbh = self._weighted_step_hidden_sbh(
                    final_interleaved_sbh,
                    stage_probs,
                    step_idx,
                    num_stages,
                )
                per_token_ce = self._per_token_ce_bsh(hidden_sbh, labels).detach()
                score = torch.sigmoid(10.0 * (per_token_ce - 0.5))
                ratio = 1.0 - score.masked_select(mask).sum() / token_count
                ratios.append(ratio.detach().float())
            return torch.stack(ratios) if ratios else None

        return None

    def _ponder_final_hidden_and_interleaved(
        self,
        stages: list[Tensor],
        position_ids: Tensor | None,
        attention_mask: Tensor | None,
        stage_weights: Tensor | None,
        stage_probs: Tensor | None,
    ) -> tuple[Tensor, Tensor, int]:
        final_sbh, num_stages = self._run_interleaved_decoder(stages, position_ids, attention_mask, stage_weights)
        if stage_probs is None:
            return final_sbh[(num_stages - 1)::num_stages].contiguous(), final_sbh, num_stages
        if stage_probs.shape[-1] < num_stages:
            raise RuntimeError(
                f"Ponder head produced {stage_probs.shape[-1]} probabilities, but final pass has {num_stages} stages."
            )
        stage_probs = stage_probs[..., :num_stages].to(dtype=final_sbh.dtype)
        hidden_bsh = final_sbh.transpose(0, 1).contiguous()
        batch_size, total_len, hidden_size = hidden_bsh.shape
        seq_len = total_len // num_stages
        staged = hidden_bsh.view(batch_size, seq_len, num_stages, hidden_size)
        final_hidden = (staged * stage_probs.unsqueeze(-1)).sum(dim=2).transpose(0, 1).contiguous()
        return final_hidden, final_sbh, num_stages

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
            raise ValueError("LlamaNewModel expects labels and returns a scalar loss.")

        if self.training and global_step is None:
            global_step = self._internal_global_step
            self._internal_global_step += 1

        token_attention_mask = self._attention_mask_2d(attention_mask, tokens.shape[1])
        initial_embeds = self._embed_tokens_sbh(tokens, position_ids)
        initial_embeds = self._ensure_sequence_parallel_sbh(
            initial_embeds,
            full_sequence_length=labels.shape[1],
        )
        stages, ponder_state = self._build_stages_with_ponder(
            initial_embeds,
            position_ids,
            token_attention_mask,
            global_step,
        )
        if ponder_state is None:
            stage_probs = None
            stage_weights = None
        else:
            stage_probs, raw_stage_weights, _, _, _ = ponder_state
            if stage_probs.shape[-1] < len(stages):
                raise RuntimeError(
                    f"Ponder head produced {stage_probs.shape[-1]} stages, but recurrent path built {len(stages)}."
                )
            stage_weights = self._stage_weights_for_count(raw_stage_weights, len(stages))

        final_hidden_sbh, final_interleaved_sbh, num_stages = self._ponder_final_hidden_and_interleaved(
            stages,
            position_ids,
            token_attention_mask,
            stage_weights,
            stage_probs,
        )
        base_loss = self._compute_shifted_loss(final_hidden_sbh.transpose(0, 1).contiguous(), labels)
        step_penalty_ratios = None
        if self._compute_lambda_min_weight_penalty(global_step) > 0.0:
            step_penalty_ratios = self._step_penalty_ratios(
                final_interleaved_sbh,
                stage_probs,
                labels,
                num_stages,
            )
        return base_loss + self._min_weight_penalty(stage_weights, global_step, step_penalty_ratios)

    def adaptive_generate(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError(
            "adaptive_generate is not ported to the Megatron/MindSpeed training backend; "
            "use the HF LLaMA-Factory implementation for adaptive early-exit decoding."
        )


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
) -> LlamaNewModel:
    if not use_transformer_engine_spec:
        raise ValueError(
            "LlamaNewModel requires the transformer_engine/MindSpeed layer spec so ponder_gate can be "
            "implemented as TE attention_bias. The local DotProductAttention path does not support attention_bias."
        )
    head_dim = int(getattr(config, "kv_channels", 0) or (config.hidden_size // config.num_attention_heads))
    config.softmax_scale = 1.0 / math.sqrt(float(head_dim + 8))
    return LlamaNewModel(
        config=config,
        model_config=model_config,
        vocab_size=vocab_size,
        max_sequence_length=max_sequence_length,
        pre_process=pre_process,
        post_process=post_process,
        parallel_output=parallel_output,
        use_transformer_engine_spec=use_transformer_engine_spec,
    )

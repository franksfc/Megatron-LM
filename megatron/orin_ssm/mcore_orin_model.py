"""Megatron Core Orin/SSM language model."""

from __future__ import annotations

import math
import os
from typing import Any

import torch
from torch import Tensor
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
from megatron.orin_ssm.mcore_loop_axis_ssm import MCoreLoopAxisSBHTPSSM, MCoreLoopAxisSSM


class OrinMCoreModel(LanguageModule):
    """MCore GPT backbone with the Orin recurrent SSM loop.

    The recurrent loop is implemented locally in ``MCoreLoopAxisSSM`` so the
    training path stays inside the Megatron/MindSpeed stack by default.
    """

    def __init__(
        self,
        config: TransformerConfig,
        orin_config: Any,
        vocab_size: int,
        max_sequence_length: int,
        pre_process: bool = True,
        post_process: bool = True,
        parallel_output: bool = True,
        use_transformer_engine_spec: bool = False,
    ) -> None:
        super().__init__(config=config)
        self.config = config
        self.orin_config = orin_config
        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        self.pre_process = pre_process
        self.post_process = post_process
        self.parallel_output = parallel_output
        self.use_transformer_engine_spec = use_transformer_engine_spec
        self.share_embeddings_and_output_weights = False
        self.use_null_attention_mask = (
            os.getenv(
                "ORIN_MCORE_USE_NULL_ATTENTION_MASK",
                "1" if use_transformer_engine_spec else "0",
            )
            == "1"
        )
        self.model_type = ModelType.encoder_or_decoder

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
            rotary_base=getattr(orin_config, "rope_theta", 10000),
            scatter_embedding_sequence_parallel=False,
        )

        self.memory_size = getattr(
            orin_config,
            "loop_ssm_state_size",
            getattr(orin_config, "memory_size", orin_config.hidden_size),
        )
        token_mamba_variant = getattr(orin_config, "loop_mamba_variant", None)
        if token_mamba_variant is None:
            patch_method = str(getattr(orin_config, "patch_method", "")).strip().lower()
            token_mamba_variant = "mamba2" if patch_method in ("orin_mamba2", "orin_mamba2_fast") else "legacy"

        loop_axis_ssm_kwargs = dict(
            hidden_size=orin_config.hidden_size,
            state_size=self.memory_size,
            rms_norm_eps=orin_config.rms_norm_eps,
            lambda_min=float(getattr(orin_config, "loop_ssm_lambda_min", 0.01)),
            lambda_max=float(getattr(orin_config, "loop_ssm_lambda_max", 4.0)),
            beta=float(getattr(orin_config, "loop_ssm_beta", 0.8)),
            out_scale=float(getattr(orin_config, "loop_ssm_out_scale", 0.3)),
            eta0=float(getattr(orin_config, "loop_ssm_eta0", 0.3)),
            token_mamba_expand=float(getattr(orin_config, "loop_mamba_expand", 2.0)),
            token_mamba_state_size=int(getattr(orin_config, "loop_mamba_state_size", 16)),
            token_mamba_conv_kernel=int(getattr(orin_config, "loop_mamba_conv_kernel", 4)),
            token_mamba_dt_rank=getattr(orin_config, "loop_mamba_dt_rank", "auto"),
            token_mamba_dt_min=float(getattr(orin_config, "loop_mamba_dt_min", 0.001)),
            token_mamba_dt_max=float(getattr(orin_config, "loop_mamba_dt_max", 0.1)),
            token_mamba_chunk_size=int(getattr(orin_config, "loop_mamba_chunk_size", 32)),
            token_mamba_head_dim=int(getattr(orin_config, "loop_mamba_head_dim", 64)),
            token_mamba_variant=token_mamba_variant,
            token_mamba_n_groups=int(getattr(orin_config, "loop_mamba_n_groups", 1)),
            token_mamba_clamp_dt=bool(getattr(orin_config, "loop_mamba_clamp_dt", False)),
            token_mamba_bias=bool(getattr(orin_config, "loop_mamba_bias", False)),
            token_mamba_conv_bias=bool(getattr(orin_config, "loop_mamba_conv_bias", True)),
            token_mamba_residual_scale=float(getattr(orin_config, "loop_mamba_residual_scale", 1.0)),
        )
        loop_layout = os.getenv("ORIN_MCORE_LOOP_LAYOUT", "bsh").strip().lower()
        if loop_layout in ("bsh", "legacy"):
            loop_axis_ssm_cls = MCoreLoopAxisSSM
        elif loop_layout in ("sbh_tp", "sbh-tp"):
            loop_axis_ssm_cls = MCoreLoopAxisSBHTPSSM
        else:
            raise ValueError(f"Unsupported ORIN_MCORE_LOOP_LAYOUT={loop_layout!r}.")
        self.loop_axis_ssm = loop_axis_ssm_cls(config=config, **loop_axis_ssm_kwargs)
        self.loop_uses_sbh_layout = bool(getattr(self.loop_axis_ssm, "uses_sbh_layout", False))

    def sharded_state_dict(
        self,
        prefix: str = "",
        sharded_offsets: tuple[tuple[int, int, int], ...] = (),
        metadata: dict | None = None,
    ) -> ShardedStateDict:
        return MegatronModule.sharded_state_dict(self, prefix, sharded_offsets, metadata)

    def set_input_tensor(self, input_tensor: Tensor) -> None:
        self.backbone.set_input_tensor(input_tensor)

    def _get_num_recurrent_iterations(self) -> int:
        train_iterations = int(getattr(self.orin_config, "more_iterations", 0) or 0)
        eval_iterations = int(getattr(self.orin_config, "more_eval_iterations", 0) or 0)
        if not self.training and eval_iterations > 0:
            return max(1, eval_iterations + 1)
        return max(1, train_iterations + 1)

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
        if getattr(self.orin_config, "scale_embeds", False):
            embeddings = embeddings * math.sqrt(embeddings.shape[-1])
        return embeddings

    def _embed_tokens(self, tokens: Tensor, position_ids: Tensor | None) -> Tensor:
        embeddings = self._embed_tokens_sbh(tokens, position_ids)
        # MCore embedding returns [seq, batch, hidden]; the legacy recurrent path uses [batch, seq, hidden].
        return embeddings.transpose(0, 1).contiguous()

    def _uses_sequence_parallel(self) -> bool:
        return self.config.sequence_parallel and parallel_state.get_tensor_model_parallel_world_size() > 1

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
                "Unexpected sequence-parallel embedding length: "
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
            raise ValueError("Orin sequence/CP parallel RoPE generation requires local position_ids.")

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

    def _decoder_forward(
        self,
        hidden_states_bsh: Tensor,
        attention_mask: Tensor | None,
        position_ids: Tensor | None,
    ) -> Tensor:
        decoder_input = hidden_states_bsh.transpose(0, 1).contiguous()
        causal_mask = None
        if not self.use_null_attention_mask:
            causal_mask = self._build_causal_mask(
                attention_mask,
                hidden_states_bsh.shape[1],
                hidden_states_bsh.device,
            )
        rotary_pos_emb = self._build_rotary_pos_emb(hidden_states_bsh.shape[1], position_ids)
        hidden_states = self.backbone.decoder(
            hidden_states=decoder_input,
            attention_mask=causal_mask,
            inference_context=None,
            rotary_pos_emb=rotary_pos_emb,
        )
        return hidden_states.transpose(0, 1).contiguous()

    def _decoder_forward_sbh(
        self,
        hidden_states_sbh: Tensor,
        attention_mask: Tensor | None,
        position_ids: Tensor | None,
        *,
        input_is_sequence_parallel: bool = False,
    ) -> Tensor:
        decoder_input = hidden_states_sbh.contiguous()
        rotary_pos_emb = self._build_rotary_pos_emb(hidden_states_sbh.shape[0], position_ids)
        use_sequence_parallel = self._uses_sequence_parallel()
        if use_sequence_parallel and not input_is_sequence_parallel:
            decoder_input = tensor_parallel.scatter_to_sequence_parallel_region(decoder_input)
        if use_sequence_parallel and not input_is_sequence_parallel:
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
        hidden_states = self.backbone.decoder(
            hidden_states=decoder_input,
            attention_mask=causal_mask,
            inference_context=None,
            rotary_pos_emb=rotary_pos_emb,
        )
        if use_sequence_parallel and not input_is_sequence_parallel:
            hidden_states = tensor_parallel.gather_from_sequence_parallel_region(
                hidden_states,
                tensor_parallel_output_grad=False,
            )
        return hidden_states

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
        if os.getenv("ORIN_DEBUG_LOSS_SHAPES", "0") == "1":
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            if rank == 0:
                print(
                    "[orin-debug] loss shapes "
                    f"hidden_sbh={tuple(hidden_states_sbh.shape)} "
                    f"logits={tuple(logits.shape)} "
                    f"labels={tuple(labels.shape)} "
                    f"mask={tuple(loss_mask.shape)} "
                    f"sp={self.config.sequence_parallel} "
                    f"tp={parallel_state.get_tensor_model_parallel_world_size()} "
                    f"cp={parallel_state.get_context_parallel_world_size()}",
                    flush=True,
                )
        if os.getenv("ORIN_LM_LOSS_UPCAST", "0") == "1":
            logits = logits.float()
        per_token_loss = self.compute_language_model_loss(labels, logits)
        loss_sum = (per_token_loss * loss_mask.to(per_token_loss.dtype)).sum()
        return loss_sum

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
        if os.getenv("ORIN_CHUNKED_LM_LOSS_CHECKPOINT", "0") != "1":
            return self._compute_lm_loss_sum(hidden_states_bsh, labels, loss_mask)
        return torch_checkpoint(
            self._compute_lm_loss_sum,
            hidden_states_bsh,
            labels,
            loss_mask,
            use_reentrant=False,
        )

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

        chunk_size = int(os.getenv("ORIN_CHUNKED_LM_LOSS_TOKENS", "0") or "0")
        token_count = loss_mask.sum(dtype=torch.float32).clamp_min(1.0)
        if chunk_size <= 0 or chunk_size >= shifted_hidden.shape[1]:
            loss = self._compute_lm_loss_sum_maybe_checkpoint(shifted_hidden, safe_labels, loss_mask)
            return loss / token_count

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
            raise ValueError("OrinMCoreModel currently expects labels and returns a scalar loss.")

        if self.loop_uses_sbh_layout:
            use_sequence_parallel = self._uses_sequence_parallel()
            embeds_sbh = self._ensure_sequence_parallel_sbh(
                self._embed_tokens_sbh(tokens, position_ids),
                full_sequence_length=labels.shape[1],
            )
            if use_sequence_parallel:
                full_seq_len = labels.shape[1]
                state, z = self.loop_axis_ssm.init_state(
                    embeds_sbh.new_empty(full_seq_len, embeds_sbh.shape[1], embeds_sbh.shape[2])
                )
            else:
                state, z = self.loop_axis_ssm.init_state(embeds_sbh)
            token_attention_mask = attention_mask if attention_mask is not None and attention_mask.dim() == 2 else None
            decoder_attention_mask = self._slice_sequence_parallel_mask(token_attention_mask)
            recurrent_attention_mask = token_attention_mask
            last_backbone_out = None
            num_loops = self._get_num_recurrent_iterations()
            decay_rate = self.loop_axis_ssm.get_decay_rate()
            beta = self.loop_axis_ssm.beta

            for loop_idx in range(num_loops):
                z_for_backbone = (
                    self.loop_axis_ssm.z_for_backbone(z)
                    if hasattr(self.loop_axis_ssm, "z_for_backbone")
                    else z
                )
                if use_sequence_parallel:
                    z_for_backbone = self._scatter_sequence_parallel_sbh(z_for_backbone)
                decoder_input = embeds_sbh + beta * z_for_backbone
                last_backbone_out_local = self._decoder_forward_sbh(
                    decoder_input,
                    decoder_attention_mask,
                    position_ids,
                    input_is_sequence_parallel=use_sequence_parallel,
                )
                last_backbone_out = self._gather_sequence_parallel_sbh(last_backbone_out_local)
                compute_next_z = loop_idx + 1 < num_loops or num_loops == 1
                state, z = self.loop_axis_ssm(
                    state=state,
                    z=z,
                    backbone_out=last_backbone_out,
                    loop_idx=loop_idx,
                    attention_mask=recurrent_attention_mask,
                    decay_rate=decay_rate,
                    compute_next_z=compute_next_z,
                )

            hidden_states_sbh = last_backbone_out + self.loop_axis_ssm.output_readout(state)
            if num_loops == 1:
                z_for_backbone = (
                    self.loop_axis_ssm.z_for_backbone(z)
                    if hasattr(self.loop_axis_ssm, "z_for_backbone")
                    else z
                )
                hidden_states_sbh = hidden_states_sbh + (0.0 * z_for_backbone)
            loss_hidden_states_sbh = self._scatter_sequence_parallel_sbh(hidden_states_sbh)
            return self._compute_shifted_loss(loss_hidden_states_sbh.transpose(0, 1).contiguous(), labels)

        embeds = self._embed_tokens(tokens, position_ids)
        token_attention_mask = attention_mask if attention_mask is not None and attention_mask.dim() == 2 else None
        state, z = self.loop_axis_ssm.init_state(embeds)
        last_backbone_out = None
        num_loops = self._get_num_recurrent_iterations()
        decay_rate = self.loop_axis_ssm.get_decay_rate()
        beta = self.loop_axis_ssm.beta

        for loop_idx in range(num_loops):
            decoder_input = embeds + beta * z
            last_backbone_out = self._decoder_forward(decoder_input, attention_mask, position_ids)
            current_attention_mask = token_attention_mask
            if current_attention_mask is not None:
                current_attention_mask = current_attention_mask[:, -last_backbone_out.shape[1] :]
            state, z = self.loop_axis_ssm(
                state=state,
                z=z,
                backbone_out=last_backbone_out,
                loop_idx=loop_idx,
                attention_mask=current_attention_mask,
                decay_rate=decay_rate,
                compute_next_z=(loop_idx + 1 < num_loops or num_loops == 1),
            )

        hidden_states = last_backbone_out + self.loop_axis_ssm.output_readout(state)
        if num_loops == 1:
            hidden_states = hidden_states + (0.0 * z)
        return self._compute_shifted_loss(hidden_states, labels)

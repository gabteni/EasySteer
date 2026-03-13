"""
GptOss vLLM model with SteerMOE support for EasySteer.

This is a copy of the upstream GptOss vLLM model (src/modeling_vllm/gpt_oss.py)
with the following additions:
  - ``GptOssForCausalLM.update_moe_manual_args()``  to push steering weights
  - ``MLPBlock.forward()`` biases router logits using those weights

Call ``easysteer.hidden_states.apply_moe_steering_weights(llm, weights)``
to activate steering before generation.

To use, register with vLLM before creating the LLM instance:

    from easysteer.modeling import register_gpt_oss
    register_gpt_oss()

    from vllm import LLM
    llm = LLM(model="path/to/gpt-oss-moe", ...)
"""

import os
from typing import Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    extract_layer_index,
    maybe_prefix,
    support_torch_compile,
)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.attention import Attention, AttentionType
from vllm.sequence import IntermediateTensors

try:
    from vllm.model_executor.layers.quantization import QuantizationConfig
except ImportError:
    QuantizationConfig = None  # type: ignore

# ---------------------------------------------------------------------------
# Config (mirrors upstream GptOssConfig)
# ---------------------------------------------------------------------------

class GptOssConfig:
    """Minimal HF-style config placeholder – populated from the JSON on disk."""

    model_type: str = "gpt_oss"

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

class OAIAttention(nn.Module):

    def __init__(self, config: GptOssConfig, prefix: str = ""):
        super().__init__()
        tp_size = get_tensor_model_parallel_world_size()
        self.layer_idx = extract_layer_index(prefix)
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scaling = self.head_dim ** -0.5
        self.sinks = getattr(config, "attention_sinks", 0)

        self.norm = RMSNorm(config.hidden_size, eps=1e-5)

        self.q_size = self.num_attention_heads * self.head_dim
        self.kv_size = self.num_key_value_heads * self.head_dim

        self.qkv = QKVParallelLinear(
            hidden_size=self.hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.num_attention_heads,
            total_num_kv_heads=self.num_key_value_heads,
            bias=True,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            input_size=self.num_attention_heads * self.head_dim,
            output_size=self.hidden_size,
            prefix=f"{prefix}.o_proj",
        )

        self.num_local_attention_heads = config.num_attention_heads // tp_size
        self.num_local_key_value_heads = config.num_key_value_heads // tp_size

        sliding_window = (
            config.sliding_window if self.layer_idx % 2 == 0 else None
        )
        self.attn = Attention(
            self.num_local_attention_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_local_key_value_heads,
            per_layer_sliding_window=sliding_window,
            attn_type=AttentionType.DECODER,
            prefix=f"{prefix}.attn",
            sinks=self.sinks,
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=getattr(config, "max_position_embeddings", 131072),
            base=getattr(config, "rope_theta", 500000.0),
        )

    def forward(
        self, hidden_states: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        t = self.norm(hidden_states)
        qkv, _ = self.qkv(t)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        v = v.contiguous()
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output + hidden_states


# ---------------------------------------------------------------------------
# MLP / MoE block  ← SteerMOE injection lives here
# ---------------------------------------------------------------------------

class MLPBlock(nn.Module):

    def __init__(
        self,
        config: GptOssConfig,
        layer_idx: int,
        quant_config=None,
        prefix: str = "",
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_experts = config.num_local_experts
        self.experts_per_token = config.num_experts_per_tok
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1

        self.norm = RMSNorm(config.hidden_size, eps=1e-5)
        self.router = nn.Linear(
            config.hidden_size,
            config.num_local_experts,
            dtype=torch.bfloat16,
        )

        assert config.intermediate_size % self.world_size == 0
        self.experts = FusedMoE(
            num_experts=config.num_local_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            reduce_results=True,
            renormalize=True,
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
            apply_router_weight_on_input=False,
            has_bias=True,
            activation="swigluoai",
        )

        # SteerMOE state – populated by GptOssForCausalLM.update_moe_manual_args()
        self.moe_manual_args: dict = {
            "moe_manual_weights": torch.zeros(
                (1, config.num_local_experts), dtype=torch.float32
            )
        }
        self.layer_idx_: int = layer_idx  # note: trailing underscore matches upstream

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = self.norm(x)
        g = self.router(t)  # [n_tokens, n_experts]

        # ------------------------------------------------------------------ #
        # SteerMOE router-logit injection                                     #
        # ------------------------------------------------------------------ #
        moe_manual_weights = self.moe_manual_args["moe_manual_weights"]
        moe_manual_weights = moe_manual_weights.to(g.device)
        layer_weights = moe_manual_weights[self.layer_idx_]  # [n_experts]

        if layer_weights.abs().max() > 0:
            g = torch.nn.functional.log_softmax(g, dim=-1)
            max_per_tok = g.max(dim=-1).values.unsqueeze(-1)   # (n_tokens, 1)
            min_per_tok = g.min(dim=-1).values.unsqueeze(-1)   # (n_tokens, 1)
            pos_mask = (layer_weights > 0)                      # [n_experts]
            neg_mask = (layer_weights < 0)                      # [n_experts]
            if pos_mask.any():
                g[:, pos_mask] = max_per_tok + 0.01
            if neg_mask.any():
                g[:, neg_mask] = min_per_tok - 0.01
        # ------------------------------------------------------------------ #

        t = self.experts(hidden_states=t, router_logits=g)
        return x + t


# ---------------------------------------------------------------------------
# Transformer block / full model
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):

    def __init__(
        self,
        config: GptOssConfig,
        quant_config=None,
        prefix: str = "",
    ):
        super().__init__()
        self.layer_idx = extract_layer_index(prefix)
        self.attn = OAIAttention(config, prefix=f"{prefix}.attn")
        self.mlp = MLPBlock(
            config,
            self.layer_idx,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )

    def forward(
        self, hidden_states: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        attn_output = self.attn(hidden_states, positions)
        output = self.mlp(attn_output)
        return output


@support_torch_compile
class GptOssModel(nn.Module):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.config = vllm_config.model_config.hf_config
        self.quant_config = vllm_config.quant_config
        self.embedding = VocabParallelEmbedding(
            self.config.vocab_size,
            self.config.hidden_size,
        )
        self.layers = nn.ModuleList([
            TransformerBlock(
                self.config,
                quant_config=self.quant_config,
                prefix=maybe_prefix(prefix, f"block.{layer_idx}"),
            )
            for layer_idx in range(self.config.num_hidden_layers)
        ])
        self.norm = RMSNorm(self.config.hidden_size, eps=1e-5)

    def forward(
        self, input_ids: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x, positions)
        return self.norm(x)


# ---------------------------------------------------------------------------
# Top-level CausalLM  ← update_moe_manual_args lives here
# ---------------------------------------------------------------------------

class GptOssForCausalLM(nn.Module):
    """
    GptOss CausalLM with SteerMOE support.

    After loading the model, call ``update_moe_manual_args`` to push
    a [num_layers, num_experts] weight tensor into every MLP block:

        model.update_moe_manual_args({
            "moe_manual_weights": weights_tensor   # shape [L, E]
        })

    Weights of 0 are no-ops.  Positive weights boost an expert (router logit
    set to max+0.01).  Negative weights suppress an expert (set to min-0.01).
    """

    def __init__(self, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config.hf_config
        self.model = GptOssModel(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
        )
        self.lm_head = ParallelLMHead(
            self.model_config.vocab_size,
            self.model_config.hidden_size,
        )
        self.logits_processor = LogitsProcessor(self.model_config.vocab_size)

        # Initialise SteerMOE with zero weights (no-op)
        self.update_moe_manual_args({
            "moe_manual_weights": torch.zeros(
                (self.model_config.num_hidden_layers,
                 self.model_config.num_local_experts),
                dtype=torch.float32,
            )
        })

    # ---------------------------------------------------------------------- #
    # SteerMOE API                                                            #
    # ---------------------------------------------------------------------- #

    def update_moe_manual_args(self, moe_manual_args: dict) -> None:
        """
        Push steering weights into every MLP block.

        Parameters
        ----------
        moe_manual_args:
            Dict containing at least ``"moe_manual_weights"``: a
            ``torch.Tensor`` of shape ``[num_layers, num_experts]``.
        """
        for layer_idx, layer in enumerate(self.model.layers):
            mlp: MLPBlock = layer.mlp
            # Deep-copy tensors so each layer has an independent reference
            per_layer_args = {}
            for k, v in moe_manual_args.items():
                per_layer_args[k] = v.clone() if isinstance(v, torch.Tensor) else v
            mlp.moe_manual_args = per_layer_args
            mlp.layer_idx_ = layer_idx

    def reset_moe_steering(self) -> None:
        """Reset all steering weights to zero (no-op)."""
        self.update_moe_manual_args({
            "moe_manual_weights": torch.zeros(
                (self.model_config.num_hidden_layers,
                 self.model_config.num_local_experts),
                dtype=torch.float32,
            )
        })

    # ---------------------------------------------------------------------- #
    # Standard vLLM interface                                                 #
    # ---------------------------------------------------------------------- #

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert intermediate_tensors is None
        assert inputs_embeds is None
        return self.model(input_ids, positions)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> torch.Tensor:
        return self.logits_processor(self.lm_head, hidden_states, sampling_metadata)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> set:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)

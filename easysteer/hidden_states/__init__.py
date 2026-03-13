# SPDX-License-Identifier: Apache-2.0
"""
Hidden States Capture for vLLM V1

A simple, clean interface for capturing hidden states from vLLM V1 models.
This module wraps vLLM V1's RPC-based hidden states capture to provide
a user-friendly API similar to V0.

Example:
    >>> import easysteer.hidden_states as hs
    >>> from vllm import LLM
    >>>
    >>> # Capture hidden states (embed task)
    >>> llm = LLM(model="meta-llama/Meta-Llama-3-8B-Instruct", task="embed")
    >>> all_hidden_states, outputs = hs.get_all_hidden_states(llm, ["Hello world"])
    >>> print(f"Captured {len(all_hidden_states)} samples")
    >>>
    >>> # Capture hidden states (generate task, for multimodal models)
    >>> llm_vlm = LLM(model="Qwen2.5-VL-7B-Instruct")
    >>> hidden_states, outputs = hs.get_all_hidden_states_generate(llm_vlm, ["Hello"])
    >>> print(f"Captured {len(hidden_states)} samples")
    >>>
    >>> # Capture MoE router logits
    >>> llm_moe = LLM(model="mistralai/Mixtral-8x7B-v0.1")
    >>> router_logits, outputs = hs.get_moe_router_logits(llm_moe, ["Hello world"])
    >>> print(f"Captured {len(router_logits)} MoE layers")
    >>>
    >>> # Apply SteerMOE weights before generation
    >>> weights = torch.zeros(32, 64)   # [num_layers, num_experts]
    >>> hs.apply_moe_steering_weights(llm_moe, weights)
"""

import logging
from typing import Any, Optional

import torch

from .capture import get_all_hidden_states, HiddenStatesCaptureV1
from .capture_generate import (
    get_all_hidden_states_generate,
    HiddenStatesCaptureGenerate,
)
from .moe_capture import (
    get_moe_router_logits,
    analyze_expert_usage,
    MoERouterLogitsCaptureV1,
)
from .moe_capture_generate import (
    get_moe_router_logits_generate,
    MoERouterLogitsCaptureGenerate,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SteerMOE application helper
# ---------------------------------------------------------------------------

def apply_moe_steering_weights(
    llm: Any,
    moe_manual_weights: torch.Tensor,
    device: Optional[str] = None,
) -> None:
    """
    Push SteerMOE weights into a running vLLM LLM instance.

    This calls ``model.update_moe_manual_args()`` on the underlying
    ``GptOssForCausalLM`` (or any other SteerMOE-patched model).  The weights
    take effect immediately for all subsequent ``llm.generate()`` calls.

    Parameters
    ----------
    llm:
        A vLLM ``LLM`` instance whose underlying model exposes
        ``update_moe_manual_args()``.
    moe_manual_weights:
        Tensor of shape ``[num_layers, num_experts]``.
          +value  → boost that expert (router logit set to max+0.01)
          -value  → suppress that expert (router logit set to min-0.01)
           0      → no change
    device:
        Optional device string to move weights to before pushing (e.g. "cuda").
        If None, the model's forward pass handles device placement.

    Raises
    ------
    AttributeError:
        If the underlying model does not have ``update_moe_manual_args``.
        Make sure you registered the SteerMOE-patched model via
        ``easysteer.modeling.register_gpt_oss()`` before creating the LLM.

    Example
    -------
        from easysteer.modeling import register_gpt_oss
        register_gpt_oss()

        from vllm import LLM
        import easysteer.steer as steer
        import easysteer.hidden_states as hs

        llm = LLM(model="path/to/gpt-oss-moe", task="embed")

        # Extract weights from examples
        weights = steer.extract_steer_moe_weights(
            llm,
            positive_texts=["great", "wonderful"],
            negative_texts=["awful", "terrible"],
        )

        # Apply and generate
        hs.apply_moe_steering_weights(llm, weights.weights)
        # ... llm.generate(...)

        # Reset to baseline
        hs.reset_moe_steering(llm)
    """
    if device is not None:
        moe_manual_weights = moe_manual_weights.to(device)

    model = _get_model(llm)
    if not hasattr(model, "update_moe_manual_args"):
        raise AttributeError(
            "The vLLM model does not expose `update_moe_manual_args`. "
            "Did you register the SteerMOE-patched model via "
            "`easysteer.modeling.register_gpt_oss()` before creating the LLM?"
        )
    model.update_moe_manual_args({"moe_manual_weights": moe_manual_weights})
    logger.info(
        f"SteerMOE weights applied: shape={list(moe_manual_weights.shape)}, "
        f"nonzero={(moe_manual_weights != 0).sum().item()}"
    )


def reset_moe_steering(llm: Any) -> None:
    """
    Reset SteerMOE weights to zero (no steering).

    Parameters
    ----------
    llm:
        A vLLM ``LLM`` instance whose model exposes ``reset_moe_steering()``.
    """
    model = _get_model(llm)
    if hasattr(model, "reset_moe_steering"):
        model.reset_moe_steering()
        logger.info("SteerMOE weights reset to zero.")
    elif hasattr(model, "update_moe_manual_args"):
        # Fallback: infer shape from the existing weights and zero them
        try:
            first_layer = next(iter(model.model.layers))
            existing = first_layer.mlp.moe_manual_args["moe_manual_weights"]
            n_layers = len(list(model.model.layers))
            n_experts = existing.shape[-1]
            zeros = torch.zeros((n_layers, n_experts), dtype=torch.float32)
            model.update_moe_manual_args({"moe_manual_weights": zeros})
            logger.info("SteerMOE weights reset to zero (fallback path).")
        except Exception as e:
            raise RuntimeError(f"Could not reset SteerMOE weights: {e}") from e
    else:
        raise AttributeError(
            "The vLLM model does not expose `reset_moe_steering` or "
            "`update_moe_manual_args`."
        )


def _get_model(llm: Any):
    """Extract the underlying nn.Module from a vLLM LLM instance."""
    try:
        # vLLM V1 path
        return (
            llm.llm_engine
               .model_executor
               .driver_worker
               .model_runner
               .model
        )
    except AttributeError:
        pass
    try:
        # Alternate path used in some vLLM versions
        return llm.llm_engine.driver_worker.model_runner.model
    except AttributeError:
        pass
    raise AttributeError(
        "Could not locate the underlying model from the vLLM LLM instance. "
        "The internal vLLM API may have changed."
    )


# ---------------------------------------------------------------------------
# Public exports
# ---------------------------------------------------------------------------

__all__ = [
    # Hidden states (embed task)
    "get_all_hidden_states",
    "HiddenStatesCaptureV1",
    # Hidden states (generate task)
    "get_all_hidden_states_generate",
    "HiddenStatesCaptureGenerate",
    # MoE router logits (embed task)
    "get_moe_router_logits",
    "analyze_expert_usage",
    "MoERouterLogitsCaptureV1",
    # MoE router logits (generate task)
    "get_moe_router_logits_generate",
    "MoERouterLogitsCaptureGenerate",
    # SteerMOE application
    "apply_moe_steering_weights",
    "reset_moe_steering",
]

__version__ = "1.0.0"

"""
SteerMOE Technique — EasySteer implementation for gpt-oss.

Captures router logits via PyTorch forward hooks on MLPBlock,
bypassing the vllm-steer RPC capture system which does not recognise
gpt-oss MLPBlock as a MoE layer.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

class MoESteeringWeights:
    """
    Per-layer expert steering weights for MoE models.
    Shape: [num_layers, num_experts]
      +value  → boost expert   (router logit set to max + 0.01)
      -value  → suppress expert (router logit set to min - 0.01)
       0      → no-op
    """

    def __init__(self, weights: torch.Tensor, model_type: str = "unknown",
                 metadata: Optional[Dict] = None):
        assert weights.ndim == 2, "weights must be [num_layers, num_experts]"
        self.weights = weights
        self.model_type = model_type
        self.metadata = metadata or {}

    @property
    def num_layers(self): return self.weights.shape[0]

    @property
    def num_experts(self): return self.weights.shape[1]

    def save(self, path: str) -> None:
        torch.save({"weights": self.weights, "model_type": self.model_type,
                    "metadata": self.metadata}, path)
        logger.info(f"MoESteeringWeights saved to {path}")

    @classmethod
    def load(cls, path: str) -> "MoESteeringWeights":
        data = torch.load(path, map_location="cpu")
        return cls(weights=data["weights"], model_type=data.get("model_type", "unknown"),
                   metadata=data.get("metadata", {}))

    def __repr__(self):
        nnz = (self.weights != 0).sum().item()
        return (f"MoESteeringWeights(model={self.model_type}, "
                f"layers={self.num_layers}, experts={self.num_experts}, nonzero={nnz})")


# ---------------------------------------------------------------------------
# One-time setup: inject capture hooks into the worker process
# ---------------------------------------------------------------------------

def _setup_gpt_oss_capture(llm: Any) -> None:
    """
    Inject everything needed for router-logit capture via RPC + forward hooks.
    Safe to call multiple times (idempotent).
    """
    import vllm.v1.worker.gpu_worker as gw_module

    # ── Step 1: patch enable to store runner ref ───────────────────────────
    if not getattr(gw_module, '_gpt_oss_patched', False):
        original_enable = gw_module.Worker.enable_moe_router_logits_capture

        def enable_and_store(self):
            gw_module._RUNNER_REF = self.model_runner
            return original_enable(self)

        gw_module.Worker.enable_moe_router_logits_capture = enable_and_store

        def setup_gpt_oss_capture(self):
            from vllm.hidden_states import VLLMTransformerLayerWrapper
            runner = gw_module._RUNNER_REF
            if runner is None:
                return "ERROR: runner not set — call enable first"

            # Remove any existing hooks
            for h in getattr(runner, '_hook_handles', []):
                h.remove()
            runner._hook_handles = []

            store = runner.moe_router_logits_store
            m = runner.model
            layers = m.model.layers if hasattr(m, 'model') else m.layers

            hooked = 0
            for layer_idx, layer in enumerate(layers):
                real_layer = (layer.base_layer
                              if isinstance(layer, VLLMTransformerLayerWrapper)
                              else layer)
                mlp = real_layer._modules.get('mlp')
                if mlp is None or not hasattr(mlp, 'router'):
                    continue

                def make_hook(layer_id, st):
                    def hook(module, inp, output):
                        if not st.capture_enabled:
                            return
                        try:
                            x = inp[0]
                            g = module.router(x)
                            if isinstance(g, tuple):
                                g = g[0]
                            st.store_router_logits(
                                layer_id, g, f"layer_{layer_id}.mlp")
                        except Exception:
                            pass
                    return hook

                handle = mlp.register_forward_hook(
                    make_hook(layer_idx, store))
                runner._hook_handles.append(handle)
                hooked += 1

            runner._moe_wrapped = True
            return f"Registered {hooked} forward hooks"

        gw_module.Worker.setup_gpt_oss_capture = setup_gpt_oss_capture
        gw_module._gpt_oss_patched = True

    def _inject(model):
        return "ok"

    llm.llm_engine.apply_model(_inject)  # ensure worker process is alive

    # ── Step 2: enable (sets _RUNNER_REF) then setup hooks ────────────────
    llm.llm_engine.engine_core.collective_rpc(
        "enable_moe_router_logits_capture")
    result = llm.llm_engine.engine_core.collective_rpc(
        "setup_gpt_oss_capture")
    logger.info(f"gpt-oss capture setup: {result}")

    # ── Step 3: disable capture until we actually want to capture ──────────
    llm.llm_engine.engine_core.collective_rpc(
        "disable_moe_router_logits_capture")


def _capture_router_logits(llm: Any, texts: List[str]) -> Dict[int, torch.Tensor]:
    """
    Run a generate pass over `texts` and return captured router logits.
    Dict[layer_id -> Tensor(n_tokens, n_experts)]
    """
    from vllm import SamplingParams
    from vllm.hidden_states import deserialize_moe_router_logits

    llm.llm_engine.engine_core.collective_rpc(
        "enable_moe_router_logits_capture")
    try:
        llm.generate(texts, SamplingParams(max_tokens=1, temperature=0.0))
        results = llm.llm_engine.engine_core.collective_rpc(
            "get_moe_router_logits")
        return deserialize_moe_router_logits(results[0])
    finally:
        llm.llm_engine.engine_core.collective_rpc("clear_moe_router_logits")
        llm.llm_engine.engine_core.collective_rpc(
            "disable_moe_router_logits_capture")


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------

class SteerMOEExtractor:
    """
    Extracts MoE steering weights from positive/negative example sets
    using router logit activation frequency differences.
    """

    @staticmethod
    def extract(
        llm: Any,
        positive_texts: List[str],
        negative_texts: List[str],
        num_pos_experts: int = 10,
        num_neg_experts: int = 10,
        top_k: int = 2,
        reverse_effect: bool = False,
        use_generate: bool = False,   # kept for API compat, ignored (always generate)
        model_type: str = "unknown",
        **kwargs,
    ) -> MoESteeringWeights:
        """
        Extract SteerMOE weights.

        Parameters
        ----------
        llm : vLLM LLM instance (gpt-oss MoE model)
        positive_texts : texts expressing desired behaviour
        negative_texts : texts expressing undesired behaviour
        num_pos_experts : experts to boost
        num_neg_experts : experts to suppress
        top_k : router top-k (used for frequency normalisation)
        reverse_effect : flip steering direction
        model_type : informational string
        """
        # ── 0. One-time setup ──────────────────────────────────────────────
        _setup_gpt_oss_capture(llm)

        logger.info(
            f"SteerMOE: capturing router logits for "
            f"{len(positive_texts)} positive / {len(negative_texts)} negative texts"
        )

        # ── 1. Capture router logits ───────────────────────────────────────
        logger.info("Capturing positive texts...")
        pos_logits = _capture_router_logits(llm, positive_texts)

        logger.info("Capturing negative texts...")
        neg_logits = _capture_router_logits(llm, negative_texts)

        if not pos_logits:
            raise ValueError(
                "No router logits captured for positive texts. "
                "Run _setup_gpt_oss_capture(llm) manually and check for errors."
            )
        if not neg_logits:
            raise ValueError("No router logits captured for negative texts.")

        layer_ids = sorted(pos_logits.keys())
        n_layers = max(layer_ids) + 1
        n_experts = pos_logits[layer_ids[0]].shape[-1]

        logger.info(
            f"Captured {len(layer_ids)} layers, {n_experts} experts each")

        # ── 2. Compute per-layer per-expert activation frequency ───────────
        def expert_freq(logits_dict):
            freq = np.zeros((n_layers, n_experts), dtype=np.float32)
            for layer_id, logits in logits_dict.items():
                probs = torch.softmax(logits.float(), dim=-1)
                _, topk_ids = torch.topk(probs, k=min(top_k, probs.shape[-1]),
                                         dim=-1)
                n_tokens = logits.shape[0]
                counts = torch.bincount(
                    topk_ids.flatten(), minlength=n_experts).float()
                freq[layer_id] = (counts / max(n_tokens * top_k, 1)).cpu().numpy()
            return freq

        pos_freq = expert_freq(pos_logits)
        neg_freq = expert_freq(neg_logits)

        # ── 3. risk_diff ───────────────────────────────────────────────────
        risk_diff = pos_freq - neg_freq
        risk_diff_abs = np.abs(risk_diff)
        if reverse_effect:
            risk_diff = -risk_diff

        # ── 4. Select top experts ──────────────────────────────────────────
        flat_abs = risk_diff_abs.flatten()
        flat_diff = risk_diff.flatten()
        sorted_idx = np.argsort(flat_abs)[::-1]

        pos_selected = neg_selected = 0
        weights_np = np.zeros((n_layers, n_experts), dtype=np.float32)

        for idx in sorted_idx:
            if pos_selected >= num_pos_experts and neg_selected >= num_neg_experts:
                break
            layer = int(idx // n_experts)
            expert = int(idx % n_experts)
            diff = float(flat_diff[idx])
            if diff > 0 and pos_selected < num_pos_experts:
                weights_np[layer, expert] = diff
                pos_selected += 1
            elif diff < 0 and neg_selected < num_neg_experts:
                weights_np[layer, expert] = diff
                neg_selected += 1

        logger.info(
            f"Selected {pos_selected} positive, {neg_selected} negative experts")

        metadata = {
            "num_pos_experts": num_pos_experts,
            "num_neg_experts": num_neg_experts,
            "top_k": top_k,
            "reverse_effect": reverse_effect,
            "n_positive_texts": len(positive_texts),
            "n_negative_texts": len(negative_texts),
            "pos_selected": pos_selected,
            "neg_selected": neg_selected,
        }

        return MoESteeringWeights(
            weights=torch.from_numpy(weights_np),
            model_type=model_type,
            metadata=metadata,
        )

"""
SteerMOE Technique
Expert-routing-based steering for Mixture-of-Experts models.

Instead of adding vectors to hidden states, SteerMOE steers by biasing
router logits at inference time: boosting "positive" experts and suppressing
"negative" experts identified from activation pattern differences.

Reference: Adobe SteerMOE (src/utils.py:steer_moe)
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

class MoESteeringWeights:
    """
    Holds per-layer expert steering weights for MoE models.

    Shape: [num_layers, num_experts]
      +value  → boost that expert  (router logit pushed to max + 0.01)
      -value  → suppress that expert (router logit pushed to min - 0.01)
       0      → leave that expert untouched
    """

    def __init__(
        self,
        weights: torch.Tensor,          # [num_layers, num_experts]
        model_type: str = "unknown",
        metadata: Optional[Dict] = None,
    ):
        assert weights.ndim == 2, "weights must be [num_layers, num_experts]"
        self.weights = weights
        self.model_type = model_type
        self.metadata = metadata or {}

    @property
    def num_layers(self) -> int:
        return self.weights.shape[0]

    @property
    def num_experts(self) -> int:
        return self.weights.shape[1]

    def save(self, path: str) -> None:
        """Save to a .pt file."""
        torch.save(
            {
                "weights": self.weights,
                "model_type": self.model_type,
                "metadata": self.metadata,
            },
            path,
        )
        logger.info(f"MoESteeringWeights saved to {path}")

    @classmethod
    def load(cls, path: str) -> "MoESteeringWeights":
        """Load from a .pt file saved by :meth:`save`."""
        data = torch.load(path, map_location="cpu")
        return cls(
            weights=data["weights"],
            model_type=data.get("model_type", "unknown"),
            metadata=data.get("metadata", {}),
        )

    def __repr__(self) -> str:
        nnz = (self.weights != 0).sum().item()
        return (
            f"MoESteeringWeights(model={self.model_type}, "
            f"layers={self.num_layers}, experts={self.num_experts}, "
            f"nonzero={nnz})"
        )


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------

class SteerMOEExtractor:
    """
    Extracts MoE steering weights from positive/negative example sets.

    Pipeline
    --------
    1. Capture router logits for positive texts  →  per-layer expert activation freq
    2. Capture router logits for negative texts  →  per-layer expert activation freq
    3. Compute ``risk_diff = freq_pos - freq_neg`` per (layer, expert)
    4. Select top-N positive experts and top-N negative experts
    5. Return :class:`MoESteeringWeights` tensor

    The resulting weights are fed into ``update_moe_manual_args()`` on the
    vLLM model (see :func:`easysteer.hidden_states.apply_moe_steering_weights`).
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
        use_generate: bool = False,
        model_type: str = "unknown",
        **kwargs,
    ) -> MoESteeringWeights:
        """
        Extract MoE steering weights.

        Parameters
        ----------
        llm:
            A vLLM ``LLM`` instance (must be a MoE model with router logit
            capture support).
        positive_texts:
            Texts that should activate the desired behaviour.
        negative_texts:
            Texts that should activate the opposite/undesired behaviour.
        num_pos_experts:
            How many top positive-risk experts to boost.
        num_neg_experts:
            How many top negative-risk experts to suppress.
        top_k:
            Number of experts selected per token by the router (used when
            computing per-expert activation frequency).
        reverse_effect:
            If True, flip the sign of ``risk_diff`` before selection (i.e.
            steer in the opposite direction).
        use_generate:
            Use ``llm.generate()`` capture instead of ``llm.embed()``.
            Required for models that don't support the embed task.
        model_type:
            String identifier stored in the returned weights object.

        Returns
        -------
        MoESteeringWeights
        """
        import easysteer.hidden_states as hs

        # ------------------------------------------------------------------
        # 1. Capture router logits
        # ------------------------------------------------------------------
        logger.info(
            f"SteerMOE: capturing router logits for "
            f"{len(positive_texts)} positive / {len(negative_texts)} negative texts"
        )

        if use_generate:
            pos_logits, _ = hs.get_moe_router_logits_generate(
                llm, positive_texts, split_by_samples=False
            )
            neg_logits, _ = hs.get_moe_router_logits_generate(
                llm, negative_texts, split_by_samples=False
            )
        else:
            pos_logits, _ = hs.get_moe_router_logits(
                llm, positive_texts, split_by_samples=False
            )
            neg_logits, _ = hs.get_moe_router_logits(
                llm, negative_texts, split_by_samples=False
            )

        # pos_logits / neg_logits: Dict[layer_id, Tensor(n_tokens, n_experts)]

        layer_ids = sorted(pos_logits.keys())
        if not layer_ids:
            raise ValueError("No router logits captured. Is the model a MoE model?")

        n_layers = max(layer_ids) + 1
        n_experts = pos_logits[layer_ids[0]].shape[-1]

        logger.info(
            f"SteerMOE: captured {len(layer_ids)} MoE layers, "
            f"{n_experts} experts each"
        )

        # ------------------------------------------------------------------
        # 2. Compute per-layer per-expert activation frequency
        # ------------------------------------------------------------------
        def expert_freq(logits_dict: Dict[int, torch.Tensor]) -> np.ndarray:
            """freq[layer, expert] = fraction of tokens that activated that expert."""
            freq = np.zeros((n_layers, n_experts), dtype=np.float32)
            for layer_id, logits in logits_dict.items():
                # logits: (n_tokens, n_experts)
                probs = torch.softmax(logits.float(), dim=-1)
                _, topk_ids = torch.topk(probs, k=top_k, dim=-1)
                n_tokens = logits.shape[0]
                counts = torch.bincount(
                    topk_ids.flatten(), minlength=n_experts
                ).float()
                freq[layer_id] = (counts / (n_tokens * top_k)).cpu().numpy()
            return freq

        pos_freq = expert_freq(pos_logits)
        neg_freq = expert_freq(neg_logits)

        # ------------------------------------------------------------------
        # 3. risk_diff = freq_pos - freq_neg
        # ------------------------------------------------------------------
        risk_diff = pos_freq - neg_freq          # [n_layers, n_experts]
        risk_diff_abs = np.abs(risk_diff)

        if reverse_effect:
            risk_diff = -risk_diff

        # ------------------------------------------------------------------
        # 4. Select top experts
        # ------------------------------------------------------------------
        # Flatten to (layer, expert) pairs sorted by |risk_diff|
        flat_abs = risk_diff_abs.flatten()
        flat_diff = risk_diff.flatten()

        sorted_idx = np.argsort(flat_abs)[::-1]  # descending

        pos_selected = 0
        neg_selected = 0
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
            f"SteerMOE: selected {pos_selected} positive experts "
            f"and {neg_selected} negative experts"
        )

        if pos_selected < num_pos_experts or neg_selected < num_neg_experts:
            logger.warning(
                f"SteerMOE: could only find {pos_selected}/{num_pos_experts} "
                f"positive and {neg_selected}/{num_neg_experts} negative experts. "
                f"Consider using more texts or reducing num_*_experts."
            )

        # ------------------------------------------------------------------
        # 5. Build and return
        # ------------------------------------------------------------------
        weights_tensor = torch.from_numpy(weights_np)

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
            weights=weights_tensor,
            model_type=model_type,
            metadata=metadata,
        )

"""
Unified Interface for Steering Methods
统一的控制向量提取方法接口
"""

from typing import Any, List, Optional

from .utils import StatisticalControlVector
from .diffmean import DiffMeanExtractor
from .pca import PCAExtractor
from .lat import LATExtractor
from .linear_probe import LinearProbeExtractor
from .steer_moe import SteerMOEExtractor, MoESteeringWeights


def extract_statistical_control_vector(
    method: str,
    all_hidden_states,
    positive_indices,
    negative_indices=None,
    **kwargs
) -> StatisticalControlVector:
    """
    统一的控制向量提取接口

    Args:
        method: 方法名称，支持的方法：
               "diffmean", "pca", "lat", "linear_probe"
        all_hidden_states: 三维列表 [样本][layer][token]
        positive_indices: 正样本的索引列表
        negative_indices: 负样本的索引列表
        **kwargs: 方法特定的参数

    Returns:
        StatisticalControlVector: 提取的控制向量
    """
    method_map = {
        "diffmean": DiffMeanExtractor,
        "pca": PCAExtractor,
        "lat": LATExtractor,
        "linear_probe": LinearProbeExtractor,
    }

    if method not in method_map:
        supported_methods = list(method_map.keys())
        raise ValueError(f"不支持的方法: {method}。支持的方法: {supported_methods}")

    extractor_class = method_map[method]
    return extractor_class.extract(
        all_hidden_states=all_hidden_states,
        positive_indices=positive_indices,
        negative_indices=negative_indices,
        **kwargs
    )


def extract_diffmean_control_vector(all_hidden_states, positive_indices, negative_indices=None, **kwargs):
    """提取DiffMean控制向量"""
    return DiffMeanExtractor.extract(all_hidden_states, positive_indices, negative_indices, **kwargs)


def extract_pca_control_vector(all_hidden_states, positive_indices, negative_indices=None, **kwargs):
    """提取PCA控制向量"""
    return PCAExtractor.extract(all_hidden_states, positive_indices, negative_indices, **kwargs)


def extract_lat_control_vector(all_hidden_states, positive_indices, negative_indices=None, **kwargs):
    """提取LAT控制向量"""
    return LATExtractor.extract(all_hidden_states, positive_indices, negative_indices, **kwargs)


def extract_linear_probe_control_vector(all_hidden_states, positive_indices, negative_indices=None, **kwargs):
    """提取LinearProbe控制向量"""
    return LinearProbeExtractor.extract(all_hidden_states, positive_indices, negative_indices, **kwargs)


def extract_steer_moe_weights(
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
    Extract SteerMOE weights from positive/negative text examples.

    This is the MoE equivalent of the hidden-state extraction functions above.
    Instead of producing a per-layer hidden-state direction vector, it produces
    a [num_layers, num_experts] weight tensor that biases router logits at
    inference time.

    Parameters
    ----------
    llm:
        A running vLLM LLM instance (MoE model with router logit capture).
    positive_texts:
        Texts that express the desired behaviour / concept.
    negative_texts:
        Texts that express the opposite / undesired behaviour.
    num_pos_experts:
        Number of top positive-risk experts to boost.
    num_neg_experts:
        Number of top negative-risk experts to suppress.
    top_k:
        Router top-k (must match the model's routing config; typically 2).
    reverse_effect:
        Flip the steering direction.
    use_generate:
        Use generate-task router logit capture instead of embed-task.
        Set True for models that do not support task="embed".
    model_type:
        Informational string stored in the result.

    Returns
    -------
    MoESteeringWeights
        Call :func:`easysteer.hidden_states.apply_moe_steering_weights` to
        push these weights into a running vLLM instance before generation.

    Example
    -------
    >>> import easysteer.steer as steer
    >>> import easysteer.hidden_states as hs
    >>> from vllm import LLM
    >>>
    >>> llm = LLM(model="path/to/gpt-oss-moe", task="embed")
    >>> weights = steer.extract_steer_moe_weights(
    ...     llm,
    ...     positive_texts=["I feel great!", "Everything is wonderful."],
    ...     negative_texts=["I feel terrible.", "Everything is awful."],
    ...     num_pos_experts=10,
    ...     num_neg_experts=10,
    ... )
    >>> weights.save("steer_moe_emotion.pt")
    >>>
    >>> # Apply before generation:
    >>> hs.apply_moe_steering_weights(llm, weights.weights)
    """
    return SteerMOEExtractor.extract(
        llm=llm,
        positive_texts=positive_texts,
        negative_texts=negative_texts,
        num_pos_experts=num_pos_experts,
        num_neg_experts=num_neg_experts,
        top_k=top_k,
        reverse_effect=reverse_effect,
        use_generate=use_generate,
        model_type=model_type,
        **kwargs,
    )

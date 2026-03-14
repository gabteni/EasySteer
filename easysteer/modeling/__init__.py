"""
EasySteer custom vLLM model registrations.

Contains SteerMOE-patched variants of upstream vLLM models.
Call the appropriate ``register_*`` helper before creating your vLLM LLM
instance.

Example
-------
    from easysteer.modeling import register_gpt_oss
    register_gpt_oss()

    from vllm import LLM
    llm = LLM(model="path/to/gpt-oss-moe", ...)
"""

from .gpt_oss import GptOssForCausalLM


def register_gpt_oss() -> None:
    """
    Register the SteerMOE-enabled GptOss model with vLLM's ModelRegistry.

    Must be called **before** constructing a vLLM ``LLM`` instance.

    Example
    -------
        from easysteer.modeling import register_gpt_oss
        register_gpt_oss()

        from vllm import LLM
        llm = LLM(model="path/to/gpt-oss-moe", task="embed")
    """
    from vllm import ModelRegistry
    ModelRegistry.register_model(
        "GptOssForCausalLM",
        "easysteer.modeling.gpt_oss:GptOssForCausalLM",
    )


__all__ = [
    "GptOssForCausalLM",
    "register_gpt_oss",
]

"""
Model registry and configuration for control_vectors_multi.

Defines supported models with chat templates, layer configurations, and metadata.
"""

from dataclasses import dataclass
from typing import Tuple, List, Dict
from enum import Enum


class ChatTemplateType(Enum):
    """Supported chat template formats."""
    CHATML = "chatml"       # Qwen-style: <|im_start|>, <|im_end|>
    OLMO = "olmo"           # OLMo-style: <|user|>, <|assistant|>


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for a supported model."""
    model_id: str                    # HuggingFace model ID
    name: str                        # Short display name
    num_layers: int                  # Total transformer layers
    template_type: ChatTemplateType  # Chat template format
    user_tag: str                    # Opening tag for user message
    asst_tag: str                    # Closing/assistant tag
    vram_gb: float                   # Approximate VRAM requirement (float16)
    recommended_layers: Tuple[float, float]  # (start_pct, end_pct) for layer selection
    notes: str = ""                  # Any special notes


# ============================================================================
# MODEL REGISTRY
# ============================================================================

MODEL_REGISTRY: Dict[str, ModelConfig] = {
    # Qwen2-1.5B-Instruct
    "qwen2-1.5b": ModelConfig(
        model_id="Qwen/Qwen2-1.5B-Instruct",
        name="Qwen2-1.5B",
        num_layers=28,
        template_type=ChatTemplateType.CHATML,
        user_tag="<|im_start|>user\n",
        asst_tag="<|im_end|>\n<|im_start|>assistant\n",
        vram_gb=3.0,
        recommended_layers=(0.45, 0.85),
        notes="Small, fast model ideal for experimentation",
    ),

    # DeepSeek-R1-Distill-Qwen-1.5B
    "deepseek-r1-1.5b": ModelConfig(
        model_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        name="DeepSeek-R1-1.5B",
        num_layers=28,
        template_type=ChatTemplateType.CHATML,
        user_tag="<|im_start|>user\n",
        asst_tag="<|im_end|>\n<|im_start|>assistant\n",
        vram_gb=3.0,
        recommended_layers=(0.45, 0.85),
        notes="DeepSeek R1 distilled into Qwen architecture",
    ),

    # OLMo-2-1124-7B-Instruct
    "olmo2-7b": ModelConfig(
        model_id="allenai/OLMo-2-1124-7B-Instruct",
        name="OLMo2-7B",
        num_layers=32,
        template_type=ChatTemplateType.OLMO,
        user_tag="<|endoftext|><|user|>\n",
        asst_tag="<|assistant|>\n",
        vram_gb=14.0,
        recommended_layers=(0.45, 0.85),
        notes="AI2's OLMo 2 model",
    ),
}

# Alias mappings for convenience
MODEL_ALIASES: Dict[str, str] = {
    "qwen": "qwen2-1.5b",
    "qwen2": "qwen2-1.5b",
    "deepseek": "deepseek-r1-1.5b",
    "deepseek-r1": "deepseek-r1-1.5b",
    "olmo": "olmo2-7b",
    "olmo2": "olmo2-7b",
}


def get_model_config(model_key: str) -> ModelConfig:
    """
    Get model configuration by key or alias.

    Args:
        model_key: Registry key or alias (e.g., "qwen2-1.5b" or "qwen")

    Returns:
        ModelConfig for the requested model

    Raises:
        ValueError: If model not found
    """
    key = model_key.lower()

    # Check aliases first
    if key in MODEL_ALIASES:
        key = MODEL_ALIASES[key]

    if key not in MODEL_REGISTRY:
        available = list(MODEL_REGISTRY.keys()) + list(MODEL_ALIASES.keys())
        raise ValueError(
            f"Unknown model '{model_key}'. Available: {', '.join(sorted(set(available)))}"
        )

    return MODEL_REGISTRY[key]


def get_recommended_layers(config: ModelConfig) -> List[int]:
    """
    Get list of recommended layer indices for a model.

    Args:
        config: ModelConfig instance

    Returns:
        List of layer indices to target for control vector training
    """
    start_pct, end_pct = config.recommended_layers
    start_layer = int(config.num_layers * start_pct)
    end_layer = int(config.num_layers * end_pct)
    return list(range(start_layer, end_layer))


def list_models() -> List[str]:
    """List all available model keys."""
    return list(MODEL_REGISTRY.keys())


def print_model_info() -> None:
    """Print a formatted table of available models."""
    print("\nAvailable Models:")
    print("-" * 80)
    print(f"{'Key':<18} {'Name':<20} {'Layers':<8} {'VRAM':<8} {'Template'}")
    print("-" * 80)

    for key, config in MODEL_REGISTRY.items():
        print(
            f"{key:<18} {config.name:<20} {config.num_layers:<8} "
            f"{config.vram_gb:<8.1f} {config.template_type.value}"
        )

    print("\nAliases:", ", ".join(f"{k}->{v}" for k, v in MODEL_ALIASES.items()))

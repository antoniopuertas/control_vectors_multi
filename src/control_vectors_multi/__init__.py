"""
Control Vectors Multi - Control vector training for multiple HuggingFace models.

Supported models:
- Qwen/Qwen2-1.5B-Instruct
- deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
- allenai/OLMo-2-1124-7B-Instruct
"""

from .models import (
    ModelConfig,
    ChatTemplateType,
    MODEL_REGISTRY,
    MODEL_ALIASES,
    get_model_config,
    get_recommended_layers,
    list_models,
    print_model_info,
)

from .dataset import (
    PersonaPair,
    PERSONA_PAIRS,
    DEFAULT_SUFFIXES,
    format_prompt,
    make_dataset,
    make_dataset_from_concept,
    list_concepts,
)

from .train import (
    load_model_and_tokenizer,
    train_control_vector,
)

from .test import (
    TestResult,
    check_coherence,
    test_vector,
    find_optimal_coefficient,
    print_comparison,
)

from .apply import (
    load_vector,
    create_controlled_model,
    generate_with_control,
)

__version__ = "0.1.0"

__all__ = [
    # Models
    "ModelConfig",
    "ChatTemplateType",
    "MODEL_REGISTRY",
    "MODEL_ALIASES",
    "get_model_config",
    "get_recommended_layers",
    "list_models",
    "print_model_info",
    # Dataset
    "PersonaPair",
    "PERSONA_PAIRS",
    "DEFAULT_SUFFIXES",
    "format_prompt",
    "make_dataset",
    "make_dataset_from_concept",
    "list_concepts",
    # Train
    "load_model_and_tokenizer",
    "train_control_vector",
    # Test
    "TestResult",
    "check_coherence",
    "test_vector",
    "find_optimal_coefficient",
    "print_comparison",
    # Apply
    "load_vector",
    "create_controlled_model",
    "generate_with_control",
]

"""
Utilities for applying trained control vectors.
"""

import torch
from pathlib import Path
from typing import Union, Optional, Any, Tuple
from repeng import ControlVector, ControlModel

from .models import get_model_config, get_recommended_layers, ModelConfig
from .train import load_model_and_tokenizer


def load_vector(path: Union[str, Path]) -> ControlVector:
    """Load a saved control vector."""
    return torch.load(path)


def create_controlled_model(
    model_key: str,
    vector_path: Union[str, Path],
    coefficient: float = 1.0,
    device: str = "auto",
) -> Tuple[ControlModel, Any, ControlVector, ModelConfig]:
    """
    Create a model with a control vector applied.

    Args:
        model_key: Registry key for the model
        vector_path: Path to saved .pt vector file
        coefficient: Control strength (positive or negative)
        device: Device to use

    Returns:
        Tuple of (control_model, tokenizer, vector, config)
    """
    model, tokenizer, config = load_model_and_tokenizer(model_key, device)
    layers = get_recommended_layers(config)

    control_model = ControlModel(model, layers)
    vector = load_vector(vector_path)

    if coefficient != 0:
        control_model.set_control(vector, coefficient)

    return control_model, tokenizer, vector, config


def generate_with_control(
    model: Any,  # ControlModel
    tokenizer: Any,
    prompt: str,
    vector: Optional[ControlVector] = None,
    coefficient: float = 0.0,
    max_new_tokens: int = 150,
    **generate_kwargs: Any,
) -> str:
    """
    Generate text with optional control vector.

    Args:
        model: ControlModel instance
        tokenizer: Tokenizer
        prompt: Formatted prompt string
        vector: Optional ControlVector to apply
        coefficient: Control strength
        max_new_tokens: Max tokens to generate
        **generate_kwargs: Additional args for model.generate()

    Returns:
        Generated text
    """
    if vector and coefficient != 0:
        model.set_control(vector, coefficient)
    else:
        model.reset()

    input_ids = tokenizer(prompt, return_tensors="pt").to(model.device)

    defaults = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "pad_token_id": tokenizer.eos_token_id,
    }
    defaults.update(generate_kwargs)

    with torch.no_grad():
        output = model.generate(**input_ids, **defaults)

    result: str = tokenizer.decode(output[0], skip_special_tokens=True)
    return result

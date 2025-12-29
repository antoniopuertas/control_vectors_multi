"""
Training logic for control vectors.
"""

import torch
from pathlib import Path
from typing import List, Optional, Union, Tuple, Any
from transformers import AutoModelForCausalLM, AutoTokenizer
from repeng import ControlVector, ControlModel

from .models import get_model_config, get_recommended_layers, ModelConfig
from .dataset import make_dataset_from_concept


def load_model_and_tokenizer(
    model_key: str,
    device: str = "auto",
    torch_dtype: Optional[torch.dtype] = None,
) -> Tuple[Any, Any, ModelConfig]:
    """
    Load model and tokenizer for a registered model.

    Args:
        model_key: Registry key (e.g., "qwen2-1.5b")
        device: Device map ("auto", "cuda", "cpu")
        torch_dtype: Torch data type for model weights (default: float16 for GPU, float32 for CPU)

    Returns:
        Tuple of (model, tokenizer, config)
    """
    config = get_model_config(model_key)

    print(f"Loading model: {config.model_id}")
    print(f"  Layers: {config.num_layers}, VRAM: ~{config.vram_gb}GB")

    tokenizer = AutoTokenizer.from_pretrained(config.model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id or 0

    # Determine dtype
    if torch_dtype is None:
        if device == "cpu":
            torch_dtype = torch.float32
        else:
            torch_dtype = torch.float16

    # Handle device mapping
    if device == "cpu":
        model = AutoModelForCausalLM.from_pretrained(
            config.model_id,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )
        model = model.to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            config.model_id,
            torch_dtype=torch_dtype,
            device_map=device,
            trust_remote_code=True,
        )

    return model, tokenizer, config


def train_control_vector(
    model_key: str,
    concept: str,
    output_path: Optional[Union[str, Path]] = None,
    layers: Optional[List[int]] = None,
    device: str = "auto",
    export_gguf: bool = False,
) -> ControlVector:
    """
    Train a control vector for a given concept.

    Args:
        model_key: Registry key (e.g., "qwen2-1.5b", "deepseek", "olmo")
        concept: Concept name from PERSONA_PAIRS
        output_path: Optional path to save the vector (.pt or .gguf)
        layers: Optional explicit layer list, otherwise uses recommendations
        device: Device to use
        export_gguf: Whether to export as GGUF for llama.cpp

    Returns:
        Trained ControlVector
    """
    # Load model
    model, tokenizer, config = load_model_and_tokenizer(model_key, device)

    # Determine layers
    if layers is None:
        layers = get_recommended_layers(config)

    print(f"Targeting layers: {layers[0]} to {layers[-1]} ({len(layers)} layers)")

    # Wrap with ControlModel
    control_model = ControlModel(model, layers)

    # Create dataset
    print(f"Creating dataset for concept: {concept}")
    dataset = make_dataset_from_concept(concept, model_key)
    print(f"Dataset size: {len(dataset)} entries")

    # Train
    print("Training control vector...")
    control_model.reset()
    vector = ControlVector.train(control_model, tokenizer, dataset)
    print("Training complete!")

    # Save
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if export_gguf:
            gguf_path = output_path.with_suffix(".gguf")
            vector.export_gguf(str(gguf_path))
            print(f"Exported GGUF to: {gguf_path}")
        else:
            torch.save(vector, output_path)
            print(f"Saved vector to: {output_path}")

    return vector

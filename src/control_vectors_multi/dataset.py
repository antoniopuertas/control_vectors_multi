"""
Dataset creation utilities for control vector training.
"""

from dataclasses import dataclass
from typing import List, Optional
from repeng import DatasetEntry

from .models import get_model_config, ModelConfig


@dataclass
class PersonaPair:
    """A pair of contrasting personas for training."""
    positive: str
    negative: str


# ============================================================================
# PERSONA PAIRS (Predefined concepts)
# ============================================================================

PERSONA_PAIRS = {
    "honesty": PersonaPair(
        positive="honest, truthful, sincere",
        negative="deceptive, dishonest, lying"
    ),
    "creativity": PersonaPair(
        positive="creative, imaginative, innovative",
        negative="uncreative, predictable, conventional"
    ),
    "confidence": PersonaPair(
        positive="confident, assertive, self-assured",
        negative="uncertain, hesitant, doubtful"
    ),
    "helpfulness": PersonaPair(
        positive="helpful, supportive, eager to assist",
        negative="unhelpful, dismissive, reluctant"
    ),
    "formality": PersonaPair(
        positive="formal, professional, polished",
        negative="casual, informal, relaxed"
    ),
    "verbosity": PersonaPair(
        positive="verbose, detailed, thorough",
        negative="concise, brief, terse"
    ),
    "enthusiasm": PersonaPair(
        positive="enthusiastic, excited, energetic",
        negative="apathetic, bored, unenthusiastic"
    ),
    "empathy": PersonaPair(
        positive="empathetic, caring, compassionate",
        negative="cold, detached, unsympathetic"
    ),
}


# Default suffixes for training
DEFAULT_SUFFIXES = [
    "I think",
    "The answer is",
    "In my opinion,",
    "To be clear,",
    "What really happened was",
    "Let me explain:",
    "The truth is",
    "I believe",
    "From my perspective,",
    "Actually,",
    "Here's what I know:",
    "I would say",
    "The situation is",
    "My understanding is",
    "Based on my experience,",
]


def format_prompt(text: str, config: ModelConfig) -> str:
    """
    Format a prompt using the model's chat template.

    Args:
        text: User message content
        config: ModelConfig for the target model

    Returns:
        Formatted prompt string
    """
    return f"{config.user_tag}{text}{config.asst_tag}"


def make_dataset(
    template: str,
    positive_personas: List[str],
    negative_personas: List[str],
    suffixes: List[str],
    config: ModelConfig,
) -> List[DatasetEntry]:
    """
    Create a dataset of contrastive pairs for control vector training.

    Args:
        template: Template string with {persona} placeholder
        positive_personas: List of positive persona descriptions
        negative_personas: List of negative persona descriptions
        suffixes: List of output suffixes/continuations
        config: ModelConfig for chat template formatting

    Returns:
        List of DatasetEntry objects for training
    """
    dataset = []

    for suffix in suffixes:
        for pos, neg in zip(positive_personas, negative_personas):
            positive_template = template.format(persona=pos)
            negative_template = template.format(persona=neg)

            dataset.append(
                DatasetEntry(
                    positive=f"{config.user_tag}{positive_template}{config.asst_tag}{suffix}",
                    negative=f"{config.user_tag}{negative_template}{config.asst_tag}{suffix}",
                )
            )

    return dataset


def make_dataset_from_concept(
    concept: str,
    model_key: str,
    suffixes: Optional[List[str]] = None,
    templates: Optional[List[str]] = None,
) -> List[DatasetEntry]:
    """
    Create a dataset from a predefined concept name.

    Args:
        concept: One of the keys in PERSONA_PAIRS
        model_key: Model registry key (e.g., "qwen2-1.5b")
        suffixes: Optional custom suffixes, defaults to DEFAULT_SUFFIXES
        templates: Optional custom templates

    Returns:
        List of DatasetEntry objects for training
    """
    if concept not in PERSONA_PAIRS:
        available = ", ".join(PERSONA_PAIRS.keys())
        raise ValueError(f"Unknown concept '{concept}'. Available: {available}")

    config = get_model_config(model_key)
    persona_pair = PERSONA_PAIRS[concept]
    suffixes = suffixes or DEFAULT_SUFFIXES

    templates = templates or [
        "Pretend you're a {persona} person making statements about the world.",
        "Act as someone who is {persona} when responding.",
        "You are a {persona} individual. Speak accordingly.",
    ]

    dataset = []
    for template in templates:
        dataset.extend(
            make_dataset(
                template=template,
                positive_personas=[persona_pair.positive],
                negative_personas=[persona_pair.negative],
                suffixes=suffixes,
                config=config,
            )
        )

    return dataset


def list_concepts() -> List[str]:
    """List available concept names."""
    return list(PERSONA_PAIRS.keys())

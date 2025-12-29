"""
Testing and validation for control vectors.
"""

import torch
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any


@dataclass
class TestResult:
    """Result from a single test."""
    prompt: str
    coefficient: float
    output: str
    coherence_score: float
    is_coherent: bool


def check_coherence(text: str) -> Tuple[float, List[str]]:
    """
    Check text for signs of degradation/incoherence.

    Returns:
        Tuple of (score 0-1, list of warning messages)
    """
    warnings = []
    score = 1.0
    words = text.split()

    # Word repetition check
    if len(words) > 10:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.3:
            warnings.append(f"High word repetition (unique: {unique_ratio:.2f})")
            score -= 0.3
        elif unique_ratio < 0.5:
            warnings.append(f"Moderate word repetition (unique: {unique_ratio:.2f})")
            score -= 0.15

    # Character repetition
    for char in "abcdefghijklmnopqrstuvwxyz.!?":
        if char * 8 in text.lower():
            warnings.append(f"Repeated character: '{char}'")
            score -= 0.2
            break

    # Truncation check
    if len(text) > 50 and not any(text.rstrip().endswith(p) for p in ".!?\"')"):
        warnings.append("Text may be truncated")
        score -= 0.1

    # Short output
    if len(words) < 5 and len(text) < 30:
        warnings.append("Very short output")
        score -= 0.1

    return max(0, score), warnings


def test_vector(
    model: Any,  # ControlModel
    tokenizer: Any,
    vector: Any,  # ControlVector
    test_prompts: List[str],
    coefficients: Optional[List[float]] = None,
    max_new_tokens: int = 150,
) -> Dict[str, Dict[float, TestResult]]:
    """
    Run A/B comparison tests on a control vector.

    Args:
        model: ControlModel instance
        tokenizer: Tokenizer
        vector: Trained ControlVector
        test_prompts: List of formatted prompts to test
        coefficients: Coefficients to test (default: [-2, -1, 0, 1, 2])
        max_new_tokens: Max tokens to generate

    Returns:
        Nested dict: prompt -> coefficient -> TestResult
    """
    coefficients = coefficients or [-2.0, -1.0, 0.0, 1.0, 2.0]
    results: Dict[str, Dict[float, TestResult]] = {}

    gen_settings = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "pad_token_id": tokenizer.eos_token_id,
    }

    for prompt in test_prompts:
        results[prompt] = {}
        input_ids = tokenizer(prompt, return_tensors="pt").to(model.device)

        for coeff in coefficients:
            if coeff == 0:
                model.reset()
            else:
                model.set_control(vector, coeff)

            with torch.no_grad():
                output = model.generate(**input_ids, **gen_settings)

            text = tokenizer.decode(output[0], skip_special_tokens=True)
            coherence_score, _ = check_coherence(text)

            results[prompt][coeff] = TestResult(
                prompt=prompt,
                coefficient=coeff,
                output=text,
                coherence_score=coherence_score,
                is_coherent=coherence_score > 0.7,
            )

        model.reset()

    return results


def find_optimal_coefficient(
    model: Any,
    tokenizer: Any,
    vector: Any,
    test_prompt: str,
    coeff_range: Tuple[float, float] = (-3.0, 3.0),
    steps: int = 13,
) -> Tuple[float, float, Dict[float, Dict[str, Any]]]:
    """
    Find optimal coefficient by sweeping a range.

    Returns:
        Tuple of (best_positive_coeff, best_negative_coeff, all_results)
    """
    import numpy as np

    coefficients = np.linspace(coeff_range[0], coeff_range[1], steps)
    results: Dict[float, Dict[str, Any]] = {}

    input_ids = tokenizer(test_prompt, return_tensors="pt").to(model.device)

    for coeff in coefficients:
        coeff = float(coeff)

        if abs(coeff) < 0.01:
            model.reset()
        else:
            model.set_control(vector, coeff)

        with torch.no_grad():
            output = model.generate(
                **input_ids,
                max_new_tokens=100,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        text = tokenizer.decode(output[0], skip_special_tokens=True)
        coherence_score, _ = check_coherence(text)
        results[coeff] = {"text": text, "coherence": coherence_score}

    model.reset()

    # Find best coherent coefficients
    positive = [(c, r) for c, r in results.items() if c > 0 and r["coherence"] > 0.7]
    negative = [(c, r) for c, r in results.items() if c < 0 and r["coherence"] > 0.7]

    best_positive = max(positive, key=lambda x: x[0])[0] if positive else 1.0
    best_negative = min(negative, key=lambda x: x[0])[0] if negative else -1.0

    return best_positive, best_negative, results


def print_comparison(results: Dict[str, Dict[float, TestResult]]) -> None:
    """Pretty print test results."""
    for prompt, coeff_results in results.items():
        print(f"\n{'='*70}")
        print(f"Prompt: {prompt[:60]}...")
        print("=" * 70)

        for coeff, result in sorted(coeff_results.items()):
            status = "OK" if result.is_coherent else "WARN"
            print(f"\n[Coeff {coeff:+.1f}] ({status}, coherence: {result.coherence_score:.2f})")
            print("-" * 40)

            response = result.output
            if prompt in response:
                response = response.split(prompt)[-1]

            if len(response) > 300:
                response = response[:300] + "..."

            print(response.strip())

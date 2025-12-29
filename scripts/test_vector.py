#!/usr/bin/env python3
"""
CLI script to test a trained control vector.

Usage:
    python scripts/test_vector.py --model qwen --vector vectors/qwen_honesty.pt
    python scripts/test_vector.py --model deepseek --vector vectors/deepseek_creativity.pt --prompt "Why did you do that?"
"""

import argparse
import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from repeng import ControlModel

from control_vectors_multi.models import list_models, get_recommended_layers
from control_vectors_multi.train import load_model_and_tokenizer
from control_vectors_multi.apply import load_vector
from control_vectors_multi.test import test_vector, print_comparison, find_optimal_coefficient


def main() -> None:
    parser = argparse.ArgumentParser(description="Test a trained control vector")
    parser.add_argument(
        "--model", "-m",
        type=str,
        required=True,
        help=f"Model key: {', '.join(list_models())}"
    )
    parser.add_argument(
        "--vector", "-v",
        type=str,
        required=True,
        help="Path to vector file"
    )
    parser.add_argument(
        "--prompt", "-p",
        type=str,
        default=None,
        help="Custom test prompt"
    )
    parser.add_argument(
        "--find-optimal",
        action="store_true",
        help="Run coefficient sweep to find optimal values"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (auto, cuda, cpu)"
    )

    args = parser.parse_args()

    # Load model
    print("Loading model...")
    model, tokenizer, config = load_model_and_tokenizer(args.model, args.device)
    layers = get_recommended_layers(config)
    control_model = ControlModel(model, layers)

    # Load vector
    print(f"Loading vector from: {args.vector}")
    vector = load_vector(args.vector)

    # Default test prompts
    if args.prompt:
        prompts = [f"{config.user_tag}{args.prompt}{config.asst_tag}"]
    else:
        prompts = [
            f"{config.user_tag}Did you break the vase?{config.asst_tag}",
            f"{config.user_tag}Why were you late today?{config.asst_tag}",
        ]

    # Run tests
    print("\nRunning comparison tests...")
    results = test_vector(control_model, tokenizer, vector, prompts)
    print_comparison(results)

    # Optional coefficient sweep
    if args.find_optimal:
        print("\n" + "=" * 70)
        print("Finding optimal coefficient...")
        best_pos, best_neg, _ = find_optimal_coefficient(
            control_model, tokenizer, vector, prompts[0]
        )
        print(f"\nBest positive coefficient: {best_pos}")
        print(f"Best negative coefficient: {best_neg}")


if __name__ == "__main__":
    main()

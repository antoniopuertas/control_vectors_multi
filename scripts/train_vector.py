#!/usr/bin/env python3
"""
CLI script to train a control vector.

Usage:
    python scripts/train_vector.py --model qwen --concept honesty
    python scripts/train_vector.py --model deepseek --concept creativity --output vectors/creativity.pt
    python scripts/train_vector.py --model olmo --concept confidence --gguf
"""

import argparse
import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from control_vectors_multi.models import list_models, print_model_info
from control_vectors_multi.dataset import list_concepts
from control_vectors_multi.train import train_control_vector


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a control vector for a specific model and concept"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        required=False,
        help=f"Model key: {', '.join(list_models())}"
    )
    parser.add_argument(
        "--concept", "-c",
        type=str,
        required=False,
        help=f"Concept: {', '.join(list_concepts())}"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output path (default: vectors/{model}_{concept}.pt)"
    )
    parser.add_argument(
        "--gguf",
        action="store_true",
        help="Export as GGUF for llama.cpp"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (auto, cuda, cpu)"
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models and exit"
    )
    parser.add_argument(
        "--list-concepts",
        action="store_true",
        help="List available concepts and exit"
    )

    args = parser.parse_args()

    if args.list_models:
        print_model_info()
        return

    if args.list_concepts:
        print("\nAvailable concepts:")
        for concept in list_concepts():
            print(f"  - {concept}")
        return

    if not args.model or not args.concept:
        parser.print_help()
        print("\nError: --model and --concept are required")
        sys.exit(1)

    output_path = args.output or f"vectors/{args.model}_{args.concept}.pt"

    train_control_vector(
        model_key=args.model,
        concept=args.concept,
        output_path=output_path,
        device=args.device,
        export_gguf=args.gguf,
    )

    print(f"\nDone! Vector saved to: {output_path}")


if __name__ == "__main__":
    main()

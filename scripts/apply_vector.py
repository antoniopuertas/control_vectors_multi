#!/usr/bin/env python3
"""
Interactive CLI for applying control vectors.

Usage:
    python scripts/apply_vector.py --model qwen --vector vectors/qwen_honesty.pt
    python scripts/apply_vector.py --model deepseek --vector vectors/deepseek_creativity.pt --coefficient 1.5
"""

import argparse
import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from control_vectors_multi.models import list_models
from control_vectors_multi.apply import create_controlled_model, generate_with_control


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive control vector demo")
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
        "--coefficient", "-c",
        type=float,
        default=2.0,
        help="Default coefficient (default: 2.0)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (auto, cuda, cpu)"
    )

    args = parser.parse_args()

    print("Loading model and vector...")
    control_model, tokenizer, vector, config = create_controlled_model(
        args.model, args.vector, coefficient=0, device=args.device
    )

    print(f"\nLoaded {config.name} with control vector")
    print(f"Default coefficient: {args.coefficient}")
    print("\nFormat: [coefficient] your prompt")
    print("Example: 2.0 Why did you miss the meeting?")
    print("Example: -1.5 Did you eat my lunch?")
    print("Type 'quit' to exit\n")

    while True:
        try:
            user_input = input("> ").strip()
            if user_input.lower() in ("quit", "exit", "q"):
                break

            if not user_input:
                continue

            # Parse coefficient if provided
            parts = user_input.split(" ", 1)
            try:
                coeff = float(parts[0])
                text = parts[1] if len(parts) > 1 else ""
            except ValueError:
                coeff = args.coefficient
                text = user_input

            if not text:
                print("Please enter a prompt.")
                continue

            prompt = f"{config.user_tag}{text}{config.asst_tag}"

            response = generate_with_control(
                control_model, tokenizer, prompt, vector, coeff
            )

            # Extract response (remove prompt echo if present)
            if config.asst_tag in response:
                response = response.split(config.asst_tag)[-1]

            print(f"\n[Coefficient: {coeff}]")
            print(response.strip())
            print()

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

    print("\nDone!")


if __name__ == "__main__":
    main()

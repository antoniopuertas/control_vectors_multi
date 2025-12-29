# Control Vectors Multi

A toolkit for training and applying **control vectors** to steer language model behavior using **representation engineering**.

## What Are Control Vectors?

Control vectors are directions in a model's activation space that correspond to high-level concepts like honesty, creativity, or confidence. By adding or subtracting these vectors during inference, you can steer the model's behavior without fine-tuning.

### How It Works

1. **Contrastive Prompts**: We create pairs of prompts that differ only in the concept we want to capture (e.g., "Act as an honest assistant" vs "Act as a deceptive assistant")

2. **Activation Extraction**: We run both prompts through the model and extract the hidden states (activations) from intermediate layers

3. **PCA on Differences**: We compute the difference between positive and negative activations and use Principal Component Analysis to find the primary direction that separates them

4. **Steering at Inference**: During generation, we add (or subtract) scaled versions of this direction to the model's hidden states, shifting its behavior toward (or away from) the concept

### What We Measure

- **Activation Differences**: The L2 norm of the difference between positive/negative hidden states per layer - higher values indicate layers where the concept is more strongly represented

- **Cosine Similarity**: How similar positive and negative activations are - lower similarity means better separation of the concept

- **Layer Importance**: Which transformer layers best capture the concept (typically middle-to-late layers, around 45-85% depth)

- **Control Strength**: The coefficient multiplier determines how strongly the vector influences generation (typical range: -3.0 to +3.0)

## Supported Models

### Small Models (for experimentation)

| Model | Key | Layers | VRAM | Notes |
|-------|-----|--------|------|-------|
| Qwen/Qwen2-1.5B-Instruct | `qwen`, `qwen2-1.5b` | 28 | ~3GB | Fast experimentation |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B | `deepseek`, `deepseek-r1-1.5b` | 28 | ~3GB | R1 distilled |
| allenai/OLMo-2-1124-7B-Instruct | `olmo`, `olmo2-7b` | 32 | ~14GB | AI2's OLMo 2 |

### Large Models (for introspection research)

| Model | Key | Layers | VRAM | Notes |
|-------|-----|--------|------|-------|
| meta-llama/Llama-3.1-70B-Instruct | `llama70b`, `llama3.1-70b` | 80 | ~140GB | Primary introspection candidate |
| Qwen/Qwen3-Next-80B-A3B-Instruct | `qwen3`, `qwen3-80b` | 62 | ~160GB | MoE (80B total, 3B active) |
| allenai/OLMo-3.1-32B-Instruct | `olmo3`, `olmo3.1-32b` | 64 | ~64GB | AI2's OLMo 3.1 |
| mistralai/Mixtral-8x22B-Instruct-v0.1 | `mixtral`, `mixtral-8x22b` | 56 | ~88GB | MoE (176B sparse) |
| meta-llama/Llama-3.1-8B-Instruct | `llama8b`, `llama3.1-8b` | 32 | ~16GB | Baseline comparison |

> **Note**: Large models require significant VRAM (160GB recommended for full suite).

## Installation

```bash
# Requirements: Python >= 3.10, GPU with sufficient VRAM
pip install -r requirements.txt

# Or install as editable package
pip install -e .
```

## Quick Start

### 1. Train a Control Vector

```bash
python scripts/train_vector.py --model qwen --concept honesty
```

This creates `vectors/qwen_honesty.pt` containing the learned direction for honesty.

### 2. Test the Vector

```bash
python scripts/test_vector.py --model qwen --vector vectors/qwen_honesty.pt
```

Compares model outputs with positive (+2.0), negative (-2.0), and no (0.0) control applied.

### 3. Interactive Demo

```bash
python scripts/apply_vector.py --model qwen --vector vectors/qwen_honesty.pt
```

Chat with the model while adjusting the control coefficient in real-time.

## Analyzing Activations

Understand which layers matter most for a concept:

### Capture Activations

```bash
# Capture contrastive activations for a concept
python scripts/capture_activations.py --model qwen --concept honesty --max-samples 10

# Capture from a single prompt
python scripts/capture_activations.py --model qwen --prompt "Tell me a story"
```

### Visualize Layer Importance

```bash
# Interactive plot showing layer-by-layer differences
python scripts/visualize_activations.py activations/qwen_honesty.pt

# Save to file
python scripts/visualize_activations.py activations/qwen_honesty.pt --output analysis.png

# Text-only summary
python scripts/visualize_activations.py activations/qwen_honesty.pt --no-plot
```

The visualization shows:
- **Difference Norm**: Higher = layer captures more concept information
- **Cosine Similarity**: Lower = better separation between positive/negative
- **Recommended Layers**: Which layers to target for control vector training

## Available Concepts

| Concept | Positive | Negative |
|---------|----------|----------|
| `honesty` | truthful, transparent | deceptive, misleading |
| `creativity` | imaginative, original | conventional, predictable |
| `confidence` | assertive, certain | hesitant, uncertain |
| `helpfulness` | supportive, thorough | dismissive, unhelpful |
| `formality` | professional, formal | casual, informal |
| `verbosity` | detailed, elaborate | terse, brief |
| `enthusiasm` | energetic, excited | apathetic, indifferent |
| `empathy` | compassionate, understanding | cold, detached |

## Python API

```python
from control_vectors_multi import (
    train_control_vector,
    create_controlled_model,
    generate_with_control,
    load_model_and_tokenizer,
)

# Train a control vector
vector = train_control_vector("qwen", "honesty", output_path="my_vector.pt")

# Load and apply
model, tokenizer, vector, config = create_controlled_model(
    "qwen",
    "my_vector.pt",
    coefficient=2.0  # Positive = more honest, negative = less honest
)

# Generate with control
prompt = f"{config.user_tag}Did you break the vase?{config.asst_tag}"
response = generate_with_control(model, tokenizer, prompt, vector, coefficient=2.0)
```

## CLI Reference

### train_vector.py
```
--model, -m       Model key (qwen, deepseek, olmo)
--concept, -c     Concept name (honesty, creativity, etc.)
--output, -o      Output path (default: vectors/{model}_{concept}.pt)
--gguf            Export as GGUF for llama.cpp
--device          Device (auto, cuda, cpu)
--list-models     Show available models
--list-concepts   Show available concepts
```

### test_vector.py
```
--model, -m       Model key
--vector, -v      Path to vector file
--prompt, -p      Custom test prompt
--find-optimal    Run coefficient sweep to find best values
--device          Device (auto, cuda, cpu)
```

### capture_activations.py
```
--model, -m       Model key
--concept, -c     Concept for contrastive capture
--prompt, -p      Single prompt to capture
--layers          Layer indices ("14,15,16" or "recommended")
--max-samples     Max samples for concept capture (default: 10)
--output, -o      Output path
--use-hooks       Use PyTorch hooks (lower memory)
```

### visualize_activations.py
```
input             Path to .pt activation file
--output, -o      Save plot to file
--no-plot         Text summary only (no GUI)
--concept         Concept name for labels
```

## References

- [Representation Engineering (Zou et al., 2023)](https://arxiv.org/abs/2310.01405) - The foundational paper on extracting and applying control vectors
- [repeng library](https://github.com/vgel/repeng) - The underlying library for control vector training
- [Representation Engineering Mistral-7B](https://vgel.me/posts/representation-engineering/) - Blog post with detailed explanations

## License

MIT

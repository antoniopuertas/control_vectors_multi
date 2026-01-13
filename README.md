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

### CUDA Compatibility

This toolkit requires matching PyTorch CUDA version with your GPU driver. Check your driver's CUDA version:

```bash
nvidia-smi  # Look for "CUDA Version" in the output
```

Then install the appropriate PyTorch version:

```bash
# For CUDA 12.1
pip install torch>=2.4.0 --index-url https://download.pytorch.org/whl/cu121

# For CUDA 12.4
pip install torch>=2.4.0 --index-url https://download.pytorch.org/whl/cu124

# For CUDA 12.6+ (newer drivers like H100 with CUDA 13.0)
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu126
```

### Recommended Environment Setup

For a clean installation (recommended):

```bash
conda create -n control_vectors python=3.11 -y
conda activate control_vectors

# Install PyTorch first (match your CUDA version)
pip install torch>=2.4.0 --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip install transformers>=4.45.0 accelerate>=0.26.0 numpy>=1.24.0 scikit-learn
pip install repeng --no-deps
pip install gguf tqdm matplotlib

# Install this package
pip install -e .
```

## Quick Start

### 1. Train a Control Vector

```bash
# Using Qwen2-1.5B (fastest, ~3GB VRAM)
python scripts/train_vector.py --model qwen --concept honesty

# Using DeepSeek-R1-Distill (reasoning-focused, ~3GB VRAM)
python scripts/train_vector.py --model deepseek --concept creativity

# Using larger models
python scripts/train_vector.py --model llama8b --concept confidence
```

This creates `vectors/{model}_{concept}.pt` containing the learned direction.

### 2. Test the Vector

```bash
# Test with default prompts
python scripts/test_vector.py --model qwen --vector vectors/qwen_honesty.pt

# Test DeepSeek-R1 with custom prompt
python scripts/test_vector.py --model deepseek --vector vectors/deepseek-r1-1.5b_creativity.pt \
    --prompt "Write a poem about the ocean"

# Find optimal coefficient values
python scripts/test_vector.py --model deepseek --vector vectors/deepseek-r1-1.5b_creativity.pt \
    --find-optimal
```

Compares model outputs with positive (+2.0), negative (-2.0), and no (0.0) control applied.

### 3. Interactive Demo

```bash
python scripts/apply_vector.py --model qwen --vector vectors/qwen_honesty.pt
python scripts/apply_vector.py --model deepseek --vector vectors/deepseek-r1-1.5b_creativity.pt
```

Chat with the model while adjusting the control coefficient in real-time.

### Example: DeepSeek-R1 Creativity Control

```bash
# Train creativity vector on DeepSeek-R1
python scripts/train_vector.py --model deepseek --concept creativity

# Test the effect
python scripts/test_vector.py --model deepseek --vector vectors/deepseek-r1-1.5b_creativity.pt

# Expected output:
# [Coeff -2.0]: Short, conventional responses
# [Coeff +0.0]: Standard responses
# [Coeff +2.0]: More elaborate, imaginative responses
```

## Analyzing Activations

Understand which layers matter most for a concept:

### Capture Activations

```bash
# Capture contrastive activations for a concept (Qwen)
python scripts/capture_activations.py --model qwen --concept honesty --max-samples 10

# Capture activations for DeepSeek-R1
python scripts/capture_activations.py --model deepseek --concept creativity --max-samples 15

# Capture from a single prompt
python scripts/capture_activations.py --model qwen --prompt "Tell me a story"

# Capture specific layers only
python scripts/capture_activations.py --model deepseek --concept confidence --layers "10,11,12,13,14"

# Use recommended layers (45-85% depth)
python scripts/capture_activations.py --model deepseek --concept honesty --layers recommended
```

### Visualize Layer Importance

```bash
# Interactive plot showing layer-by-layer differences
python scripts/visualize_activations.py activations/qwen_honesty.pt

# Visualize DeepSeek-R1 activations
python scripts/visualize_activations.py activations/deepseek-r1-1.5b_creativity.pt

# Save to file (useful for remote servers)
python scripts/visualize_activations.py activations/deepseek-r1-1.5b_creativity.pt --output creativity_analysis.png

# Text-only summary (no GUI required)
python scripts/visualize_activations.py activations/qwen_honesty.pt --no-plot

# Add concept name to plot title
python scripts/visualize_activations.py activations/deepseek-r1-1.5b_creativity.pt --concept creativity --output deepseek_creativity.png
```

### Example: Full Analysis Pipeline

```bash
# 1. Capture activations for DeepSeek-R1 honesty concept
python scripts/capture_activations.py --model deepseek --concept honesty --max-samples 20

# 2. Visualize to find best layers
python scripts/visualize_activations.py activations/deepseek-r1-1.5b_honesty.pt --no-plot

# Output example:
# Layer Analysis Summary:
#   Layer 12: diff_norm=2.34, cosine_sim=0.82 (recommended)
#   Layer 13: diff_norm=2.56, cosine_sim=0.79 (recommended)
#   Layer 14: diff_norm=2.41, cosine_sim=0.81 (recommended)

# 3. Train vector targeting those layers
python scripts/train_vector.py --model deepseek --concept honesty

# 4. Test the trained vector
python scripts/test_vector.py --model deepseek --vector vectors/deepseek-r1-1.5b_honesty.pt
```

The visualization shows:
- **Difference Norm**: Higher = layer captures more concept information
- **Cosine Similarity**: Lower = better separation between positive/negative
- **Recommended Layers**: Which layers to target for control vector training (typically 45-85% depth)

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

# Train a control vector (Qwen)
vector = train_control_vector("qwen", "honesty", output_path="vectors/qwen_honesty.pt")

# Train with DeepSeek-R1
vector = train_control_vector("deepseek", "creativity", output_path="vectors/deepseek_creativity.pt")

# Load and apply
model, tokenizer, vector, config = create_controlled_model(
    "qwen",
    "vectors/qwen_honesty.pt",
    coefficient=2.0  # Positive = more honest, negative = less honest
)

# Generate with control
prompt = f"{config.user_tag}Did you break the vase?{config.asst_tag}"
response = generate_with_control(model, tokenizer, prompt, vector, coefficient=2.0)
```

### DeepSeek-R1 Example

```python
from control_vectors_multi import train_control_vector, load_model_and_tokenizer
from control_vectors_multi.apply import load_vector
from repeng import ControlModel
import torch

# Train creativity vector
train_control_vector("deepseek", "creativity", output_path="vectors/deepseek_creativity.pt")

# Load model and vector
model, tokenizer, config = load_model_and_tokenizer("deepseek")
vector = load_vector("vectors/deepseek_creativity.pt")

# Create controlled model
from control_vectors_multi.models import get_recommended_layers
layers = get_recommended_layers(config)
control_model = ControlModel(model, layers)

# Generate with different creativity levels
prompt = f"{config.user_tag}Write a haiku about programming{config.asst_tag}"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

for coeff in [-2.0, 0.0, 2.0]:
    if coeff == 0:
        control_model.reset()
    else:
        control_model.set_control(vector, coeff)

    with torch.no_grad():
        output = control_model.generate(**inputs, max_new_tokens=100, do_sample=False)

    print(f"[Coeff {coeff:+.1f}]")
    print(tokenizer.decode(output[0], skip_special_tokens=True))
    print()
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

## Troubleshooting

### Gibberish Output / Model Produces Random Tokens

If the model generates nonsensical output (e.g., `!!!!!!!!` or random characters), this is usually a **CUDA version mismatch**:

1. Check your GPU driver's CUDA version: `nvidia-smi`
2. Check PyTorch's CUDA version: `python -c "import torch; print(torch.version.cuda)"`
3. If they don't match closely, reinstall PyTorch with the correct CUDA version (see Installation section)

**Quick test** to verify your setup works:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-1.5B-Instruct", torch_dtype=torch.float32).cuda()
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-1.5B-Instruct")

inputs = tokenizer("The capital of France is", return_tensors="pt").to("cuda")
out = model.generate(**inputs, max_new_tokens=10, do_sample=False)
print(tokenizer.decode(out[0]))  # Should mention "Paris"
```

### NaN Values During Training

If you see `ValueError: Input X contains NaN`, this indicates numerical instability:

- Try using `torch_dtype=torch.float32` instead of `bfloat16`
- Ensure CUDA versions match (see above)
- On newer GPUs (H100, etc.), use PyTorch nightly with CUDA 12.6+

### Package Version Conflicts

The `repeng` package requires `numpy<2.0.0`. If you see numpy conflicts:

```bash
pip install numpy>=1.24.0,<2.0.0
pip install repeng --no-deps
```

### Model Loading Issues

If models fail to load or produce errors with `trust_remote_code`:

- The toolkit now loads models without `trust_remote_code=True` by default
- This improves compatibility with newer GPU drivers
- If a specific model requires custom code, you may need to modify `train.py`

## References

- [Representation Engineering (Zou et al., 2023)](https://arxiv.org/abs/2310.01405) - The foundational paper on extracting and applying control vectors
- [repeng library](https://github.com/vgel/repeng) - The underlying library for control vector training
- [Representation Engineering Mistral-7B](https://vgel.me/posts/representation-engineering/) - Blog post with detailed explanations

## License

MIT

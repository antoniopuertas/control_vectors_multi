# How to Capture Model Activations

This guide explains how to capture and save internal activations (hidden states) from transformer models using `capture_activations.py`.

## Prerequisites

```bash
pip install -r requirements.txt
```

## Command Line Usage

### Capture from a single prompt

```bash
python scripts/capture_activations.py --model qwen --prompt "Tell me a story"
```

### Capture contrastive activations for a concept

Captures both positive and negative prompt activations for comparison:

```bash
python scripts/capture_activations.py --model qwen --concept honesty --max-samples 10
```

Available concepts: `honesty`, `creativity`, `confidence`, `helpfulness`, `formality`, `verbosity`, `enthusiasm`, `empathy`

### Capture only specific layers

```bash
# Specific layers
python scripts/capture_activations.py --model qwen --prompt "Hello" --layers "14,15,16,17"

# Recommended layers (45%-85% of model depth)
python scripts/capture_activations.py --model qwen --prompt "Hello" --layers recommended
```

### Use PyTorch hooks (lower memory usage)

```bash
python scripts/capture_activations.py --model qwen --prompt "Hello" --use-hooks
```

### Specify a different model

```bash
python scripts/capture_activations.py --model deepseek --prompt "Hello"
python scripts/capture_activations.py --model olmo --concept creativity
```

### List available models and concepts

```bash
python scripts/capture_activations.py --model qwen --list-models
python scripts/capture_activations.py --model qwen --list-concepts
```

## Output Files

| File | Contents |
|------|----------|
| `*.pt` | Tensor data (activations, input_ids, tokens) |
| `*.json` | Metadata and statistics (mean, std, norm per layer) |

Default output location: `activations/{model}_{concept}.pt`

## Visualizing Activations

After capturing activations, visualize them with:

```bash
# Interactive plot
python scripts/visualize_activations.py activations/qwen_honesty.pt

# Save to file
python scripts/visualize_activations.py activations/qwen_honesty.pt --output honesty_analysis.png

# Text-only summary (no GUI required)
python scripts/visualize_activations.py activations/qwen_honesty.pt --no-plot
```

The visualization shows:
- **Difference norm per layer** - Higher values indicate layers more important for the concept
- **Cosine similarity** - Lower values indicate more separation between positive/negative
- **Layer ranking** - Ranked list of most important layers

## Programmatic Usage

### Basic capture

```python
from scripts.capture_activations import capture_activations
from control_vectors_multi import load_model_and_tokenizer

# Load model
model, tokenizer, config = load_model_and_tokenizer("qwen", "auto")

# Capture activations
data = capture_activations(model, tokenizer, "Your prompt here")

# Access activations per layer
for layer_idx, hidden_states in data["activations"].items():
    print(f"Layer {layer_idx}: {hidden_states.shape}")
    # Shape: (batch_size, sequence_length, hidden_dim)
```

### Capture specific layers only

```python
data = capture_activations(
    model,
    tokenizer,
    "Your prompt here",
    layers=[14, 15, 16, 17, 18]
)
```

### Capture contrastive activations

```python
from scripts.capture_activations import capture_contrastive_activations

positive_prompts = [
    "<|im_start|>user\nAct as an honest assistant. Hello!<|im_end|>\n<|im_start|>assistant\n",
    "<|im_start|>user\nAct as an honest assistant. How can I help?<|im_end|>\n<|im_start|>assistant\n",
]
negative_prompts = [
    "<|im_start|>user\nAct as a deceptive assistant. Hello!<|im_end|>\n<|im_start|>assistant\n",
    "<|im_start|>user\nAct as a deceptive assistant. How can I help?<|im_end|>\n<|im_start|>assistant\n",
]

data = capture_contrastive_activations(
    model, tokenizer,
    positive_prompts,
    negative_prompts,
    layers=[14, 15, 16]
)

# Access results
for i, (pos, neg) in enumerate(zip(data["positive"], data["negative"])):
    print(f"Pair {i}: pos shape = {pos['activations'][14].shape}")
```

### Capture from predefined concept

```python
from scripts.capture_activations import capture_from_concept

data = capture_from_concept(
    model, tokenizer,
    concept="honesty",
    model_key="qwen",
    layers=[14, 15, 16],
    max_samples=5
)
```

### Compute statistics

```python
from scripts.capture_activations import compute_activation_stats

stats = compute_activation_stats(data)
for layer_idx, layer_stats in stats.items():
    print(f"Layer {layer_idx}:")
    print(f"  Mean: {layer_stats['mean']:.4f}")
    print(f"  Std:  {layer_stats['std']:.4f}")
    print(f"  Norm: {layer_stats['norm']:.4f}")
```

### Compute contrastive difference

```python
from scripts.capture_activations import compute_contrastive_diff

pos_data = capture_activations(model, tokenizer, "Act as honest. Hello!")
neg_data = capture_activations(model, tokenizer, "Act as deceptive. Hello!")

diffs = compute_contrastive_diff(pos_data, neg_data)
for layer_idx, diff_info in diffs.items():
    print(f"Layer {layer_idx}:")
    print(f"  Diff norm: {diff_info['diff_norm']:.4f}")
    print(f"  Cosine similarity: {diff_info['cosine_sim']:.4f}")
```

### Save and load activations

```python
from scripts.capture_activations import save_activations
import torch

# Save
save_activations(data, "my_activations")
# Creates: my_activations.pt and my_activations.json

# Load
loaded_data = torch.load("my_activations.pt")
```

## Data Structure

The captured data dictionary contains:

```python
{
    "activations": {
        0: tensor(...),   # Layer 0 hidden states
        1: tensor(...),   # Layer 1 hidden states
        # ... more layers
    },
    "input_ids": tensor(...),      # Tokenized input
    "tokens": ["<s>", "Hello", ...],  # Token strings
    "prompt": "Original prompt",
    "num_layers": 28,
    "captured_layers": [0, 1, 2, ...]
}
```

Each activation tensor has shape: `(batch_size, sequence_length, hidden_dim)`

## Contrastive Data Structure

For concept-based captures:

```python
{
    "positive": [
        {"activations": {...}, "prompt": "...", ...},
        {"activations": {...}, "prompt": "...", ...},
    ],
    "negative": [
        {"activations": {...}, "prompt": "...", ...},
        {"activations": {...}, "prompt": "...", ...},
    ],
    "num_pairs": 10,
    "concept": "honesty",
    "layers": [12, 13, 14, ...]
}
```

## Tips

- **Memory**: Use `--layers recommended` or specific layers to capture only what you need
- **Speed**: Use `--use-hooks` for slightly faster capture with less memory overhead
- **Analysis**: The last token position (`[:, -1, :]`) is typically most relevant for control vectors
- **Comparison**: Use `--concept` to capture contrastive pairs for control vector analysis
- **Visualization**: Use `visualize_activations.py` to find the most important layers

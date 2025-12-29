# Control Vectors Multi

Control vector training for multiple HuggingFace models using representation engineering.

## Supported Models

| Model | Key | Layers | VRAM | Notes |
|-------|-----|--------|------|-------|
| Qwen/Qwen2-1.5B-Instruct | `qwen`, `qwen2-1.5b` | 28 | ~3GB | Fast experimentation |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B | `deepseek`, `deepseek-r1-1.5b` | 28 | ~3GB | R1 distilled |
| allenai/OLMo-2-1124-7B-Instruct | `olmo`, `olmo2-7b` | 32 | ~14GB | AI2's OLMo 2 |

## Requirements

- Python >= 3.10
- GPU with sufficient VRAM (see model table above)

## Installation

```bash
pip install -r requirements.txt
```

Or install as a package:

```bash
pip install -e .
```

## Quick Start

### Train a control vector

```bash
python scripts/train_vector.py --model qwen --concept honesty
```

### Test a trained vector

```bash
python scripts/test_vector.py --model qwen --vector vectors/qwen_honesty.pt
```

### Interactive demo

```bash
python scripts/apply_vector.py --model qwen --vector vectors/qwen_honesty.pt
```

## Available Concepts

- `honesty` - truthful vs deceptive
- `creativity` - imaginative vs conventional
- `confidence` - assertive vs hesitant
- `helpfulness` - supportive vs dismissive
- `formality` - professional vs casual
- `verbosity` - detailed vs terse
- `enthusiasm` - energetic vs apathetic
- `empathy` - compassionate vs cold

## Python API

```python
from control_vectors_multi import train_control_vector, create_controlled_model

# Train
vector = train_control_vector("qwen", "honesty", "my_vector.pt")

# Apply
model, tokenizer, vector, config = create_controlled_model("qwen", "my_vector.pt", coefficient=2.0)

# Generate
from control_vectors_multi import generate_with_control
response = generate_with_control(model, tokenizer, prompt, vector, coefficient=2.0)
```

## CLI Options

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
--find-optimal    Run coefficient sweep
--device          Device (auto, cuda, cpu)
```

### apply_vector.py

```
--model, -m       Model key
--vector, -v      Path to vector file
--coefficient, -c Default coefficient (default: 2.0)
--device          Device (auto, cuda, cpu)
```

## License

MIT

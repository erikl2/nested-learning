# Nested Learning: Implementation Study

## Version 0.3.1 - December 2025

Implementation of the NeurIPS 2025 paper "Nested Learning" by Behrouz et al. (Google Research).

**Mathematical verification status:**
- ✅ **Paper-exact**: Delta rule, L2RegressionAttention, LinearAttention, CMS update scheduling
- ✅ **Paper-exact (optional)**: DMGD with `internal_loss_mode='l2_regression'` (Eq 21-23)
- Default: DMGD uses practical surrogate loss (stable, conceptually aligned)

This version includes paper-exact modes, mathematical correctness tests, and inference-time adaptation.

---

## What's Implemented

| Component | Status | Description |
|-----------|--------|-------------|
| **DeepMomentumGD** | ✅ Complete | Memory modules trained via internal loss L^(2) |
| **SelfModifyingLinear** | ✅ Complete | Paper-exact (`normalized=False`) and stable modes |
| **L2RegressionAttention** | ✅ New | Paper's L2 regression variant (Eq 27-29) |
| **ContinuumMemorySystem** | ✅ Complete | Paper-exact nesting (`use_residual=False`) available |
| **LinearAttention** | ✅ Fixed | Per-sequence memory (not batch-averaged) |
| **HOPE Model** | ✅ Complete | Full integration with all components |
| **Math Correctness Tests** | ✅ New | Finite difference gradient verification |
| **Benchmarks** | ✅ Complete | WikiText-103 and LAMBADA evaluation scripts |

See [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md) for detailed component documentation.

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/erikl2/nested-learning.git
cd nested-learning

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Validate Installation

```bash
# Run all tests (27+ tests including math correctness)
python -m pytest tests/ -v

# Run the demo
python examples/nested_learning_demo.py
```

### Run Benchmarks

```bash
# Synthetic benchmarks (fast, no data download)
python experiments/benchmark_pattern_learning.py      # DMGD vs SGD
python experiments/benchmark_continual_learning.py    # HOPE vs Vanilla

# WikiText-103 benchmark (Paper Table 1)
python experiments/benchmark_wikitext.py --test

# LAMBADA zero-shot evaluation
python experiments/benchmark_lambada.py --test
```

## Architecture

```
nested-learning/
├── src/nested_learning/
│   ├── optimizers/          # DeepMomentumGD, NestedDeepMomentumGD
│   ├── memory/              # ContinuumMemorySystem, AssociativeMemory
│   ├── models/              # HOPE, SelfModifyingAttention
│   ├── training/            # NestedLearningTrainer
│   └── utils/               # AMP utilities, helpers
├── experiments/             # Benchmark scripts
├── examples/                # Demo scripts
├── tests/                   # Comprehensive test suite
└── docs/                    # Documentation
```

## Usage Examples

### DeepMomentumGD with True Nested Optimization

```python
from nested_learning.optimizers import DeepMomentumGD

# Memory modules are trained via internal loss every step
optimizer = DeepMomentumGD(
    params=model.parameters(),
    lr=1e-3,
    memory_lr=1e-4,           # Learning rate for memory modules
    use_shared_memory=True,    # Efficient memory pooling
    gradient_checkpointing=True,  # Memory efficient (v0.3.0)
    use_factorized_memory=True,   # Parameter efficient (v0.3.0)
)

for x, y in dataloader:
    optimizer.zero_grad()
    loss = criterion(model(x), y)
    loss.backward()
    optimizer.step()  # Trains both model AND memory modules
```

### Meta-Learning with NestedDeepMomentumGD

```python
from nested_learning.optimizers import NestedDeepMomentumGD

optimizer = NestedDeepMomentumGD(
    params=model.parameters(),
    lr=0.01,
    memory_lr=0.001,
    meta_learning=True,
)

# Inner loop: training steps
for _ in range(inner_steps):
    loss = model(train_batch)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step(create_graph=True)  # Preserve graph for meta-learning

# Outer loop: meta-update based on validation
val_loss = model(val_batch)
optimizer.meta_step(val_loss)  # Updates memory modules
```

### HOPE Model with Self-Modifying Attention

```python
from nested_learning.models import HOPE

model = HOPE(
    dim=512,
    n_layers=12,
    n_heads=8,
    vocab_size=50257,
    use_self_modification=True,  # Enable delta-rule attention
)

# Weights update during forward pass (works in training AND inference)
logits = model(input_ids, enable_self_modification=True)

# Apply pending weight updates after backward pass
model.apply_pending_updates()
```

### Safe Online Adaptation with adaptation_scope

```python
# adaptation_scope allows temporary self-modification that reverts on exit
# Useful for inference-time adaptation without permanent weight drift

with model.adaptation_scope():
    # Self-modification is active within this scope
    output = model.generate(prompt, max_new_tokens=100)
    # Weights adapt to the prompt context

# Weights are automatically restored to pre-scope values
# Safe to reuse model for unrelated prompts
```

### Paper-Exact Mode (for Reproduction)

```python
from nested_learning.models.titans import SelfModifyingLinear, L2RegressionAttention
from nested_learning.memory import ContinuumMemorySystem

# Paper-exact self-modification (Eq 28-29)
layer = SelfModifyingLinear(512, 512, normalized=False)

# Paper-exact CMS nesting (Eq 30)
cms = ContinuumMemorySystem(dim=512, num_levels=3)
output = cms(x, use_residual=False)  # True nesting: MLP_k(MLP_{k-1}(...))

# Paper-exact L2 regression attention (Eq 27-29)
attn = L2RegressionAttention(dim=512, num_heads=8, normalized=False)
```

**Paper-exactness quick reference:**

| Component | Paper-Exact | Stable Default |
|-----------|-------------|----------------|
| SelfModifyingLinear | `normalized=False` | `normalized=True` |
| CMS | `use_residual=False` | `use_residual=True` |
| L2RegressionAttention | `normalized=False` | `normalized=True` |
| DMGD internal loss | `internal_loss_mode='l2_regression'` | `'surrogate'` |

See [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md) for detailed configuration guide.

### Mixed Precision Training (v0.3.0)

```python
from nested_learning.utils.amp import NestedAMPWrapper

amp = NestedAMPWrapper(enabled=True, dtype=torch.bfloat16)

with amp.model_autocast():
    loss = model(batch)

amp.backward(loss)
amp.unscale_and_clip(optimizer, max_norm=1.0)
amp.step(optimizer)
amp.update()
```

### NestedLearningTrainer

```python
from nested_learning.training import NestedLearningTrainer

trainer = NestedLearningTrainer(
    model=model,
    optimizer=optimizer,
    train_dataloader=train_loader,
    val_dataloader=val_loader,
)

# Trains with CMS multi-frequency updates and self-modification
trainer.train(num_epochs=10)
```

## Key Concepts

### Nested Optimization (L^(2) Internal Loss)

The core innovation: memory modules learn to compress gradients through a self-supervised internal loss.

**Note**: The current DMGD internal loss is a **practical surrogate** (not the literal L² regression on K-V matrices from the paper). It uses:

1. **Reconstruction Loss**: Memory output should capture gradient direction (cosine similarity)
2. **Magnitude Preservation**: Output magnitude proportional to input gradient
3. **Temporal Smoothness**: Smooth changes over consecutive steps

This is *conceptually aligned* with the paper's internal objective but not mathematically identical. For paper-exact L² regression, see `L2RegressionAttention`.

### Multi-Frequency Updates (CMS)

Different MLP levels update at different rates:
- **Level 0**: Every step (working memory)
- **Level 1**: Every 10 steps (short-term patterns)
- **Level 2**: Every 100 steps (long-term knowledge)

### Self-Modifying Attention

Weights change during forward pass via delta rule (Equations 28-29):
```
# Normalized mode (default, more stable):
W -= lr * (W @ x @ x^T) / (x^T @ x)

# Paper-exact mode (normalized=False):
W -= lr * (W @ x @ x^T)
```

Updates are deferred until after backward pass to preserve gradient computation.
**New in v0.3.1**: Works during both training AND inference for online adaptation.

## Scalability Features (v0.3.0)

### Gradient Checkpointing
Trade compute for memory - useful for large models:
```python
optimizer = DeepMomentumGD(..., gradient_checkpointing=True)
```

### Factorized Memory
Low-rank factorization for large parameter tensors (4x parameter reduction):
```python
optimizer = DeepMomentumGD(..., use_factorized_memory=True, factorized_rank=16)
```

### Mixed Precision
BF16/FP16 training with gradient scaling:
```python
from nested_learning.utils.amp import AMPTrainer, AMPConfig
trainer = AMPTrainer(model, optimizer, amp_config=AMPConfig(dtype=torch.bfloat16))
```

## Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test suites
python -m pytest tests/test_math_correctness.py -v  # Mathematical verification
python -m pytest tests/test_meta_learning.py -v     # Meta-learning validation
python -m pytest tests/test_scalability.py -v       # Scalability features
python -m pytest tests/test_optimizers.py -v        # Optimizer tests
```

**Test Results**: 27+ passed (including 12 math correctness tests)

## Remaining Work

- [ ] Distributed training (multi-GPU with DDP)
- [ ] Continual learning benchmark
- [ ] Long-context benchmark (100K+ tokens)
- [ ] Hyperparameter search for exact paper reproduction

## Citation

If you use this implementation, please cite the original paper:

```bibtex
@inproceedings{behrouz2025nested,
  title={Nested Learning: The Illusion of Deep Learning Architectures},
  author={Behrouz, Ali and Razaviyayn, Meisam and Zhong, Peiling and Mirrokni, Vahab},
  booktitle={Advances in Neural Information Processing Systems},
  year={2025}
}
```

## Documentation

- [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md) - Detailed component documentation
- [CONTRIBUTING.md](CONTRIBUTING.md) - Contribution guidelines
- [docs/CONCEPTS.md](docs/CONCEPTS.md) - Paper concepts explained

## License

MIT License - See LICENSE file for details

## Acknowledgments

This implementation is based on the NeurIPS 2025 paper by Google Research. All credit for the theoretical contributions goes to the original authors.

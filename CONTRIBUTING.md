# Contributing to Nested Learning

Thank you for your interest in contributing! This repo is a partial implementation of the NeurIPS 2025 paper with documented gaps. Contributions to close those gaps are especially welcome.

## Getting Started

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/erikl2/nested-learning.git
   cd nested-learning
   ```

3. Install in development mode:
   ```bash
   pip install -e ".[dev]"
   ```

4. Read [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md) to understand what's implemented and what's missing.

## Development Guidelines

### Code Style

- Follow PEP 8 guidelines
- Use Black for code formatting: `black src/`
- Use type hints where appropriate
- Add docstrings to all functions and classes
- **Be honest in docstrings** - if something doesn't work as the paper describes, say so

### Testing

- Write tests for new features
- Run tests before submitting PR:
  ```bash
  pytest tests/
  ```
- Note: Current tests are smoke tests. Consider adding tests that validate paper correctness.

### Pull Request Process

1. Create a new branch for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and commit:
   ```bash
   git add .
   git commit -m "Description of changes"
   ```

3. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

4. Open a Pull Request with:
   - Clear description of changes
   - Reference to any related issues
   - Test results
   - How this relates to the paper (if applicable)

---

## Wanted Contributions

This repo is missing core paper concepts. **High-value contributions:**

### 1. Nested Optimization (High Impact)

The paper's core concept is that optimizers should have internal losses and be trained via nested optimization. Currently, our DeepMomentumGD just uses static random MLPs.

**What's needed:**
- Define internal loss `L^(2)` for memory modules (Eq. 23)
- Add inner optimization loop for memory training
- Alternate between task loss and memory loss
- Or implement MAML-style meta-learning for optimizer pretraining

**Files to modify:**
- `src/nested_learning/optimizers/deep_momentum.py`
- Possibly add new `src/nested_learning/meta/` module

### 2. Multi-Frequency Training Integration (Medium Impact)

ContinuumMemorySystem has multi-frequency update methods, but they're never called.

**What's needed:**
- Modify training loop to call `get_update_levels(step)`
- Accumulate gradients per level
- Apply accumulated gradients at correct frequencies
- Test that different levels actually update at different rates

**Files to modify:**
- `src/nested_learning/memory/continuum.py` (verify methods work)
- `src/nested_learning/utils/training.py` or create new trainer
- Add integration test

### 3. SelfModifyingTitan Validation (Medium Impact)

We have a basic delta-rule implementation, but it needs validation against the paper.

**What's needed:**
- Verify delta-rule math matches Eq. 28-29
- Add tests comparing against expected behavior
- Benchmark on sequence tasks
- Document any deviations from paper

**Files to modify:**
- `src/nested_learning/models/titans.py`
- `tests/test_models.py`

### 4. Experimental Reproduction (High Impact)

No one has tried to reproduce the paper's main results.

**What's needed:**
- Scripts reproducing Table 1 (Language Modeling benchmarks)
- Scripts reproducing Table 2 (Continual Learning)
- Scripts reproducing Table 3 (Long-context tasks)
- Match hyperparameters from paper
- Document results vs. paper claims

**Files to create:**
- `experiments/reproduce_table1.py`
- `experiments/reproduce_table2.py`
- etc.

### 5. Documentation Improvements (Low-Medium Impact)

- Add architecture diagrams
- Write tutorials explaining paper concepts
- Improve inline documentation
- Create Jupyter notebooks demonstrating components

---

## Areas NOT Needing Contribution

- **Over-engineering**: Don't add features beyond the paper's scope
- **Cosmetic changes**: Focus on functionality over style
- **Claiming completeness**: Don't update docs to claim things work if they don't

---

## Questions?

Open an issue for questions or discussions! We especially welcome:
- Clarifications about paper concepts
- Ideas for implementing missing features
- Reports of bugs or incorrect implementations

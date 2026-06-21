# TorchCTS — Validate Your PyTorch Backend

[![PyPI Version](https://img.shields.io/pypi/v/torchcts?style=flat-square&color=3B8BF6&label=PyPI)](https://pypi.org/project/torchcts/)
[![License](https://img.shields.io/github/license/TorchCTS/TorchCTS?style=flat-square&color=gray)](LICENSE)

TorchCTS is a comprehensive conformance test suite that stress-tests operators, autograd, memory, training pipelines, and `torch.compile` — across every dtype and layout — against CPU references. It is built specifically for backend developers shipping CUDA, MPS, XPU, or custom PrivateUse1 backends.

---

## Why TorchCTS?

- **🔬 Correctness Over Everything**: Every registered ATen operator is tested against CPU references across all dtypes, strides, and layouts. Non-contiguous memory, channels-last, overlapping strides — nothing is skipped unless explicitly configured.
- **📊 Actionable Scorecards**: Self-contained reports with pass/fail per capability, dtype coverage matrices, and regression diffs. Understand what failed and why without needing to rerun.
- **⚡ Manifest-Driven**: Declare your backend capabilities in a single file (`manifest.py`). The suite automatically skips unsupported features. Pick a template — *minimal*, *inference*, *training*, or *complete* — and customize from there.

---

## Quick Start (How It Works)

### 1. Install
Add TorchCTS to your project (requires Python ≥ 3.10 and PyTorch ≥ 2.12):
```bash
pip install torchcts
```

### 2. Init
Initialize a manifest file in your directory by choosing one of the available templates (`complete`, `training`, `inference`, `minimal`):
```bash
torchcts init
```

### 3. Run
Execute the test suite against your targeted backend:
```bash
torchcts run --device mps
```
*Note: Run `torchcts show-skips` for a collection-only dry-run to print which tests will be skipped and why.*

### 4. Report
Generate or update the comprehensive HTML/Markdown scorecard and validation reports from the test execution results:
```bash
torchcts report
```

---

## CLI Reference

TorchCTS provides a CLI with the following subcommands:

- **`init`**: Initialize `manifest.py` from a template.
- **`run`**: Run the test suite against the target backend.
- **`show-skips`**: Dry-run collection to show which tests will be skipped and why.
- **`report`**: Regenerate scorecards and reports from JSON results.
- **`sync-opinfo`**: Force-rebuild the OpInfo registry cache.
- **`check-manifest`**: Validate `manifest.py` syntax and schema.

---

## Project Structure & Development

- The package entry point is `torchcts`.
- Manifest templates are located in `torchcts/templates/`.
- Test execution results are saved under the `./results/` directory.

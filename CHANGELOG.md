# Changelog

## [Unreleased]

### Added

- Initial package layout with installable distribution via `pyproject.toml`.
- Symbolic model builders for MLP, LSTM, GRU, GCN, and Transformer encoder blocks.
- Derivative export utilities for Jacobian and Hessian in JSON and LaTeX formats.
- CLI entry points (`python -m symbolic_neural_networks.cli`, `symbolic-nn`) with model-specific options.
- Batch example script for generating reproducible symbolic artifacts.
- Compatibility module `symbolic.py` for legacy import compatibility.

### Changed

- Replaced monolithic script usage with package-first architecture:
  - Core implementation in `symbolic_neural_networks/symbolic.py`
  - CLI in `symbolic_neural_networks/cli.py`
  - Package exports via `symbolic_neural_networks/__init__.py`
- Reworked user documentation with package-centric usage and API reference.

## [0.1.0] - 2026-02-25

### Added

- First versioned package baseline with support for symbolic MLP, LSTM, GRU, GCN, and Transformer construction.
- Public Python API and CLI support.

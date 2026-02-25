# symbolic-neural-networks

A Python package for deriving **symbolic expressions** and **exact derivatives** for common neural-network families with `sympy`.
This is intended for analysis pipelines, formal verification research, and symbolic reasoning experiments.

Latest changes and release history are tracked in [`CHANGELOG.md`](./CHANGELOG.md).

## Package scope

The package currently supports:

- MLP (`build_mlp_model`)
- LSTM (`build_lstm_model`)
- GRU (`build_gru_model`)
- GCN (`build_gcn_model`)
- Transformer encoder-style models (`build_transformer_model`)

## Installation

Create and activate a virtual environment, then install the package in editable mode:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Or install only runtime requirements:

```bash
pip install -r requirements.txt
```

## Use as a library

```python
from symbolic_neural_networks import (
    build_mlp_model,
    build_lstm_model,
    build_gru_model,
    build_gcn_model,
    build_transformer_model,
)

mlp = build_mlp_model(input_size=3, hidden_layers=[4, 2])
first_order, second_order = mlp.gradients()

print("f(x) =", mlp.output)
print("gradients =", first_order)
print("hessian diag =", [second_order[i][i] for i in range(len(first_order))])
```

### Exporting derivatives

```python
from symbolic_neural_networks import derivatives_to_dict, derivatives_to_latex

payload = derivatives_to_dict(mlp)
latex = derivatives_to_latex(mlp)

print(payload["parameter_count"], payload["jacobian"]["x_1"])
```

## Command-line interface

After installation, use either:

```bash
python -m symbolic_neural_networks.cli mlp
# or
symbolic-nn mlp
```

Or use the compatibility entrypoint:

```bash
python main.py mlp
```

Examples:

```bash
python -m symbolic_neural_networks.cli mlp --input-size 3 --hidden-layers 4 3
python -m symbolic_neural_networks.cli lstm --input-size 2 --hidden-size 4 --sequence-length 5 --lstm-layers 2
python -m symbolic_neural_networks.cli gru --input-size 2 --hidden-size 3 --sequence-length 5
python -m symbolic_neural_networks.cli gcn --node-count 4 --feature-dim 2 --gcn-hidden-dim 3
python -m symbolic_neural_networks.cli transformer --sequence-length 4 --model-dim 8 --transformer-heads 4
```

Common options:

- `--show-full-hessian`
- `--latex`
- `--show-params`
- `--simplify`
- `--export-json <file>`
- `--export-latex <file>`

`--help` lists all architecture-specific options.

## Changelog and contribution

- Release notes: [`CHANGELOG.md`](./CHANGELOG.md)
- Contributing guide: [`CONTRIBUTING.md`](./CONTRIBUTING.md)

## Release workflow

This repository includes `.github/workflows/release.yml`:

- Tag a release as `vX.Y.Z` (for example, `v0.1.0`) to run build checks, validate version/changelog alignment, and publish to PyPI.
- Manual release can be started from the Actions UI with `publish` set to `true`/`false`.
- Ensure `PYPI_API_TOKEN` repository secret is configured for publish runs.

## API reference

Each builder returns a `SymbolicModel` with:

- `output`: scalar symbolic expression
- `input_symbols`: input symbols used in the model
- `parameters`: generated parameter symbols
- `gradients()`: computes first- and second-order derivatives

Builder functions:

- `build_mlp_model(input_size, hidden_layers, activation=soft_relu, output_activation=None)`
- `build_lstm_model(input_size, hidden_size, sequence_length, num_layers=1, output_layer=1, output_time=None, output_projection=True, output_hidden_index=1, hidden_activation=tanh, gate_activation=sigmoid)`
- `build_gru_model(input_size, hidden_size, sequence_length, num_layers=1, output_layer=1, output_time=None, output_projection=True, output_hidden_index=1, hidden_activation=tanh, gate_activation=sigmoid)`
- `build_gcn_model(node_count, feature_dim, hidden_dim=2, num_layers=1, output_layer=1, output_node=1, output_projection=True, output_hidden_index=1, activation=relu, use_degree_normalization=True, include_self_loops=True)`
- `build_transformer_model(sequence_length, model_dim, num_heads=2, feed_forward_dim=None, num_layers=1, include_positional_embeddings=True, output_position=1, ff_activation=relu)`
- `derivatives_to_dict(model, first_order=None, second_order=None, simplify_output=False)`
- `derivatives_to_latex(model, first_order=None, second_order=None, simplify_output=False)`

## Examples script

Run:

```bash
python examples/symbolic_inspection.py
```

This creates JSON and LaTeX artifacts under `examples/outputs/` for:

- `mlp`
- `lstm`
- `gru`
- `gcn`
- `transformer`

## Project structure

```text
symbolic_neural_networks/
  __init__.py
  symbolic.py
  cli.py
main.py
examples/
  symbolic_inspection.py
.github/
  workflows/
    release.yml
pyproject.toml
requirements.txt
symbolic.py
CHANGELOG.md
CONTRIBUTING.md
```

## Developer extension guide

To add a new architecture:

1. Implement a `build_<arch>_model(...)` function in `symbolic_neural_networks/symbolic.py`.
2. Add a network branch in `symbolic_neural_networks/cli.py`.
3. Update examples and README sections for the new model.

# symbolic-neural-networks

A lightweight repository for deriving **symbolic expressions** and derivatives for neural network architectures with `sympy`. It is intentionally structured for use in formal-verification style workflows where closed-form algebraic expressions are useful for reasoning, checking identities, and generating proof obligations.

## Current scope

- `symbolic.py` provides symbolic model builders and derivative utilities.
- `main.py` provides a CLI to choose between model families:
  - `mlp`
  - `lstm`
  - `gru`
  - `gcn`
  - `transformer`
- `examples/symbolic_inspection.py` performs batch symbolic runs and writes derivative exports.
- `requirements.txt` declares project dependencies.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Run one architecture at a time and print first- and second-order gradients.

```bash
python main.py mlp
python main.py mlp --input-size 3 --hidden-layers 4 3 --activation tanh
python main.py lstm --input-size 2 --hidden-size 4 --sequence-length 5 --lstm-layers 2
python main.py lstm --input-size 2 --hidden-size 3 --sequence-length 4 --lstm-output-time 3 --lstm-no-projection
python main.py gru --input-size 2 --hidden-size 3 --sequence-length 4 --gru-layers 2 --gru-output-time 3
python main.py gru --input-size 2 --hidden-size 3 --sequence-length 5 --gru-no-projection --gru-output-hidden 2
python main.py gcn --node-count 4 --feature-dim 2 --gcn-layers 1 --gcn-hidden-dim 3
python main.py gcn --node-count 4 --feature-dim 2 --gcn-layers 2 --gcn-output-node 2 --gcn-disable-degree-normalization --gcn-no-self-loops
python main.py transformer --sequence-length 3 --model-dim 4 --transformer-heads 2
python main.py transformer --sequence-length 4 --model-dim 8 --transformer-heads 4 --transformer-ffn-dim 16 --transformer-layers 2
```

CLI options:

- `--show-full-hessian`: print all Hessian entries instead of only diagonal.
- `--latex`: prints equations as LaTeX strings in the terminal output.
- `--show-params`: prints all generated parameter symbols.
- `--simplify`: applies `sympy.simplify` on exported derivative expressions.
- `--export-json <file>`: writes full gradients as JSON.
- `--export-latex <file>`: writes full gradients in LaTeX text format.

LSTM-specific:

- `--lstm-layers`: number of stacked LSTM layers.
- `--lstm-output-layer`: which layer output is consumed for final scalar output (1-indexed).
- `--lstm-output-time`: time step to read output from (1-indexed, `0` for last).
- `--lstm-output-hidden`: hidden dimension index when projection is disabled.
- `--lstm-no-projection`: skip output projection and return one hidden-unit entry directly.

GRU-specific:

- `--gru-layers`: number of stacked GRU layers.
- `--gru-output-layer`: which layer output is consumed for final scalar output (1-indexed).
- `--gru-output-time`: time step to read output from (1-based, `0` for last).
- `--gru-output-hidden`: hidden dimension index when projection is disabled.
- `--gru-no-projection`: skip output projection and return one hidden-unit entry directly.

GCN-specific:

- `--gcn-layers`: number of stacked GCN layers.
- `--gcn-hidden-dim`: node-hidden feature width for each layer.
- `--gcn-output-layer`: which layer output is used for final readout (1-indexed).
- `--gcn-output-node`: node index used for final readout.
- `--gcn-output-hidden`: hidden feature index when projection is disabled.
- `--gcn-activation`: layer activation function.
- `--gcn-no-projection`: skip final scalar projection and return one node feature directly.
- `--gcn-no-self-loops`: disable self-loop terms in the adjacency.
- `--gcn-disable-degree-normalization`: use raw adjacency without normalization.

Transformer-specific:

- `--transformer-heads`: number of attention heads.
- `--transformer-ffn-dim`: feed-forward inner dimension (defaults to `4 * model_dim`).
- `--transformer-layers`: number of encoder-like layers.
- `--transformer-output-position`: choose the token position used for scalar output.
- `--disable-transformer-positional-embeddings`: remove additive positional symbols.

## API notes

The module exposes a small reusable API.

- `build_mlp_model(input_size, hidden_layers, activation=soft_relu, output_activation=None)`
- `build_lstm_model(
    input_size,
    hidden_size,
    sequence_length,
    num_layers=1,
    output_layer=1,
    output_time=None,
    output_projection=True,
    output_hidden_index=1,
    hidden_activation=tanh,
    gate_activation=sigmoid,
  )`
- `build_gru_model(
    input_size,
    hidden_size,
    sequence_length,
    num_layers=1,
    output_layer=1,
    output_time=None,
    output_projection=True,
    output_hidden_index=1,
    hidden_activation=tanh,
    gate_activation=sigmoid,
  )`
- `build_gcn_model(
    node_count,
    feature_dim,
    hidden_dim=2,
    num_layers=1,
    output_layer=1,
    output_node=1,
    output_projection=True,
    output_hidden_index=1,
    activation=relu,
    use_degree_normalization=True,
    include_self_loops=True,
  )`
- `build_transformer_model(
    sequence_length,
    model_dim,
    num_heads=2,
    feed_forward_dim=None,
    num_layers=1,
    include_positional_embeddings=True,
    output_position=1,
    ff_activation=relu,
  )`
- `derivatives_to_dict(model, first_order=None, second_order=None, simplify_output=False)`
- `derivatives_to_latex(model, first_order=None, second_order=None, simplify_output=False)`
- `SymbolicModel` data class with:
  - `output`: scalar symbolic output
  - `input_symbols`: tuple of input symbols
  - `parameters`: parameter symbol table
  - `gradients()`: returns first-order gradient vector and full Hessian matrix entries

The Transformer builder follows the structure in *Attention Is All You Need*:

- multi-head scaled dot-product attention
- residual connections
- layer normalization
- position-wise feed-forward network
- optional positional embeddings
- final scalar projection

Exports are structured as:

- `model`: network family name
- `input_symbols`: ordered list of symbolic inputs
- `parameter_count`: number of symbolic parameters
- `output`: output expression
- `jacobian`: input -> derivative map
- `hessian`: full Hessian matrix values

## Example output

`main.py` can print:

- model name
- symbolic output expression
- `∂f/∂x_i` for each input symbol
- diagonal Hessian entries or full Hessian with `--show-full-hessian`
- serialized exports when requested

## Batch symbolic inspection

```bash
python examples/symbolic_inspection.py
```

This generates:

- `examples/outputs/mlp.json`
- `examples/outputs/mlp.tex`
- `examples/outputs/lstm.json`
- `examples/outputs/lstm.tex`
- `examples/outputs/gru.json`
- `examples/outputs/gru.tex`
- `examples/outputs/gcn.json`
- `examples/outputs/gcn.tex`
- `examples/outputs/transformer.json`
- `examples/outputs/transformer.tex`

## Extensibility

The module is structured so each architecture is a standalone builder function returning a `SymbolicModel`. To add a new architecture, implement another `build_<arch>_model(...)` function in `symbolic.py` and dispatch to it from `main.py`.

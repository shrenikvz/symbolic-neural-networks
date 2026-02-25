"""Symbolic neural-network expression builders for gradient-based analysis.

This module intentionally stays dependency-light so it can be used in CLI scripts,
experiments, or formal-verification pipelines without interactive-only imports.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Sequence, Tuple

import sympy as sp

Expression = sp.Expr
Activation = Callable[[Expression], Expression]


def soft_relu(z: Expression) -> Expression:
    return sp.log(1 + sp.exp(z))


def relu(z: Expression) -> Expression:
    return sp.Max(0, z)


def tanh(z: Expression) -> Expression:
    return (sp.exp(z) - sp.exp(-z)) / (sp.exp(z) + sp.exp(-z))


def sigmoid(z: Expression) -> Expression:
    return 1 / (1 + sp.exp(-z))


def _make_symbols(prefix: str, count: int, *, start: int = 1) -> Tuple[sp.Symbol, ...]:
    return tuple(sp.symbols(f"{prefix}_{i}", real=True) for i in range(start, start + count))


def _stringify(expr: Expression, simplify_output: bool = False) -> str:
    value = sp.simplify(expr) if simplify_output else expr
    return str(value)


def _to_latex(expr: Expression, simplify_output: bool = False) -> str:
    value = sp.simplify(expr) if simplify_output else expr
    return sp.latex(value)


def _layer_norm(
    vector: Sequence[Expression],
    gamma: Sequence[Expression],
    beta: Sequence[Expression],
    epsilon: Expression = sp.Rational(1, 10**12),
) -> Tuple[Expression, ...]:
    dim = len(vector)
    mean = sum(vector) / sp.Integer(dim)
    variance = sum((value - mean) ** 2 for value in vector) / sp.Integer(dim)
    inv_std = 1 / sp.sqrt(variance + epsilon)
    return tuple(
        gamma[index] * (value - mean) * inv_std + beta[index]
        for index, value in enumerate(vector)
    )


def _vector_add(left: Sequence[Expression], right: Sequence[Expression]) -> Tuple[Expression, ...]:
    return tuple(lhs + rhs_i for lhs, rhs_i in zip(left, right))


@dataclass(frozen=True)
class SymbolicModel:
    """Container for a symbolic scalar output and its input symbols."""

    name: str
    output: Expression
    input_symbols: Tuple[sp.Symbol, ...]
    parameters: Dict[str, sp.Symbol]

    def gradients(self) -> Tuple[Tuple[Expression, ...], Tuple[Tuple[Expression, ...], ...]]:
        first_order = tuple(sp.diff(self.output, x) for x in self.input_symbols)
        second_order = tuple(
            tuple(sp.diff(dx, xj) for xj in self.input_symbols)
            for dx in first_order
        )
        return first_order, second_order

    def jacobian(self) -> Tuple[Expression, ...]:
        return self.gradients()[0]

    def hessian(self) -> Tuple[Tuple[Expression, ...], ...]:
        return self.gradients()[1]

    def derivative_payload(
        self,
        *,
        simplify_output: bool = False,
    ) -> Dict[str, object]:
        first_order, second_order = self.gradients()
        return derivatives_to_dict(self, first_order, second_order, simplify_output=simplify_output)


def _lstm_gate(
    gate_name: str,
    x_t: Sequence[Expression],
    h_prev: Sequence[Expression],
    params: Dict[str, sp.Symbol],
    layer_index: int,
    input_size: int,
    hidden_size: int,
    gate_activation: Activation = sigmoid,
) -> Tuple[Expression, ...]:
    gate_outputs = []
    for h in range(1, hidden_size + 1):
        weighted_input = sum(
            params[f"W_{layer_index}_{gate_name}_{i}_{h}"] * x_t[i - 1]
            for i in range(1, input_size + 1)
        )
        recurrent_input = sum(
            params[f"U_{layer_index}_{gate_name}_{j}_{h}"] * h_prev[j - 1]
            for j in range(1, hidden_size + 1)
        )
        raw_gate = weighted_input + recurrent_input + params[f"b_{layer_index}_{gate_name}_{h}"]
        gate_outputs.append(gate_activation(raw_gate))
    return tuple(gate_outputs)


def _lstm_step(
    x_t: Sequence[Expression],
    h_prev: Sequence[Expression],
    c_prev: Sequence[Expression],
    params: Dict[str, sp.Symbol],
    layer_index: int,
    input_size: int,
    hidden_size: int,
    gate_activation: Activation = sigmoid,
    hidden_activation: Activation = tanh,
) -> Tuple[Tuple[Expression, ...], Tuple[Expression, ...]]:
    i_t = _lstm_gate(
        "i",
        x_t,
        h_prev,
        params,
        layer_index,
        input_size,
        hidden_size,
        gate_activation=gate_activation,
    )
    f_t = _lstm_gate(
        "f",
        x_t,
        h_prev,
        params,
        layer_index,
        input_size,
        hidden_size,
        gate_activation=gate_activation,
    )
    o_t = _lstm_gate(
        "o",
        x_t,
        h_prev,
        params,
        layer_index,
        input_size,
        hidden_size,
        gate_activation=gate_activation,
    )
    g_t = _lstm_gate(
        "g",
        x_t,
        h_prev,
        params,
        layer_index,
        input_size,
        hidden_size,
        gate_activation=hidden_activation,
    )

    c_next = tuple(
        f_t[idx] * c_prev[idx] + i_t[idx] * g_t[idx] for idx in range(hidden_size)
    )
    h_next = tuple(o_t[idx] * hidden_activation(c_next[idx]) for idx in range(hidden_size))
    return h_next, c_next


def _gru_gate(
    gate_name: str,
    x_t: Sequence[Expression],
    h_prev: Sequence[Expression],
    params: Dict[str, sp.Symbol],
    layer_index: int,
    input_size: int,
    hidden_size: int,
    gate_activation: Activation = sigmoid,
) -> Tuple[Expression, ...]:
    gate_outputs = []
    for h in range(1, hidden_size + 1):
        weighted_input = sum(
            params[f"W_{layer_index}_{gate_name}_{i}_{h}"] * x_t[i - 1]
            for i in range(1, input_size + 1)
        )
        recurrent_input = sum(
            params[f"U_{layer_index}_{gate_name}_{j}_{h}"] * h_prev[j - 1]
            for j in range(1, hidden_size + 1)
        )
        raw_gate = weighted_input + recurrent_input + params[f"b_{layer_index}_{gate_name}_{h}"]
        gate_outputs.append(gate_activation(raw_gate))
    return tuple(gate_outputs)


def _gru_step(
    x_t: Sequence[Expression],
    h_prev: Sequence[Expression],
    params: Dict[str, sp.Symbol],
    layer_index: int,
    input_size: int,
    hidden_size: int,
    gate_activation: Activation = sigmoid,
    hidden_activation: Activation = tanh,
) -> Tuple[Expression, ...]:
    z_t = _gru_gate(
        "z",
        x_t,
        h_prev,
        params,
        layer_index,
        input_size,
        hidden_size,
        gate_activation=gate_activation,
    )
    r_t = _gru_gate(
        "r",
        x_t,
        h_prev,
        params,
        layer_index,
        input_size,
        hidden_size,
        gate_activation=gate_activation,
    )
    h_candidate = []
    for h in range(1, hidden_size + 1):
        weighted_input = sum(
            params[f"W_{layer_index}_h_{i}_{h}"] * x_t[i - 1]
            for i in range(1, input_size + 1)
        )
        recurrent_input = sum(
            params[f"U_{layer_index}_h_{j}_{h}"] * (r_t[j - 1] * h_prev[j - 1])
            for j in range(1, hidden_size + 1)
        )
        h_candidate.append(
            hidden_activation(
                weighted_input + recurrent_input + params[f"b_{layer_index}_h_{h}"]
            )
        )

    h_candidate = tuple(h_candidate)
    h_next = tuple(
        z_t[idx] * h_prev[idx] + (1 - z_t[idx]) * h_candidate[idx]
        for idx in range(hidden_size)
    )
    return h_next


def _activation_from_name(name: str) -> Activation:
    mapping = {"soft_relu": soft_relu, "tanh": tanh, "sigmoid": sigmoid, "relu": relu}
    try:
        return mapping[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported activation '{name}'.") from exc


def build_mlp_model(
    input_size: int,
    hidden_layers: Sequence[int],
    activation: Activation = soft_relu,
    output_activation: Activation | None = None,
) -> SymbolicModel:
    """Create a fully symbolic MLP with one scalar output.

    Parameters
    ----------
    input_size:
        Number of input features.
    hidden_layers:
        Hidden layer widths in order. The final output layer is always width 1.
    activation:
        Activation for hidden layers.
    output_activation:
        Optional activation for the output neuron. If omitted, output is linear.
    """
    if input_size < 1:
        raise ValueError("input_size must be a positive integer")
    if any(layer_size < 1 for layer_size in hidden_layers):
        raise ValueError("hidden_layers must only contain positive integers")

    params: Dict[str, sp.Symbol] = {}
    x = _make_symbols("x", input_size)
    layer_sizes = tuple(hidden_layers) + (1,)
    layer_input: Tuple[Expression, ...] = x

    for layer_index, layer_width in enumerate(layer_sizes, start=1):
        is_output_layer = layer_index == len(layer_sizes)
        layer_output = []

        for out_neuron in range(1, layer_width + 1):
            affine = sum(
                params.setdefault(
                    f"W_{layer_index}_{in_neuron}_{out_neuron}",
                    sp.Symbol(f"W_{layer_index}_{in_neuron}_{out_neuron}", real=True),
                )
                * layer_input[in_neuron - 1]
                for in_neuron in range(1, len(layer_input) + 1)
            ) + params.setdefault(
                f"b_{layer_index}_{out_neuron}",
                sp.Symbol(f"b_{layer_index}_{out_neuron}", real=True),
            )

            if is_output_layer and output_activation is None:
                layer_output.append(affine)
            elif is_output_layer:
                layer_output.append(output_activation(affine))
            else:
                layer_output.append(activation(affine))

        layer_input = tuple(layer_output)

    return SymbolicModel(
        name="MLP",
        output=layer_input[0],
        input_symbols=x,
        parameters=params,
    )


def build_lstm_model(
    input_size: int,
    hidden_size: int,
    sequence_length: int,
    num_layers: int = 1,
    output_layer: int = 1,
    output_time: int | None = None,
    output_projection: bool = True,
    output_hidden_index: int = 1,
    hidden_activation: Activation = tanh,
    gate_activation: Activation = sigmoid,
) -> SymbolicModel:
    """Create a symbolic stacked LSTM encoder-style network."""
    if input_size < 1 or hidden_size < 1 or sequence_length < 1 or num_layers < 1:
        raise ValueError(
            "input_size, hidden_size, sequence_length, and num_layers must be positive"
        )
    if output_layer < 1 or output_layer > num_layers:
        raise ValueError("output_layer must be between 1 and num_layers")
    if output_time is not None and not 1 <= output_time <= sequence_length:
        raise ValueError("output_time must be between 1 and sequence_length")
    if not output_projection and not 1 <= output_hidden_index <= hidden_size:
        raise ValueError("output_hidden_index must be between 1 and hidden_size")

    params: Dict[str, sp.Symbol] = {}
    for layer_index in range(1, num_layers + 1):
        layer_input_size = input_size if layer_index == 1 else hidden_size
        for gate_name in ("i", "f", "o", "g"):
            for i in range(1, layer_input_size + 1):
                for h in range(1, hidden_size + 1):
                    params.setdefault(
                        f"W_{layer_index}_{gate_name}_{i}_{h}",
                        sp.Symbol(f"W_{layer_index}_{gate_name}_{i}_{h}", real=True),
                    )
            for j in range(1, hidden_size + 1):
                for h in range(1, hidden_size + 1):
                    params.setdefault(
                        f"U_{layer_index}_{gate_name}_{j}_{h}",
                        sp.Symbol(f"U_{layer_index}_{gate_name}_{j}_{h}", real=True),
                    )
            for h in range(1, hidden_size + 1):
                params.setdefault(
                    f"b_{layer_index}_{gate_name}_{h}",
                    sp.Symbol(f"b_{layer_index}_{gate_name}_{h}", real=True),
                )

    x = _make_symbols("x", input_size * sequence_length)
    inputs_by_time: list[Tuple[Expression, ...]] = [
        x[t * input_size : (t + 1) * input_size] for t in range(sequence_length)
    ]

    selected_layer_outputs: list[Tuple[Expression, ...]] = []
    current_layer_inputs: list[Tuple[Expression, ...]] = inputs_by_time

    for layer_index in range(1, num_layers + 1):
        layer_input_size = input_size if layer_index == 1 else hidden_size
        h_t = tuple(sp.Integer(0) for _ in range(hidden_size))
        c_t = tuple(sp.Integer(0) for _ in range(hidden_size))
        layer_outputs: list[Tuple[Expression, ...]] = []

        for t in range(sequence_length):
            step_inputs = current_layer_inputs[t]
            if len(step_inputs) != layer_input_size:
                raise RuntimeError("Inconsistent LSTM layer input size during construction.")

            h_t, c_t = _lstm_step(
                step_inputs=step_inputs,
                h_prev=h_t,
                c_prev=c_t,
                params=params,
                layer_index=layer_index,
                input_size=layer_input_size,
                hidden_size=hidden_size,
                gate_activation=gate_activation,
                hidden_activation=hidden_activation,
            )
            layer_outputs.append(h_t)

        current_layer_inputs = layer_outputs
        if layer_index == output_layer:
            selected_layer_outputs = layer_outputs

    assert selected_layer_outputs
    selected_time = sequence_length if output_time is None else output_time
    selected_step = selected_layer_outputs[selected_time - 1]

    if output_projection:
        for h in range(1, hidden_size + 1):
            params.setdefault(
                f"W_out_{output_layer}_{h}",
                sp.Symbol(f"W_out_{output_layer}_{h}", real=True),
            )
        params.setdefault("b_out", sp.Symbol("b_out", real=True))
        output = (
            sum(params[f"W_out_{output_layer}_{h}"] * selected_step[h - 1] for h in range(1, hidden_size + 1))
            + params["b_out"]
        )
    else:
        output = selected_step[output_hidden_index - 1]

    return SymbolicModel(
        name="LSTM",
        output=output,
        input_symbols=x,
        parameters=params,
    )


def build_gru_model(
    input_size: int,
    hidden_size: int,
    sequence_length: int,
    num_layers: int = 1,
    output_layer: int = 1,
    output_time: int | None = None,
    output_projection: bool = True,
    output_hidden_index: int = 1,
    hidden_activation: Activation = tanh,
    gate_activation: Activation = sigmoid,
) -> SymbolicModel:
    """Create a symbolic stacked GRU encoder-style network."""
    if input_size < 1 or hidden_size < 1 or sequence_length < 1 or num_layers < 1:
        raise ValueError(
            "input_size, hidden_size, sequence_length, and num_layers must be positive"
        )
    if output_layer < 1 or output_layer > num_layers:
        raise ValueError("output_layer must be between 1 and num_layers")
    if output_time is not None and not 1 <= output_time <= sequence_length:
        raise ValueError("output_time must be between 1 and sequence_length")
    if not output_projection and not 1 <= output_hidden_index <= hidden_size:
        raise ValueError("output_hidden_index must be between 1 and hidden_size")

    params: Dict[str, sp.Symbol] = {}
    for layer_index in range(1, num_layers + 1):
        layer_input_size = input_size if layer_index == 1 else hidden_size
        for gate_name in ("z", "r"):
            for i in range(1, layer_input_size + 1):
                for h in range(1, hidden_size + 1):
                    params.setdefault(
                        f"W_{layer_index}_{gate_name}_{i}_{h}",
                        sp.Symbol(f"W_{layer_index}_{gate_name}_{i}_{h}", real=True),
                    )
            for j in range(1, hidden_size + 1):
                for h in range(1, hidden_size + 1):
                    params.setdefault(
                        f"U_{layer_index}_{gate_name}_{j}_{h}",
                        sp.Symbol(f"U_{layer_index}_{gate_name}_{j}_{h}", real=True),
                    )
            for h in range(1, hidden_size + 1):
                params.setdefault(
                    f"b_{layer_index}_{gate_name}_{h}",
                    sp.Symbol(f"b_{layer_index}_{gate_name}_{h}", real=True),
                )

        for i in range(1, layer_input_size + 1):
            for h in range(1, hidden_size + 1):
                params.setdefault(
                    f"W_{layer_index}_h_{i}_{h}",
                    sp.Symbol(f"W_{layer_index}_h_{i}_{h}", real=True),
                )
        for j in range(1, hidden_size + 1):
            for h in range(1, hidden_size + 1):
                params.setdefault(
                    f"U_{layer_index}_h_{j}_{h}",
                    sp.Symbol(f"U_{layer_index}_h_{j}_{h}", real=True),
                )
        for h in range(1, hidden_size + 1):
            params.setdefault(
                f"b_{layer_index}_h_{h}",
                sp.Symbol(f"b_{layer_index}_h_{h}", real=True),
            )

    x = _make_symbols("x", input_size * sequence_length)
    inputs_by_time: list[Tuple[Expression, ...]] = [
        x[t * input_size : (t + 1) * input_size] for t in range(sequence_length)
    ]

    selected_layer_outputs: list[Tuple[Expression, ...]] = []
    current_layer_inputs: list[Tuple[Expression, ...]] = inputs_by_time

    for layer_index in range(1, num_layers + 1):
        layer_input_size = input_size if layer_index == 1 else hidden_size
        h_t = tuple(sp.Integer(0) for _ in range(hidden_size))
        layer_outputs: list[Tuple[Expression, ...]] = []

        for t in range(sequence_length):
            step_inputs = current_layer_inputs[t]
            if len(step_inputs) != layer_input_size:
                raise RuntimeError("Inconsistent GRU layer input size during construction.")

            h_t = _gru_step(
                x_t=step_inputs,
                h_prev=h_t,
                params=params,
                layer_index=layer_index,
                input_size=layer_input_size,
                hidden_size=hidden_size,
                gate_activation=gate_activation,
                hidden_activation=hidden_activation,
            )
            layer_outputs.append(h_t)

        current_layer_inputs = layer_outputs
        if layer_index == output_layer:
            selected_layer_outputs = layer_outputs

    assert selected_layer_outputs
    selected_time = sequence_length if output_time is None else output_time
    selected_step = selected_layer_outputs[selected_time - 1]

    if output_projection:
        for h in range(1, hidden_size + 1):
            params.setdefault(
                f"W_out_{output_layer}_{h}",
                sp.Symbol(f"W_out_{output_layer}_{h}", real=True),
            )
        params.setdefault("b_out", sp.Symbol("b_out", real=True))
        output = (
            sum(params[f"W_out_{output_layer}_{h}"] * selected_step[h - 1] for h in range(1, hidden_size + 1))
            + params["b_out"]
        )
    else:
        output = selected_step[output_hidden_index - 1]

    return SymbolicModel(
        name="GRU",
        output=output,
        input_symbols=x,
        parameters=params,
    )


def _gcn_normalized_edge_weight(
    source: int,
    target: int,
    params: Dict[str, sp.Symbol],
    degrees: Tuple[Expression, ...],
    use_degree_normalization: bool,
    include_self_loops: bool,
) -> Expression:
    raw = params[f"A_{source}_{target}"] + (1 if include_self_loops and source == target else 0)
    if not use_degree_normalization:
        return raw

    source_degree = degrees[source - 1]
    target_degree = degrees[target - 1]
    return raw / sp.sqrt(source_degree * target_degree)


def _gcn_layer(
    node_features: Sequence[Tuple[Expression, ...]],
    params: Dict[str, sp.Symbol],
    layer_index: int,
    node_count: int,
    input_size: int,
    output_size: int,
    activation: Activation = relu,
    use_degree_normalization: bool = True,
    include_self_loops: bool = True,
) -> Tuple[Tuple[Expression, ...], ...]:
    if len(node_features) != node_count or any(len(features) != input_size for features in node_features):
        raise RuntimeError("Inconsistent GCN node feature dimensions during construction.")

    # Precompute degrees once per layer.
    degrees = tuple(
        sum(
            params[f"A_{source}_{target}"] + (1 if include_self_loops and source == target else 0)
            for target in range(1, node_count + 1)
        )
        for source in range(1, node_count + 1)
    )

    layer_outputs: list[Tuple[Expression, ...]] = []
    for source in range(1, node_count + 1):
        aggregated_features = []
        for in_feature in range(1, input_size + 1):
            aggregated_features.append(
                sum(
                    _gcn_normalized_edge_weight(
                        source=source,
                        target=target,
                        params=params,
                        degrees=degrees,
                        use_degree_normalization=use_degree_normalization,
                        include_self_loops=include_self_loops,
                    )
                    * node_features[target - 1][in_feature - 1]
                    for target in range(1, node_count + 1)
                )
            )

        node_output = tuple(
            activation(
                sum(
                    params[f"W_gcn_{layer_index}_{in_feature}_{out_feature}"]
                    * aggregated_features[in_feature - 1]
                    for in_feature in range(1, input_size + 1)
                )
                + params[f"b_gcn_{layer_index}_{out_feature}"]
            )
            for out_feature in range(1, output_size + 1)
        )
        layer_outputs.append(node_output)

    assert len(layer_outputs) == node_count
    return tuple(layer_outputs)


def build_gcn_model(
    node_count: int,
    feature_dim: int,
    hidden_dim: int = 2,
    num_layers: int = 1,
    output_layer: int = 1,
    output_node: int = 1,
    output_projection: bool = True,
    output_hidden_index: int = 1,
    activation: Activation = relu,
    use_degree_normalization: bool = True,
    include_self_loops: bool = True,
) -> SymbolicModel:
    """Create a stacked symbolic graph-convolutional network.

    Parameters
    ----------
    node_count:
        Number of graph nodes.
    feature_dim:
        Input feature width per node.
    hidden_dim:
        Hidden feature width per node for each GCN layer.
    num_layers:
        Number of stacked graph convolution layers.
    output_layer:
        Layer index used for final readout.
    output_node:
        Node index used for scalar readout if projection is disabled.
    output_projection:
        If True, applies a final scalar projection over selected node hidden features.
    output_hidden_index:
        Hidden index used when projection is disabled.
    activation:
        Nonlinearity for each GCN layer.
    use_degree_normalization:
        Enable Kipf-style degree normalization (A_hat = D^{-1/2} A D^{-1/2}).
    include_self_loops:
        Add self-connections in the adjacency.
    """
    if node_count < 1:
        raise ValueError("node_count must be a positive integer")
    if feature_dim < 1 or hidden_dim < 1:
        raise ValueError("feature_dim and hidden_dim must be positive integers")
    if num_layers < 1:
        raise ValueError("num_layers must be a positive integer")
    if output_layer < 1 or output_layer > num_layers:
        raise ValueError("output_layer must be between 1 and num_layers")
    if output_node < 1 or output_node > node_count:
        raise ValueError("output_node must be between 1 and node_count")
    if not output_projection and not 1 <= output_hidden_index <= hidden_dim:
        raise ValueError("output_hidden_index must be between 1 and hidden_dim")

    params: Dict[str, sp.Symbol] = {}
    for source in range(1, node_count + 1):
        for target in range(1, node_count + 1):
            params.setdefault(f"A_{source}_{target}", sp.Symbol(f"A_{source}_{target}", real=True))

    for layer_index in range(1, num_layers + 1):
        in_dim = feature_dim if layer_index == 1 else hidden_dim
        for inp in range(1, in_dim + 1):
            for out in range(1, hidden_dim + 1):
                params.setdefault(
                    f"W_gcn_{layer_index}_{inp}_{out}",
                    sp.Symbol(f"W_gcn_{layer_index}_{inp}_{out}", real=True),
                )
        for out in range(1, hidden_dim + 1):
            params.setdefault(f"b_gcn_{layer_index}_{out}", sp.Symbol(f"b_gcn_{layer_index}_{out}", real=True))

    x = _make_symbols("x", node_count * feature_dim)
    node_features: list[Tuple[Expression, ...]] = [
        x[(node - 1) * feature_dim : node * feature_dim] for node in range(1, node_count + 1)
    ]

    selected_layer_features: list[Tuple[Expression, ...]] = []
    current_features = node_features
    for layer_index in range(1, num_layers + 1):
        in_dim = feature_dim if layer_index == 1 else hidden_dim
        layer_output = _gcn_layer(
            node_features=current_features,
            params=params,
            layer_index=layer_index,
            node_count=node_count,
            input_size=in_dim,
            output_size=hidden_dim,
            activation=activation,
            use_degree_normalization=use_degree_normalization,
            include_self_loops=include_self_loops,
        )
        current_features = list(layer_output)
        if layer_index == output_layer:
            selected_layer_features = list(layer_output)

    selected_node_features = selected_layer_features[output_node - 1]

    if output_projection:
        for hidden_index in range(1, hidden_dim + 1):
            params.setdefault(
                f"W_gcn_out_{output_layer}_{hidden_index}",
                sp.Symbol(f"W_gcn_out_{output_layer}_{hidden_index}", real=True),
            )
        params.setdefault("b_gcn_out", sp.Symbol("b_gcn_out", real=True))
        output = (
            sum(
                params[f"W_gcn_out_{output_layer}_{hidden_index}"]
                * selected_node_features[hidden_index - 1]
                for hidden_index in range(1, hidden_dim + 1)
            )
            + params["b_gcn_out"]
        )
    else:
        output = selected_node_features[output_hidden_index - 1]

    return SymbolicModel(
        name="GCN",
        output=output,
        input_symbols=x,
        parameters=params,
    )


def _multi_head_attention(
    x: Sequence[Tuple[Expression, ...]],
    params: Dict[str, sp.Symbol],
    layer_index: int,
    model_dim: int,
    num_heads: int,
    head_dim: int,
    sequence_length: int,
) -> Tuple[Tuple[Expression, ...], ...]:
    """Return symbolic Multi-Head attention outputs for one layer."""
    per_head_queries: list[list[Tuple[Expression, ...]]] = []
    per_head_keys: list[list[Tuple[Expression, ...]]] = []
    per_head_values: list[list[Tuple[Expression, ...]]] = []

    for head in range(1, num_heads + 1):
        queries = []
        keys = []
        values = []

        for token in x:
            q = tuple(
                sum(
                    params[f"WQ_{layer_index}_{head}_{inp}_{out}"] * token[inp - 1]
                    for inp in range(1, model_dim + 1)
                )
                for out in range(1, head_dim + 1)
            )
            k = tuple(
                sum(
                    params[f"WK_{layer_index}_{head}_{inp}_{out}"] * token[inp - 1]
                    for inp in range(1, model_dim + 1)
                )
                for out in range(1, head_dim + 1)
            )
            v = tuple(
                sum(
                    params[f"WV_{layer_index}_{head}_{inp}_{out}"] * token[inp - 1]
                    for inp in range(1, model_dim + 1)
                )
                for out in range(1, head_dim + 1)
            )

            queries.append(q)
            keys.append(k)
            values.append(v)

        per_head_queries.append(queries)
        per_head_keys.append(keys)
        per_head_values.append(values)

    scaled_attention_outputs: list[list[Tuple[Expression, ...]]] = []
    scale = sp.sqrt(sp.Integer(head_dim))

    for head in range(num_heads):
        head_outputs = []
        q = per_head_queries[head]
        k = per_head_keys[head]
        v = per_head_values[head]

        for t in range(sequence_length):
            scores = []
            for u in range(sequence_length):
                scores.append(sum(q[t][d] * k[u][d] for d in range(head_dim)) / scale)
            denominator = sum(sp.exp(score) for score in scores)
            weights = tuple(sp.exp(score) / denominator for score in scores)
            context = tuple(
                sum(weights[u] * v[u][d] for u in range(sequence_length))
                for d in range(head_dim)
            )
            head_outputs.append(context)
        scaled_attention_outputs.append(head_outputs)

    attention_output: list[Tuple[Expression, ...]] = []
    for t in range(sequence_length):
        concat = ()
        for head in range(num_heads):
            concat += scaled_attention_outputs[head][t]

        head_output = tuple(
            sum(
                params[f"WO_{layer_index}_{out}_{inner}"] * concat[inner - 1]
                for inner in range(1, model_dim + 1)
            )
            for out in range(1, model_dim + 1)
        )
        attention_output.append(head_output)

    return tuple(attention_output)


def build_transformer_model(
    sequence_length: int,
    model_dim: int,
    num_heads: int = 2,
    feed_forward_dim: int | None = None,
    num_layers: int = 1,
    include_positional_embeddings: bool = True,
    output_position: int = 1,
    ff_activation: Activation = relu,
) -> SymbolicModel:
    """Create a symbolic Transformer-style encoder stack.

    The implementation follows the core structure from
    "Attention Is All You Need":
    multi-head self-attention, residual connections, layer normalization,
    and position-wise feed-forward layers.
    """
    if sequence_length < 1:
        raise ValueError("sequence_length must be a positive integer")
    if model_dim < 1:
        raise ValueError("model_dim must be a positive integer")
    if num_heads < 1:
        raise ValueError("num_heads must be a positive integer")
    if model_dim % num_heads != 0:
        raise ValueError("model_dim must be divisible by num_heads")
    if num_layers < 1:
        raise ValueError("num_layers must be a positive integer")
    if not 1 <= output_position <= sequence_length:
        raise ValueError("output_position must be within [1, sequence_length]")

    params: Dict[str, sp.Symbol] = {}
    x = _make_symbols("x", sequence_length * model_dim)

    head_dim = model_dim // num_heads
    ff_dim = feed_forward_dim if feed_forward_dim is not None else max(1, 4 * model_dim)
    if ff_dim < 1:
        raise ValueError("feed_forward_dim must be a positive integer")

    # token embeddings (plus optional sinusoidal-like symbolic positional terms)
    token_embeddings: list[Tuple[Expression, ...]] = []
    for t in range(sequence_length):
        base = x[t * model_dim : (t + 1) * model_dim]
        if include_positional_embeddings:
            token = tuple(
                base[d]
                + params.setdefault(
                    f"P_{t + 1}_{d + 1}",
                    sp.Symbol(f"P_{t + 1}_{d + 1}", real=True),
                )
                for d in range(model_dim)
            )
        else:
            token = base
        token_embeddings.append(token)

    current = token_embeddings

    for layer_index in range(1, num_layers + 1):
        # Attention block parameters
        for head in range(1, num_heads + 1):
            for inp in range(1, model_dim + 1):
                for out in range(1, head_dim + 1):
                    params.setdefault(
                        f"WQ_{layer_index}_{head}_{inp}_{out}",
                        sp.Symbol(f"WQ_{layer_index}_{head}_{inp}_{out}", real=True),
                    )
                    params.setdefault(
                        f"WK_{layer_index}_{head}_{inp}_{out}",
                        sp.Symbol(f"WK_{layer_index}_{head}_{inp}_{out}", real=True),
                    )
                    params.setdefault(
                        f"WV_{layer_index}_{head}_{inp}_{out}",
                        sp.Symbol(f"WV_{layer_index}_{head}_{inp}_{out}", real=True),
                    )

            for out in range(1, model_dim + 1):
                for inner in range(1, model_dim + 1):
                    params.setdefault(
                        f"WO_{layer_index}_{out}_{inner}",
                        sp.Symbol(f"WO_{layer_index}_{out}_{inner}", real=True),
                    )

        for index in range(1, model_dim + 1):
            params.setdefault(f"gamma1_{layer_index}_{index}", sp.Symbol(f"gamma1_{layer_index}_{index}", real=True))
            params.setdefault(f"beta1_{layer_index}_{index}", sp.Symbol(f"beta1_{layer_index}_{index}", real=True))
            params.setdefault(f"gamma2_{layer_index}_{index}", sp.Symbol(f"gamma2_{layer_index}_{index}", real=True))
            params.setdefault(f"beta2_{layer_index}_{index}", sp.Symbol(f"beta2_{layer_index}_{index}", real=True))

        for ff_out in range(1, ff_dim + 1):
            for inp in range(1, model_dim + 1):
                params.setdefault(
                    f"W1_{layer_index}_{inp}_{ff_out}",
                    sp.Symbol(f"W1_{layer_index}_{inp}_{ff_out}", real=True),
                )
            params.setdefault(f"b1_{layer_index}_{ff_out}", sp.Symbol(f"b1_{layer_index}_{ff_out}", real=True))

        for out in range(1, model_dim + 1):
            for ff in range(1, ff_dim + 1):
                params.setdefault(
                    f"W2_{layer_index}_{ff}_{out}",
                    sp.Symbol(f"W2_{layer_index}_{ff}_{out}", real=True),
                )
            params.setdefault(f"b2_{layer_index}_{out}", sp.Symbol(f"b2_{layer_index}_{out}", real=True))

        attention = _multi_head_attention(
            x=current,
            params=params,
            layer_index=layer_index,
            model_dim=model_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            sequence_length=sequence_length,
        )

        after_attention = [
            _vector_add(token, addend)
            for token, addend in zip(current, attention)
        ]
        norm1_gamma = tuple(params[f"gamma1_{layer_index}_{idx}"] for idx in range(1, model_dim + 1))
        norm1_beta = tuple(params[f"beta1_{layer_index}_{idx}"] for idx in range(1, model_dim + 1))
        after_attention = [
            _layer_norm(token, norm1_gamma, norm1_beta)
            for token in after_attention
        ]

        ff_output: list[Tuple[Expression, ...]] = []
        for token in after_attention:
            hidden = tuple(
                ff_activation(
                    sum(
                        params[f"W1_{layer_index}_{inp}_{ff}"] * token[inp - 1]
                        for inp in range(1, model_dim + 1)
                    )
                    + params[f"b1_{layer_index}_{ff}"]
                )
                for ff in range(1, ff_dim + 1)
            )
            projected = tuple(
                sum(params[f"W2_{layer_index}_{ff}_{out}"] * hidden[ff - 1]
                    for ff in range(1, ff_dim + 1))
                + params[f"b2_{layer_index}_{out}"]
                for out in range(1, model_dim + 1)
            )
            ff_output.append(projected)

        after_ff = [
            _vector_add(token, addend) for token, addend in zip(after_attention, ff_output)
        ]
        norm2_gamma = tuple(params[f"gamma2_{layer_index}_{idx}"] for idx in range(1, model_dim + 1))
        norm2_beta = tuple(params[f"beta2_{layer_index}_{idx}"] for idx in range(1, model_dim + 1))
        current = [_layer_norm(token, norm2_gamma, norm2_beta) for token in after_ff]

    # final projection to scalar output for symbolic differentiation
    for out_dim in range(1, model_dim + 1):
        params.setdefault(f"Wfinal_{out_dim}", sp.Symbol(f"Wfinal_{out_dim}", real=True))
    params.setdefault("bfinal", sp.Symbol("bfinal", real=True))
    final_token = current[output_position - 1]
    output = sum(params[f"Wfinal_{d}"] * final_token[d - 1] for d in range(1, model_dim + 1)) + params["bfinal"]

    return SymbolicModel(
        name="Transformer (Attention Is All You Need)",
        output=output,
        input_symbols=x,
        parameters=params,
    )

def derivatives_to_dict(
    model: SymbolicModel,
    first_order: Tuple[Expression, ...] | None = None,
    second_order: Tuple[Tuple[Expression, ...], ...] | None = None,
    *,
    simplify_output: bool = False,
) -> Dict[str, object]:
    """Serialize a model and its full derivatives to a JSON-serializable dict."""
    if first_order is None or second_order is None:
        first_order, second_order = model.gradients()

    if first_order is None or second_order is None:
        raise RuntimeError("Could not compute first and second-order derivatives")

    jacobian = {
        str(symbol): _stringify(grad, simplify_output=simplify_output)
        for symbol, grad in zip(model.input_symbols, first_order)
    }

    hessian = [
        [_stringify(val, simplify_output=simplify_output) for val in row]
        for row in second_order
    ]

    return {
        "model": model.name,
        "input_symbols": [str(symbol) for symbol in model.input_symbols],
        "parameter_count": len(model.parameters),
        "output": _stringify(model.output, simplify_output=simplify_output),
        "jacobian": jacobian,
        "hessian": hessian,
    }


def derivatives_to_latex(
    model: SymbolicModel,
    first_order: Tuple[Expression, ...] | None = None,
    second_order: Tuple[Tuple[Expression, ...], ...] | None = None,
    *,
    simplify_output: bool = False,
) -> str:
    """Serialize symbolic derivatives to a LaTeX-friendly text block."""
    if first_order is None or second_order is None:
        first_order, second_order = model.gradients()

    if first_order is None or second_order is None:
        raise RuntimeError("Could not compute first and second-order derivatives")

    lines: list[str] = []
    lines.append(r"\textbf{" + f"Model: {model.name}" + r"}")
    lines.append(r"f(\mathbf{x}) = " + _to_latex(model.output, simplify_output=simplify_output))

    lines.append(r"\\")
    lines.append(r"\textbf{Jacobian}")
    for symbol, grad in zip(model.input_symbols, first_order):
        lines.append(r"\frac{\partial f}{\partial " + f"{symbol}} = " + _to_latex(grad, simplify_output=simplify_output))

    lines.append(r"\\")
    lines.append(r"\textbf{Hessian}")
    for i, row in enumerate(second_order, start=1):
        equations = []
        for j, val in enumerate(row, start=1):
            equations.append(
                r"\frac{\partial^2 f}{\partial "
                + f"x_{i} \partial x_{j}} = "
                + _to_latex(val, simplify_output=simplify_output)
            )
        lines.append(r"\\
".join(equations))

    return "\n".join(lines)


__all__ = [
    "Expression",
    "Activation",
    "SymbolicModel",
    "soft_relu",
    "relu",
    "tanh",
    "sigmoid",
    "_activation_from_name",
    "build_mlp_model",
    "build_lstm_model",
    "build_gru_model",
    "build_gcn_model",
    "build_transformer_model",
    "derivatives_to_dict",
    "derivatives_to_latex",
]

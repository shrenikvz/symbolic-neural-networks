"""CLI for selecting symbolic neural-network families and computing gradients."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import sympy as sp

from symbolic import (
    _activation_from_name,
    build_gru_model,
    build_gcn_model,
    build_lstm_model,
    build_mlp_model,
    build_transformer_model,
    derivatives_to_dict,
    derivatives_to_latex,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build symbolic expressions for neural networks and compute derivatives."
    )
    parser.add_argument(
        "network",
        choices=["mlp", "lstm", "gru", "gcn", "transformer"],
        help="Neural architecture to construct.",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=2,
        help="Input width (per-time-step input for recurrent models)",
    )
    parser.add_argument("--hidden-layers", nargs="+", type=int, default=[2, 3], help="MLP hidden widths")
    parser.add_argument("--hidden-size", type=int, default=2, help="LSTM/GRU hidden width")
    parser.add_argument("--lstm-layers", type=int, default=1, help="LSTM layer count")
    parser.add_argument("--lstm-output-layer", type=int, default=1, help="LSTM layer used for final output")
    parser.add_argument(
        "--lstm-output-time",
        type=int,
        default=0,
        help="LSTM time step for output (1-based); 0 means last time step",
    )
    parser.add_argument("--lstm-output-hidden", type=int, default=1, help="LSTM output hidden index if no projection")
    parser.add_argument(
        "--lstm-no-projection",
        action="store_true",
        help="Disable final scalar projection and return one hidden unit directly",
    )
    parser.add_argument("--gru-layers", type=int, default=1, help="GRU layer count")
    parser.add_argument("--gru-output-layer", type=int, default=1, help="GRU layer used for final output")
    parser.add_argument(
        "--gru-output-time",
        type=int,
        default=0,
        help="GRU time step for output (1-based); 0 means last time step",
    )
    parser.add_argument("--gru-output-hidden", type=int, default=1, help="GRU output hidden index if no projection")
    parser.add_argument(
        "--gru-no-projection",
        action="store_true",
        help="Disable final scalar projection and return one hidden unit directly",
    )
    parser.add_argument("--gcn-layers", type=int, default=1, help="GCN layer count")
    parser.add_argument("--gcn-hidden-dim", type=int, default=2, help="GCN hidden feature size")
    parser.add_argument("--gcn-output-layer", type=int, default=1, help="GCN layer used for final output")
    parser.add_argument("--gcn-output-node", type=int, default=1, help="Node index for final GCN readout")
    parser.add_argument("--gcn-output-hidden", type=int, default=1, help="GCN hidden index if no projection")
    parser.add_argument(
        "--gcn-no-projection",
        action="store_true",
        help="Disable final scalar projection and return one hidden feature",
    )
    parser.add_argument(
        "--gcn-no-self-loops",
        action="store_true",
        help="Disable self-loop terms in adjacency",
    )
    parser.add_argument(
        "--gcn-disable-degree-normalization",
        action="store_true",
        help="Disable degree-normalized adjacency weights",
    )
    parser.add_argument(
        "--gcn-activation",
        choices=["soft_relu", "tanh", "sigmoid", "relu"],
        default="relu",
        help="Activation for GCN layers",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=3,
        help="Sequence length for LSTM or Transformer",
    )
    parser.add_argument("--model-dim", type=int, default=2, help="Transformer model dimension")
    parser.add_argument("--transformer-heads", type=int, default=2, help="Transformer number of attention heads")
    parser.add_argument(
        "--transformer-ffn-dim",
        type=int,
        default=0,
        help="Transformer feed-forward inner dimension (default 4*model_dim)",
    )
    parser.add_argument("--transformer-layers", type=int, default=1, help="Transformer number of encoder layers")
    parser.add_argument(
        "--transformer-output-position",
        type=int,
        default=1,
        help="Transformer position index used for final scalar projection",
    )
    parser.add_argument(
        "--disable-transformer-positional-embeddings",
        action="store_true",
        help="Disable positional embeddings in transformer input construction",
    )
    parser.add_argument("--node-count", type=int, default=3, help="Number of graph nodes")
    parser.add_argument("--feature-dim", type=int, default=2, help="Graph node feature dimension")
    parser.add_argument(
        "--activation",
        choices=["soft_relu", "tanh", "sigmoid", "relu"],
        default="soft_relu",
        help="Activation for MLP hidden layers",
    )
    parser.add_argument("--show-params", action="store_true", help="Print parameter symbols")
    parser.add_argument("--latex", action="store_true", help="Print LaTeX-formatted equations")
    parser.add_argument(
        "--show-full-hessian",
        action="store_true",
        help="Print full Hessian matrix entries (not only diagonal)",
    )
    parser.add_argument(
        "--export-json",
        type=Path,
        help="Path to write serialized derivatives as JSON",
    )
    parser.add_argument(
        "--export-latex",
        type=Path,
        help="Path to write derivatives in LaTeX text format",
    )
    parser.add_argument(
        "--simplify",
        action="store_true",
        help="Apply sympy.simplify to exported expressions",
    )
    return parser.parse_args()


def _print_expr(prefix: str, expr: sp.Expr, latex: bool = False) -> None:
    if latex:
        print(f"{prefix} = ${sp.latex(expr)}$")
    else:
        print(f"{prefix} = {sp.simplify(expr)}")


def _build_model(args: argparse.Namespace):
    if args.network == "mlp":
        return build_mlp_model(
            input_size=args.input_size,
            hidden_layers=args.hidden_layers,
            activation=_activation_from_name(args.activation),
        )
    if args.network == "lstm":
        output_time = args.lstm_output_time if args.lstm_output_time > 0 else None
        return build_lstm_model(
            input_size=args.input_size,
            hidden_size=args.hidden_size,
            sequence_length=args.sequence_length,
            num_layers=args.lstm_layers,
            output_layer=args.lstm_output_layer,
            output_time=output_time,
            output_projection=not args.lstm_no_projection,
            output_hidden_index=args.lstm_output_hidden,
            )
    if args.network == "gru":
        output_time = args.gru_output_time if args.gru_output_time > 0 else None
        return build_gru_model(
            input_size=args.input_size,
            hidden_size=args.hidden_size,
            sequence_length=args.sequence_length,
            num_layers=args.gru_layers,
            output_layer=args.gru_output_layer,
            output_time=output_time,
            output_projection=not args.gru_no_projection,
            output_hidden_index=args.gru_output_hidden,
        )
    if args.network == "gcn":
        return build_gcn_model(
            node_count=args.node_count,
            feature_dim=args.feature_dim,
            hidden_dim=args.gcn_hidden_dim,
            num_layers=args.gcn_layers,
            output_layer=args.gcn_output_layer,
            output_node=args.gcn_output_node,
            output_projection=not args.gcn_no_projection,
            output_hidden_index=args.gcn_output_hidden,
            activation=_activation_from_name(args.gcn_activation),
            include_self_loops=not args.gcn_no_self_loops,
            use_degree_normalization=not args.gcn_disable_degree_normalization,
        )
    if args.network == "transformer":
        ffn_dim = args.transformer_ffn_dim if args.transformer_ffn_dim > 0 else None
        return build_transformer_model(
            sequence_length=args.sequence_length,
            model_dim=args.model_dim,
            num_heads=args.transformer_heads,
            feed_forward_dim=ffn_dim,
            num_layers=args.transformer_layers,
            include_positional_embeddings=not args.disable_transformer_positional_embeddings,
            output_position=args.transformer_output_position,
        )
    raise RuntimeError("Unsupported network type")


def _write_exports(model, first_order, second_order, args: argparse.Namespace) -> None:
    if args.export_json:
        payload = derivatives_to_dict(
            model,
            first_order,
            second_order,
            simplify_output=args.simplify,
        )
        args.export_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote JSON derivatives to {args.export_json}")

    if args.export_latex:
        latex_payload = derivatives_to_latex(
            model,
            first_order,
            second_order,
            simplify_output=args.simplify,
        )
        args.export_latex.write_text(latex_payload, encoding="utf-8")
        print(f"Wrote LaTeX derivatives to {args.export_latex}")


def main() -> None:
    args = _parse_args()
    model = _build_model(args)

    print(f"Model: {model.name}")
    print(f"Inputs: {', '.join(str(s) for s in model.input_symbols)}")

    first_order, second_order = model.gradients()

    _print_expr("f(inputs)", model.output, latex=args.latex)

    print("\nFirst-order gradients:")
    for symbol, grad in zip(model.input_symbols, first_order):
        _print_expr(f"∂f/∂{symbol}", grad, latex=args.latex)

    if args.show_full_hessian:
        print("\nFull Hessian:")
        for symbol_row, row in zip(model.input_symbols, second_order):
            for symbol_col, value in zip(model.input_symbols, row):
                _print_expr(f"∂²f/∂{symbol_row}∂{symbol_col}", value, latex=args.latex)
    else:
        print("\nSecond-order derivatives (diagonal Hessian):")
        for idx, symbol in enumerate(model.input_symbols):
            _print_expr(f"∂²f/∂{symbol}²", second_order[idx][idx], latex=args.latex)

    if args.show_params:
        print("\nParameters:")
        for key in sorted(model.parameters):
            print(f"  {key}")

    _write_exports(model, first_order, second_order, args)


if __name__ == "__main__":
    main()

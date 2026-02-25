"""Batch symbolic inspection utilities for all supported model families.

This script is intended as a reproducible starting point for research experiments.
It builds compact models, prints quick summaries, and writes JSON/LaTeX derivative
exports to `examples/outputs/`.
"""

from __future__ import annotations

import json
from pathlib import Path

from symbolic_neural_networks import (
    build_lstm_model,
    build_mlp_model,
    build_gcn_model,
    build_gru_model,
    build_transformer_model,
    derivatives_to_dict,
    derivatives_to_latex,
)

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def _run(name: str, model) -> None:
    first_order, second_order = model.gradients()

    payload = derivatives_to_dict(
        model,
        first_order,
        second_order,
        simplify_output=False,
    )

    json_path = OUTPUT_DIR / f"{name}.json"
    latex_path = OUTPUT_DIR / f"{name}.tex"

    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    latex_path.write_text(derivatives_to_latex(model, first_order, second_order), encoding="utf-8")

    print(f"[{name}] output expression:")
    print(f"  {model.output}")
    print(f"  jacobian: {len(first_order)} entries")
    print(f"  hessian: {len(second_order)}x{len(second_order)} matrix")
    print(f"  artifacts -> {json_path.name}, {latex_path.name}")


def main() -> None:
    models = {
        "mlp": build_mlp_model(input_size=2, hidden_layers=[2, 3]),
        "lstm": build_lstm_model(
            input_size=2,
            hidden_size=2,
            sequence_length=3,
            num_layers=2,
            output_time=3,
            output_projection=True,
        ),
        "gru": build_gru_model(
            input_size=2,
            hidden_size=2,
            sequence_length=3,
            num_layers=1,
            output_time=3,
            output_projection=True,
        ),
        "gcn": build_gcn_model(
            node_count=3,
            feature_dim=2,
            hidden_dim=3,
            num_layers=1,
            output_layer=1,
            output_node=1,
            output_projection=True,
        ),
        "transformer": build_transformer_model(
            sequence_length=2,
            model_dim=2,
            num_heads=1,
            num_layers=1,
            feed_forward_dim=4,
        ),
    }

    for name, model in models.items():
        _run(name, model)


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import shutil
from pathlib import Path
from datetime import datetime, timezone

import numpy as np

from surrogate_tool.contracts.run_config import load_run_config
from surrogate_tool.contracts.status import write_status
from surrogate_tool.paths import runs_root


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _to_modelica_array_1d(vec) -> str:
    return "{" + ",".join(f"{float(x):.16g}" for x in vec) + "}"


def _to_modelica_array_2d(mat) -> str:
    rows = []
    for row in mat:
        rows.append("{" + ",".join(f"{float(x):.16g}" for x in row) + "}")
    return "{" + ",".join(rows) + "}"


def _clean_package_dir(pkg_dir: Path) -> None:
    """
    Delete generated package folder to avoid stale .mo conflicts.
    """
    if pkg_dir.exists():
        shutil.rmtree(pkg_dir, ignore_errors=True)
    pkg_dir.mkdir(parents=True, exist_ok=True)


def _act_to_layer_func(act: str) -> str:
    return "relu" if (act or "").lower() == "relu" else "identity"


def _pin_y_positions(n: int, y_top: int = 80, y_bot: int = -80) -> list[int]:
    if n <= 1:
        return [0]
    step = (y_top - y_bot) / (n - 1)
    return [int(round(y_top - i * step)) for i in range(n)]


def _placement(extent: tuple[tuple[int, int], tuple[int, int]]) -> str:
    """
    Placement annotation with iconTransformation so pins show in OMEdit Diagram/Icon view.
    """
    (x1, y1), (x2, y2) = extent
    # Emit literal braces safely (no Python f-string brace errors)
    return (
        f'annotation(Placement(transformation(extent={{{{{x1},{y1}}},{{{x2},{y2}}}}}), '
        f'iconTransformation(extent={{{{{x1},{y1}}},{{{x2},{y2}}}}})))'
    )


def _load_weights_as_modelica_Wb(export: dict) -> tuple[list[np.ndarray], list[np.ndarray], list[str]]:
    """
    Deterministic conversion:
      - mlp_weights.json stores Dense weights as Keras convention: W_tf shape (in_dim, out_dim)
      - Modelica dense() expects: W shape (out_dim, in_dim)
    So: W_modelica = W_tf.T ALWAYS.

    Returns:
      Ws: list of W_modelica arrays (out_dim, in_dim)
      bs: list of b arrays (out_dim,)
      acts: list of activations mapped to relu/identity
    """
    layers = export.get("layers", [])
    if not layers:
        raise ValueError("mlp_weights.json contains no layers.")

    Ws: list[np.ndarray] = []
    bs: list[np.ndarray] = []
    acts: list[str] = []

    for li, L in enumerate(layers, start=1):
        W_tf = np.array(L["W"], dtype=float)  # (in,out)
        b = np.array(L["b"], dtype=float)     # (out,)

        if W_tf.ndim != 2:
            raise ValueError(f"Layer {li}: weights must be 2D, got shape={W_tf.shape}")
        if b.ndim != 1:
            raise ValueError(f"Layer {li}: bias must be 1D, got shape={b.shape}")

        W = W_tf.T  # (out,in)

        if W.shape[0] != b.shape[0]:
            raise ValueError(
                f"Layer {li}: after transpose expected W.shape[0]==len(b) "
                f"({b.shape[0]}), got W.shape={W.shape} from W_tf.shape={W_tf.shape}"
            )

        Ws.append(W)
        bs.append(b)
        acts.append(_act_to_layer_func(L.get("activation", "identity")))

    return Ws, bs, acts


def export_modelica(run_id: str, attempt_num: int) -> dict:
    """
    Export a Modelica package for the trained surrogate:
      - scalar pins u1..uN and y1..yM
      - pins visible in OMEdit via Placement + iconTransformation
      - deterministic weights transpose (removes layout inference errors)
      - clean export folder
    """
    run_dir = runs_root() / run_id
    cfg_path = run_dir / "input" / "run_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"run_config.json not found at: {cfg_path}")

    cfg = load_run_config(cfg_path)

    status_path = run_dir / "status.json"
    write_status(status_path, state="EXPORT_MODELICA", message="Preparing Modelica export...", progress=0.1)

    attempt_dir = run_dir / "attempts" / f"attempt_{attempt_num:03d}"
    processed_dir = attempt_dir / "processed"
    modelica_root = attempt_dir / "modelica"

    weights_path = processed_dir / "mlp_weights.json"
    if not weights_path.exists():
        raise FileNotFoundError(f"Missing {weights_path}. Run training first.")

    export = json.loads(weights_path.read_text(encoding="utf-8"))
    pkg_name = cfg.package_name

    pkg_dir = modelica_root / pkg_name
    _clean_package_dir(pkg_dir)

    MODELICA_VERSION = "4.0.0"
    MODEL_NAME = "SurrogateMLP"
    EX_WRAPPER = "ValidateSurrogateMLP"
    EX_CONST = "ValidateSurrogateMLP_Constant"

    # IMPORTANT: training-exported order (this defines u1..uN mapping)
    input_cols = export["input_cols"]
    output_cols = export["target_cols"]

    x_mean = np.array(export["x_scaler"]["mean"], dtype=float)
    x_scale = np.array(export["x_scaler"]["scale"], dtype=float)
    y_mean = np.array(export["y_scaler"]["mean"], dtype=float)
    y_scale = np.array(export["y_scaler"]["scale"], dtype=float)

    n_in = int(len(x_mean))
    n_out = int(len(y_mean))

    Ws, bs, acts = _load_weights_as_modelica_Wb(export)

    # ---------------- Top package ----------------
    _write_text(pkg_dir / "package.mo", f"""within ;
package {pkg_name}
  annotation(uses(Modelica(version="{MODELICA_VERSION}")));
end {pkg_name};
""")
    _write_text(pkg_dir / "package.order", "Layers\nNetworks\nExamples\n")

    # ---------------- Layers package ----------------
    layers_dir = pkg_dir / "Layers"
    _write_text(layers_dir / "package.mo", f"""within {pkg_name};
package Layers
end Layers;
""")
    _write_text(layers_dir / "package.order", "dense\nrelu\naffine_scale\naffine_unscale\nidentity\n")

    _write_text(layers_dir / "identity.mo", f"""within {pkg_name}.Layers;
function identity
  input Real x[:];
  output Real y[size(x,1)];
algorithm
  y := x;
end identity;
""")

    _write_text(layers_dir / "dense.mo", f"""within {pkg_name}.Layers;
function dense
  input Real x[:];
  input Real W[:, :]; // W[n_out, n_in]
  input Real b[:];    // b[n_out]
  output Real y[size(W,1)];
algorithm
  for i in 1:size(W,1) loop
    y[i] := b[i];
    for j in 1:size(W,2) loop
      y[i] := y[i] + W[i,j] * x[j];
    end for;
  end for;
end dense;
""")

    _write_text(layers_dir / "relu.mo", f"""within {pkg_name}.Layers;
function relu
  input Real x[:];
  output Real y[size(x,1)];
algorithm
  for i in 1:size(x,1) loop
    y[i] := noEvent(if x[i] > 0 then x[i] else 0);
  end for;
end relu;
""")

    _write_text(layers_dir / "affine_scale.mo", f"""within {pkg_name}.Layers;
function affine_scale
  input Real x[:];
  input Real mean[:];
  input Real scale[:];
  output Real y[size(x,1)];
protected
  constant Real eps = 1e-12;
  Real denom;
algorithm
  for i in 1:size(x,1) loop
    denom := noEvent(if abs(scale[i]) > eps then scale[i] else 1.0);
    y[i] := (x[i] - mean[i]) / denom;
  end for;
end affine_scale;
""")

    _write_text(layers_dir / "affine_unscale.mo", f"""within {pkg_name}.Layers;
function affine_unscale
  input Real x[:];
  input Real mean[:];
  input Real scale[:];
  output Real y[size(x,1)];
algorithm
  for i in 1:size(x,1) loop
    y[i] := x[i] * scale[i] + mean[i];
  end for;
end affine_unscale;
""")

    # ---------------- Networks package ----------------
    networks_dir = pkg_dir / "Networks"
    _write_text(networks_dir / "package.mo", f"""within {pkg_name};
package Networks
end Networks;
""")
    _write_text(networks_dir / "package.order", f"{MODEL_NAME}\n")

    # weights/biases constants
    param_lines = []
    for k, (W, b) in enumerate(zip(Ws, bs), start=1):
        param_lines.append(f"  constant Real W{k}[{W.shape[0]},{W.shape[1]}] = {_to_modelica_array_2d(W.tolist())};")
        param_lines.append(f"  constant Real b{k}[{b.shape[0]}] = {_to_modelica_array_1d(b.tolist())};")

    # pin placement + declarations
    y_in = _pin_y_positions(n_in)
    y_out = _pin_y_positions(n_out)

    in_decl = []
    for i, (nm, yy) in enumerate(zip(input_cols, y_in), start=1):
        in_decl.append(
            f'  Modelica.Blocks.Interfaces.RealInput u{i} "{nm}" '
            f'{_placement(((-120, yy-10), (-80, yy+10)))};'
        )

    out_decl = []
    for i, (nm, yy) in enumerate(zip(output_cols, y_out), start=1):
        out_decl.append(
            f'  Modelica.Blocks.Interfaces.RealOutput y{i} "{nm}" '
            f'{_placement(((80, yy-10), (120, yy+10)))};'
        )

    # Wiring map comment (helps prevent wrong connections)
    wiring_map = []
    wiring_map.append("  // ================= WIRING MAP (ORDER MATTERS) =================")
    wiring_map.append("  // u1..uN correspond to input columns in this exact order:")
    for idx, nm in enumerate(input_cols, start=1):
        wiring_map.append(f"  //   u{idx} = {nm}")
    wiring_map.append("  // y1..yM correspond to output columns in this exact order:")
    for idx, nm in enumerate(output_cols, start=1):
        wiring_map.append(f"  //   y{idx} = {nm}")
    wiring_map.append("  // ==============================================================")

    # protected vectors
    prot = ["protected"]
    prot.append(f"  Real uVec[{n_in}];")
    prot.append(f"  Real x_s[{n_in}];")
    for li in range(1, len(Ws)):
        prot.append(f"  Real h{li}[{Ws[li-1].shape[0]}];")
    prot.append(f"  Real y_s[{n_out}];")
    prot.append(f"  Real yVec[{n_out}];")

    # equations
    eq = []
    eq.append("  // Pack scalar inputs into a vector (training order)")
    eq.append("  uVec = {" + ",".join([f"u{i}" for i in range(1, n_in + 1)]) + "};")
    eq.append("")
    eq.append("  // Scale inputs")
    eq.append("  x_s = Layers.affine_scale(uVec, x_mean, x_scale);")
    eq.append("")
    eq.append("  // Forward pass")

    n_layers = len(Ws)
    if n_layers == 0:
        raise ValueError("No Dense layers found in export.")

    # Hidden layers
    for li in range(1, n_layers):
        src = "x_s" if li == 1 else f"h{li-1}"
        if acts[li - 1] == "relu":
            eq.append(f"  h{li} = Layers.relu(Layers.dense({src}, W{li}, b{li}));")
        else:
            eq.append(f"  h{li} = Layers.identity(Layers.dense({src}, W{li}, b{li}));")

    # Output layer
    out_src = "x_s" if n_layers == 1 else f"h{n_layers - 1}"
    if acts[n_layers - 1] == "relu":
        eq.append(f"  y_s = Layers.relu(Layers.dense({out_src}, W{n_layers}, b{n_layers}));")
    else:
        eq.append(f"  y_s = Layers.dense({out_src}, W{n_layers}, b{n_layers});")

    eq.append("")
    eq.append("  // Unscale outputs")
    eq.append("  yVec = Layers.affine_unscale(y_s, y_mean, y_scale);")
    eq.append("")
    eq.append("  // Unpack vector outputs to scalar pins")
    for i in range(1, n_out + 1):
        eq.append(f"  y{i} = yVec[{i}];")

    network_text = f"""within {pkg_name}.Networks;
model {MODEL_NAME}
  import {pkg_name}.Layers;

{chr(10).join(in_decl)}
{chr(10).join(out_decl)}

{chr(10).join(wiring_map)}

  // Learned scalers
  constant Real x_mean[{n_in}] = {_to_modelica_array_1d(x_mean)};
  constant Real x_scale[{n_in}] = {_to_modelica_array_1d(x_scale)};
  constant Real y_mean[{n_out}] = {_to_modelica_array_1d(y_mean)};
  constant Real y_scale[{n_out}] = {_to_modelica_array_1d(y_scale)};

  // Dense weight layout: Modelica W[out,in] = transpose(Keras W[in,out])
{chr(10).join(param_lines)}
{chr(10).join(prot)}
equation
{chr(10).join(eq)}
end {MODEL_NAME};
"""
    _write_text(networks_dir / f"{MODEL_NAME}.mo", network_text)

    # ---------------- Examples package ----------------
    examples_dir = pkg_dir / "Examples"
    _write_text(examples_dir / "package.mo", f"""within {pkg_name};
package Examples
end Examples;
""")
    _write_text(examples_dir / "package.order", f"{EX_WRAPPER}\n{EX_CONST}\n")

    # Wrapper model: provides array ports u[:] and y[:] and connects to scalar pins
    _write_text(examples_dir / f"{EX_WRAPPER}.mo", f"""within {pkg_name}.Examples;
model {EX_WRAPPER}
  Modelica.Blocks.Interfaces.RealInput u[{n_in}];
  Modelica.Blocks.Interfaces.RealOutput y[{n_out}];
  {pkg_name}.Networks.{MODEL_NAME} nn;
equation
""" + "\n".join([f"  connect(u[{i}], nn.u{i});" for i in range(1, n_in + 1)]) + "\n" +
"\n".join([f"  connect(nn.y{i}, y[{i}]);" for i in range(1, n_out + 1)]) + f"""
end {EX_WRAPPER};
""")

    # Constant smoke-test: use training data mean per input (statistically meaningful defaults)
    u_test = x_mean.tolist()
    const_sources = []
    const_connects = []
    for i in range(1, n_in + 1):
        const_sources.append(f"  Modelica.Blocks.Sources.Constant c{i}(k={u_test[i-1]});")
        const_connects.append(f"  connect(c{i}.y, nn.u{i});")

    _write_text(examples_dir / f"{EX_CONST}.mo", f"""within {pkg_name}.Examples;
model {EX_CONST}
  {pkg_name}.Networks.{MODEL_NAME} nn;
{chr(10).join(const_sources)}
equation
{chr(10).join(['  ' + c for c in const_connects])}
  annotation(experiment(StopTime=1.0));
end {EX_CONST};
""")

    write_status(status_path, state="EXPORT_MODELICA_DONE", message="Modelica export complete.", progress=1.0)

    return {
        "timestamp_utc": _utc_now(),
        "run_id": run_id,
        "attempt": f"attempt_{attempt_num:03d}",
        "package_name": pkg_name,
        "package_dir": str(pkg_dir),
        "model_file": str(networks_dir / f"{MODEL_NAME}.mo"),
    }
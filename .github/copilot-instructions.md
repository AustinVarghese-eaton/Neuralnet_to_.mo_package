# SurrogateGenerator — Copilot Instructions

## What This Project Does

Trains a TensorFlow MLP on engineering tabular I/O data (CSV/XLSX) and exports it as a **Modelica surrogate package (`.mo`)**. Target users: simulation engineers using Dymola/Modelica toolchains.

Two interfaces:
- **ExcelUI** (`ExcelUI/SurrogateGeneratorUI.xlsm`) — VBA front-end calling Python via `WScript.Shell`
- **CLI** — `python -m surrogate_tool <command>`

---

## Repository Layout

```
Engine/src/surrogate_tool/   ← Python backend (installable package)
  cli.py                     ← All argparse subcommands
  paths.py                   ← project_root() / runs_root()
  pipeline/                  ← 7 ordered stages
    preprocess.py
    eda.py
    split_scale.py
    train.py
    modelica_export.py
    fmu_export.py            ← NEW: FMU 2.0 Model Exchange generation + DLL compilation
    report.py
    orchestrator.py          ← run_full_attempt(run_id, attempt_num)
  contracts/
    run_config.py            ← Pydantic RunConfig
    status.py                ← write_status() → status.json
  io/dataset_loader.py       ← preview_headers(), load_dataset()
  attempts/manager.py        ← select best attempt by physical RMSE

ExcelUI/
  SurrogateGeneratorUI.xlsm  ← VBA-driven front-end
  build_ui_v2.py             ← Rebuilds .xlsm from scratch (idempotent)

runs/                        ← Runtime output; never commit contents
samples/                     ← Example datasets
```

---

## Run Directory Structure

```
runs/<run_id>/
  input/run_config.json          ← RunConfig (input_columns, output_columns, package_name)
  status.json                    ← StatusMessage (state, progress 0–1, message)
  latest.json                    ← Best attempt metadata
  attempts/attempt_001/
    processed/                   ← cleaned_dataset.csv, scalers, mlp_weights.json, .keras
    reports/                     ← metrics.json, report.html, figures/
    logs/pipeline.log
    modelica/<PackageName>/      ← Generated Modelica package (.mo files)
    fmu/<PackageName>.fmu        ← FMU 2.0 Model Exchange archive
      modelDescription.xml       ← FMI 2.0 compliant variable definitions
      sources/model.c            ← MLP forward pass in C (weights embedded)
      binaries/win64/<Name>.dll  ← Pre-compiled Windows DLL (requires gcc)
```

- `runs/latest_run.json` always points to the most recent run.
- Attempt numbering is **1-based, zero-padded**: `attempt_001`, `attempt_002`, …
- **Best attempt** = lowest physical-unit RMSE (not scaled RMSE).

---

## Key CLI Commands

```powershell
python -m surrogate_tool preview-headers --dataset path/to/data.csv
python -m surrogate_tool make-run --dataset path/to/data.csv --package MyModel --gcc-path "C:\path\to\gcc.exe"
python -m surrogate_tool retrain --run-id 20260203_093642
python -m surrogate_tool create-fmu --run-id 20260203_093642 --attempt 1
python -m surrogate_tool best-attempt --run-id 20260203_093642
python -m surrogate_tool report --run-id 20260203_093642 --attempt 1
```

Install: `pip install -r Engine/requirements.txt` (venv at `.venv/`)

---

## Pipeline Stages (orchestrator.py order)

1. **preprocess** — remove NaN/duplicates/outliers, validate columns
2. **eda** — correlation heatmaps, distributions, outlier plots
3. **split_scale** — 70/15/15 train/val/test; `StandardScaler` fit on train only (no data leakage)
4. **train** — TensorFlow MLP with early stopping; saves `surrogate_model.keras` + `mlp_weights.json`
5. **modelica_export** — reads `mlp_weights.json`, generates nested Modelica package
6. **create_fmu** — generates FMU 2.0 Model Exchange (`.fmu`) from `mlp_weights.json`; compiles Windows DLL via gcc
7. **report** — Jinja2 HTML with RMSE/MAE, loss curves, figures

---

## Key Data Contracts

### RunConfig (`contracts/run_config.py`)
Pydantic model — the central config for every run.

```python
RunConfig(
    run_id="20260203_093642",
    dataset_path="...",          # Must exist; .csv or .xlsx only
    dataset_format="csv",
    input_columns=["Temp", "Voltage"],
    output_columns=["PowerLoss"],
    package_name="IGBT_NN",      # Regex: ^[A-Za-z][A-Za-z0-9_]*$
    random_seed=42,
    gcc_path=None,               # Optional: path to gcc.exe for FMU DLL compilation
                                 # Auto-detected from OpenModelica install if None
)
```

- `package_name` is validated against `^[A-Za-z][A-Za-z0-9_]*$` — always validate before writing.
- JSON reads use `utf-8-sig` (handles Windows BOM); writes use `utf-8`.
- Use `load_run_config(path)` / `save_run_config(cfg, path)` — never read the JSON manually.

### StatusMessage (`contracts/status.py`)
Written to `status.json`; polled by ExcelUI for progress.

```python
write_status(status_path, state="RUNNING", message="Training...", progress=0.6)
```

States: `READY`, `RUNNING`, `DONE`, `ERROR`

### mlp_weights.json
Serialized trained network: `input_cols`, `target_cols`, `layers[]` (W, b, activation), `x_scaler`, `y_scaler`. This is what gets compiled into Modelica and FMU.
- Scaler keys are `mean` and `scale` (not `mean_` / `scale_`) — do not confuse with sklearn attribute names.
- Layer weight matrix `W` is shape `[n_out_layer][n_in_layer]` (already transposed for direct matmul).

### FMU 2.0 Model Exchange (`fmu_export.py`)
- Output: `attempts/attempt_001/fmu/<PackageName>.fmu` — a zip containing `modelDescription.xml`, `sources/model.c`, and optionally `binaries/win64/<PackageName>.dll`.
- gcc auto-detection order: `C:\OpenModelica*\tools\msys\ucrt64\bin\gcc.exe` (OM 1.24+) → `mingw64` variant → system PATH → `cfg.gcc_path`.
- gcc's own bin directory is always prepended to subprocess PATH so `cc1`/`as`/`ld` are found.
- FMI 2.0 XML rules enforced: continuous inputs must NOT have `initial="exact"`, must have `<Real start="0.0"/>`, and `<ModelExchange>` must NOT have `needsDirectionDerivatives` attribute.
- `numberOfContinuousStates="0"` — pure algebraic model, no ODE states.

---

## Coding Conventions

- **`from __future__ import annotations`** at the top of every Python file.
- **Pathlib exclusively** — never `os.path`. All paths are `Path` objects.
- **No hardcoded paths** — always use `runs_root()` and `project_root()` from `paths.py`.
- **UTF-8-SIG for reads, UTF-8 for writes** — all JSON files from Windows tools may have BOM.
- **matplotlib headless** — always call `matplotlib.use("Agg")` before importing pyplot; the tool runs in Excel/headless context.
- **Pydantic for validation** — use `@field_validator`, not manual if/raise patterns.
- **`from __future__ import annotations`** is already set; all type hints use lowercase generics (`list[str]`, `dict[str, int]`).

---

## ExcelUI ↔ Python Integration

VBA calls Python via `WScript.Shell` using `.venv\Scripts\python.exe`.

**9-button workflow:**

| # | Button | VBA Action | CLI Command |
|---|--------|-----------|-------------|
| 1 | Pick Dataset | `Action_PickDataset` | — |
| 2 | Load Headers | `Action_LoadHeaders` | `preview-headers` |
| 3 | Build Mapping | `Action_BuildMapping` | — |
| 4 | Process (green) | `Action_Process` | `make-run` + `retrain` |
| 5 | Retrain | `Action_Retrain` | `retrain` |
| 6 | Save Best .mo | `Action_SaveBestMo` | — |
| 7 | Open Report | `Action_OpenReport` | — |
| 8 | Open Run Folder | `Action_OpenRunFolder` | — |

**Named ranges (col J of "SurrogateGenerator" sheet):**
- `UI_HeadersCache` — pipe-delimited column names from dataset
- `UI_InputCols` — comma-separated selected input columns
- `UI_OutputCols` — comma-separated selected output columns

**Column mapping sheet:**
- Created dynamically by `Action_BuildMapping`; named `"mapping"`
- Headers cached in hidden col E (DV source) to avoid the 255-char Excel formula limit
- Row counts stored in hidden col G so `Action_ConfirmMapping` knows how many rows to read

To rebuild the `.xlsm` from scratch: `python ExcelUI/build_ui_v2.py` (idempotent).

---

## Common Pitfalls

- **Never commit `runs/`** — runtime output only; gitignored.
- **Attempt numbering is 1-based** — `attempt_001` not `attempt_000`.
- **Best attempt = physical RMSE**, not scaled RMSE — check `attempts/manager.py`.
- **No tests yet** — `Engine/tests/` is empty. Write pytest tests there.
- **`create-fmu` is part of `run_full_attempt()`** — runs automatically between `modelica_export` and `report`; also callable standalone via `python -m surrogate_tool create-fmu --run-id <id> --attempt 1`.
- **FMU XML pitfalls** — continuous inputs: no `initial="exact"`, require `<Real start="0.0"/>`. `<ModelExchange>` must not have `needsDirectionDerivatives`. Violating these causes OMEdit FMI2XML errors even though simulation may still run.
- **gcc subprocess needs its own bin on PATH** — always prepend `Path(gcc_exe).parent` to `env["PATH"]` when calling `subprocess.run`; otherwise `cc1`/`as` not found and compile silently fails with rc=1.
- **mlp_weights.json scaler keys** — are `mean`/`scale`, not `mean_`/`scale_`.
- **`package_name` regex** — forgetting to validate before writing `RunConfig` causes a Pydantic error at construction time, not at export time.
- **XLSX sheet name** — must be passed explicitly if not `Sheet1`; defaults to `None` in `RunConfig` (loader picks first sheet).

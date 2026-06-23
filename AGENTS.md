# SurrogateGenerator — Agent Instructions

**Project:** One-click pipeline that trains an MLP on engineering I/O data and exports it as a Modelica surrogate model.  
**Users:** Simulation engineers working with Modelica/Dymola toolchains (e.g., IGBT power-loss lookup tables).

---

## Repository Layout

```
Engine/                   ← Python backend (installable package)
  src/surrogate_tool/
    cli.py                ← All CLI subcommands (argparse)
    pipeline/             ← 6 ordered stages (orchestrator.py drives them)
    contracts/            ← Pydantic models: RunConfig, StatusMessage
    io/                   ← dataset_loader.py (CSV / XLSX)
    attempts/             ← manager.py: select best attempt by physical RMSE
    paths.py              ← runs_root() → <repo>/runs/
ExcelUI/                  ← VBA-driven Excel front-end (SurrogateGeneratorUI.xlsm)
runs/                     ← Runtime output (never edit manually)
samples/                  ← Example datasets
```

Detailed architecture: see [HOW_TO_USE.md](HOW_TO_USE.md).

---

## Setup & Running

```powershell
# Install (from repo root, venv already at .venv/)
pip install -r Engine/requirements.txt

# Run CLI
python -m surrogate_tool <command> [args]
```

Key CLI commands (all defined in [Engine/src/surrogate_tool/cli.py](Engine/src/surrogate_tool/cli.py)):

| Command | Purpose |
|---|---|
| `preview-headers` | List column names from a dataset without loading it fully |
| `make-run` | Create `runs/<timestamp>/` workspace, copy dataset |
| `retrain --run-id <id>` | Run full pipeline end-to-end (EDA → preprocess → train → export) |
| `best-attempt --run-id <id>` | Return attempt with lowest physical RMSE |
| `report --run-id <id> --attempt <n>` | Generate HTML report |

---

## Pipeline Stages (in order)

Orchestrated by `pipeline/orchestrator.py` → `run_full_attempt(run_id, attempt_num)`:

1. **Preprocess** — remove NaN/duplicates/outliers, validate columns
2. **EDA** — correlation heatmaps, distributions, outlier plots (matplotlib/seaborn)
3. **Split & Scale** — 70/15/15 train/val/test + `StandardScaler` fit on train only
4. **Train** — TensorFlow MLP with early stopping; saves `surrogate_model.keras` + `mlp_weights.json`
5. **Export Modelica** — `mlp_weights.json` → nested Modelica package (`package.mo`, dense blocks)
6. **Report** — Jinja2 HTML with metrics, loss curves, figures

---

## Run Directory Structure

```
runs/<run_id>/
  input/run_config.json          ← RunConfig (input_columns, output_columns, package_name)
  status.json                    ← StatusMessage (state, progress 0–1, message)
  latest.json                    ← Best attempt metadata after pipeline
  attempts/attempt_001/
    processed/                   ← cleaned_dataset.csv, scalers, mlp_weights.json, .keras
    reports/                     ← metrics.json, report.html, figures/
    logs/pipeline.log
    modelica/<PackageName>/      ← Generated Modelica package
```

`runs/latest_run.json` always points to the most recent run.

---

## Key Contracts

- **`RunConfig`** ([contracts/run_config.py](Engine/src/surrogate_tool/contracts/run_config.py)): Pydantic model. `package_name` must match `^[A-Za-z][A-Za-z0-9_]*$`.
- **`StatusMessage`** ([contracts/status.py](Engine/src/surrogate_tool/contracts/status.py)): Written to `status.json`; ExcelUI polls this for progress.
- **`mlp_weights.json`**: Serialized MLP — `input_cols`, `target_cols`, `layers[]` (W, b, activation), `x_scaler`, `y_scaler`.

---

## ExcelUI ↔ Python Integration

- ExcelUI (`SurrogateGeneratorUI.xlsm`) calls CLI via VBA's `Shell`/`WScript.Shell` using the `.venv` Python executable.
- Button layout (8 buttons): Pick Dataset → Load Headers → Build Mapping → **Process** → **Retrain** → Save Best → Open Report → Open Run Folder.
- See [/memories/repo/features.md](/memories/repo/features.md) for current button-to-action mapping and UI conventions.
- Headers are cached pipe-delimited in named range `UI_HeadersCache` (col J). Column mappings stored in `UI_InputCols` / `UI_OutputCols`.
- `build_ui_v2.py` rebuilds the `.xlsm` from scratch — it is idempotent and safe to re-run.

---

## Conventions & Pitfalls

- **No tests yet** — `Engine/tests/` is empty. Write tests under `Engine/tests/` using pytest.
- **`runs/` is runtime output** — never commit contents; contents are gitignored.
- **Attempt numbering** is 1-based, zero-padded: `attempt_001`, `attempt_002`, …
- **Best attempt** = lowest physical-unit RMSE (not scaled RMSE).
- **Excel DV limit**: Data Validation formula strings max 255 chars — mapping sheet uses a cell-range source on the mapping sheet, not an inline formula.
- Python entry point: `python -m surrogate_tool` (via `__main__.py`).

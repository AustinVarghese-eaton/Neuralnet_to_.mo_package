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
- Layer weight matrix `W` is stored as Keras convention: shape `[n_in, n_out]` (NOT pre-transposed).
- **Consumers must transpose before use:** `modelica_export.py` does `W = W_tf.T` → `[n_out, n_in]`; `fmu_export.py` does `W_T = np.array(layer["W"]).T.tolist()` before flattening into C. Both are correct post-fix (May 2026).

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

**10-button workflow:**

| # | Button | VBA Action | CLI Command |
|---|--------|-----------|-------------|
| 1 | Pick Dataset | `UI_PickDataset` | — |
| 2 | Load Headers | `UI_LoadHeaders` | `preview-headers` |
| 3 | Build Mapping | `UI_BuildMapping` | — |
| 4 | Process (blue) | `UI_Process` | `make-run` + pipeline stages |
| 5 | Retrain | `UI_Retrain` | `retrain` |
| 6 | Create FMU (green) | `UI_CreateFMU` | `create-fmu --run-id <id> --attempt <n>` |
| 7 | Save Best .mo + Report | `UI_SaveBestModelica` | — |
| 8 | Open Report | `UI_OpenReport` | — |
| 9 | Open Run Folder | `UI_OpenRunFolder` | — |
| 10 | Reset (red) | `UI_Reset` | — |

- **`UI_CreateFMU`** reads `runs/<run_id>/latest.json` to get `best_attempt`, strips the `"attempt_"` prefix to derive the attempt number, then calls `create-fmu`. Updates status cells during execution and shows the `.fmu` path on success.
- Rebuild button layout after any VBA change by running `UX_RebuildDashboardV2` (`Alt+F8` → select → Run).

**Named ranges (col J of "SurrogateGenerator" sheet):**
- `UI_HeadersCache` — pipe-delimited column names from dataset
- `UI_InputCols` — comma-separated selected input columns
- `UI_OutputCols` — comma-separated selected output columns

**Column mapping sheet:**
- Created dynamically by `Action_BuildMapping`; named `"mapping"`
- Headers cached in hidden col E (DV source) to avoid the 255-char Excel formula limit
- Row counts stored in hidden col G so `Action_ConfirmMapping` knows how many rows to read

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
- **mlp_weights.json W shape** — stored as Keras `[n_in, n_out]`. Both `modelica_export.py` and `fmu_export.py` transpose to `[n_out, n_in]` before use. Do NOT remove the transpose — it will silently produce wrong predictions.
- **`package_name` regex** — forgetting to validate before writing `RunConfig` causes a Pydantic error at construction time, not at export time.
- **XLSX sheet name** — must be passed explicitly if not `Sheet1`; defaults to `None` in `RunConfig` (loader picks first sheet).

---

## Environment Setup & Runs Folder Troubleshooting

### Symptom: `make-run` fails or run folders are created in the wrong repo

This happens when `C:\SG_ENV` (or any venv the UI calls) has the surrogate_tool package installed from a **different repo** (e.g. `Automation_pipeline\SurrogateGenerator\`). The `paths.py` `project_root()` then resolves to that repo and all `runs/` output goes there instead of here.

**Diagnosis — run in VS Code terminal (PowerShell):**
```powershell
& "C:\SG_ENV\Scripts\python.exe" -c "from surrogate_tool.paths import runs_root; print(runs_root())"
```
Expected output: `C:\Users\E0849595\Desktop\IMP\.mo_FMU\runs`

If it prints a different path, the venv is pointing to the wrong repo.

**Fix — re-point the venv to this repo (one-time):**
```powershell
& "C:\SG_ENV\Scripts\pip.exe" install -e "C:\Users\E0849595\Desktop\IMP\.mo_FMU\Engine"
```

**Verify:**
```powershell
& "C:\SG_ENV\Scripts\python.exe" -c "from surrogate_tool.paths import project_root; print(project_root())"
# Should print: C:\Users\E0849595\Desktop\IMP\.mo_FMU
```

After this, hitting **Process** in the UI creates run folders under `runs/` in this repo.

### Symptom: `.mo` package created but no FMU

The old repo's `orchestrator.py` did not include the `create_fmu` stage. After re-pointing the venv (above), new runs will produce both `.mo` and `.fmu` automatically.

To generate the FMU for an existing run (standalone):
```powershell
& "C:\SG_ENV\Scripts\python.exe" -m surrogate_tool create-fmu --run-id <run_id> --attempt 1
```

### How `project_root()` is resolved (`paths.py`)

`paths.py` walks up from `__file__` looking for a `.surrogate_root` marker file (placed at the repo root). This uniquely identifies **this** repo even if multiple repos with the same structure exist on the machine. The `.surrogate_root` file is empty — do not delete it.

---

## Improvements

Efficiency improvements identified (May 2026) in priority order:

### 1. Add Tests (Highest Priority — Enables Everything Else Safely)

**Status:** Implementation complete (May 2026).  
Test files live in `Engine/tests/`. Run with: `cd Engine && python -m pytest tests/ -v` (omits slow) or `python -m pytest tests/ -v -m slow` for round-trip.

#### Test files

| File | What it tests |
|---|---|
| `tests/conftest.py` | Shared fixtures: `synthetic_csv`, `minimal_mlp_weights`, `slow` marker |
| `tests/test_contracts.py` | `validate_modelica_name`, `load_run_config`/`save_run_config` BOM round-trip, `write_status` |
| `tests/test_dataset_loader.py` | `infer_format`, `preview_headers`, `load_dataset` |
| `tests/test_preprocess.py` | `make_attempt_paths`, `validate_and_clean` (duplicates, coercion) |
| `tests/test_split_scale.py` | `_split_data` ratios (70/15/15 ±1 row), `_scale_xy` no-data-leakage |
| `tests/test_attempts_manager.py` | `score_attempt_rmse_mean_physical`, `select_best_attempt` picks lowest RMSE |
| `tests/test_fmu_export.py` | `_resolve_gcc` (cfg override, not-found), FMI 2.0 XML rules (variable counts, `<Real start>`, no `initial="exact"`, no `needsDirectionDerivatives`), C code weight arrays |
| `tests/test_modelica_export.py` | `_load_weights_as_modelica_Wb` transpose correctness, `export_modelica` file creation |
| `tests/test_pipeline_roundtrip.py` | `@pytest.mark.slow` — 200-row synthetic CSV → `run_full_attempt()` (DLL compile skipped) → asserts `mlp_weights.json` shape, finite RMSE, `package.mo` exists |

#### Test conventions
- Private functions tested directly (`_split_data`, `_scale_xy`, `_resolve_gcc`, `_load_weights_as_modelica_Wb`) — this is where bugs hide.
- `pytest-mock` (`mocker`) used for `_resolve_gcc` filesystem mocking.
- `train.py` excluded from unit tests — TF import is heavy; only the round-trip test exercises TF.
- Round-trip test mocks `_compile_dll` to skip gcc requirement.
- No coverage threshold — green tests first.

### 2. Skip EDA on Retrain
`orchestrator.py` always runs `run_eda()` as stage 2, even on `retrain`. EDA figures (heatmaps, distributions) don't change between attempts on the same dataset, yet they regenerate every time (~10–30s overhead per attempt).

**Fix:** In `run_eda()`, check if figures already exist in the run's `reports/figures/` directory and skip regeneration if so.

### 3. Drop Intermediate Scaled CSVs
`split_scale.py` writes six CSVs (`X_train_scaled.csv`, etc.) to disk; `train.py` reads them all back immediately. This is pure disk I/O overhead.

**Fix:** Replace with a single `dataset_splits.npz` (already defined in the run structure) and load via `np.load()`, or pass arrays via a shared `ProcessedDataset` dataclass if stages run in the same process.

### 4. Cache gcc Resolution
`fmu_export.py` `_resolve_gcc()` runs `glob.glob()` across multiple root patterns on every FMU creation. On large filesystems this is slow.

**Fix:** Cache the resolved path in a module-level variable after first successful resolution, or resolve once during `make-run` and persist `gcc_path` in `run_config.json`.

### 5. Hyperparameter Variation Across Attempts
`train.py` uses a hardcoded `[128, 128, 64]` architecture for every attempt. The `retrain` command already supports multiple attempts — nothing varies between them.

**Fix:** Pass different `hidden` layer configs per attempt (e.g., attempt 1: `[64, 64]`, attempt 2: `[128, 128, 64]`, attempt 3: `[256, 128, 64]`). The best-attempt selector already picks the lowest physical RMSE, so this gives free architecture search.

### 6. Consistent Path Construction Across Stages
`preprocess.py` defines `make_attempt_paths()` but `train.py` and other stages reconstruct `processed_dir`, `reports_dir`, `logs_dir` inline independently.

**Fix:** All stages should call `make_attempt_paths()` from `preprocess.py` instead of duplicating path logic.

### 7. Add a `cleanup` CLI Subcommand
`runs/` accumulates indefinitely (25+ folders as of May 2026), each containing `.keras` models, CSVs, and `.fmu` files.

**Fix:** Add `python -m surrogate_tool cleanup --keep-latest N` that deletes all but the N most recent run folders.

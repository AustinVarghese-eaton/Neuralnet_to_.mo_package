# SurrogateGenerator — Full Project Context

> **Purpose:** One-click pipeline that trains a TensorFlow MLP on engineering tabular I/O data (CSV/XLSX) and exports it as a **Modelica surrogate package (`.mo`)** and/or **FMU 2.0 Model Exchange (`.fmu`)**.
> **Target Users:** Simulation engineers using Dymola / OpenModelica / Simulink toolchains.

---

## Table of Contents

1. [What Problem This Solves](#1-what-problem-this-solves)
2. [Repository Layout](#2-repository-layout)
3. [Setup & Installation](#3-setup--installation)
4. [Two Interfaces: ExcelUI vs CLI](#4-two-interfaces-exceluivs-cli)
5. [Full Pipeline — Stage by Stage](#5-full-pipeline--stage-by-stage)
6. [Run Directory Structure](#6-run-directory-structure)
7. [Key Data Contracts](#7-key-data-contracts)
8. [Neural Network Architecture](#8-neural-network-architecture)
9. [NN → Modelica Export](#9-nn--modelica-export)
10. [NN → FMU Export (Deep Dive)](#10-nn--fmu-export-deep-dive)
11. [ExcelUI ↔ Python Integration](#11-exceluivs-python-integration)
12. [Attempt Management & Best-Attempt Selection](#12-attempt-management--best-attempt-selection)
13. [Tests](#13-tests)
14. [Coding Conventions](#14-coding-conventions)
15. [Common Pitfalls & Known Issues](#15-common-pitfalls--known-issues)
16. [Environment / Venv Troubleshooting](#16-environment--venv-troubleshooting)
17. [Planned Improvements](#17-planned-improvements)
18. [Full CLI Reference](#18-full-cli-reference)

---

## 1. What Problem This Solves

In power electronics and thermal simulation, engineers run expensive physics simulations (e.g., IGBT power loss as a function of temperature and current). These are too slow for real-time or system-level Modelica/Dymola simulations.

**SurrogateGenerator replaces those expensive lookups** with a trained neural network that:
- Runs in microseconds vs. minutes
- Is exported as a standard FMI 2.0 FMU or Modelica package
- Requires zero Python/TensorFlow at runtime — pure compiled C or Modelica equations

---

## 2. Repository Layout

```
.mo_FMU/
├── .surrogate_root              ← Marker file: identifies this repo root (do NOT delete)
├── AGENTS.md                    ← Agent/Copilot instructions
├── HOW_TO_USE.md                ← End-user setup guide
├── PROJECT_CONTEXT.md           ← This file
│
├── Engine/                      ← Python backend (pip-installable package)
│   ├── pyproject.toml
│   ├── requirements.txt
│   └── src/
│       └── surrogate_tool/
│           ├── __init__.py
│           ├── __main__.py          ← Entry point: python -m surrogate_tool
│           ├── cli.py               ← All argparse subcommands
│           ├── paths.py             ← project_root() / runs_root()
│           ├── logging_config.py
│           ├── version.py
│           ├── contracts/
│           │   ├── run_config.py    ← Pydantic RunConfig model
│           │   └── status.py        ← write_status() → status.json
│           ├── io/
│           │   └── dataset_loader.py ← preview_headers(), load_dataset()
│           ├── attempts/
│           │   └── manager.py       ← select_best_attempt() by physical RMSE
│           └── pipeline/
│               ├── orchestrator.py  ← run_full_attempt(run_id, attempt_num)
│               ├── preprocess.py    ← Stage 1
│               ├── eda.py           ← Stage 2
│               ├── split_scale.py   ← Stage 3
│               ├── train.py         ← Stage 4
│               ├── modelica_export.py ← Stage 5
│               ├── fmu_export.py    ← Stage 6
│               └── report.py        ← Stage 7
│
├── Engine/tests/                ← pytest test suite
│   ├── conftest.py
│   ├── test_contracts.py
│   ├── test_dataset_loader.py
│   ├── test_preprocess.py
│   ├── test_split_scale.py
│   ├── test_attempts_manager.py
│   ├── test_fmu_export.py
│   ├── test_modelica_export.py
│   └── test_pipeline_roundtrip.py  ← @pytest.mark.slow
│
├── ExcelUI/
│   └── SurrogateGeneratorUI.xlsm   ← VBA-driven Excel front-end
│
├── runs/                        ← Runtime output (gitignored, never commit)
│   └── latest_run.json          ← Pointer to most recent run
│
└── samples/                     ← Example datasets (CSV/XLSX)
```

---

## 3. Setup & Installation

### Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.10+ |
| pip | bundled with Python |
| OpenModelica *(optional, for gcc)* | 1.24+ recommended |
| Microsoft Excel | Required for ExcelUI |

### Install

```powershell
# From repo root — activate venv first
python -m venv .venv
.venv\Scripts\Activate.ps1

# Install all dependencies
pip install -r Engine/requirements.txt

# Install the package in editable mode
pip install -e Engine/
```

### Key Dependencies (`Engine/requirements.txt`)

```
pydantic>=2.6          # Data validation & RunConfig schema
pandas>=2.2            # Data loading and manipulation
openpyxl>=3.1          # XLSX support
numpy>=1.26            # Array math
matplotlib>=3.8        # Plotting (headless / Agg backend)
seaborn>=0.13          # Correlation heatmaps
scikit-learn>=1.4      # StandardScaler, train_test_split
joblib>=1.3            # Scaler serialization
tensorflow-cpu>=2.16   # MLP training (CPU version)
pytest>=8.0            # Testing
pytest-mock>=3.12      # Mocking in tests
```

### Verify Installation

```powershell
python -m surrogate_tool --help
```

---

## 4. Two Interfaces: ExcelUI vs CLI

### ExcelUI (Recommended for Engineers)

Located at `ExcelUI/SurrogateGeneratorUI.xlsm`. Open in Excel with macros enabled.

**10-button workflow:**

| # | Button | VBA Function | What it does |
|---|---|---|---|
| 1 | Pick Dataset | `UI_PickDataset` | File dialog → stores path |
| 2 | Load Headers | `UI_LoadHeaders` | Calls `preview-headers` CLI |
| 3 | Build Mapping | `UI_BuildMapping` | Creates dynamic mapping sheet |
| 4 | **Process** | `UI_Process` | Calls `make-run` then full pipeline |
| 5 | **Retrain** | `UI_Retrain` | Calls `retrain` on existing run |
| 6 | **Create FMU** | `UI_CreateFMU` | Calls `create-fmu` standalone |
| 7 | Save Best .mo + Report | `UI_SaveBestModelica` | Copies best attempt outputs |
| 8 | Open Report | `UI_OpenReport` | Opens `report.html` in browser |
| 9 | Open Run Folder | `UI_OpenRunFolder` | Opens `runs/<run_id>/` in Explorer |
| 10 | Reset | `UI_Reset` | Clears all UI state |

**Named Ranges (sheet "SurrogateGenerator", col J):**

| Named Range | Content |
|---|---|
| `UI_HeadersCache` | Pipe-delimited column names from dataset |
| `UI_InputCols` | Comma-separated selected input columns |
| `UI_OutputCols` | Comma-separated selected output columns |

**Column Mapping Sheet:**
- Created dynamically by `Action_BuildMapping`; named `"mapping"`
- Headers cached in hidden col E (DV source) to avoid the 255-char Excel formula limit
- Row counts stored in hidden col G so `Action_ConfirmMapping` knows how many rows to read

### CLI

```powershell
python -m surrogate_tool <subcommand> [options]
```

See [Full CLI Reference](#18-full-cli-reference) for all commands.

---

## 5. Full Pipeline — Stage by Stage

Orchestrated by `pipeline/orchestrator.py → run_full_attempt(run_id, attempt_num)`.

```
Input CSV/XLSX
      │
      ▼
┌─────────────────────────────────────────────────────────────────┐
│  Stage 1: Preprocess          preprocess.py                     │
│  - Drop NaN rows                                                │
│  - Remove duplicate rows                                        │
│  - IQR-based outlier removal                                    │
│  - Validate that all selected columns exist and are numeric     │
│  - Output: cleaned_dataset.csv                                  │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│  Stage 2: EDA                 eda.py                            │
│  - Correlation heatmap                                          │
│  - Distribution plots per column                               │
│  - Outlier scatter plots                                        │
│  - Saves figures to reports/figures/                            │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│  Stage 3: Split & Scale       split_scale.py                    │
│  - 70% train / 15% val / 15% test split                         │
│  - StandardScaler fit on TRAIN only (no data leakage)          │
│  - Saves: X/Y_train/val/test_scaled.csv, scalers.joblib         │
│  - Also saves dataset_splits.npz                               │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│  Stage 4: Train               train.py                          │
│  - TensorFlow MLP: Input → Dense(128,ReLU) → Dense(128,ReLU)   │
│                            → Dense(64,ReLU) → Output(Linear)   │
│  - Adam optimizer, MSE loss, EarlyStopping (patience=30)       │
│  - ReduceLROnPlateau (factor=0.5, patience=10, min_lr=1e-6)    │
│  - Saves: surrogate_model.keras + mlp_weights.json             │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│  Stage 5: Modelica Export     modelica_export.py                │
│  - Reads mlp_weights.json                                       │
│  - Generates nested Modelica package (.mo files)               │
│  - Includes Dense block components, NeuralNetwork model        │
│  - Output: modelica/<PackageName>/package.mo                   │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│  Stage 6: FMU Export          fmu_export.py                     │
│  - Reads mlp_weights.json                                       │
│  - Generates modelDescription.xml (FMI 2.0 compliant)         │
│  - Generates model.c (MLP forward pass in pure C)              │
│  - Compiles model.c → win64/<Name>.dll via gcc                 │
│  - Packages into <PackageName>.fmu (ZIP archive)               │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│  Stage 7: Report              report.py                         │
│  - Jinja2 HTML report                                           │
│  - RMSE/MAE metrics (scaled + physical units)                  │
│  - Loss curves, prediction vs actual plots                     │
│  - Output: reports/report.html                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 6. Run Directory Structure

```
runs/
├── latest_run.json                     ← Always points to most recent run
└── <run_id>/                           ← e.g., 20260504_134650
    ├── input/
    │   └── run_config.json             ← RunConfig (input/output columns, package name)
    ├── status.json                     ← StatusMessage (state, progress 0–1, message)
    ├── latest.json                     ← Best attempt metadata after pipeline
    └── attempts/
        └── attempt_001/               ← 1-based, zero-padded
            ├── processed/
            │   ├── cleaned_dataset.csv
            │   ├── X_train_scaled.csv
            │   ├── X_val_scaled.csv
            │   ├── X_test_scaled.csv
            │   ├── Y_train_scaled.csv
            │   ├── Y_val_scaled.csv
            │   ├── Y_test_scaled.csv
            │   ├── x_scaler.joblib
            │   ├── y_scaler.joblib
            │   ├── dataset_splits.npz
            │   ├── surrogate_model.keras
            │   └── mlp_weights.json    ← Serialized NN weights + scalers
            ├── reports/
            │   ├── report.html
            │   ├── metrics.json
            │   ├── train_history.csv
            │   ├── preprocess_report.json
            │   └── figures/            ← Heatmaps, distributions, loss curves
            ├── logs/
            │   └── pipeline.log
            ├── modelica/
            │   └── <PackageName>/      ← package.mo + Dense block files
            └── fmu/
                └── <PackageName>.fmu   ← FMU 2.0 Model Exchange archive
                    ├── modelDescription.xml
                    ├── sources/model.c
                    └── binaries/win64/<PackageName>.dll
```

**Key rules:**
- Attempt numbering is **1-based, zero-padded**: `attempt_001`, `attempt_002`, …
- `runs/` is **gitignored** — never commit runtime output
- `runs/latest_run.json` always points to the most recent run

---

## 7. Key Data Contracts

### RunConfig (`contracts/run_config.py`)

Pydantic model — central config for every run.

```python
class RunConfig(BaseModel):
    run_id: str                        # e.g., "20260504_134650"
    created_utc: str                   # ISO timestamp
    dataset_path: str                  # Must exist; .csv or .xlsx only
    dataset_format: Literal["csv","xlsx"]
    sheet_name: Optional[str]          # For XLSX only; None = first sheet
    input_columns: list[str]           # e.g., ["Temp", "Voltage"]
    output_columns: list[str]          # e.g., ["PowerLoss"]
    n_inputs: int
    n_outputs: int
    package_name: str                  # Regex: ^[A-Za-z][A-Za-z0-9_]*$
    random_seed: int = 42
    engine_version: str = "0.1.0"
    gcc_path: Optional[str]            # Path to gcc.exe for DLL compilation
```

**Rules:**
- `package_name` validated against `^[A-Za-z][A-Za-z0-9_]*$` — Pydantic raises at construction if invalid
- JSON reads use `utf-8-sig` (handles Windows BOM); writes use `utf-8`
- Always use `load_run_config(path)` / `save_run_config(cfg, path)` — never read JSON manually

### StatusMessage (`contracts/status.py`)

Written to `status.json`; polled by ExcelUI for progress updates.

```python
write_status(status_path, state="RUNNING", message="Training...", progress=0.6)
# States: "READY" | "RUNNING" | "TRAINING" | "SPLIT_SCALE" | "DONE" | "ERROR"
```

### mlp_weights.json

The serialized trained network — the single source of truth for both Modelica and FMU export.

```json
{
  "input_cols": ["Temp", "Voltage"],
  "target_cols": ["PowerLoss"],
  "activation": "relu",
  "layers": [
    {
      "name": "dense_1",
      "W": [[...], ...],      ← shape [n_in, n_out] (Keras convention)
      "b": [...],             ← shape [n_out]
      "activation": "relu"
    },
    ...
  ],
  "x_scaler": {
    "mean": [...],            ← NOT mean_ (sklearn attribute vs JSON key)
    "scale": [...]            ← NOT scale_
  },
  "y_scaler": {
    "mean": [...],
    "scale": [...]
  }
}
```

**Critical shape note:** `W` is stored as Keras `[n_in, n_out]`. Both `modelica_export.py` and `fmu_export.py` **transpose** to `[n_out, n_in]` before use. Never remove those transposes.

---

## 8. Neural Network Architecture

**File:** `pipeline/train.py`

```
Input Layer  (n_inputs neurons)
     │
Dense(128, activation="relu")    ← hidden layer 1
     │
Dense(128, activation="relu")    ← hidden layer 2
     │
Dense(64,  activation="relu")    ← hidden layer 3
     │
Output Layer (n_outputs neurons, activation="linear")
```

**Training hyperparameters (defaults):**

| Param | Default | Notes |
|---|---|---|
| Hidden layers | `[128, 128, 64]` | Hardcoded per attempt (improvement #5: vary per attempt) |
| Learning rate | `1e-3` | Adam optimizer |
| Batch size | `32` | — |
| Max epochs | `500` | EarlyStopping kicks in before this |
| Patience | `30` | EarlyStopping on `val_loss` |
| LR reduction | `factor=0.5, patience=10, min_lr=1e-6` | ReduceLROnPlateau |
| Random seed | `42` (from RunConfig) | Both TF and NumPy seeded |

**Data split:** 70% train / 15% validation / 15% test

**Scaling:** `StandardScaler` fit on **train only** to prevent data leakage. Applied identically to val and test.

---

## 9. NN → Modelica Export

**File:** `pipeline/modelica_export.py`

### What it generates

A nested Modelica package that can be directly opened in OMEdit or Dymola:

```
modelica/<PackageName>/
├── package.mo          ← Top-level package declaration
├── NeuralNetwork.mo    ← Top-level model with input/output connectors
└── DenseBlock_1.mo     ← One file per Dense layer
    DenseBlock_2.mo
    DenseBlock_3.mo
    OutputLayer.mo
```

### Weight Transposition

```python
# mlp_weights.json stores W as Keras shape: (n_in, n_out)
# Modelica dense() expects: W as (n_out, n_in)
W_modelica = W_tf.T   # Always transpose
```

### Modelica math (per Dense layer)

$$y = W \cdot x + b$$

Then ReLU: $y_i = \max(0, y_i)$ for hidden layers, or identity for the output layer.

The input normalization and output denormalization are embedded as Modelica `parameter` declarations using the scaler `mean` and `scale` arrays from `mlp_weights.json`.

---

## 10. NN → FMU Export (Deep Dive)

**File:** `pipeline/fmu_export.py`

An FMU (Functional Mock-up Unit) is a **ZIP archive** following the FMI 2.0 standard. Any FMI-compatible simulation tool can import and simulate it without Python or TensorFlow.

### What's Inside the FMU

```
<PackageName>.fmu
├── modelDescription.xml       ← "Instruction manual": declares inputs, outputs, GUID
├── sources/
│   └── model.c                ← Complete MLP forward pass in pure C
└── binaries/
    └── win64/
        └── <PackageName>.dll  ← Pre-compiled Windows shared library
```

### Step 1: Generate `modelDescription.xml`

```python
# Function: _gen_model_description_xml()
```

Declares to the simulation tool:
- **Inputs:** `causality="input"`, `variability="continuous"`, `<Real start="0.0"/>` (no `initial="exact"`)
- **Outputs:** `causality="output"`, `initial="calculated"`, `<Real/>`
- **Model type:** `<ModelExchange modelIdentifier="<PackageName>"/>` — pure algebraic (no ODE states)
- **GUID:** UUID4 uniquely identifying this exact model instance
- `numberOfEventIndicators="0"` and `numberOfContinuousStates="0"` (algebraic only)

**FMI 2.0 Rules (violating any of these causes OMEdit FMI2XML errors):**

| Rule | Reason |
|---|---|
| Inputs must NOT have `initial="exact"` | They are driven externally |
| Inputs MUST have `<Real start="0.0"/>` | FMI spec requires a start value |
| `<ModelExchange>` must NOT have `needsDirectionDerivatives` | Not supported / causes parse error |
| `numberOfContinuousStates="0"` | Pure algebraic model |

### Step 2: Generate `model.c`

```python
# Function: _gen_model_c()
```

This translates the entire trained NN into plain C with all weights **hardcoded as static arrays**.

#### Weight layout in C

```c
/* Keras stores W as [n_in, n_out]. C matmul needs row i = output neuron i.
   So transpose to [n_out, n_in] before writing to C. */
W_T = np.array(layer["W"]).T.tolist()   # shape [n_out, n_in]

// In C:
static const double L0_W[] = {w00, w01, ..., w_nout_nin};  // row-major
static const double L0_b[] = {b0, b1, ..., b_nout};
static const double X_MEAN[]  = {...};
static const double X_SCALE[] = {...};
static const double Y_MEAN[]  = {...};
static const double Y_SCALE[] = {...};
```

#### Forward pass in C (the `_compute()` function)

```c
static void _compute(ModelInstance* inst) {
    double tmp[MAX_NEURONS];
    double buf[MAX_NEURONS];

    // 1. Normalize inputs
    for (i = 0; i < N_IN; i++)
        tmp[i] = (inst->u[i] - X_MEAN[i]) / X_SCALE[i];

    // 2. For each layer: matmul + bias + activation
    for (i = 0; i < N_NEURONS_L; i++) {
        buf[i] = L_b[i];
        for (j = 0; j < N_PREV; j++)
            buf[i] += L_W[i * N_PREV + j] * tmp[j];   // row-major access
        buf[i] = buf[i] > 0.0 ? buf[i] : 0.0;         // ReLU (hidden only)
    }
    // copy buf → tmp, repeat for next layer...

    // 3. Denormalize outputs
    for (i = 0; i < N_OUT; i++)
        inst->y[i] = tmp[i] * Y_SCALE[i] + Y_MEAN[i];
}
```

Math summary:

$$\hat{x}_i = \frac{x_i - \mu_{x,i}}{\sigma_{x,i}}$$

$$h^{(l)}_i = \text{ReLU}\!\left(\sum_j W^{(l)}_{ij} \cdot h^{(l-1)}_j + b^{(l)}_i\right)$$

$$\hat{y}_i = h^{(\text{last})}_i \cdot \sigma_{y,i} + \mu_{y,i}$$

#### FMI 2.0 API in C

The C file also implements all mandatory FMI 2.0 functions:

| FMI Function | What it does |
|---|---|
| `fmi2Instantiate` | Allocates a `ModelInstance` struct (holds `u[]` inputs, `y[]` outputs, `isDirty` flag) |
| `fmi2SetReal` | Simulation tool writes new input values → sets `isDirty = 1` |
| `fmi2GetReal` | Simulation tool reads outputs → if `isDirty`, calls `_compute()` first |
| `fmi2EnterInitializationMode` | No-op (algebraic model) |
| `fmi2ExitInitializationMode` | Calls `_compute()` |
| `fmi2DoStep` | Not applicable (Model Exchange, not Co-Simulation) |
| `fmi2Terminate`, `fmi2FreeInstance` | Cleanup / free memory |

### Step 3: Compile to DLL

```python
# Function: _compile_dll()
gcc_cmd = [gcc_exe, "-O2", "-shared", "-fPIC", "-o", dll_path, "model.c", "-lm"]
env["PATH"] = str(Path(gcc_exe).parent) + os.pathsep + env["PATH"]
subprocess.run(gcc_cmd, env=env, check=True)
```

**gcc auto-detection order:**
1. `C:\OpenModelica*\tools\msys\ucrt64\bin\gcc.exe` (OM 1.24+)
2. `C:\OpenModelica*\tools\msys\mingw64\bin\gcc.exe` (older OM)
3. OneDrive/Documents variants of OpenModelica installs
4. System `PATH` (`shutil.which("gcc")`)
5. Explicit `cfg.gcc_path` from RunConfig

**Why prepend gcc's own bin to PATH?** gcc needs `cc1`, `as`, and `ld` from the same directory. Without this, compilation silently fails with return code 1.

### Step 4: Package into ZIP

```python
with zipfile.ZipFile(fmu_path, "w") as zf:
    zf.write(xml_path, "modelDescription.xml")
    zf.write(c_path,   "sources/model.c")
    if dll_path.exists():
        zf.write(dll_path, f"binaries/win64/{pkg_name}.dll")
```

Output: `runs/<run_id>/attempts/attempt_001/fmu/<PackageName>.fmu`

---

## 11. ExcelUI ↔ Python Integration

VBA calls Python via `WScript.Shell` using `.venv\Scripts\python.exe` (or `C:\SG_ENV\Scripts\python.exe` if the global venv is used).

### How `UI_CreateFMU` works

1. Reads `runs/<run_id>/latest.json` to get `best_attempt` key
2. Strips the `"attempt_"` prefix → derives integer attempt number
3. Calls: `python -m surrogate_tool create-fmu --run-id <id> --attempt <n>`
4. Updates Excel status cells during execution
5. Shows the `.fmu` path on success

### How `UI_Process` works

1. Calls `make-run` → creates `runs/<timestamp>/input/run_config.json`
2. Calls full pipeline (`retrain` equivalent) which runs `run_full_attempt()`
3. Polls `status.json` every N ms to update the progress bar

---

## 12. Attempt Management & Best-Attempt Selection

**File:** `attempts/manager.py`

Each call to `retrain` creates a new `attempt_NNN` directory. Multiple attempts can coexist in one run.

**Best attempt = lowest mean physical-unit RMSE** (not scaled RMSE).

```python
def select_best_attempt(run_id: str) -> AttemptInfo:
    # Reads metrics.json from each attempt
    # Computes mean of rmse_physical values across all outputs
    # Returns attempt with the lowest mean physical RMSE
```

The best attempt info is written to `runs/<run_id>/latest.json`:

```json
{
  "run_id": "20260504_134650",
  "best_attempt": "attempt_001",
  "best_attempt_num": 1,
  "rmse_physical_mean": 0.00312,
  "selected_at_utc": "2026-05-04T13:47:22+00:00"
}
```

**Why physical RMSE and not scaled?** Scaled RMSE is normalized and dimensionless — it tells you how well the model learned relative to variance. Physical RMSE tells you the actual error in engineering units (watts, degrees, etc.), which is what matters to the engineer.

---

## 13. Tests

**Location:** `Engine/tests/`  
**Run:** `cd Engine && python -m pytest tests/ -v` (fast tests only)  
**Slow tests:** `python -m pytest tests/ -v -m slow`

| Test File | What it covers |
|---|---|
| `conftest.py` | Shared fixtures: `synthetic_csv`, `minimal_mlp_weights`, `slow` marker |
| `test_contracts.py` | `validate_modelica_name`, `load_run_config`/`save_run_config` BOM round-trip, `write_status` |
| `test_dataset_loader.py` | `infer_format`, `preview_headers`, `load_dataset` |
| `test_preprocess.py` | `make_attempt_paths`, `validate_and_clean` (duplicates, NaN, coercion) |
| `test_split_scale.py` | `_split_data` ratios (70/15/15 ±1 row), `_scale_xy` no-data-leakage |
| `test_attempts_manager.py` | `score_attempt_rmse_mean_physical`, `select_best_attempt` picks lowest RMSE |
| `test_fmu_export.py` | `_resolve_gcc`, FMI 2.0 XML rules, C code weight arrays, no `needsDirectionDerivatives` |
| `test_modelica_export.py` | `_load_weights_as_modelica_Wb` transpose correctness, `export_modelica` file creation |
| `test_pipeline_roundtrip.py` | `@pytest.mark.slow` — 200-row CSV → `run_full_attempt()` → asserts RMSE finite, `.fmu` exists |

**Test conventions:**
- Private functions tested directly (`_split_data`, `_resolve_gcc`, etc.) — this is where bugs hide
- `pytest-mock` (`mocker`) used for filesystem mocking
- `train.py` excluded from unit tests (TF import is heavy); only round-trip test exercises TF
- Round-trip test mocks `_compile_dll` to skip gcc requirement

---

## 14. Coding Conventions

| Convention | Detail |
|---|---|
| `from __future__ import annotations` | At the top of every Python file |
| Pathlib exclusively | Never `os.path`; all paths are `Path` objects |
| No hardcoded paths | Always use `runs_root()` and `project_root()` from `paths.py` |
| UTF-8-SIG for reads | All JSON from Windows tools may have BOM |
| UTF-8 for writes | Standard UTF-8 (no BOM) |
| matplotlib headless | Always `matplotlib.use("Agg")` before importing pyplot |
| Pydantic for validation | Use `@field_validator`, not manual if/raise patterns |
| Type hints | Lowercase generics: `list[str]`, `dict[str, int]` (Python 3.10+) |

---

## 15. Common Pitfalls & Known Issues

### Weight Transposition

Keras `W` shape is `[n_in, n_out]`. Both Modelica and FMU exporters **must** transpose to `[n_out, n_in]` before use. If you remove the transpose, the model produces completely wrong numbers with no error.

```python
# modelica_export.py
W_modelica = W_tf.T

# fmu_export.py
W_T = np.array(layer["W"]).T.tolist()
```

### mlp_weights.json Scaler Keys

Keys are `mean` and `scale` — **not** `mean_` and `scale_` (sklearn attribute names use underscore, but the JSON keys do not).

### FMU XML Pitfalls

| Wrong | Correct | Why |
|---|---|---|
| `initial="exact"` on inputs | No `initial` attribute | Inputs are driven externally |
| No `start` on input `<Real>` | `<Real start="0.0"/>` | FMI spec requires it |
| `needsDirectionDerivatives` in `<ModelExchange>` | Remove it | Causes parse error in OMEdit |

### gcc Subprocess PATH

Always prepend `Path(gcc_exe).parent` to `env["PATH"]` before calling subprocess. Without this, gcc can't find `cc1`/`as`/`ld` and compilation silently fails with exit code 1.

### Attempt Numbering

Attempts are **1-based**: `attempt_001`, never `attempt_000`.

### `package_name` Validation

`package_name` must match `^[A-Za-z][A-Za-z0-9_]*$`. Pydantic raises at `RunConfig` construction time — not at export time. Always validate before writing.

### Excel DV Limit

Data Validation formula strings in Excel max out at 255 chars. The mapping sheet uses a **cell-range source** instead of inline formulas to work around this.

### matplotlib Must Be Headless

Always call `matplotlib.use("Agg")` before any pyplot import. The tool runs in Excel/WScript context with no display.

---

## 16. Environment / Venv Troubleshooting

### Symptom: `make-run` creates folders in wrong repo

Happens when the venv (`C:\SG_ENV` or `.venv`) has `surrogate_tool` installed from a **different repo**. `paths.py` walks up from its `__file__` looking for `.surrogate_root` — if installed from another repo, all `runs/` output goes there.

**Diagnose:**
```powershell
& "C:\SG_ENV\Scripts\python.exe" -c "from surrogate_tool.paths import runs_root; print(runs_root())"
# Expected: C:\Users\E0849595\Desktop\IMP\.mo_FMU\runs
```

**Fix:**
```powershell
& "C:\SG_ENV\Scripts\pip.exe" install -e "C:\Users\E0849595\Desktop\IMP\.mo_FMU\Engine"
```

**Verify:**
```powershell
& "C:\SG_ENV\Scripts\python.exe" -c "from surrogate_tool.paths import project_root; print(project_root())"
# Should print: C:\Users\E0849595\Desktop\IMP\.mo_FMU
```

### How `project_root()` works

`paths.py` walks upward from `__file__` looking for a `.surrogate_root` marker file (empty file at the repo root). This uniquely identifies **this** repo even if multiple repos with the same structure exist on the machine. **Do NOT delete `.surrogate_root`.**

### Symptom: `.mo` created but no FMU

The old repo's orchestrator did not include the `create_fmu` stage. After re-pointing the venv, new runs automatically produce both.

To generate FMU for an existing run:
```powershell
python -m surrogate_tool create-fmu --run-id <run_id> --attempt 1
```

---

## 17. Planned Improvements

In priority order (as of May 2026):

| # | Improvement | Status |
|---|---|---|
| 1 | **Add Tests** | ✅ Complete (`Engine/tests/`) |
| 2 | **Skip EDA on Retrain** | Not started — check if figures already exist before regenerating |
| 3 | **Drop Intermediate Scaled CSVs** | Not started — replace 6 CSVs with single `dataset_splits.npz` |
| 4 | **Cache gcc Resolution** | Not started — module-level cache after first successful find |
| 5 | **Hyperparameter Variation Across Attempts** | Not started — vary `[64,64]` / `[128,128,64]` / `[256,128,64]` per attempt |
| 6 | **Consistent Path Construction** | Not started — all stages call `make_attempt_paths()` from `preprocess.py` |
| 7 | **`cleanup` CLI Subcommand** | Not started — `cleanup --keep-latest N` to prune old run folders |

---

## 18. Full CLI Reference

```powershell
# Preview column headers from a dataset (no full load)
python -m surrogate_tool preview-headers --dataset path/to/data.csv

# Create a new run workspace and start full pipeline
python -m surrogate_tool make-run \
    --dataset path/to/data.csv \
    --package MyModel \
    --inputs Temp Voltage \
    --outputs PowerLoss \
    --gcc-path "C:\path\to\gcc.exe"   # optional

# Re-run full pipeline on existing run (creates new attempt)
python -m surrogate_tool retrain --run-id 20260504_134650

# Generate FMU only (standalone, skips training)
python -m surrogate_tool create-fmu --run-id 20260504_134650 --attempt 1

# Show which attempt has the best physical RMSE
python -m surrogate_tool best-attempt --run-id 20260504_134650

# Generate HTML report for a specific attempt
python -m surrogate_tool report --run-id 20260504_134650 --attempt 1
```

---

## Quick Reference Card

```
Dataset (CSV/XLSX)
       │
       ▼
  [1] Preprocess → cleaned_dataset.csv
  [2] EDA        → figures/
  [3] Split/Scale→ *_scaled.csv + scalers.joblib
  [4] Train      → surrogate_model.keras + mlp_weights.json
  [5] Modelica   → modelica/<Name>/package.mo
  [6] FMU        → fmu/<Name>.fmu  (modelDesc.xml + model.c + .dll)
  [7] Report     → reports/report.html

Key Files:
  mlp_weights.json    ← All weights + scalers (W shape: [n_in, n_out])
  run_config.json     ← RunConfig Pydantic model
  status.json         ← Progress tracking (polled by ExcelUI)
  latest.json         ← Best attempt pointer

Both Modelica and FMU exporters TRANSPOSE W before use:
  W_export = W_json.T    →  [n_out, n_in]
```

---

*Generated: 2026-06-08 | SurrogateGenerator v0.1.0*

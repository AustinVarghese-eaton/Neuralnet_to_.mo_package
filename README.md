# 🧠 Neural Network → Modelica Surrogate Generator

> Train a neural network on your tabular data and automatically export a production-ready **Modelica `.mo` package** — no manual coding required.

Built for engineers who work with simulation tools like **Dymola** and **OpenModelica** and need fast, accurate surrogate models from experimental or simulation data.

---

## ✨ What It Does

| Stage | Description |
|-------|-------------|
| 📥 **Load** | Accepts CSV or XLSX datasets with numeric input/output columns |
| 🔍 **Analyze** | Runs exploratory data analysis — correlations, distributions, boxplots |
| ⚙️ **Preprocess** | Cleans data, handles missing values, splits 80/20 train/test, normalizes |
| 🤖 **Train** | Trains a configurable TensorFlow neural network |
| 📦 **Export** | Generates a ready-to-use Modelica `.mo` package with scaling constants |
| 📊 **Report** | Produces an interactive HTML report with RMSE, MAE, R², and training plots |

---

## 🖥️ Two Ways to Use It

### Option 1 — Excel UI *(Recommended for non-technical users)*
Open `ExcelUI/SurrogateGeneratorUI.xlsm`, load your dataset, configure columns and model settings, and click **"Generate Surrogate"**. No terminal needed.

### Option 2 — Command Line *(For advanced users & automation)*
```bash
# Create a run
python -m surrogate_tool make-run --dataset "data.csv" --package "MyModel"

# Run the full pipeline
python -m surrogate_tool retrain --run-id <timestamp>
```

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone <repository-url>
cd SurrogateGenerator
```

### 2. Create and activate a virtual environment
```bash
# Windows (PowerShell)
python -m venv venv
venv\Scripts\Activate.ps1

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r Engine/requirements.txt
```

### 4. Run your first surrogate
```bash
python -m surrogate_tool make-run --dataset "samples/my_data.csv" --package "QuickTest"
python -m surrogate_tool retrain --run-id <timestamp_from_above>
```

Results are saved to `runs/<timestamp>/attempts/attempt_001/`.

---

## 📦 Generated Modelica Package

After a successful run, your Modelica package is at:
```
runs/<timestamp>/attempts/attempt_001/modelica/<PackageName>/
```

It contains:
```
MyModel/
├── package.mo           # Main package file
├── NeuralNetwork.mo     # The surrogate model
├── Records/
│   ├── InputScaling.mo  # Normalization constants for inputs
│   └── OutputScaling.mo # Normalization constants for outputs
└── Example.mo           # Example usage in Dymola / OpenModelica
```

---

## 📁 Project Structure

```
SurrogateGenerator/
├── Engine/                          # Python backend
│   ├── src/surrogate_tool/
│   │   ├── cli.py                   # Command-line interface
│   │   └── pipeline/
│   │       ├── preprocess.py        # Data cleaning
│   │       ├── eda.py               # Exploratory data analysis
│   │       ├── split_scale.py       # Train/test split & normalization
│   │       ├── train.py             # Neural network training
│   │       ├── modelica_export.py   # .mo file generation
│   │       ├── report.py            # HTML report generation
│   │       └── orchestrator.py      # Full pipeline runner
│   ├── requirements.txt
│   └── pyproject.toml
├── ExcelUI/
│   └── SurrogateGeneratorUI.xlsm    # Excel-based UI
├── runs/                            # Auto-generated outputs
├── samples/                         # Example datasets
└── HOW_TO_USE.md                    # Detailed usage guide
```

---

## ⚙️ Key Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--hidden` | `128,64,32` | Hidden layer architecture (neurons per layer) |
| `--epochs` | `100` | Number of training iterations |
| `--lr` | `0.001` | Learning rate |
| `--batch-size` | `32` | Samples per gradient update |
| `--patience` | `20` | Early stopping patience (epochs) |

---

## 🧰 Tech Stack

| Library | Purpose |
|---------|---------|
| TensorFlow 2.16+ | Neural network training |
| pandas / NumPy | Data processing |
| scikit-learn | Scaling, train/test split, metrics |
| matplotlib / seaborn | Visualizations |
| openpyxl | Excel file support |
| Pydantic | Configuration schemas |

**Requirements:** Python 3.10+

---

## 📋 Prerequisites

- Python 3.10 or higher — [download here](https://www.python.org/downloads/)
- A dataset in **CSV** or **XLSX** format with numeric input and output columns
- Microsoft Excel (for the Excel UI method)

---

## 📖 Full Documentation

See **[HOW_TO_USE.md](HOW_TO_USE.md)** for:
- Step-by-step Excel UI walkthrough
- All CLI commands with examples
- Troubleshooting guide
- Excel-Python VBA integration guide

---

## 📝 Notes

- First run may take slightly longer due to TensorFlow compilation
- Recommended dataset size: **1,000 – 1,000,000 rows**
- All input/output columns must be **numeric**
- Generated Modelica code is production-ready for **Dymola** and **OpenModelica**

---

**Version:** 0.1.0 &nbsp;|&nbsp; **Language:** Python &nbsp;|&nbsp; **Last Updated:** April 2026

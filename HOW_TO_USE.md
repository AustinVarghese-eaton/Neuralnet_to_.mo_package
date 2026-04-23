# How to Use the Surrogate Generator

This guide covers both **Excel UI** and **VS Code CLI** methods to generate Modelica surrogates from your data.

---

## 📋 Prerequisites

- **Python 3.10+** (download from [python.org](https://www.python.org/downloads/))
- **pip** (comes with Python)
- **Your dataset** in CSV or XLSX format with:
  - Input columns (features/parameters)
  - Output columns (targets/responses)

---

## 🚀 Installation

### Step 1: Clone the Repository
```bash
git clone <repository-url>
cd SurrogateGenerator
```

### Step 2: Set Up Virtual Environment
```bash
# Windows (PowerShell)
python -m venv venv
venv\Scripts\Activate.ps1

# macOS/Linux (Bash)
python -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r Engine/requirements.txt
```

This installs:
- TensorFlow (neural network training)
- pandas, scikit-learn (data processing)
- openpyxl (Excel support)
- matplotlib, seaborn (visualization)
- And more...

---

## 📊 Method 1: Excel UI (Recommended for Non-Technical Users)

### Opening the Excel File
1. Navigate to `ExcelUI/` folder
2. Open **`SurrogateGeneratorUI.xlsm`** in Microsoft Excel
3. Enable macros when prompted (required for functionality)

### Step-by-Step Workflow

#### 1️⃣ **Load Your Dataset**
- Click the **"Load Dataset"** button
- Browse and select your CSV or XLSX file
- If XLSX, specify the sheet name (default: `Sheet1`)
- Headers from your dataset will be auto-populated

#### 2️⃣ **Select Input & Output Columns**
- In the **Input Columns** section, check the columns that are **parameters/features**
- In the **Output Columns** section, check the columns that are **targets/responses**
  - Example:
    - **Inputs**: Temperature, Pressure, Voltage
    - **Outputs**: Power_Loss, Efficiency

#### 3️⃣ **Configure Model Parameters**
- **Modelica Package Name**: Enter a valid name (e.g., `MyModel`, `IGBT_NN`)
  - Must start with a letter, use only alphanumeric + underscore
- **Hidden Layers**: Define network architecture
  - Default: `128,64,32` (3 hidden layers with 128, 64, 32 neurons)
  - Example: `256,128,64,32` for a deeper network
- **Training Epochs**: Number of training iterations (default: 100)
- **Learning Rate**: Model learning speed (default: 0.001)
- **Batch Size**: Samples per update (default: 32)

#### 4️⃣ **Click "Generate Surrogate"**
- The engine processes your data through:
  1. **Preprocessing** → Data cleaning & validation
  2. **EDA** → Exploratory analysis with visualizations
  3. **Split & Scale** → 80/20 train/test split, normalization
  4. **Training** → Neural network model training
  5. **Export** → Generate Modelica `.mo` package
  6. **Report** → HTML analysis report
- Status updates appear in real-time
- Watch the **Progress Bar** fill as each stage completes

#### 5️⃣ **Review Results**
- Once complete, results are saved to `runs/<timestamp>/`
- Open the **HTML Report** for:
  - Data profiling
  - Correlation heatmaps
  - Model performance metrics (RMSE, MAE)
  - Training history plots

#### 6️⃣ **Get Your `.mo` Package**
- Navigate to `runs/<timestamp>/attempts/attempt_001/modelica/<PackageName>/`
- Your Modelica package is ready to use in Dymola or OpenModelica!

---

## 💻 Method 2: Command Line (Advanced)

### Overview of Available Commands

#### Create a New Run
```bash
python -m surrogate_tool make-run \
  --dataset "path/to/data.csv" \
  --package "MyModel" \
  --sheet "Sheet1"
```
- Generates a `run_id` (timestamp-based)
- Creates directory structure in `runs/`

#### Preview Dataset Headers
```bash
python -m surrogate_tool preview-headers \
  --dataset "path/to/data.csv" \
  --sheet "Sheet1"
```
- Lists all available columns
- Helpful to confirm column names before preprocessing

#### Run Individual Pipeline Stages

**Preprocess**
```bash
python -m surrogate_tool preprocess \
  --run-id "20260224_123030" \
  --attempt 1 \
  --inputs "Temperature,Pressure,Voltage" \
  --outputs "Power_Loss,Efficiency"
```

**Exploratory Data Analysis**
```bash
python -m surrogate_tool eda \
  --run-id "20260224_123030" \
  --attempt 1
```
- Generates correlation heatmaps, distributions, boxplots

**Split & Scale**
```bash
python -m surrogate_tool split-scale \
  --run-id "20260224_123030" \
  --attempt 1
```

**Train Model**
```bash
python -m surrogate_tool train \
  --run-id "20260224_123030" \
  --attempt 1 \
  --hidden "256,128,64" \
  --epochs 150 \
  --lr 0.001 \
  --batch-size 32 \
  --patience 20
```

**Export to Modelica**
```bash
python -m surrogate_tool export-modelica \
  --run-id "20260224_123030" \
  --attempt 1
```

**Generate Report**
```bash
python -m surrogate_tool report \
  --run-id "20260224_123030" \
  --attempt 1
```

#### Run Everything at Once
```bash
python -m surrogate_tool retrain \
  --run-id "20260224_123030"
```
- Runs the complete pipeline
- Creates a new attempt and finds the best performing model

#### List All Attempts
```bash
python -m surrogate_tool list-attempts "20260224_123030"
```

#### Find Best Attempt
```bash
python -m surrogate_tool best-attempt "20260224_123030"
```
- Returns the highest-performing model metrics

---

## 🔧 Engine Command (Advanced) - Integrating with Excel UI

This section is for advanced users who want to **manually configure Excel macros** to execute Python backend commands or troubleshoot Excel-Python integration.

### Understanding the Integration

The Excel UI communicates with the Python engine by:
1. Writing your configuration to `runs/<run_id>/input/run_config.json`
2. Executing Python commands via command line
3. Reading results from `runs/` directory
4. Updating the Excel sheet with status and outputs

### Step 1: Test Command in Terminal

Before adding a command to Excel, **always test it in the terminal first**:

```bash
# Activate your environment
venv\Scripts\Activate.ps1

# Test a command (example: preview headers)
python -m surrogate_tool preview-headers --dataset "C:\path\to\data.csv"

# You should see JSON output with column names
```

**✅ If it works**, you're ready to add it to Excel.  
**❌ If it fails**, fix the command before proceeding (see Troubleshooting below).

### Step 2: Copy the Working Command

Once your command works in the terminal:

1. Select and copy the entire command (everything after `python -m surrogate_tool`)
   ```
   preview-headers --dataset "C:\path\to\data.csv"
   ```

2. **Note the full command path** you'll need:
   - Python executable: `C:\Users\<YourUsername>\SurrogateGenerator\venv\Scripts\python.exe`
   - Command: `m surrogate_tool <your_command>`

### Step 3: Add Command to Excel UI (VBA Macro)

#### Option A: Using VBA in Excel

1. **Open Excel file**: `ExcelUI/SurrogateGeneratorUI.xlsm`

2. **Press `Alt + F11`** to open VBA Editor

3. **Find or create a module** for your macro:
   - Right-click on "Microsoft Excel Objects"
   - Select "Insert" → "Module"

4. **Paste this template** and modify with your command:

```vba
Sub RunEngineCommand()
    Dim shell As Object
    Dim cmd As String
    Dim pythonExe As String
    Dim workingDir As String
    Dim result As String
    
    ' Set paths
    pythonExe = "C:\Users\E0849595\SurrogateGenerator\venv\Scripts\python.exe"
    workingDir = "C:\Users\E0849595\Desktop\IMP\Automation_pipeline\SurrogateGenerator"
    
    ' Build command (modify the surrogate_tool command as needed)
    cmd = pythonExe & " -m surrogate_tool retrain --run-id 20260224_123030"
    
    ' Execute command
    Set shell = CreateObject("WScript.Shell")
    On Error Resume Next
    shell.CurrentDirectory = workingDir
    shell.Exec cmd
    On Error GoTo 0
    
    ' Feedback to user
    MsgBox "Command executed. Check the runs/ folder for results.", vbInformation
End Sub
```

5. **Customize for your command**:
   - Replace `20260224_123030` with actual run ID
   - Replace the `surrogate_tool` command with your specific command
   - Update `pythonExe` path to match your installation

6. **Attach to a Button**:
   - In Excel, insert a button (Insert → Button)
   - Right-click → Assign Macro
   - Select `RunEngineCommand`
   - Click OK

#### Option B: Using Shell Command Directly (Simpler)

For Excel 2019+ or Office 365:

1. **Create a new worksheet** called "Commands"

2. **Add this formula** in a cell:

```
=SHELL("C:\Users\E0849595\SurrogateGenerator\venv\Scripts\python.exe -m surrogate_tool retrain --run-id 20260224_123030")
```

**Note:** This requires enabling `SHELL` function in your Excel settings.

### Step 4: Verify Integration

Once you've added the command to Excel:

1. **Run the macro** (press your button or run from VBA)

2. **Check the Terminal** for any error messages:
   - If errors appear, copy them
   - Adjust the command accordingly
   - Re-test in terminal first

3. **Monitor the runs/ folder**:
   - Look for new directories with timestamp names
   - Check `status.json` for pipeline progress

4. **Review output** in Excel or the HTML report

### Common Engine Commands for Excel Integration

Here are the most common commands you might want to add to Excel:

#### Generate Full Surrogate (Most Common)
```
python -m surrogate_tool retrain --run-id {RUN_ID}
```
**Use when:** You want to generate a complete surrogate from dataset to Modelica code

#### Preview Dataset Headers
```
python -m surrogate_tool preview-headers --dataset "{DATASET_PATH}" --sheet "{SHEET_NAME}"
```
**Use when:** You want to see available columns before selection

#### Run Preprocessing Only
```
python -m surrogate_tool preprocess --run-id {RUN_ID} --attempt 1 --inputs "col1,col2,col3" --outputs "col4,col5"
```
**Use when:** You want to test data cleaning before training

#### Train with Custom Parameters
```
python -m surrogate_tool train --run-id {RUN_ID} --attempt 1 --hidden "256,128,64" --epochs 200 --lr 0.001 --batch-size 32
```
**Use when:** You want to fine-tune model architecture

#### Export Only
```
python -m surrogate_tool export-modelica --run-id {RUN_ID} --attempt 1
```
**Use when:** You've trained a model and want to convert it to Modelica

### Placeholder Variables for Excel

When adding commands to Excel, use these variables to make them dynamic:

| Variable | What to Replace With | Example |
|----------|---------------------|---------|
| `{RUN_ID}` | Timestamp from `latest_run.json` | `20260224_123030` |
| `{DATASET_PATH}` | Full path to your CSV/XLSX | `C:\Data\my_dataset.csv` |
| `{SHEET_NAME}` | Sheet name (XLSX only) | `Sheet1` |
| `{ATTEMPT}` | Attempt number | `1`, `2`, `3` |
| `{PACKAGE_NAME}` | Your Modelica package name | `MyModel` |

**In VBA, you can read cell values to make it dynamic:**

```vba
Sub RunDynamicCommand()
    Dim runId As String
    Dim pythonExe As String
    
    ' Read run ID from Excel cell
    runId = Sheet1.Range("A1").Value
    
    ' Build dynamic command
    pythonExe = "C:\Users\E0849595\SurrogateGenerator\venv\Scripts\python.exe"
    Dim cmd As String
    cmd = pythonExe & " -m surrogate_tool retrain --run-id " & runId
    
    ' Execute
    Dim shell As Object
    Set shell = CreateObject("WScript.Shell")
    shell.Exec cmd
End Sub
```

### Troubleshooting Excel-Python Integration

#### Issue: "Python not found" error
**Solution:**
1. Verify the exact path to `python.exe`:
   ```bash
   where python
   ```
2. Copy the full path from terminal
3. Update the path in your VBA code

#### Issue: Command runs but no output appears in Excel
**Solution:**
1. Commands run in the background
2. Check `runs/` folder for output files
3. Add a status message after execution:
   ```vba
   MsgBox "Pipeline started. Results will be in runs/ folder.", vbInformation
   ```

#### Issue: "Permission denied" or "Access denied"
**Solution:**
1. Make sure you're not already running a pipeline
2. Check that `runs/` folder is not write-protected
3. Try running Excel as Administrator

#### Issue: Command times out (for long-running jobs)
**Solution:**
- Use `shell.Exec()` instead of `shell.Run()` (non-blocking)
- Split into smaller commands (preprocess first, then train separately)
- Let the command run in background and check results later

#### Issue: Environment variables not working in Excel
**Solution:**
1. Always use **full absolute paths**, not relative paths
2. Use backslashes: `C:\Users\...\venv\Scripts\python.exe`
3. Quote paths with spaces: `"C:\Program Files\..."`

### Testing Your Setup

Run this **verification command** to test your complete setup:

```bash
# Activate environment
venv\Scripts\Activate.ps1

# Test preview command (should show your dataset columns)
python -m surrogate_tool preview-headers --dataset "samples/my_data.csv"

# Test make-run command
python -m surrogate_tool make-run --dataset "samples/my_data.csv" --package "TestModel"

# If both work, you're ready to integrate into Excel!
```

### Best Practices

✅ **DO:**
- Test commands in terminal first
- Use full absolute paths
- Keep commands simple (one task per button)
- Add error handling and user feedback
- Document which command each button runs

❌ **DON'T:**
- Use relative paths in Excel macros
- Mix multiple commands in one button
- Forget to activate the Python environment
- Run overlapping commands simultaneously
- Hardcode paths (use absolute paths instead)

---

## 📁 Project Structure

```
SurrogateGenerator/
├── Engine/                          # Python backend
│   ├── src/surrogate_tool/
│   │   ├── cli.py                   # Command-line interface
│   │   ├── pipeline/                # Processing stages
│   │   │   ├── preprocess.py        # Data cleaning
│   │   │   ├── eda.py               # Data analysis & plots
│   │   │   ├── split_scale.py       # Train/test split & normalization
│   │   │   ├── train.py             # Model training
│   │   │   ├── modelica_export.py   # Generate .mo files
│   │   │   ├── report.py            # HTML report generation
│   │   │   └── orchestrator.py      # Full pipeline runner
│   │   ├── contracts/               # Data schemas (RunConfig, Status)
│   │   └── io/                      # Dataset loading utilities
│   ├── requirements.txt             # Python dependencies
│   └── pyproject.toml               # Package configuration
│
├── ExcelUI/                         # User interface
│   └── SurrogateGeneratorUI.xlsm    # Main Excel application
│
├── runs/                            # Execution outputs (auto-generated)
│   ├── 20260224_123030/            # Run 1
│   │   ├── input/
│   │   │   └── run_config.json      # Your configuration
│   │   ├── attempts/
│   │   │   └── attempt_001/
│   │   │       ├── modelica/        # Generated .mo package
│   │   │       ├── metrics.json     # Model performance
│   │   │       └── figures/         # Plots & visualizations
│   │   └── status.json              # Pipeline status
│   └── latest_run.json              # Pointer to last run
│
├── samples/                         # Example datasets
└── HOW_TO_USE.md                    # This file
```

---

## 📊 Understanding Output

### Generated Files for Each Run

After running a surrogate generation, you'll find:

**Run Directory** (`runs/<run_id>/`)
- `run_config.json` — Your input configuration
- `status.json` — Pipeline execution status
- `latest_run.json` — Metadata about this run

**Attempt Directory** (`runs/<run_id>/attempts/attempt_001/`)
- `cleaned_data.csv` — Preprocessed dataset
- `metrics.json` — Model performance (RMSE, MAE, R² per output)
- `train_history.json` — Loss/accuracy per epoch
- `figures/` — EDA plots (distributions, correlations, boxplots)
- `modelica/<PackageName>/` — **Your generated Modelica package**
- `report.html` — Interactive analysis report

### Modelica Package Contents

Inside `modelica/<PackageName>/`:
```
MyModel/
├── package.mo          # Main package file
├── Records/            # Input/output normalization constants
│   ├── InputScaling.mo
│   └── OutputScaling.mo
├── NeuralNetwork.mo    # The surrogate model
└── Example.mo          # Example usage in Dymola
```

---

## ⚙️ Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'tensorflow'"
**Solution:**
```bash
pip install --upgrade tensorflow-cpu>=2.16
# Or use GPU version if you have CUDA
pip install tensorflow-gpu>=2.16
```

### Issue: "Excel file not opening or macros not working"
**Solution:**
1. Ensure you're opening `.xlsm` file (macro-enabled)
2. Enable macros in Excel security settings
3. If using LibreOffice, try re-saving as Excel format

### Issue: Dataset not loading in Excel UI
**Solution:**
1. Verify the file path is correct
2. Check file format is CSV or XLSX
3. For XLSX: specify the correct sheet name
4. Ensure no special characters in column names

### Issue: Training crashes with out-of-memory error
**Solution:**
1. Reduce `batch_size` (e.g., 16 instead of 32)
2. Reduce `hidden` layers (e.g., `128,64` instead of `256,128,64,32`)
3. Reduce dataset size (if very large >100k rows, sample 50k)
4. Close other applications to free RAM

### Issue: Model has poor accuracy (high RMSE)
**Solution:**
1. Ensure input/output columns are correct (no mixing features with targets)
2. Check data quality (missing values, outliers)
3. Increase `epochs` (default 100 → try 200-500)
4. Adjust `hidden` layers (try `256,128,64,32` for more capacity)
5. Try different `learning_rate` (0.001 → 0.0005 or 0.01)
6. Review the EDA report for data issues

### Issue: Modelica export fails
**Solution:**
1. Check your package name is valid (starts with letter, alphanumeric + underscore)
2. Ensure training completed successfully (check RMSE is reasonable)
3. Review `attempt_001/modelica/` directory exists

---

## 📞 Getting Help

1. **Review the HTML Report** — Generated after each run, contains diagnostic plots
2. **Check logs** — See `runs/<run_id>/status.json` for pipeline status
3. **Sample data** — Try with sample datasets in `samples/` folder first
4. **VS Code debugging** — Use CLI commands one at a time to isolate issues

---

## 🎯 Quick Start Example

### Excel UI
1. Open `ExcelUI/SurrogateGeneratorUI.xlsm`
2. Click "Load Dataset" → select `samples/my_data.csv`
3. Select 3 inputs, 1 output
4. Enter package name: `QuickTest`
5. Click "Generate Surrogate"
6. Wait for completion
7. Find results in `runs/` folder

### CLI
```bash
# Activate environment
venv\Scripts\Activate.ps1

# Create run
python -m surrogate_tool make-run --dataset "samples/my_data.csv" --package "QuickTest"

# Run full pipeline
python -m surrogate_tool retrain --run-id <timestamp_from_above>

# View results
# Check runs/<timestamp>/attempts/attempt_001/modelica/QuickTest/
```

---

## 📝 Notes

- **First run takes longer** due to dependency downloads and model compilation
- **Keep dataset size reasonable** (1,000 - 1,000,000 rows recommended)
- **Input/output columns should be numeric** (exclude non-numeric features)
- **Generated Modelica code** is production-ready and can be integrated into Dymola/OpenModelica

---

**Version:** 0.1.0  
**Last Updated:** April 2026

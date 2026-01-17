# Running Flask App with UV and VENV

This guide shows how to set up and run the Flask GSM Fabric Predictor using `uv` (fast Python package manager) and `venv` (Python virtual environment).

---

## Option 1: Using UV (Recommended - Fastest)

### 1. Install UV
If you don't have `uv` installed, install it first:

```powershell
# Using pip
pip install uv

# Or using Windows Package Manager (Winget)
winget install astral-sh.uv
```

### 2. Create Virtual Environment with UV
```powershell
cd C:\Users\I769816\Desktop\GSM_fabric\fabric_gsm_pipeline\flask_app

# Create venv with uv
uv venv

# Activate the virtual environment
# On Windows:
.\.venv\Scripts\Activate.ps1

# On Linux/Mac:
source .venv/bin/activate
```

### 3. Install Dependencies with UV
```powershell
# Install all packages from requirements.txt
uv pip install -r requirements.txt

# Or install specific packages
uv pip install flask torch pandas opencv-python
```

### 4. Run the Flask App
```powershell
# Make sure venv is activated first
python app.py

# Or use Flask directly
flask run
```

---

## Option 2: Using VENV (Standard Python)

### 1. Create Virtual Environment
```powershell
cd C:\Users\I769816\Desktop\GSM_fabric\fabric_gsm_pipeline\flask_app

# Create venv
python -m venv .venv
```

### 2. Activate Virtual Environment
```powershell
# On Windows (PowerShell):
.\.venv\Scripts\Activate.ps1

# On Windows (Command Prompt):
.\.venv\Scripts\activate.bat

# On Linux/Mac:
source .venv/bin/activate
```

### 3. Upgrade pip and Install Dependencies
```powershell
# Upgrade pip (important!)
python -m pip install --upgrade pip

# Install from requirements.txt
pip install -r requirements.txt
```

### 4. Run the Flask App
```powershell
# Make sure venv is activated (you should see (.venv) in terminal)
python app.py

# Or run with Flask
flask run
```

---

## Option 3: Using UV with Sync (Lock File Approach)

For production-grade dependency management:

### 1. Create `pyproject.toml`
```toml
[build-system]
requires = ["setuptools", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "gsm-fabric-predictor"
version = "1.0.0"
description = "Flask app for GSM fabric prediction"
requires-python = ">=3.8"
dependencies = [
    "flask==2.3.3",
    "flask-cors==4.0.0",
    "pillow==10.0.0",
    "opencv-python==4.8.0.76",
    "pandas==2.0.3",
    "numpy==1.24.3",
    "scikit-learn==1.3.0",
    "catboost==1.2.2",
    "scipy==1.11.2",
    "torch==2.0.1",
    "werkzeug==2.3.7",
]
```

### 2. Create Lock File and Environment
```powershell
# Create lock file
uv pip compile pyproject.toml -o requirements.lock

# Create venv
uv venv

# Activate venv
.\.venv\Scripts\Activate.ps1

# Install from lock file (reproducible)
uv pip install -r requirements.lock
```

### 3. Run the App
```powershell
python app.py
```

---

## Quick Reference Commands

### UV Commands
```powershell
# Create venv
uv venv

# Activate (Windows)
.\.venv\Scripts\Activate.ps1

# Install packages
uv pip install flask torch

# Install from file
uv pip install -r requirements.txt

# List installed packages
uv pip list

# Remove venv
Remove-Item .venv -Recurse
```

### VENV Commands
```powershell
# Create venv
python -m venv .venv

# Activate (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# Activate (Windows CMD)
.\.venv\Scripts\activate.bat

# Install packages
pip install flask torch

# Install from file
pip install -r requirements.txt

# List installed packages
pip list

# Deactivate
deactivate

# Remove venv
Remove-Item .venv -Recurse
```

---

## Troubleshooting

### PowerShell Execution Policy Error
If you get "cannot be loaded because running scripts is disabled":

```powershell
# Temporarily allow script execution
Set-ExecutionPolicy -ExecutionPolicy RemoteS igned -Scope CurrentUser

# Or use CMD instead:
.\.venv\Scripts\activate.bat
```

### PyTorch Installation Issues
If PyTorch is slow to install:

```powershell
# Use UV which is faster
uv pip install torch

# Or specify CPU-only version (lighter)
uv pip install torch --index-strategy unsafe-best-match
```

### Model File Not Found
Make sure the model is in the correct location:
- Expected: `../Model/gsm_regressor.pt`
- The model directory should be at the parent level of flask_app

---

## Verify Installation

```powershell
# Activate venv first
.\.venv\Scripts\Activate.ps1

# Run setup verification
python setup.py

# Or test imports directly
python -c "import flask, torch, cv2, pandas; print('✅ All dependencies installed')"
```

---

## Running the Flask App

Once everything is installed:

```powershell
# Activate venv
.\.venv\Scripts\Activate.ps1

# Run Flask app
python app.py

# App will be available at: http://localhost:5000
```

Open your browser to **http://localhost:5000** to access the GSM Fabric Predictor web interface.

---

## Summary

| Method | Speed | Reproducibility | Command |
|--------|-------|-----------------|---------|
| **UV** | ⚡⚡⚡ Fast | High | `uv venv && .\.venv\Scripts\Activate.ps1 && uv pip install -r requirements.txt` |
| **VENV** | ⚡ Slower | Medium | `python -m venv .venv && pip install -r requirements.txt` |
| **UV + Lock** | ⚡⚡ Fast | ⚡⚡⚡ Highest | `uv pip compile && uv pip install -r requirements.lock` |

**Recommendation**: Use **UV** for fastest setup and installation.


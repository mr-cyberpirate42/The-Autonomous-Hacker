# 🤖 Autonomous Hacker Agent

> **ADK-Compatible AI Agent for Kaggle Competition Domination**

![ADK](https://img.shields.io/badge/ADK-Compatible-success)
![Kaggle](https://img.shields.io/badge/Kaggle-Competition-20BEFF)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)

# 🤖 Autonomous Hacker Agent

> **ADK-Compatible AI Agent for Kaggle Competition Domination**

![ADK](https://img.shields.io/badge/ADK-Compatible-success)
![Kaggle](https://img.shields.io/badge/Kaggle-Competition-20BEFF)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)

## 🚀 Quick Start

This project targets modern Python (3.10+ recommended). The instructions below use PowerShell on Windows.

### 1. Create and activate a virtual environment

```powershell
# create venv (one-time)
python -m venv .venv

# activate venv (PowerShell)
.\.venv\Scripts\Activate.ps1
```

If you prefer the system Python (e.g., Python 3.13) you can skip the venv step, but using a venv is recommended.

### 2. Upgrade pip and tooling

```powershell
python -m pip install --upgrade pip setuptools wheel
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

### Common Warnings and How to Resolve Them

You may see a few common warnings during installation; here are the typical ones and how to resolve them:

- **"You are using pip version X; however version Y is available."**
	- Fix: Run `python -m pip install --upgrade pip` (see step 2 above).

- **Warnings or errors mentioning `asyncio`**
	- Cause: `asyncio` is part of the Python standard library on modern Python and should not be included in `requirements.txt`.
	- Fix: Remove `asyncio` from `requirements.txt`. This repository's `requirements.txt` has been updated accordingly.

- **Binary wheel / compilation errors for packages such as `xgboost`, `catboost`, `lightgbm`**
	- Cause: These packages provide platform-specific wheels and may fail to build on 32-bit Python or when compatible wheels are not available.
	- Fixes:
		- Use a 64-bit Python installation (recommended).
		- Make sure `pip` is up-to-date so it can download prebuilt wheels.
		- On Windows, consider using `conda` to install heavy ML packages if `pip` builds fail.

### 4. Run the agent

From the project root (with venv activated):

```powershell
python agent.py
```

Or use the VS Code launch/task which is preconfigured to use the project venv.

---

If you still see warnings when running `pip install -r requirements.txt`, copy the
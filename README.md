# Identra-.-007

A PalmPay-style biometric payments demo built with Flask and SQLite.

## Project notes
- Keep the website behavior and UI as-is.
- Do not commit large model binaries, local databases, virtual environments, or generated files.
- Repository size is intentionally kept lean for GitHub uploads.

## Local run
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

## Git cleanup strategy
This repo uses a lean .gitignore to avoid pushing:
- `__pycache__`, compiled Python files
- local SQLite DBs and uploaded data
- ONNX model files and large model folders
- virtual environments and OS/editor junk

If you later add large assets, keep them outside Git or add a more specific ignore rule for them.

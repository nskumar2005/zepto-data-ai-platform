"""Convenience runner for the three capstone modules."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

commands = [
    [sys.executable, str(ROOT / "data_pipeline" / "pipeline.py")],
    [sys.executable, str(ROOT / "analytics" / "01_eda.py")],
    [sys.executable, str(ROOT / "analytics" / "02_modeling.py")],
    [sys.executable, str(ROOT / "support_assistant" / "ingest.py")],
]

for cmd in commands:
    print("\n$", " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)

print("\nAll runnable batch stages completed. Start FastAPI separately with uvicorn.")

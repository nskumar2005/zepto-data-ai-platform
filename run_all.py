"""Run the three modules in order where practical."""
import subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
subprocess.run([sys.executable,str(ROOT/"data_pipeline/pipeline.py")],check=True)
subprocess.run([sys.executable,str(ROOT/"analytics/01_eda.py")],check=True)
subprocess.run([sys.executable,str(ROOT/"analytics/02_modeling.py")],check=True)
subprocess.run([sys.executable,str(ROOT/"support_assistant/ingest.py")],check=True)
print("All modules completed.")

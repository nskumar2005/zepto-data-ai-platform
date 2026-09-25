"""Static repository completeness checker for the capstone submission."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
required = [
    "README.md", "requirements.txt", "run_all.py",
    "data_pipeline/README.md", "data_pipeline/pipeline.py", "data_pipeline/sql_queries.sql",
    "analytics/README.md", "analytics/01_eda.py", "analytics/02_modeling.py", "analytics/titanic.csv",
    "support_assistant/README.md", "support_assistant/Dockerfile", "support_assistant/requirements.txt",
    "support_assistant/main.py", "support_assistant/ingest.py", "support_assistant/rag.py", "support_assistant/prompt.py",
]
required += [f"support_assistant/docs/doc_{i:02d}.txt" for i in range(1, 9)]

missing = [p for p in required if not (ROOT / p).exists()]
print(f"Required paths checked: {len(required)}")
if missing:
    print("MISSING:")
    for p in missing:
        print(" -", p)
    raise SystemExit(1)

print("PASS: required repository structure is present.")
print("Note: this checker does not replace running each module and reviewing its generated outputs.")

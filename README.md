# Zepto Data & AI Platform

End-to-end AI/ML capstone implementing three connected modules in one repository:

1. **Data Pipeline** — scrape, clean, convert, normalize and query catalog data.
2. **Analytics** — profile and clean Titanic data, perform EDA, train/evaluate classifiers, tune Random Forest, and run a regression side-task.
3. **Support Assistant** — local embeddings + ChromaDB retrieval + LangGraph routing + FastAPI, with a deterministic `MOCK_LLM=1` baseline.

This repository follows the capstone requirement of one public repository containing `/data_pipeline`, `/analytics`, and `/support_assistant`. The assignment explicitly requires one root README and an end-to-end setup/run description. See the supplied capstone brief for the full grading criteria.

## Repository structure

```text
zepto-data-ai-platform/
├── README.md
├── requirements.txt
├── run_all.py
├── verify_submission.py
├── .gitignore
├── data_pipeline/
│   ├── README.md
│   ├── pipeline.py
│   ├── sql_queries.sql
│   └── output/
│       └── .gitkeep
├── analytics/
│   ├── README.md
│   ├── 01_eda.py
│   ├── 02_modeling.py
│   ├── titanic.csv
│   ├── figures/
│   │   └── .gitkeep
│   ├── models/
│   │   └── .gitkeep
│   └── output/
│       └── .gitkeep
└── support_assistant/
    ├── README.md
    ├── Dockerfile
    ├── requirements.txt
    ├── main.py
    ├── ingest.py
    ├── rag.py
    ├── prompt.py
    ├── docs/
    │   ├── doc_01.txt
    │   ├── doc_02.txt
    │   ├── doc_03.txt
    │   ├── doc_04.txt
    │   ├── doc_05.txt
    │   ├── doc_06.txt
    │   ├── doc_07.txt
    │   └── doc_08.txt
    └── chroma_db/
```

## Setup

Python 3.10+ is recommended.

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# Windows CMD
# .venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate

pip install -r requirements.txt
```

There is one consolidated root `requirements.txt`. The support assistant also contains a small module-specific `requirements.txt` for Docker/local isolation.

## Run Module 1 — Data Pipeline

From the repository root:

```bash
python data_pipeline/pipeline.py
```

The script scrapes at least 60 books from at least 3 categories, cleans the fields, applies the fixed `1 GBP = 105.50 INR` rate, creates a normalized SQLite database, executes the required SQL queries, and saves query outputs under `data_pipeline/output/`.

The conversion rate is intentionally fixed by the assignment; it is not a live market rate.

## Run Module 2 — Analytics

The raw dataset is loaded with `sns.load_dataset('titanic')` exactly once by `01_eda.py`. Immediately after loading, the script saves the committed offline fallback `analytics/titanic.csv`. The modeling script then reads that CSV instead of calling `sns.load_dataset` again.

```bash
python analytics/01_eda.py
python analytics/02_modeling.py
```

Outputs include charts in `analytics/figures/`, written metrics/results in `analytics/output/`, and the fitted end-to-end pipeline in `analytics/models/`.

If the first `01_eda.py` run cannot reach the Seaborn dataset repository, use the committed `analytics/titanic.csv` as the grading/offline fallback. The assignment's intended online path is the single `sns.load_dataset('titanic')` call.

## Run Module 3 — Support Assistant

First build the ChromaDB collection and local embeddings:

```bash
python support_assistant/ingest.py
```

Then start FastAPI with the required default mock mode:

```bash
uvicorn support_assistant.main:app --reload --port 7860
```

Example policy request:

```bash
curl -X POST http://127.0.0.1:7860/ask ^
  -H "Content-Type: application/json" ^
  -d "{\"query\":\"What is the standard delivery fee?\"}"
```

Example general request:

```bash
curl -X POST http://127.0.0.1:7860/ask ^
  -H "Content-Type: application/json" ^
  -d "{\"query\":\"What is the capital of France?\"}"
```

Keep `MOCK_LLM` unset or set it to `1` for the graded deterministic baseline. Set `MOCK_LLM=0` only for the optional real-LLM extension.

## Docker — Support Assistant

From the repository root:

```bash
docker build -f support_assistant/Dockerfile -t zepto-support .
docker run --rm -p 7860:7860 zepto-support
```

Then call `POST /ask` on `http://127.0.0.1:7860`.

## Design decisions

### Data Pipeline
- `requests` + BeautifulSoup are used for scraping.
- The pipeline handles at least 3 categories and at least 60 books.
- Rating text is mapped from One–Five to integers 1–5.
- Availability is converted to a boolean `in_stock`.
- Numeric parsing failures are handled with median imputation.
- A two-table normalized SQLite schema enforces a category PK/FK relationship.
- SQL query strings and their outputs are saved for auditability.

### Analytics
- EDA and modeling are one connected workflow based on the same cleaned dataset.
- Missing-value handling in EDA follows the assignment's percentage thresholds.
- The modeling pipeline uses a stratified split and a train-only `ColumnTransformer`/`Pipeline` for imputation, encoding and scaling.
- Three classifiers are evaluated using the required metrics.
- Imbalance strategies are compared, SMOTE is applied only to the training data, and Random Forest tuning uses `oob_score=True`.
- The saved model is the complete preprocessing + estimator pipeline, not a bare estimator.

### Support Assistant
- The 8 supplied policy documents are chunked and embedded with `all-MiniLM-L6-v2`.
- ChromaDB stores the vectors.
- LangGraph has `classify_intent`, `retrieve_and_answer`, and `direct_answer` nodes.
- Retrieval runs in both mock and real modes; only generation branches on `MOCK_LLM`.
- Final output is validated by a Pydantic schema containing `answer`, `sources`, and `confidence`.

## Git workflow required by the brief

The capstone requires a feature branch with at least two commits and a merge back into `main`. Use:

```bash
git checkout -b feature/capstone-implementation
git add .
git commit -m "feat: add capstone project scaffold"
# make/verify additional changes
git add .
git commit -m "feat: implement three capstone modules"
git checkout main
git merge --no-ff feature/capstone-implementation -m "merge: capstone implementation"
git log --graph --oneline --all
```

Do not squash away the feature-branch history because the branch/merge history is part of the grading requirement.

## Verification

```bash
python verify_submission.py
```

This performs structural checks and reports missing required paths/files without pretending that a full live run has completed.

# Zepto Data & AI Platform — Capstone Project

This repository implements the three connected modules specified in the capstone brief:

- `/data_pipeline` — scrape → clean → convert → normalized SQLite → SQL + pandas validation.
- `/analytics` — one-load Titanic profiling/EDA → train-only preprocessing → three classifiers → imbalance study → RF tuning/OOB → regression → saved end-to-end pipeline.
- `/support_assistant` — local embeddings + ChromaDB → LangGraph intent routing → deterministic offline mock baseline → FastAPI → Docker.

The three modules are intentionally independent at runtime but are submitted as **one repository**.

## Setup

Use Python 3.10+.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

No paid service is required.

## Module 1 — Data pipeline

```bash
cd data_pipeline
python pipeline.py
```

The script:
1. Scrapes at least 60 books across five catalogue pages using `requests` + `BeautifulSoup`.
2. Cleans price/rating/availability.
3. Converts GBP → INR using the required fixed rate **1 GBP = 105.50 INR**.
4. Creates `zepto_catalog.db` with `categories` and `books`.
5. Runs five required SQL demonstrations plus a JOIN.
6. Reads query results with `pd.read_sql` and reproduces the JOIN with `pd.merge`.
7. Writes outputs to `output/sql_outputs.txt` and `output/pandas_join_comparison.csv`.

## Module 2 — Analytics

The raw Titanic data must be loaded once. `01_eda.py` first tries the required `sns.load_dataset("titanic")`; if network/cache access is unavailable, it uses `analytics/titanic.csv` as the committed/offline fallback when that file is present. The modeling stage never calls `sns.load_dataset` again.

```bash
cd analytics
python 01_eda.py
python 02_modeling.py
```

Outputs include EDA figures, model metrics, imbalance comparison, GridSearchCV results, regression diagnostics, and `models/best_pipeline.joblib`.

### Dataset fallback note

The official Seaborn example repository contains the Titanic dataset used by `sns.load_dataset("titanic")`. In an offline environment, place that same `titanic.csv` in `/analytics` before running. The project code is deliberately written so the raw dataset is loaded only once by `01_eda.py`; `02_modeling.py` reads the committed CSV.

## Module 3 — Support assistant

Default mode is completely offline and deterministic:

```bash
cd support_assistant
python ingest.py
uvicorn main:app --reload --port 7860
```

Then:

```bash
curl -X POST http://127.0.0.1:7860/ask ^
  -H "Content-Type: application/json" ^
  -d "{\"query\":\"What is the delivery fee below INR 149?\"}"

curl -X POST http://127.0.0.1:7860/ask ^
  -H "Content-Type: application/json" ^
  -d "{\"query\":\"Tell me a joke.\"}"
```

The default `MOCK_LLM` behavior requires no API key. Retrieval still uses real local embeddings and ChromaDB.

Optional real-LLM mode:

```bash
set MOCK_LLM=0
set GROQ_API_KEY=your_key
set GROQ_MODEL=llama-3.1-8b-instant
```

The real-LLM path is optional and never required for the graded baseline.

### Docker

```bash
cd support_assistant
docker build -t zepto-support-assistant .
docker run --rm -p 7860:7860 zepto-support-assistant
```

## Architecture

```text
                 DATA PIPELINE
books.toscrape.com
       ↓
requests + BeautifulSoup
       ↓
clean / type conversion / GBP→INR
       ↓
categories ──< books (SQLite PK/FK)
       ↓
SQL + pandas validation

                 ANALYTICS
sns.load_dataset("titanic") OR offline titanic.csv
       ↓
single cleaned DataFrame
       ├── EDA + charts + correlation + z-score sanity check
       └── stratified split
              ↓
       ColumnTransformer
       (imputer + encoder + scaler)
              ↓
 Logistic / Tree / Random Forest
              ↓
 metrics + ROC/AUC + imbalance + GridSearch/OOB
              ↓
 complete fitted pipeline (.joblib)

              SUPPORT ASSISTANT
8 policy documents
       ↓
chunking → SentenceTransformer embeddings
       ↓
ChromaDB collection
       ↓
LangGraph classify_intent
       ├── policy_question → retrieve_and_answer
       └── general_question → direct_answer
       ↓
Pydantic AnswerResponse
       ↓
FastAPI /ask → local Docker container
```

`MOCK_LLM` branches only generation/classification generation steps. In the required default state, routing is keyword-based and answer generation is deterministic. Embedding and Chroma retrieval are real in both modes.

## Design decisions

### Data pipeline
A normalized two-table schema separates category entities from book records and avoids repeated category strings. Parsing failures are handled without crashing; numeric price/rating failures use median values, while rows with unusable title/category are dropped because they cannot form a reliable catalog record.

### Analytics
The EDA cleaning rule follows the assignment's percentage thresholds. Modeling uses a `Pipeline`/`ColumnTransformer` so imputation, encoding, and scaling are structurally fit on training data only. SMOTE is placed inside an imbalanced-learn pipeline and is therefore applied only to the training folds.

### Support assistant
The corpus is intentionally kept local. Each policy document is chunked at document level, embedded with `all-MiniLM-L6-v2`, and indexed in ChromaDB. The LangGraph state carries intent, retrieved chunks, answer, sources, and confidence. The mock path makes grading deterministic and independent of cloud APIs.

## Git workflow requirement

The assignment requires at least one feature branch with two commits and a merge back to `main`. The repository code cannot manufacture that GitHub history by itself. After creating the Git repository, use:

```bash
git init
git add .
git commit -m "chore: initial capstone scaffold"
git branch -M main
git checkout -b feature/data-platform
git add .
git commit -m "feat: implement capstone modules"
git add README.md
git commit -m "docs: complete capstone runbook"
git checkout main
git merge --no-ff feature/data-platform -m "merge: data platform feature"
```

Then push the single repository publicly.

# Module 2 — Analytics Pipeline

## Run order

```bash
python analytics/01_eda.py
python analytics/02_modeling.py
```

`01_eda.py` performs the one allowed `sns.load_dataset('titanic')` call and immediately writes `analytics/titanic.csv`. `02_modeling.py` reads that committed CSV and never calls the Seaborn loader.

## Required analysis coverage

The scripts cover profiling, missingness thresholds, IQR outliers, fare mean/median/mode and skewness, survival breakdowns, the exact six-column correlation matrix, four-plus multivariate charts with written interpretations, z-score sanity checks, stratified train/test splitting, train-only preprocessing, three classifiers, full classification metrics, imbalance comparison including training-only SMOTE, Random Forest GridSearchCV with `oob_score=True`, linear regression with adjusted R², residual analysis, final metric tables, and persistence/reload of the complete fitted pipeline.

Generated evidence is written to `figures/`, `output/`, and `models/`.

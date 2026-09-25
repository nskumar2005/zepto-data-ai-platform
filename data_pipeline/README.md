# Module 1 — Data Pipeline

## Objective
Scrape catalog data from Books to Scrape, clean it, apply the fixed `1 GBP = 105.50 INR` conversion, load a normalized SQLite database, and demonstrate SQL + pandas querying.

## Run

```bash
python data_pipeline/pipeline.py
```

## Outputs
- `output/books.sqlite3` — generated normalized database.
- `output/clean_books.csv` — cleaned dataset.
- `output/sql_outputs.txt` — query strings and printed outputs.
- `output/sql_outputs.txt` — comparison of the SQL join result with `pd.merge` is included at the end of this file.

## Cleaning decisions
- Price is parsed into `price_gbp` as float.
- Star words One–Five are mapped to 1–5.
- Availability becomes boolean `in_stock`.
- If numeric parsing produces missing values, the numeric column median is used.
- Rows with missing non-numeric required fields are dropped because they cannot safely participate in the relational output.

## Schema

`categories(category_id PK, category_name UNIQUE)`

`books(book_id PK, title, price_gbp, price_inr, rating, in_stock, category_id FK)`

The join query is reproduced in memory using `pd.merge` and compared against a `pd.read_sql` result.

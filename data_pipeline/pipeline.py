"""Books-to-SQLite end-to-end data pipeline."""

from pathlib import Path
import re
import sqlite3
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output"
OUT.mkdir(exist_ok=True)

DB_PATH = OUT / "books.sqlite3"

BASE_URL = "https://books.toscrape.com/"
RATE_GBP_TO_INR = 105.50

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 Chrome/140 Safari/537.36"
}


def fetch(url: str) -> str:
    """Fetch a URL and return its HTML."""
    response = requests.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()
    return response.text


def parse_category_links(html: str):
    """Extract category names and URLs from the Books to Scrape homepage."""
    soup = BeautifulSoup(html, "html.parser")

    categories = []

    for anchor in soup.select("div.side_categories ul li ul li a"):
        name = anchor.get_text(" ", strip=True)
        href = anchor.get("href")

        if name and isinstance(href, str):
            categories.append((name, urljoin(BASE_URL, href)))

    return categories


def parse_listing(html: str, category: str):
    """Parse all books from one category page."""
    soup = BeautifulSoup(html, "html.parser")
    rows = []

    for card in soup.select("article.product_pod"):
        # Title
        title = ""
        if card.h3 and card.h3.a:
            title_value = card.h3.a.get("title")
            title = title_value.strip() if isinstance(title_value, str) else ""

        # Price
        price_node = card.select_one("p.price_color")
        price_text = price_node.get_text(" ", strip=True) if price_node else ""

        # Rating
        rating = ""
        rating_node = card.select_one("p.star-rating")

        if rating_node:
            rating_classes = rating_node.get("class")
            if not isinstance(rating_classes, list):
                rating_classes = []
            for value in ["One", "Two", "Three", "Four", "Five"]:
                if value in rating_classes:
                    rating = value
                    break

        # Availability
        availability_node = card.select_one("p.instock.availability")
        availability = (
            availability_node.get_text(" ", strip=True)
            if availability_node
            else ""
        )

        rows.append(
            {
                "title": title,
                "price_gbp_raw": price_text,
                "star_rating": rating,
                "availability": availability,
                "category": category,
            }
        )

    next_link = soup.select_one("li.next a")
    next_href = next_link.get("href") if next_link else None

    return rows, next_href


def scrape(min_rows: int = 60) -> pd.DataFrame:
    """Scrape at least 60 books across at least 3 categories."""
    home_html = fetch(BASE_URL)
    categories = parse_category_links(home_html)

    if len(categories) < 3:
        raise RuntimeError(
            f"Only {len(categories)} categories were found on the site."
        )

    rows = []

    for category, category_url in categories:
        page_url = category_url

        while page_url and len(rows) < min_rows:
            html = fetch(page_url)

            page_rows, next_href = parse_listing(
                html,
                category
            )

            rows.extend(page_rows)

            if not next_href:
                break

            page_url = urljoin(page_url, str(next_href))

        category_count = len(
            {row["category"] for row in rows}
        )

        if len(rows) >= min_rows and category_count >= 3:
            break

    df = pd.DataFrame(rows)

    if df.empty:
        raise RuntimeError("Scraping returned zero rows.")

    df = df.drop_duplicates(
        subset=["title", "category"]
    ).reset_index(drop=True)

    category_count = df["category"].nunique()

    if len(df) < min_rows or category_count < 3:
        raise RuntimeError(
            f"Scrape scope too small: "
            f"{len(df)} rows / {category_count} categories"
        )

    return df


def parse_price(value) -> float:
    """
    Extract a numeric GBP value from scraped price text.

    Examples:
        '£51.77' -> 51.77
        ' £12.99 ' -> 12.99
        'GBP £8.50' -> 8.50
    """
    if pd.isna(value):
        return float("nan")

    text = str(value).strip()

    # Extract the first numeric value, including decimals.
    match = re.search(r"\d+(?:\.\d+)?", text)

    if not match:
        return float("nan")

    try:
        return float(match.group())
    except ValueError:
        return float("nan")


def parse_rating(value):
    """Convert text rating One..Five into integer 1..5."""
    rating_map = {
        "One": 1,
        "Two": 2,
        "Three": 3,
        "Four": 4,
        "Five": 5,
    }

    if pd.isna(value):
        return pd.NA

    return rating_map.get(str(value).strip(), pd.NA)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Clean scraped fields and apply the required GBP -> INR conversion."""
    df = df.copy()

    # Price
    df["price_gbp"] = df["price_gbp_raw"].apply(parse_price)

    # Rating
    df["rating"] = df["star_rating"].apply(parse_rating)

    # Stock status
    df["in_stock"] = (
        df["availability"]
        .astype("string")
        .str.contains("In stock", case=False, na=False)
    )

    # Numeric conversion
    df["price_gbp"] = pd.to_numeric(
        df["price_gbp"],
        errors="coerce"
    )

    df["rating"] = pd.to_numeric(
        df["rating"],
        errors="coerce"
    )

    # Print diagnostic information before imputation.
    print("\nCleaning diagnostics")
    print("--------------------")
    print(
        "price_gbp missing before imputation:",
        int(df["price_gbp"].isna().sum())
    )
    print(
        "rating missing before imputation:",
        int(df["rating"].isna().sum())
    )

    # Numeric imputation using the median.
    # This follows the assignment's required approach for malformed numeric
    # values.
    for column in ["price_gbp", "rating"]:
        missing_count = int(df[column].isna().sum())

        if missing_count > 0:
            median_value = df[column].median()

            if pd.isna(median_value):
                raise RuntimeError(
                    f"All values in {column} failed to parse. "
                    "The scraper returned no usable numeric values."
                )

            df[column] = df[column].fillna(median_value)

    # Drop rows missing required text fields.
    df = df.dropna(
        subset=["title", "category"]
    ).copy()

    # Enforce expected data types.
    df["price_gbp"] = df["price_gbp"].astype(float)
    df["rating"] = df["rating"].round().astype(int)
    df["in_stock"] = df["in_stock"].astype(bool)

    # Required fixed project conversion.
    df["price_inr"] = (
        df["price_gbp"] * RATE_GBP_TO_INR
    )

    cleaned = df[
        [
            "title",
            "price_gbp",
            "price_inr",
            "rating",
            "in_stock",
            "category",
        ]
    ].copy()

    # Final validation before SQLite.
    if cleaned["price_gbp"].isna().any():
        raise RuntimeError(
            "price_gbp still contains NaN values after cleaning."
        )

    if cleaned["price_inr"].isna().any():
        raise RuntimeError(
            "price_inr contains NaN values after conversion."
        )

    if cleaned["rating"].isna().any():
        raise RuntimeError(
            "rating still contains NaN values after cleaning."
        )

    print(
        f"\nCleaned dataset: {len(cleaned)} rows "
        f"across {cleaned['category'].nunique()} categories."
    )

    return cleaned


def load_sqlite(df: pd.DataFrame):
    """Create the normalized SQLite database and load cleaned data."""
    if DB_PATH.exists():
        DB_PATH.unlink()

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("PRAGMA foreign_keys = ON")

        conn.execute(
            """
            CREATE TABLE categories (
                category_id INTEGER PRIMARY KEY,
                category_name TEXT UNIQUE NOT NULL
            )
            """
        )

        conn.execute(
            """
            CREATE TABLE books (
                book_id INTEGER PRIMARY KEY,
                title TEXT NOT NULL,
                price_gbp REAL NOT NULL,
                price_inr REAL NOT NULL,
                rating INTEGER NOT NULL,
                in_stock INTEGER NOT NULL,
                category_id INTEGER NOT NULL,
                FOREIGN KEY(category_id)
                    REFERENCES categories(category_id)
            )
            """
        )

        # Insert categories.
        categories = pd.DataFrame(
            {
                "category_name": sorted(
                    df["category"].unique()
                )
            }
        )

        categories.to_sql(
            "categories",
            conn,
            if_exists="append",
            index=False
        )

        # Read category IDs.
        category_lookup = pd.read_sql(
            """
            SELECT category_id, category_name
            FROM categories
            """,
            conn
        )

        # Join category IDs to books.
        books = df.merge(
            category_lookup,
            left_on="category",
            right_on="category_name",
            how="inner"
        )

        books = books[
            [
                "title",
                "price_gbp",
                "price_inr",
                "rating",
                "in_stock",
                "category_id",
            ]
        ].copy()

        # Final SQLite validation.
        if books["price_gbp"].isna().any():
            raise RuntimeError(
                "Attempting to insert NULL price_gbp into SQLite."
            )

        if books["price_inr"].isna().any():
            raise RuntimeError(
                "Attempting to insert NULL price_inr into SQLite."
            )

        books.to_sql(
            "books",
            conn,
            if_exists="append",
            index=False
        )

        count = conn.execute(
            "SELECT COUNT(*) FROM books"
        ).fetchone()[0]

        print(f"SQLite rows inserted: {count}")


def run_queries(df: pd.DataFrame):
    """Run required SQL queries and compare SQL JOIN with pandas merge."""
    queries = [
        (
            "SELECT_WHERE",
            """
            SELECT title, price_gbp, rating
            FROM books
            WHERE price_gbp > 30;
            """
        ),
        (
            "ORDER_BY",
            """
            SELECT title, price_gbp
            FROM books
            ORDER BY price_gbp DESC;
            """
        ),
        (
            "LIMIT",
            """
            SELECT title, rating
            FROM books
            ORDER BY rating DESC, title
            LIMIT 10;
            """
        ),
        (
            "DISTINCT",
            """
            SELECT DISTINCT rating
            FROM books
            ORDER BY rating;
            """
        ),
        (
            "BETWEEN",
            """
            SELECT title, price_gbp
            FROM books
            WHERE price_gbp BETWEEN 10 AND 20
            ORDER BY price_gbp;
            """
        ),
        (
            "JOIN",
            """
            SELECT
                c.category_name,
                b.title,
                b.rating,
                b.price_inr
            FROM books b
            JOIN categories c
                ON b.category_id = c.category_id
            ORDER BY
                b.rating DESC,
                c.category_name,
                b.title
            LIMIT 10;
            """
        ),
    ]

    blocks = []

    with sqlite3.connect(DB_PATH) as conn:

        for name, sql in queries:
            result = pd.read_sql(sql, conn)

            blocks.append(
                f"=== {name} ===\n"
                f"{sql.strip()}\n"
                f"{result.to_string(index=False)}\n"
            )

        # SQL JOIN result
        sql_join = pd.read_sql(
            queries[-1][1],
            conn
        )

        # Reproduce the JOIN using pandas.
        category_lookup = (
            df[["category"]]
            .drop_duplicates()
            .sort_values("category")
            .reset_index(drop=True)
        )

        category_lookup["category_id"] = (
            category_lookup.index + 1
        )

        mem_books = df.merge(
            category_lookup,
            on="category",
            how="inner"
        )

        mem_join = mem_books[
            [
                "category",
                "title",
                "rating",
                "price_inr",
            ]
        ].rename(
            columns={
                "category": "category_name"
            }
        )

        mem_join = (
            mem_join
            .sort_values(
                [
                    "rating",
                    "category_name",
                    "title",
                ],
                ascending=[
                    False,
                    True,
                    True,
                ],
            )
            .head(10)
            .reset_index(drop=True)
        )

        sql_norm = (
            sql_join
            .sort_values(
                [
                    "rating",
                    "category_name",
                    "title",
                ],
                ascending=[
                    False,
                    True,
                    True,
                ],
            )
            .reset_index(drop=True)
        )

        # Make numeric types consistent for comparison.
        sql_norm["rating"] = sql_norm["rating"].astype(int)
        mem_join["rating"] = mem_join["rating"].astype(int)

        sql_norm["price_inr"] = sql_norm["price_inr"].astype(float)
        mem_join["price_inr"] = mem_join["price_inr"].astype(float)

        equivalent = sql_norm.equals(mem_join)

        blocks.append(
            "=== pandas SQL-vs-merge equivalence ===\n"
            "SQL result:\n"
            + sql_norm.to_string(index=False)
            + "\n\n"
            "pd.merge result:\n"
            + mem_join.to_string(index=False)
            + f"\n\nEquivalent: {equivalent}\n"
        )

    (OUT / "sql_outputs.txt").write_text(
        "\n".join(blocks),
        encoding="utf-8"
    )

    df.to_csv(
        OUT / "clean_books.csv",
        index=False
    )


def main():
    print("Starting Books-to-Scrape pipeline...")

    raw_df = scrape()

    print(
        f"Scraped {len(raw_df)} rows across "
        f"{raw_df['category'].nunique()} categories."
    )

    clean_df = clean(raw_df)

    load_sqlite(clean_df)

    run_queries(clean_df)

    print("\nPipeline completed successfully.")
    print(f"Clean CSV: {OUT / 'clean_books.csv'}")
    print(f"SQLite DB: {DB_PATH}")
    print(f"SQL output: {OUT / 'sql_outputs.txt'}")
    print(f"GBP → INR fixed rate: {RATE_GBP_TO_INR}")


if __name__ == "__main__":
    main()
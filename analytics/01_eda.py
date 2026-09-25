"""Titanic EDA and cleaning stage.
Loads sns.load_dataset('titanic') exactly once.
"""

from pathlib import Path
import io
import contextlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"
OUT = ROOT / "output"

FIG.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)


def savefig(name):
    """Save the current matplotlib figure."""
    plt.tight_layout()
    plt.savefig(
        FIG / name,
        dpi=160,
        bbox_inches="tight"
    )
    plt.close()


def clean(df):
    """
    Clean missing values using the assignment threshold rule:

    <5% missing:
        Drop affected rows.

    5%-30% missing:
        Numeric -> median imputation
        Categorical -> 'Unknown'

    >30% missing:
        Preserve information using an explicit 'Unknown' category.
    """
    df = df.copy()

    # Measure missingness BEFORE cleaning.
    missing = df.isna().mean() * 100
    missing = missing[missing > 0].sort_values(ascending=False)

    strategies = {}

    for col in missing.index:
        pct = missing[col]

        # Less than 5% -> drop rows.
        if pct < 5:

            df = df.dropna(subset=[col])

            strategies[col] = (
                f"Drop rows: {pct:.2f}% missing "
                f"(<5% threshold)."
            )

        # 5%-30% -> impute.
        elif pct <= 30:

            # Numeric -> median.
            if pd.api.types.is_numeric_dtype(df[col]):

                median_value = df[col].median()

                df[col] = df[col].fillna(
                    median_value
                )

                strategies[col] = (
                    f"Median imputation: {pct:.2f}% missing "
                    f"(5%-30%)."
                )

            # Categorical -> Unknown.
            else:

                if isinstance(
                    df[col].dtype,
                    pd.CategoricalDtype
                ):

                    if (
                        "Unknown"
                        not in df[col].cat.categories
                    ):
                        df[col] = (
                            df[col]
                            .cat
                            .add_categories(["Unknown"])
                        )

                df[col] = df[col].fillna("Unknown")

                strategies[col] = (
                    f"Category 'Unknown': {pct:.2f}% missing "
                    f"(5%-30%)."
                )

        # >30% -> explicit missing category.
        else:

            if isinstance(
                df[col].dtype,
                pd.CategoricalDtype
            ):

                if (
                    "Unknown"
                    not in df[col].cat.categories
                ):
                    df[col] = (
                        df[col]
                        .cat
                        .add_categories(["Unknown"])
                    )

            df[col] = df[col].fillna("Unknown")

            strategies[col] = (
                f"Encode missing as 'Unknown': "
                f"{pct:.2f}% missing (>30%)."
            )

    return df, strategies, missing


def main():

    # ---------------------------------------------------------
    # 1. LOAD TITANIC DATASET EXACTLY ONCE
    # ---------------------------------------------------------

    df = sns.load_dataset("titanic")

    # Required offline fallback.
    df.to_csv(
        ROOT / "titanic.csv",
        index=False
    )

    profile = []

    # ---------------------------------------------------------
    # 2. DATA PROFILE
    # ---------------------------------------------------------

    profile.append("=== df.info() ===\n")

    buf = io.StringIO()

    with contextlib.redirect_stdout(buf):
        df.info()

    profile.append(
        buf.getvalue()
    )

    profile.append(
        "=== df.describe() ===\n"
        + df.describe(
            include="all"
        ).to_string()
    )

    profile.append(
        f"\n=== shape ===\n{df.shape}"
    )

    # ---------------------------------------------------------
    # 3. CLEANING
    # ---------------------------------------------------------

    cleaned, strategies, missing = clean(df)

    profile.append(
        "\n=== missing-value percentages "
        "before cleaning ==="
    )

    profile.append(
        missing.to_string()
    )

    profile.append(
        "\n=== strategies ==="
    )

    # IMPORTANT:
    # strategies is a dictionary:
    # column -> strategy
    #
    # missing contains:
    # column -> percentage

    profile.extend(
        f"{c}: {missing[c]:.2f}% -> {s}"
        for c, s in strategies.items()
    )

    # ---------------------------------------------------------
    # 4. UNIVARIATE ANALYSIS
    # ---------------------------------------------------------

    for col in ["age", "fare"]:

        # Histogram
        plt.figure(figsize=(7, 4))

        sns.histplot(
            cleaned[col],
            kde=True
        )

        plt.title(
            f"{col.title()} distribution"
        )

        savefig(
            f"{col}_hist.png"
        )

        # Box plot
        plt.figure(figsize=(7, 3))

        sns.boxplot(
            x=cleaned[col]
        )

        plt.title(
            f"{col.title()} box plot"
        )

        savefig(
            f"{col}_box.png"
        )

        # IQR outliers
        q1, q3 = cleaned[col].quantile(
            [0.25, 0.75]
        )

        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        n = int(
            (
                (cleaned[col] < lower_bound)
                |
                (cleaned[col] > upper_bound)
            ).sum()
        )

        profile.append(
            f"IQR outliers — {col}: {n}"
        )

    # ---------------------------------------------------------
    # 5. FARE SKEWNESS
    # ---------------------------------------------------------

    mode_fare = (
        cleaned["fare"]
        .mode()
        .iloc[0]
    )

    mean_fare = (
        cleaned["fare"]
        .mean()
    )

    median_fare = (
        cleaned["fare"]
        .median()
    )

    if (
        mean_fare
        >
        median_fare
        >
        mode_fare
    ):
        skew = "right-skewed"

    elif (
        mean_fare
        <
        median_fare
        <
        mode_fare
    ):
        skew = "left-skewed"

    else:
        skew = (
            "not strictly monotonic by "
            "mean/median/mode"
        )

    profile.append(
        "Fare mean="
        f"{mean_fare:.4f}, "
        "median="
        f"{median_fare:.4f}, "
        "mode="
        f"{mode_fare:.4f}; "
        "distribution conclusion: "
        f"{skew}."
    )

    # ---------------------------------------------------------
    # 6. BIVARIATE SURVIVAL ANALYSIS
    # ---------------------------------------------------------

    sex_rate = (
        cleaned
        .groupby(
            "sex",
            observed=True
        )["survived"]
        .mean()
    )

    pclass_rate = (
        cleaned
        .groupby(
            "pclass",
            observed=True
        )["survived"]
        .mean()
    )

    sex_class_rate = (
        cleaned
        .groupby(
            ["sex", "pclass"],
            observed=True
        )["survived"]
        .mean()
    )

    profile.append(
        "\n=== survival rate by sex ===\n"
        + sex_rate.to_string()
    )

    profile.append(
        "\n=== survival rate by pclass ===\n"
        + pclass_rate.to_string()
    )

    profile.append(
        "\n=== survival rate by sex+pclass ===\n"
        + sex_class_rate.to_string()
    )

    # ---------------------------------------------------------
    # 7. EXACT SIX-COLUMN CORRELATION MATRIX
    # ---------------------------------------------------------

    corr_cols = [
        "survived",
        "pclass",
        "age",
        "sibsp",
        "parch",
        "fare"
    ]

    corr = cleaned[corr_cols].corr()

    plt.figure(figsize=(8, 6))

    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="vlag",
        center=0
    )

    plt.title(
        "Titanic numeric correlation matrix"
    )

    savefig(
        "correlation_heatmap.png"
    )

    # Strongest two absolute correlations
    pairs = []

    for i, a in enumerate(corr_cols):

        for b in corr_cols[i + 1:]:

            pairs.append(
                (
                    abs(corr.loc[a, b]),
                    a,
                    b,
                    corr.loc[a, b]
                )
            )

    top2 = sorted(
        pairs,
        reverse=True
    )[:2]

    profile.append(
        "\n=== two strongest absolute "
        "off-diagonal correlations ==="
    )

    profile.extend(
        f"{a} vs {b}: r={r:.4f}"
        for _, a, b, r in top2
    )

    # ---------------------------------------------------------
    # 8. MULTIVARIATE CHART 1
    # ---------------------------------------------------------

    plt.figure(figsize=(7, 4))

    sns.barplot(
        data=cleaned,
        x="sex",
        y="survived",
        hue="pclass"
    )

    plt.title(
        "Survival by sex and passenger class"
    )

    savefig(
        "survival_sex_pclass.png"
    )

    profile.append(
        "Interpretation — Survival varies "
        "strongly by sex and pclass; the grouped "
        "bars show how class modifies the "
        "sex-based survival pattern."
    )

    # ---------------------------------------------------------
    # 9. MULTIVARIATE CHART 2
    # ---------------------------------------------------------

    plt.figure(figsize=(7, 4))

    sns.boxplot(
        data=cleaned,
        x="survived",
        y="fare",
        hue="sex"
    )

    plt.title(
        "Fare by survival and sex"
    )

    savefig(
        "fare_survival_sex.png"
    )

    profile.append(
        "Interpretation — Fare distributions "
        "differ across survival groups and sex, "
        "showing that ticket price is associated "
        "with passenger outcomes while also "
        "reflecting passenger segment differences."
    )

    # ---------------------------------------------------------
    # 10. MULTIVARIATE CHART 3
    # ---------------------------------------------------------

    plt.figure(figsize=(7, 4))

    sns.scatterplot(
        data=cleaned,
        x="age",
        y="fare",
        hue="survived",
        alpha=0.65
    )

    plt.title(
        "Age vs fare by survival"
    )

    savefig(
        "age_fare_survival.png"
    )

    profile.append(
        "Interpretation — The scatter plot combines "
        "age and fare with survival status and helps "
        "reveal whether high-fare and age regions "
        "contain different survival patterns."
    )

    # ---------------------------------------------------------
    # 11. MULTIVARIATE CHART 4
    # ---------------------------------------------------------

    plt.figure(figsize=(7, 4))

    sns.pointplot(
        data=cleaned,
        x="pclass",
        y="survived",
        hue="sex",
        errorbar=None
    )

    plt.title(
        "Survival rate by class and sex"
    )

    savefig(
        "survival_pointplot.png"
    )

    profile.append(
        "Interpretation — Survival rates are "
        "compared across the class hierarchy "
        "separately for each sex, providing a "
        "compact multivariate view of the main "
        "categorical relationships."
    )

    # ---------------------------------------------------------
    # 12. EDA-ONLY STANDARDIZATION CHECK
    # ---------------------------------------------------------

    scaler = StandardScaler()

    z = scaler.fit_transform(
        cleaned[
            ["age", "fare"]
        ]
    )

    zdf = pd.DataFrame(
        z,
        columns=[
            "age_z",
            "fare_z"
        ]
    )

    profile.append(
        "\n=== EDA-only z-score check ===\n"
        f"age mean/std: "
        f"{zdf['age_z'].mean():.6f}/"
        f"{zdf['age_z'].std(ddof=0):.6f}\n"
        f"fare mean/std: "
        f"{zdf['fare_z'].mean():.6f}/"
        f"{zdf['fare_z'].std(ddof=0):.6f}"
    )

    # ---------------------------------------------------------
    # 13. SAVE STRATEGY TABLE
    # ---------------------------------------------------------

    strategy_rows = []

    for col, strategy in strategies.items():

        strategy_rows.append(
            {
                "column": col,
                "missing_pct": float(
                    missing[col]
                ),
                "strategy": strategy
            }
        )

    strategy_df = pd.DataFrame(
        strategy_rows,
        columns=[
            "column",
            "missing_pct",
            "strategy"
        ]
    )

    strategy_df.to_csv(
        OUT / "missingness_strategies.csv",
        index=False
    )

    # ---------------------------------------------------------
    # 14. SAVE REPORT + CLEANED DATA
    # ---------------------------------------------------------

    (
        OUT / "eda_report.txt"
    ).write_text(
        "\n".join(profile),
        encoding="utf-8"
    )

    cleaned.to_csv(
        OUT / "cleaned_titanic.csv",
        index=False
    )

    print(
        "EDA complete. See "
        "analytics/output/eda_report.txt"
    )


if __name__ == "__main__":
    main()
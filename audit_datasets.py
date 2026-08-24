# =========================================================
# SMARTDOSEAI - DATASET AUDIT
# =========================================================

import os
import json
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif


# =========================================================
# DIRECTORIES
# =========================================================

os.makedirs("model_results/dataset_audit", exist_ok=True)


# =========================================================
# DATASET CONFIGURATION
# =========================================================

DATASETS = {

    "kidney": {
        "file": "datasets/kidney_dataset.csv",
        "target": "Kidney_Disease_Type"
    },

    "dialysis": {
        "file": "datasets/kidney_dataset.csv",
        "target": "Dialysis_Risk"
    },

    "diabetes": {
        "file": "datasets/diabetes_dataset.csv",
        "target": "Diabetes_Type"
    },

    "heart": {
        "file": "datasets/heart_dataset.csv",
        "target": "Heart_Disease_Type"
    },

    "hypertension": {
        "file": "datasets/hypertension_dataset.csv",
        "target": "Hypertension_Type"
    }

}


# =========================================================
# HELPER
# =========================================================

def safe_float(value):

    try:
        return float(value)

    except Exception:
        return None


# =========================================================
# AUDIT ONE DATASET
# =========================================================

def audit_dataset(
    dataset_name,
    file_path,
    target_column
):

    print("\n")
    print("=" * 80)
    print(
        f"DATASET AUDIT: {dataset_name.upper()}"
    )
    print("=" * 80)

    # -----------------------------------------------------
    # LOAD DATA
    # -----------------------------------------------------

    df = pd.read_csv(
        file_path
    )

    print(
        f"\nDataset shape: {df.shape}"
    )

    # -----------------------------------------------------
    # BASIC INFORMATION
    # -----------------------------------------------------

    print("\nColumns:")

    for column in df.columns:

        print(
            f"  - {column}"
        )

    # -----------------------------------------------------
    # TARGET
    # -----------------------------------------------------

    if target_column not in df.columns:

        print(
            f"\nERROR: Target '{target_column}' "
            f"not found."
        )

        return

    y = df[target_column]

    print(
        f"\nTarget column: {target_column}"
    )

    print(
        "\nTarget distribution:"
    )

    target_counts = (
        y.value_counts()
        .sort_index()
    )

    target_percentages = (
        y.value_counts(
            normalize=True
        )
        .sort_index()
        * 100
    )

    for label in target_counts.index:

        print(
            f"  {label}: "
            f"{target_counts[label]} "
            f"({target_percentages[label]:.2f}%)"
        )

    # -----------------------------------------------------
    # MISSING VALUES
    # -----------------------------------------------------

    print(
        "\nMissing values:"
    )

    missing = (
        df.isnull()
        .sum()
        .sort_values(
            ascending=False
        )
    )

    missing_found = False

    for column, count in missing.items():

        if count > 0:

            missing_found = True

            percentage = (
                count /
                len(df) *
                100
            )

            print(
                f"  {column}: "
                f"{count} "
                f"({percentage:.2f}%)"
            )

    if not missing_found:

        print(
            "  None"
        )

    # -----------------------------------------------------
    # DUPLICATES
    # -----------------------------------------------------

    duplicate_count = (
        df.duplicated()
        .sum()
    )

    duplicate_percentage = (
        duplicate_count /
        len(df) *
        100
    )

    print(
        "\nDuplicate rows:"
    )

    print(
        f"  {duplicate_count} "
        f"({duplicate_percentage:.2f}%)"
    )

    # -----------------------------------------------------
    # TARGET DUPLICATE PATTERNS
    # -----------------------------------------------------

    print(
        "\nTarget + feature uniqueness:"
    )

    unique_summary = []

    for column in df.columns:

        if column == target_column:
            continue

        unique_count = (
            df[column]
            .nunique(
                dropna=False
            )
        )

        unique_summary.append({

            "feature":
                column,

            "unique_values":
                unique_count,

            "unique_ratio":
                round(
                    unique_count /
                    len(df),
                    4
                )

        })

    unique_df = (
        pd.DataFrame(
            unique_summary
        )
        .sort_values(
            "unique_ratio",
            ascending=False
        )
    )

    print(
        unique_df.to_string(
            index=False
        )
    )

    # -----------------------------------------------------
    # NUMERICAL FEATURE ANALYSIS
    # -----------------------------------------------------

    numerical_columns = (
        df.select_dtypes(
            include=np.number
        )
        .columns
        .tolist()
    )

    numerical_columns = [

        column

        for column in numerical_columns

        if column != target_column

    ]

    numerical_report = []

    print(
        "\nNumerical feature analysis:"
    )

    for column in numerical_columns:

        values = df[column]

        numerical_report.append({

            "feature":
                column,

            "mean":
                safe_float(
                    values.mean()
                ),

            "std":
                safe_float(
                    values.std()
                ),

            "min":
                safe_float(
                    values.min()
                ),

            "max":
                safe_float(
                    values.max()
                )

        })

    numerical_df = pd.DataFrame(
        numerical_report
    )

    if not numerical_df.empty:

        print(
            numerical_df.to_string(
                index=False
            )
        )

    # -----------------------------------------------------
    # TARGET-WISE NUMERICAL DISTRIBUTIONS
    # -----------------------------------------------------

    if numerical_columns:

        print(
            "\nTarget-wise numerical means:"
        )

        grouped_means = (
            df.groupby(
                target_column
            )[numerical_columns]
            .mean()
            .round(3)
        )

        print(
            grouped_means.to_string()
        )

        grouped_means.to_csv(

            f"model_results/"
            f"dataset_audit/"
            f"{dataset_name}_"
            f"target_feature_means.csv"

        )

    # -----------------------------------------------------
    # CATEGORICAL FEATURE ANALYSIS
    # -----------------------------------------------------

    categorical_columns = (
        df.select_dtypes(
            include=[
                "object",
                "category"
            ]
        )
        .columns
        .tolist()
    )

    categorical_columns = [

        column

        for column in categorical_columns

        if column != target_column

    ]

    print(
        "\nCategorical feature analysis:"
    )

    categorical_report = []

    for column in categorical_columns:

        unique_values = (
            df[column]
            .dropna()
            .unique()
        )

        print(
            f"\n{column}:"
        )

        print(
            f"  Unique values: "
            f"{len(unique_values)}"
        )

        print(
            f"  Values: "
            f"{list(unique_values)[:20]}"
        )

        # Crosstab
        crosstab = pd.crosstab(

            df[column],

            df[target_column],

            normalize="index"

        ) * 100

        print(
            "\n  Target distribution by category (%):"
        )

        print(
            crosstab.round(2)
            .to_string()
        )

        crosstab.to_csv(

            f"model_results/"
            f"dataset_audit/"
            f"{dataset_name}_"
            f"{column}_target_distribution.csv"

        )

        categorical_report.append({

            "feature":
                column,

            "unique_values":
                len(unique_values)

        })

    # -----------------------------------------------------
    # MUTUAL INFORMATION
    # -----------------------------------------------------

    print(
        "\nFeature-target association "
        "(Mutual Information):"
    )

    X_mi = df.drop(
        columns=[target_column]
    ).copy()

    y_mi = y.copy()

    # Encode target
    target_encoder = LabelEncoder()

    y_encoded = (
        target_encoder.fit_transform(
            y_mi.astype(str)
        )
    )

    # Encode categorical features
    categorical_mask = []

    for column in X_mi.columns:

        if (
            X_mi[column].dtype ==
            "object"
        ):

            encoder = LabelEncoder()

            X_mi[column] = (
                encoder.fit_transform(
                    X_mi[column]
                    .astype(str)
                )
            )

            categorical_mask.append(
                True
            )

        else:

            X_mi[column] = (
                X_mi[column]
                .fillna(
                    X_mi[column]
                    .median()
                )
            )

            categorical_mask.append(
                False
            )

    # Fill remaining missing values
    X_mi = X_mi.fillna(0)

    try:

        mi_scores = (
            mutual_info_classif(

                X_mi,

                y_encoded,

                discrete_features=
                    categorical_mask,

                random_state=42

            )
        )

        mi_df = pd.DataFrame({

            "feature":
                X_mi.columns,

            "mutual_information":
                mi_scores

        })

        mi_df = (
            mi_df
            .sort_values(
                "mutual_information",
                ascending=False
            )
        )

        print(
            mi_df.to_string(
                index=False
            )
        )

        mi_df.to_csv(

            f"model_results/"
            f"dataset_audit/"
            f"{dataset_name}_"
            f"mutual_information.csv",

            index=False

        )

    except Exception as error:

        print(
            "\nMutual information calculation "
            f"failed: {error}"
        )

        mi_df = pd.DataFrame()

    # -----------------------------------------------------
    # TRAIN / TEST OVERLAP CHECK
    # -----------------------------------------------------

    print(
        "\nTrain/Test overlap check:"
    )

    X = df.drop(
        columns=[target_column]
    )

    y = df[target_column]

    try:

        X_train, X_test, y_train, y_test = (
            train_test_split(

                X,

                y,

                test_size=0.20,

                random_state=42,

                stratify=y

            )
        )

        train_indices = set(
            X_train.index
        )

        test_indices = set(
            X_test.index
        )

        overlapping_indices = (
            train_indices
            .intersection(
                test_indices
            )
        )

        print(
            f"  Exact row-index overlap: "
            f"{len(overlapping_indices)}"
        )

    except Exception as error:

        print(
            f"  Overlap check failed: "
            f"{error}"
        )

    # -----------------------------------------------------
    # SUMMARY
    # -----------------------------------------------------

    top_features = []

    if not mi_df.empty:

        top_features = (

            mi_df
            .head(10)
            .to_dict(
                orient="records"
            )

        )

    summary = {

        "dataset":
            dataset_name,

        "rows":
            int(df.shape[0]),

        "columns":
            int(df.shape[1]),

        "target":
            target_column,

        "target_classes":
            [
                str(value)
                for value in
                target_counts.index
            ],

        "duplicate_rows":
            int(duplicate_count),

        "missing_values":
            int(
                df.isnull()
                .sum()
                .sum()
            ),

        "top_mutual_information_features":
            top_features

    }

    summary_path = (

        f"model_results/"
        f"dataset_audit/"
        f"{dataset_name}_summary.json"

    )

    with open(
        summary_path,
        "w"
    ) as file:

        json.dump(
            summary,
            file,
            indent=4
        )

    print(
        f"\nSaved audit summary:"
        f"\n{summary_path}"
    )


# =========================================================
# RUN ALL AUDITS
# =========================================================

for dataset_name, config in DATASETS.items():

    audit_dataset(

        dataset_name,

        config["file"],

        config["target"]

    )


# =========================================================
# COMPLETE
# =========================================================

print("\n")
print("=" * 80)
print("DATASET AUDIT COMPLETED")
print("=" * 80)

print(
    "\nAudit reports are available in:"
)

print(
    "model_results/dataset_audit/"
)
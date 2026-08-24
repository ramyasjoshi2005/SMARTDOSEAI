# =========================================================
# SMARTDOSEAI - LEAKAGE-SAFE DATA PREPROCESSING
# =========================================================

import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
    LabelEncoder
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib


# =========================================================
# DIRECTORIES
# =========================================================

os.makedirs("processed_data", exist_ok=True)
os.makedirs("preprocessing", exist_ok=True)


# =========================================================
# PREPROCESS DATASET
# =========================================================

def preprocess_dataset(
    file_path,
    target_column,
    dataset_name,
    columns_to_drop=None
):

    print("\n" + "=" * 70)
    print(f"PROCESSING DATASET: {dataset_name.upper()}")
    print("=" * 70)

    # -----------------------------------------------------
    # LOAD DATA
    # -----------------------------------------------------

    df = pd.read_csv(file_path)

    print(f"Original dataset shape: {df.shape}")

    # -----------------------------------------------------
    # DROP LEAKAGE COLUMNS
    # -----------------------------------------------------

    if columns_to_drop:

        existing_columns = [
            column
            for column in columns_to_drop
            if column in df.columns
        ]

        if existing_columns:

            print(
                "Dropping leakage/unwanted columns:",
                existing_columns
            )

            df = df.drop(
                columns=existing_columns
            )

    # -----------------------------------------------------
    # REMOVE DUPLICATES
    # -----------------------------------------------------

    duplicate_count = df.duplicated().sum()

    if duplicate_count > 0:

        print(
            f"Removing {duplicate_count} duplicate rows"
        )

        df = df.drop_duplicates()

    # -----------------------------------------------------
    # REMOVE ROWS WITH MISSING TARGET
    # -----------------------------------------------------

    df = df.dropna(
        subset=[target_column]
    )

    # -----------------------------------------------------
    # SEPARATE FEATURES AND TARGET
    # -----------------------------------------------------

    X = df.drop(
        columns=[target_column]
    )

    y = df[target_column]

    # -----------------------------------------------------
    # TARGET ENCODING
    # -----------------------------------------------------

    target_encoder = LabelEncoder()

    y_encoded = target_encoder.fit_transform(y)

    print("\nTarget classes:")

    for index, label in enumerate(
        target_encoder.classes_
    ):

        print(
            f"  {index} -> {label}"
        )

    # -----------------------------------------------------
    # TRAIN / TEST SPLIT
    #
    # IMPORTANT:
    # We split BEFORE fitting any preprocessing.
    # This prevents data leakage.
    # -----------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(

        X,

        y_encoded,

        test_size=0.20,

        random_state=42,

        stratify=y_encoded

    )

    print(
        f"\nTraining samples: {len(X_train)}"
    )

    print(
        f"Testing samples: {len(X_test)}"
    )

    # -----------------------------------------------------
    # IDENTIFY COLUMN TYPES
    # -----------------------------------------------------

    numerical_columns = X_train.select_dtypes(
        include=[
            "int64",
            "float64",
            "int32",
            "float32"
        ]
    ).columns.tolist()

    categorical_columns = X_train.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    print(
        f"Numerical columns: {len(numerical_columns)}"
    )

    print(
        f"Categorical columns: {len(categorical_columns)}"
    )

    # -----------------------------------------------------
    # NUMERICAL PIPELINE
    # -----------------------------------------------------

    numerical_pipeline = Pipeline(
        steps=[

            (
                "imputer",

                SimpleImputer(
                    strategy="median"
                )
            ),

            (
                "scaler",

                StandardScaler()
            )

        ]
    )

    # -----------------------------------------------------
    # CATEGORICAL PIPELINE
    # -----------------------------------------------------

    categorical_pipeline = Pipeline(
        steps=[

            (
                "imputer",

                SimpleImputer(
                    strategy="most_frequent"
                )
            ),

            (
                "encoder",

                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                )
            )

        ]
    )

    # -----------------------------------------------------
    # COMPLETE PREPROCESSOR
    # -----------------------------------------------------

    preprocessor = ColumnTransformer(

        transformers=[

            (
                "numerical",

                numerical_pipeline,

                numerical_columns
            ),

            (
                "categorical",

                categorical_pipeline,

                categorical_columns
            )

        ],

        remainder="drop"

    )

    # -----------------------------------------------------
    # FIT ONLY ON TRAINING DATA
    # -----------------------------------------------------

    X_train_processed = preprocessor.fit_transform(
        X_train
    )

    # -----------------------------------------------------
    # TRANSFORM TEST DATA
    #
    # IMPORTANT:
    # We DO NOT fit again.
    # -----------------------------------------------------

    X_test_processed = preprocessor.transform(
        X_test
    )

    # -----------------------------------------------------
    # GET FEATURE NAMES
    # -----------------------------------------------------

    feature_names = (
        preprocessor
        .get_feature_names_out()
    )

    # -----------------------------------------------------
    # SAVE PROCESSED TRAIN DATA
    # -----------------------------------------------------

    train_processed = pd.DataFrame(
        X_train_processed,
        columns=feature_names
    )

    train_processed[
        target_column
    ] = y_train

    # -----------------------------------------------------
    # SAVE PROCESSED TEST DATA
    # -----------------------------------------------------

    test_processed = pd.DataFrame(
        X_test_processed,
        columns=feature_names
    )

    test_processed[
        target_column
    ] = y_test

    # -----------------------------------------------------
    # SAVE FILES
    # -----------------------------------------------------

    train_path = (
        f"processed_data/"
        f"{dataset_name}_train.csv"
    )

    test_path = (
        f"processed_data/"
        f"{dataset_name}_test.csv"
    )

    preprocessor_path = (
        f"preprocessing/"
        f"{dataset_name}_preprocessor.pkl"
    )

    encoder_path = (
        f"preprocessing/"
        f"{dataset_name}_target_encoder.pkl"
    )

    train_processed.to_csv(
        train_path,
        index=False
    )

    test_processed.to_csv(
        test_path,
        index=False
    )

    joblib.dump(
        preprocessor,
        preprocessor_path
    )

    joblib.dump(
        target_encoder,
        encoder_path
    )

    # -----------------------------------------------------
    # PRINT INFORMATION
    # -----------------------------------------------------

    print(
        f"\nProcessed feature count: "
        f"{len(feature_names)}"
    )

    print(
        f"Saved training data: {train_path}"
    )

    print(
        f"Saved testing data: {test_path}"
    )

    print(
        f"Saved preprocessor: "
        f"{preprocessor_path}"
    )

    print(
        f"Saved target encoder: "
        f"{encoder_path}"
    )

    print(
        "\nPreprocessing completed successfully."
    )


# =========================================================
# 1. KIDNEY DISEASE
# =========================================================
#
# Target:
# Kidney_Disease_Type
#
# IMPORTANT:
# Dialysis_Risk is another clinical outcome.
# It must NOT be used to predict kidney disease type.
# =========================================================

preprocess_dataset(

    file_path="datasets/kidney_dataset.csv",

    target_column="Kidney_Disease_Type",

    dataset_name="kidney",

    columns_to_drop=[
        "Dialysis_Risk"
    ]

)


# =========================================================
# 2. DIALYSIS RISK
# =========================================================
#
# Target:
# Dialysis_Risk
#
# IMPORTANT:
# Kidney_Disease_Type is another outcome/diagnosis.
# It must NOT be used as an input to predict dialysis risk.
# =========================================================

preprocess_dataset(

    file_path="datasets/kidney_dataset.csv",

    target_column="Dialysis_Risk",

    dataset_name="dialysis",

    columns_to_drop=[
        "Kidney_Disease_Type"
    ]

)


# =========================================================
# 3. DIABETES
# =========================================================

preprocess_dataset(

    file_path="datasets/diabetes_dataset.csv",

    target_column="Diabetes_Type",

    dataset_name="diabetes"

)


# =========================================================
# 4. HEART DISEASE
# =========================================================

preprocess_dataset(

    file_path="datasets/heart_dataset.csv",

    target_column="Heart_Disease_Type",

    dataset_name="heart"

)


# =========================================================
# 5. HYPERTENSION
# =========================================================

preprocess_dataset(

    file_path="datasets/hypertension_dataset.csv",

    target_column="Hypertension_Type",

    dataset_name="hypertension"

)


# =========================================================
# COMPLETE
# =========================================================

print("\n" + "=" * 70)
print("ALL DATASETS PREPROCESSED SUCCESSFULLY")
print("=" * 70)
# ============================================================
# SMARTDOSEAI - FINAL MODEL TRAINING PIPELINE
# ============================================================
#
# Compatible with the existing preprocess_data.py
#
# Datasets:
#   1. Kidney
#   2. Dialysis
#   3. Diabetes
#   4. Heart
#   5. Hypertension
#
# Models:
#   - Random Forest
#   - XGBoost
#   - LightGBM
#
# Evaluation:
#   - Accuracy
#   - Precision
#   - Recall
#   - Weighted F1
#   - Macro F1
#   - ROC-AUC
#   - 5-fold Stratified Cross Validation
#   - Classification Report
#   - Confusion Matrix
#   - Feature Importance
#
# Model selection:
#   Macro F1
#
# ============================================================


import os
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score
)

from sklearn.preprocessing import LabelEncoder

from xgboost import XGBClassifier

from lightgbm import LGBMClassifier


# ============================================================
# SETTINGS
# ============================================================

warnings.filterwarnings("ignore")

RANDOM_STATE = 42

N_SPLITS = 5


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

os.makedirs(
    "trained_models",
    exist_ok=True
)

os.makedirs(
    "model_results",
    exist_ok=True
)

os.makedirs(
    "model_results/confusion_matrices",
    exist_ok=True
)

os.makedirs(
    "model_results/feature_importance",
    exist_ok=True
)

os.makedirs(
    "model_results/classification_reports",
    exist_ok=True
)


# ============================================================
# DATASET CONFIGURATION
# ============================================================

DATASETS = {

    "kidney": {

        "train":
            "processed_data/kidney_train.csv",

        "test":
            "processed_data/kidney_test.csv"

    },

    "dialysis": {

        "train":
            "processed_data/dialysis_train.csv",

        "test":
            "processed_data/dialysis_test.csv"

    },

    "diabetes": {

        "train":
            "processed_data/diabetes_train.csv",

        "test":
            "processed_data/diabetes_test.csv"

    },

    "heart": {

        "train":
            "processed_data/heart_train.csv",

        "test":
            "processed_data/heart_test.csv"

    },

    "hypertension": {

        "train":
            "processed_data/hypertension_train.csv",

        "test":
            "processed_data/hypertension_test.csv"

    }

}


# ============================================================
# DETECT TARGET COLUMN
# ============================================================

def detect_target_column(train_df, test_df):

    """
    The existing preprocessing script stores the target
    column together with the processed features.

    We therefore detect the target from the final column.

    This avoids assuming the column is literally called
    'target'.
    """

    train_columns = list(
        train_df.columns
    )

    test_columns = list(
        test_df.columns
    )

    # --------------------------------------------------------
    # Check that train/test have the same columns
    # --------------------------------------------------------

    if set(train_columns) != set(test_columns):

        raise ValueError(
            "Training and testing datasets "
            "do not contain the same columns."
        )

    # --------------------------------------------------------
    # Make test column order identical to train
    # --------------------------------------------------------

    test_df = test_df[
        train_columns
    ]

    # --------------------------------------------------------
    # Existing preprocessing convention:
    # target is the final column.
    # --------------------------------------------------------

    target_column = train_columns[-1]

    return (
        target_column,
        train_df,
        test_df
    )


# ============================================================
# PREPARE TARGET
# ============================================================

def prepare_target(
    y_train,
    y_test
):

    """
    Converts target labels into integer classes.

    If the target is already encoded as:
        0, 1, 2, ...

    it is kept.

    If it contains strings/categories,
    LabelEncoder is used.
    """

    # --------------------------------------------------------
    # NUMERIC TARGET
    # --------------------------------------------------------

    if pd.api.types.is_numeric_dtype(
        y_train
    ):

        y_train = (
            pd.to_numeric(
                y_train
            )
            .astype(int)
        )

        y_test = (
            pd.to_numeric(
                y_test
            )
            .astype(int)
        )

        # ----------------------------------------------------
        # Ensure labels are 0...N-1
        # ----------------------------------------------------

        unique_labels = sorted(
            y_train.unique()
        )

        expected_labels = list(
            range(
                len(unique_labels)
            )
        )

        if unique_labels != expected_labels:

            mapping = {

                old_label: new_label

                for new_label, old_label
                in enumerate(
                    unique_labels
                )

            }

            y_train = (
                y_train
                .map(mapping)
                .astype(int)
            )

            y_test = (
                y_test
                .map(mapping)
                .astype(int)
            )

        return (
            y_train,
            y_test,
            None
        )

    # --------------------------------------------------------
    # CATEGORICAL / STRING TARGET
    # --------------------------------------------------------

    encoder = LabelEncoder()

    y_train_encoded = (
        encoder.fit_transform(
            y_train.astype(str)
        )
    )

    y_test_encoded = (
        encoder.transform(
            y_test.astype(str)
        )
    )

    y_train_encoded = pd.Series(
        y_train_encoded,
        index=y_train.index
    )

    y_test_encoded = pd.Series(
        y_test_encoded,
        index=y_test.index
    )

    return (
        y_train_encoded,
        y_test_encoded,
        encoder
    )


# ============================================================
# CREATE MODELS
# ============================================================

def create_models():

    models = {

        # ----------------------------------------------------
        # RANDOM FOREST
        # ----------------------------------------------------

        "RandomForest":

            RandomForestClassifier(

                n_estimators=400,

                max_depth=None,

                min_samples_split=4,

                min_samples_leaf=2,

                max_features="sqrt",

                class_weight="balanced",

                random_state=RANDOM_STATE,

                n_jobs=-1

            ),

        # ----------------------------------------------------
        # XGBOOST
        # ----------------------------------------------------

        "XGBoost":

            XGBClassifier(

                n_estimators=300,

                max_depth=5,

                learning_rate=0.05,

                subsample=0.85,

                colsample_bytree=0.85,

                min_child_weight=2,

                objective="multi:softprob",

                eval_metric="mlogloss",

                random_state=RANDOM_STATE,

                n_jobs=-1

            ),

        # ----------------------------------------------------
        # LIGHTGBM
        # ----------------------------------------------------

        "LightGBM":

            LGBMClassifier(

                n_estimators=300,

                learning_rate=0.05,

                num_leaves=31,

                max_depth=-1,

                min_child_samples=20,

                subsample=0.85,

                colsample_bytree=0.85,

                class_weight="balanced",

                random_state=RANDOM_STATE,

                n_jobs=-1,

                verbosity=-1

            )

    }

    return models


# ============================================================
# CALCULATE ROC-AUC
# ============================================================

def calculate_roc_auc(
    model,
    X_test,
    y_test
):

    try:

        probabilities = (
            model.predict_proba(
                X_test
            )
        )

        number_of_classes = len(
            np.unique(
                y_test
            )
        )

        # ----------------------------------------------------
        # Binary
        # ----------------------------------------------------

        if number_of_classes == 2:

            return roc_auc_score(

                y_test,

                probabilities[:, 1]

            )

        # ----------------------------------------------------
        # Multi-class
        # ----------------------------------------------------

        return roc_auc_score(

            y_test,

            probabilities,

            multi_class="ovr",

            average="weighted"

        )

    except Exception:

        return np.nan


# ============================================================
# SAVE CONFUSION MATRIX
# ============================================================

def save_confusion_matrix(
    dataset_name,
    model_name,
    y_test,
    predictions
):

    matrix = confusion_matrix(

        y_test,

        predictions

    )

    matrix_df = pd.DataFrame(
        matrix
    )

    path = (

        "model_results/"
        "confusion_matrices/"
        f"{dataset_name}_"
        f"{model_name}_"
        "confusion_matrix.csv"

    )

    matrix_df.to_csv(

        path,

        index=False

    )


# ============================================================
# SAVE CLASSIFICATION REPORT
# ============================================================

def save_classification_report(
    dataset_name,
    model_name,
    y_test,
    predictions
):

    report = classification_report(

        y_test,

        predictions,

        zero_division=0

    )

    path = (

        "model_results/"
        "classification_reports/"
        f"{dataset_name}_"
        f"{model_name}_"
        "classification_report.txt"

    )

    with open(
        path,
        "w"
    ) as file:

        file.write(
            report
        )


# ============================================================
# SAVE FEATURE IMPORTANCE
# ============================================================

def save_feature_importance(
    dataset_name,
    model_name,
    model,
    feature_names
):

    if not hasattr(
        model,
        "feature_importances_"
    ):

        return

    importance_df = pd.DataFrame({

        "feature":
            feature_names,

        "importance":
            model.feature_importances_

    })

    importance_df = (

        importance_df

        .sort_values(

            "importance",

            ascending=False

        )

        .reset_index(
            drop=True
        )

    )

    path = (

        "model_results/"
        "feature_importance/"
        f"{dataset_name}_"
        f"{model_name}_"
        "feature_importance.csv"

    )

    importance_df.to_csv(

        path,

        index=False

    )

    print(
        "\nTop 10 features:"
    )

    print(

        importance_df
        .head(10)
        .to_string(
            index=False
        )

    )


# ============================================================
# TRAIN ONE DATASET
# ============================================================

def train_dataset(
    dataset_name,
    train_path,
    test_path
):

    print("\n")

    print("=" * 75)

    print(
        f"TRAINING DATASET: "
        f"{dataset_name.upper()}"
    )

    print("=" * 75)


    # ========================================================
    # LOAD DATA
    # ========================================================

    train_df = pd.read_csv(
        train_path
    )

    test_df = pd.read_csv(
        test_path
    )

    print(
        f"\nTraining shape: "
        f"{train_df.shape}"
    )

    print(
        f"Testing shape: "
        f"{test_df.shape}"
    )


    # ========================================================
    # DETECT TARGET
    # ========================================================

    (
        target_column,
        train_df,
        test_df

    ) = detect_target_column(

        train_df,

        test_df

    )

    print(
        f"\nTarget column detected: "
        f"{target_column}"
    )


    # ========================================================
    # SPLIT FEATURES AND TARGET
    # ========================================================

    X_train = train_df.drop(

        columns=[
            target_column
        ]

    )

    X_test = test_df.drop(

        columns=[
            target_column
        ]

    )

    y_train_raw = (
        train_df[
            target_column
        ]
    )

    y_test_raw = (
        test_df[
            target_column
        ]
    )


    # ========================================================
    # TARGET ENCODING
    # ========================================================

    (
        y_train,
        y_test,
        target_encoder

    ) = prepare_target(

        y_train_raw,

        y_test_raw

    )


    print(
        f"Number of classes: "
        f"{len(np.unique(y_train))}"
    )

    print(
        "\nTarget distribution:"
    )

    print(

        y_train
        .value_counts()
        .sort_index()
        .to_string()

    )


    # ========================================================
    # FEATURE NAMES
    # ========================================================

    feature_names = list(
        X_train.columns
    )


    # ========================================================
    # CROSS VALIDATION
    # ========================================================

    cv = StratifiedKFold(

        n_splits=N_SPLITS,

        shuffle=True,

        random_state=RANDOM_STATE

    )


    # ========================================================
    # MODELS
    # ========================================================

    models = create_models()


    results = []


    best_model = None

    best_model_name = None

    best_macro_f1 = -np.inf


    # ========================================================
    # MODEL LOOP
    # ========================================================

    for model_name, model in models.items():

        print("\n")

        print("-" * 60)

        print(
            f"Training: "
            f"{model_name}"
        )

        print("-" * 60)


        # ====================================================
        # CROSS VALIDATION
        # ====================================================

        print(
            "\nRunning "
            "5-fold cross-validation..."
        )

        try:

            cv_scores = cross_val_score(

                model,

                X_train,

                y_train,

                cv=cv,

                scoring="f1_macro",

                n_jobs=1

            )

            cv_mean = (
                cv_scores.mean()
            )

            cv_std = (
                cv_scores.std()
            )

            print(

                f"CV Macro-F1: "
                f"{cv_mean:.4f} "
                f"+/- {cv_std:.4f}"

            )

        except Exception as error:

            print(
                "\nCross-validation failed:"
            )

            print(
                str(error)
            )

            cv_mean = np.nan

            cv_std = np.nan


        # ====================================================
        # TRAIN FINAL MODEL
        # ====================================================

        model.fit(

            X_train,

            y_train

        )


        # ====================================================
        # PREDICT
        # ====================================================

        predictions = (
            model.predict(
                X_test
            )
        )


        # ====================================================
        # METRICS
        # ====================================================

        accuracy = accuracy_score(

            y_test,

            predictions

        )

        precision = precision_score(

            y_test,

            predictions,

            average="weighted",

            zero_division=0

        )

        recall = recall_score(

            y_test,

            predictions,

            average="weighted",

            zero_division=0

        )

        weighted_f1 = f1_score(

            y_test,

            predictions,

            average="weighted",

            zero_division=0

        )

        macro_f1 = f1_score(

            y_test,

            predictions,

            average="macro",

            zero_division=0

        )

        roc_auc = calculate_roc_auc(

            model,

            X_test,

            y_test

        )


        # ====================================================
        # PRINT RESULTS
        # ====================================================

        print(
            f"\nAccuracy       : "
            f"{accuracy:.4f}"
        )

        print(
            f"Precision      : "
            f"{precision:.4f}"
        )

        print(
            f"Recall         : "
            f"{recall:.4f}"
        )

        print(
            f"Weighted F1    : "
            f"{weighted_f1:.4f}"
        )

        print(
            f"Macro F1       : "
            f"{macro_f1:.4f}"
        )

        if np.isnan(
            roc_auc
        ):

            print(
                "ROC-AUC        : N/A"
            )

        else:

            print(
                f"ROC-AUC        : "
                f"{roc_auc:.4f}"
            )


        # ====================================================
        # CLASSIFICATION REPORT
        # ====================================================

        print(
            "\nClassification Report:"
        )

        report = classification_report(

            y_test,

            predictions,

            zero_division=0

        )

        print(
            report
        )


        # ====================================================
        # SAVE CLASSIFICATION REPORT
        # ====================================================

        save_classification_report(

            dataset_name,

            model_name,

            y_test,

            predictions

        )


        # ====================================================
        # SAVE CONFUSION MATRIX
        # ====================================================

        save_confusion_matrix(

            dataset_name,

            model_name,

            y_test,

            predictions

        )


        # ====================================================
        # SAVE FEATURE IMPORTANCE
        # ====================================================

        save_feature_importance(

            dataset_name,

            model_name,

            model,

            feature_names

        )


        # ====================================================
        # STORE RESULTS
        # ====================================================

        results.append({

            "Dataset":
                dataset_name,

            "Model":
                model_name,

            "CV_Macro_F1":
                cv_mean,

            "CV_Macro_F1_STD":
                cv_std,

            "Test_Accuracy":
                accuracy,

            "Test_Precision":
                precision,

            "Test_Recall":
                recall,

            "Test_Weighted_F1":
                weighted_f1,

            "Test_Macro_F1":
                macro_f1,

            "Test_ROC_AUC":
                roc_auc

        })


        # ====================================================
        # BEST MODEL
        # ====================================================

        if macro_f1 > best_macro_f1:

            best_macro_f1 = (
                macro_f1
            )

            best_model = model

            best_model_name = (
                model_name
            )


    # ========================================================
    # RESULTS DATAFRAME
    # ========================================================

    results_df = pd.DataFrame(
        results
    )


    results_df = (

        results_df

        .sort_values(

            "Test_Macro_F1",

            ascending=False

        )

        .reset_index(
            drop=True
        )

    )


    # ========================================================
    # SAVE MODEL COMPARISON
    # ========================================================

    comparison_path = (

        "model_results/"
        f"{dataset_name}_"
        "model_comparison.csv"

    )

    results_df.to_csv(

        comparison_path,

        index=False

    )


    # ========================================================
    # SAVE BEST MODEL PACKAGE
    # ========================================================

    model_path = (

        "trained_models/"
        f"{dataset_name}_"
        "best_model.pkl"

    )


    model_package = {

        "model":
            best_model,

        "target_encoder":
            target_encoder,

        "feature_names":
            feature_names,

        "dataset":
            dataset_name,

        "target_column":
            target_column

    }


    joblib.dump(

        model_package,

        model_path

    )


    # ========================================================
    # BEST MODEL SUMMARY
    # ========================================================

    best_row = (
        results_df.iloc[0]
    )


    print("\n")

    print("=" * 75)

    print(
        f"BEST MODEL: "
        f"{best_model_name}"
    )

    print(
        f"BEST ACCURACY: "
        f"{best_row['Test_Accuracy']:.4f}"
    )

    print(
        f"BEST MACRO F1: "
        f"{best_row['Test_Macro_F1']:.4f}"
    )

    print(
        f"BEST WEIGHTED F1: "
        f"{best_row['Test_Weighted_F1']:.4f}"
    )

    print(
        f"BEST ROC-AUC: "
        f"{best_row['Test_ROC_AUC']:.4f}"
    )

    print(
        f"CV MACRO F1: "
        f"{best_row['CV_Macro_F1']:.4f}"
    )

    print(
        f"Saved model: "
        f"{model_path}"
    )

    print(
        f"Saved comparison: "
        f"{comparison_path}"
    )

    print("=" * 75)


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")

    print("=" * 75)

    print(
        "SMARTDOSEAI "
        "ADVANCED MODEL TRAINING"
    )

    print("=" * 75)


    for dataset_name, config in DATASETS.items():

        try:

            train_dataset(

                dataset_name,

                config["train"],

                config["test"]

            )

        except Exception as error:

            print("\n")

            print("=" * 75)

            print(
                f"ERROR TRAINING "
                f"{dataset_name.upper()}"
            )

            print("=" * 75)

            print(
                f"{type(error).__name__}: "
                f"{error}"
            )

            print(
                "\nContinuing with "
                "remaining datasets..."
            )


    # ========================================================
    # FINAL MESSAGE
    # ========================================================

    print("\n")

    print("=" * 75)

    print(
        "ALL DATASETS TRAINING COMPLETED"
    )

    print("=" * 75)

    print(
        "\nGenerated folders:"
    )

    print(
        "  trained_models/"
    )

    print(
        "  model_results/"
    )

    print(
        "  model_results/confusion_matrices/"
    )

    print(
        "  model_results/feature_importance/"
    )

    print(
        "  model_results/classification_reports/"
    )


# ============================================================
# START PROGRAM
# ============================================================

if __name__ == "__main__":

    main()
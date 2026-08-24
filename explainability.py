# ============================================================
# SMARTDOSEAI - MULTI-DISEASE SHAP EXPLAINABILITY
# ============================================================

import os
import warnings
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


# ============================================================
# SETTINGS
# ============================================================

DATASETS = [
    "diabetes",
    "heart",
    "kidney",
    "hypertension",
    "dialysis"
]

SAMPLE_SIZE = 200


# ============================================================
# HELPER - MAKE FEATURE NAMES HUMAN READABLE
# ============================================================

def make_readable_feature_name(feature_name):

    name = str(feature_name)

    # Remove sklearn transformer prefixes
    if name.startswith("numerical__"):
        name = name.replace(
            "numerical__",
            "",
            1
        )

    elif name.startswith("categorical__"):
        name = name.replace(
            "categorical__",
            "",
            1
        )

    # Make one-hot encoded names easier to read
    name = name.replace("_", " ")

    return name


# ============================================================
# HELPER - GET NUMBER OF CLASSES
# ============================================================

def get_number_of_classes(
    model,
    y_test
):

    if hasattr(
        model,
        "classes_"
    ):

        return len(
            model.classes_
        )

    return len(
        np.unique(y_test)
    )


# ============================================================
# HELPER - NORMALIZE SHAP ARRAY
# ============================================================

def calculate_global_shap_importance(
    shap_values,
    number_of_classes,
    number_of_features
):

    """
    SHAP 0.52 for multi-class tree models commonly returns:

        samples x features x classes

    Older versions may return:

        list[class] containing samples x features

    This function handles both.
    """

    # --------------------------------------------------------
    # LIST FORMAT
    # --------------------------------------------------------

    if isinstance(
        shap_values,
        list
    ):

        shap_array = np.stack(
            shap_values,
            axis=0
        )

        # class x samples x features

        mean_abs_shap = np.mean(
            np.abs(shap_array),
            axis=(0, 1)
        )

        return (
            shap_array,
            mean_abs_shap
        )


    # --------------------------------------------------------
    # NUMPY FORMAT
    # --------------------------------------------------------

    if isinstance(
        shap_values,
        np.ndarray
    ):

        shap_array = shap_values

        # ----------------------------------------------------
        # MULTI-CLASS
        # ----------------------------------------------------

        if shap_array.ndim == 3:

            # Expected SHAP 0.52 format:
            # samples x features x classes

            if (
                shap_array.shape[0] > 1
                and
                shap_array.shape[1]
                == number_of_features
                and
                shap_array.shape[2]
                == number_of_classes
            ):

                mean_abs_shap = np.mean(
                    np.abs(shap_array),
                    axis=(0, 2)
                )

                return (
                    shap_array,
                    mean_abs_shap
                )


            # Alternative:
            # classes x samples x features

            if (
                shap_array.shape[0]
                == number_of_classes
                and
                shap_array.shape[2]
                == number_of_features
            ):

                mean_abs_shap = np.mean(
                    np.abs(shap_array),
                    axis=(0, 1)
                )

                return (
                    shap_array,
                    mean_abs_shap
                )


        # ----------------------------------------------------
        # BINARY / SINGLE OUTPUT
        # ----------------------------------------------------

        if shap_array.ndim == 2:

            mean_abs_shap = np.mean(
                np.abs(shap_array),
                axis=0
            )

            return (
                shap_array,
                mean_abs_shap
            )


    raise ValueError(
        "Unsupported SHAP value format: "
        f"{type(shap_values)}"
    )


# ============================================================
# HELPER - GET PATIENT SHAP VALUES
# ============================================================

def get_patient_shap_values(
    shap_values,
    predicted_class,
    number_of_classes,
    number_of_features
):

    # --------------------------------------------------------
    # LIST FORMAT
    # --------------------------------------------------------

    if isinstance(
        shap_values,
        list
    ):

        return np.asarray(
            shap_values[
                int(predicted_class)
            ][0]
        )


    # --------------------------------------------------------
    # NUMPY FORMAT
    # --------------------------------------------------------

    shap_array = np.asarray(
        shap_values
    )

    # --------------------------------------------------------
    # SHAP 0.52:
    # samples x features x classes
    # --------------------------------------------------------

    if (
        shap_array.ndim == 3
        and
        shap_array.shape[0] == 1
        and
        shap_array.shape[1]
        == number_of_features
        and
        shap_array.shape[2]
        == number_of_classes
    ):

        return shap_array[
            0,
            :,
            int(predicted_class)
        ]


    # --------------------------------------------------------
    # Alternative:
    # classes x samples x features
    # --------------------------------------------------------

    if (
        shap_array.ndim == 3
        and
        shap_array.shape[0]
        == number_of_classes
        and
        shap_array.shape[2]
        == number_of_features
    ):

        return shap_array[
            int(predicted_class),
            0,
            :
        ]


    # --------------------------------------------------------
    # Binary / single output
    # --------------------------------------------------------

    if shap_array.ndim == 2:

        return shap_array[0]


    raise ValueError(
        "Unable to determine patient SHAP values."
    )


# ============================================================
# HELPER - DECODE CLASS LABEL
# ============================================================

def decode_class_label(
    class_index,
    target_encoder
):

    if target_encoder is not None:

        try:

            return str(
                target_encoder.inverse_transform(
                    [int(class_index)]
                )[0]
            )

        except Exception:

            pass

    return str(
        class_index
    )


# ============================================================
# TRAINED MODEL ANALYSIS
# ============================================================

def analyze_dataset(
    dataset_name
):

    print("\n")
    print("=" * 75)
    print(
        f"SHAP ANALYSIS: "
        f"{dataset_name.upper()}"
    )
    print("=" * 75)


    # ========================================================
    # PATHS
    # ========================================================

    model_path = (
        f"trained_models/"
        f"{dataset_name}_best_model.pkl"
    )

    test_path = (
        f"processed_data/"
        f"{dataset_name}_test.csv"
    )

    output_dir = (
        f"model_results/"
        f"shap/"
        f"{dataset_name}"
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )


    # ========================================================
    # CHECK FILES
    # ========================================================

    if not os.path.exists(
        model_path
    ):

        print(
            f"ERROR: Model not found:"
            f" {model_path}"
        )

        return


    if not os.path.exists(
        test_path
    ):

        print(
            f"ERROR: Test data not found:"
            f" {test_path}"
        )

        return


    # ========================================================
    # LOAD MODEL
    # ========================================================

    print(
        "\nLoading model..."
    )

    model_package = joblib.load(
        model_path
    )

    model = model_package[
        "model"
    ]

    feature_names = model_package[
        "feature_names"
    ]

    target_encoder = model_package[
        "target_encoder"
    ]

    target_column = model_package[
        "target_column"
    ]


    print(
        "Model:",
        type(model).__name__
    )

    print(
        "Features:",
        len(feature_names)
    )

    print(
        "Target:",
        target_column
    )


    # ========================================================
    # LOAD TEST DATA
    # ========================================================

    print(
        "\nLoading test data..."
    )

    test_df = pd.read_csv(
        test_path
    )

    X_test = test_df.drop(
        columns=[
            target_column
        ]
    )

    y_test = test_df[
        target_column
    ]


    # Ensure exact feature order
    X_test = X_test[
        feature_names
    ]


    print(
        "Test samples:",
        len(X_test)
    )


    # ========================================================
    # SAMPLE DATA
    # ========================================================

    sample_size = min(
        SAMPLE_SIZE,
        len(X_test)
    )

    X_sample = X_test.iloc[
        :sample_size
    ].copy()


    print(
        "SHAP samples:",
        len(X_sample)
    )


    # ========================================================
    # CREATE EXPLAINER
    # ========================================================

    print(
        "\nCreating TreeExplainer..."
    )

    try:

        explainer = shap.TreeExplainer(
            model
        )

    except Exception as error:

        print(
            "\nERROR creating SHAP explainer:"
        )

        print(
            str(error)
        )

        return


    # ========================================================
    # CALCULATE SHAP
    # ========================================================

    print(
        "Calculating SHAP values..."
    )

    try:

        shap_values = (
            explainer.shap_values(
                X_sample
            )
        )

    except Exception as error:

        print(
            "\nERROR calculating SHAP:"
        )

        print(
            str(error)
        )

        return


    number_of_classes = (
        get_number_of_classes(
            model,
            y_test
        )
    )


    print(
        "Number of classes:",
        number_of_classes
    )


    # ========================================================
    # SHAP ARRAY + GLOBAL IMPORTANCE
    # ========================================================

    try:

        shap_array, mean_abs_shap = (
            calculate_global_shap_importance(
                shap_values,
                number_of_classes,
                len(feature_names)
            )
        )

    except Exception as error:

        print(
            "\nERROR processing SHAP values:"
        )

        print(
            str(error)
        )

        return


    print(
        "SHAP values generated successfully."
    )

    print(
        "SHAP array shape:",
        np.asarray(
            shap_array
        ).shape
    )


    # ========================================================
    # GLOBAL FEATURE IMPORTANCE
    # ========================================================

    importance_df = pd.DataFrame({

        "feature":
            feature_names,

        "readable_feature":
            [
                make_readable_feature_name(
                    feature
                )
                for feature in feature_names
            ],

        "mean_abs_shap":
            mean_abs_shap

    })


    importance_df = (
        importance_df
        .sort_values(
            "mean_abs_shap",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )


    # ========================================================
    # SAVE GLOBAL IMPORTANCE
    # ========================================================

    importance_path = (
        f"{output_dir}/"
        "global_feature_importance.csv"
    )

    importance_df.to_csv(
        importance_path,
        index=False
    )


    print(
        "\nSaved:",
        importance_path
    )


    # ========================================================
    # PRINT TOP FEATURES
    # ========================================================

    print(
        "\nTop 10 SHAP features:"
    )

    print(
        importance_df[
            [
                "readable_feature",
                "mean_abs_shap"
            ]
        ]
        .head(10)
        .to_string(
            index=False
        )
    )


    # ========================================================
    # GLOBAL BAR PLOT
    # ========================================================

    plt.figure(
        figsize=(10, 7)
    )

    top_features = (
        importance_df
        .head(10)
        .iloc[::-1]
    )

    plt.barh(
        top_features[
            "readable_feature"
        ],

        top_features[
            "mean_abs_shap"
        ]
    )

    plt.xlabel(
        "Mean |SHAP value|"
    )

    plt.ylabel(
        "Feature"
    )

    plt.title(
        f"{dataset_name.title()} "
        "Model - SHAP Feature Importance"
    )

    plt.tight_layout()


    bar_path = (
        f"{output_dir}/"
        "global_feature_importance.png"
    )

    plt.savefig(
        bar_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


    print(
        "Saved:",
        bar_path
    )


    # ========================================================
    # SHAP SUMMARY PLOT
    # ========================================================

    print(
        "\nGenerating SHAP summary plot..."
    )

    try:

        # SHAP 0.52 multi-class:
        # samples x features x classes

        if (
            np.asarray(
                shap_array
            ).ndim == 3
        ):

            if (
                shap_array.shape[0]
                == len(X_sample)
                and
                shap_array.shape[1]
                == len(feature_names)
            ):

                # Aggregate across classes
                summary_values = np.mean(
                    np.abs(
                        shap_array
                    ),
                    axis=2
                )

            else:

                # class x samples x features
                summary_values = np.mean(
                    np.abs(
                        shap_array
                    ),
                    axis=0
                )

        else:

            summary_values = (
                shap_array
            )


        plt.figure(
            figsize=(10, 7)
        )

        shap.summary_plot(
            summary_values,
            X_sample,
            feature_names=[
                make_readable_feature_name(
                    feature
                )
                for feature in feature_names
            ],
            show=False
        )

        plt.tight_layout()


        summary_path = (
            f"{output_dir}/"
            "shap_summary.png"
        )

        plt.savefig(
            summary_path,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()


        print(
            "Saved:",
            summary_path
        )

    except Exception as error:

        print(
            "\nWARNING:"
            " Summary plot failed."
        )

        print(
            str(error)
        )


    # ========================================================
    # INDIVIDUAL PATIENT
    # ========================================================

    print(
        "\nGenerating "
        "individual-patient explanation..."
    )


    patient = X_test.iloc[
        0:1
    ]


    # ========================================================
    # PREDICTION
    # ========================================================

    prediction = model.predict(
        patient
    )[0]


    probabilities = (
        model.predict_proba(
            patient
        )[0]
    )


    prediction_label = (
        decode_class_label(
            prediction,
            target_encoder
        )
    )


    print(
        "\nPrediction:",
        prediction_label
    )


    # ========================================================
    # PROBABILITIES
    # ========================================================

    probability_rows = []


    print(
        "\nClass probabilities:"
    )


    for index, probability in enumerate(
        probabilities
    ):

        label = (
            decode_class_label(
                index,
                target_encoder
            )
        )

        print(
            f"  {label}: "
            f"{probability:.4f}"
        )

        probability_rows.append({

            "class_index":
                index,

            "class_label":
                label,

            "probability":
                probability

        })


    probability_df = pd.DataFrame(
        probability_rows
    )


    probability_path = (
        f"{output_dir}/"
        "patient_1_probabilities.csv"
    )

    probability_df.to_csv(
        probability_path,
        index=False
    )


    # ========================================================
    # PATIENT SHAP
    # ========================================================

    patient_shap_raw = (
        explainer.shap_values(
            patient
        )
    )


    try:

        patient_shap = (
            get_patient_shap_values(
                patient_shap_raw,
                prediction,
                number_of_classes,
                len(feature_names)
            )
        )

    except Exception as error:

        print(
            "\nERROR extracting "
            "patient SHAP values:"
        )

        print(
            str(error)
        )

        return


    # ========================================================
    # PATIENT EXPLANATION TABLE
    # ========================================================

    patient_explanation = pd.DataFrame({

        "feature":
            feature_names,

        "readable_feature":
            [
                make_readable_feature_name(
                    feature
                )
                for feature in feature_names
            ],

        "model_input_value":
            patient.iloc[
                0
            ].values,

        "shap_value":
            patient_shap

    })


    patient_explanation[
        "absolute_shap"
    ] = (
        patient_explanation[
            "shap_value"
        ].abs()
    )


    patient_explanation = (
        patient_explanation
        .sort_values(
            "absolute_shap",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )


    # ========================================================
    # DIRECTION
    # ========================================================

    patient_explanation[
        "direction"
    ] = np.where(

        patient_explanation[
            "shap_value"
        ] > 0,

        "Supports prediction",

        "Opposes prediction"

    )


    # ========================================================
    # SAVE PATIENT EXPLANATION
    # ========================================================

    patient_path = (
        f"{output_dir}/"
        "patient_1_explanation.csv"
    )


    patient_explanation.to_csv(
        patient_path,
        index=False
    )


    print(
        "\nSaved:",
        patient_path
    )


    # ========================================================
    # PRINT TOP PATIENT FACTORS
    # ========================================================

    print(
        "\nTop factors influencing "
        "this patient's prediction:"
    )


    print(
        patient_explanation[
            [
                "readable_feature",
                "model_input_value",
                "shap_value",
                "direction"
            ]
        ]
        .head(10)
        .to_string(
            index=False
        )
    )


    # ========================================================
    # PATIENT SHAP BAR PLOT
    # ========================================================

    top_patient = (
        patient_explanation
        .head(10)
        .copy()
        .iloc[::-1]
    )


    plt.figure(
        figsize=(10, 7)
    )


    plt.barh(
        top_patient[
            "readable_feature"
        ],

        top_patient[
            "shap_value"
        ]
    )


    plt.axvline(
        0,
        linewidth=1
    )


    plt.xlabel(
        "SHAP value"
    )


    plt.ylabel(
        "Feature"
    )


    plt.title(
        f"{dataset_name.title()} - "
        "Individual Patient SHAP Explanation"
    )


    plt.tight_layout()


    patient_plot_path = (
        f"{output_dir}/"
        "patient_1_shap.png"
    )


    plt.savefig(
        patient_plot_path,
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()


    print(
        "Saved:",
        patient_plot_path
    )


    # ========================================================
    # COMPLETE
    # ========================================================

    print(
        "\n"
        + "-" * 75
    )

    print(
        f"{dataset_name.upper()} "
        "SHAP ANALYSIS COMPLETED"
    )

    print(
        "-" * 75
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 75)
    print(
        "SMARTDOSEAI - MULTI-DISEASE "
        "EXPLAINABLE AI"
    )
    print("=" * 75)

    print(
        "\nSHAP version:",
        shap.__version__
    )

    for dataset_name in DATASETS:

        try:

            analyze_dataset(
                dataset_name
            )

        except Exception as error:

            print("\n")
            print("=" * 75)

            print(
                f"ERROR ANALYZING "
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


    print("\n")
    print("=" * 75)

    print(
        "ALL SHAP ANALYSIS COMPLETED"
    )

    print("=" * 75)

    print(
        "\nGenerated folders:"
    )

    for dataset_name in DATASETS:

        print(
            f"  model_results/shap/"
            f"{dataset_name}/"
        )


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    main()

# ============================================================
# REAL-TIME EXPLAINABILITY FOR WEB APP
# ============================================================

_EXPLAINERS = {}


def _json_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def influence_strength(abs_shap, max_abs):
    if max_abs <= 0:
        return "Mild"
    ratio = abs_shap / max_abs
    if ratio >= 0.60:
        return "Strong"
    if ratio >= 0.25:
        return "Moderate"
    return "Mild"


def load_global_explanation(disease_type, top_n=8):
    from clinical_units import format_clinical_value, readable_feature_name, strip_transformer_prefix

    path = os.path.join(
        "model_results",
        "shap",
        (disease_type or "").lower(),
        "global_feature_importance.csv",
    )
    if not os.path.exists(path):
        return []
    try:
        frame = pd.read_csv(path)
    except Exception:
        return []
    rows = []
    seen = set()
    for _, item in frame.iterrows():
        raw_name = item.get("feature") or item.get("readable_feature")
        label = readable_feature_name(raw_name)
        key = strip_transformer_prefix(raw_name)
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "feature": key,
                "readable_feature": label,
                "mean_abs_shap": round(_json_float(item.get("mean_abs_shap")), 4),
            }
        )
        if len(rows) >= top_n:
            break
    if not rows:
        return rows
    max_abs = max(row["mean_abs_shap"] for row in rows) or 1
    for row in rows:
        row["strength"] = influence_strength(row["mean_abs_shap"], max_abs)
    return rows


def generate_patient_explanation_realtime(
    model_package,
    model_input_df,
    predicted_class,
    number_of_features,
    original_parameters=None,
    raw_feature_df=None,
    disease_type=None,
):
    """
    SHAP is computed on preprocessed model input.
    Display values come from the original clinical parameters.
    """
    from clinical_units import format_clinical_value, readable_feature_name, strip_transformer_prefix

    model = model_package["model"]
    feature_names = list(model_package["feature_names"])
    original_parameters = original_parameters or {}

    probabilities = model.predict_proba(model_input_df)[0]
    probabilities = [_json_float(x) for x in np.asarray(probabilities).ravel()]

    cache_key = id(model)
    explainer = _EXPLAINERS.get(cache_key)
    if explainer is None:
        explainer = shap.TreeExplainer(model)
        _EXPLAINERS[cache_key] = explainer

    shap_values_raw = explainer.shap_values(model_input_df)
    number_of_classes = get_number_of_classes(model, None)
    try:
        patient_shap = get_patient_shap_values(
            shap_values_raw,
            predicted_class,
            number_of_classes,
            number_of_features,
        )
    except Exception:
        patient_shap = np.zeros(len(feature_names))

    raw_lookup = {}
    if raw_feature_df is not None and len(raw_feature_df):
        raw_lookup = raw_feature_df.iloc[0].to_dict()

    grouped = {}
    for i, feature in enumerate(feature_names):
        shap_val = _json_float(patient_shap[i] if i < len(patient_shap) else 0)
        clinical_key = strip_transformer_prefix(feature)
        clinical_value = original_parameters.get(clinical_key)
        if clinical_key not in original_parameters or original_parameters[clinical_key] in (None, "", "N/A", "n/a"):
            clinical_value = None
        bucket = grouped.setdefault(
            clinical_key,
            {
                "feature": clinical_key,
                "readable_feature": readable_feature_name(feature),
                "shap_value": 0.0,
                "clinical_value": clinical_value,
            },
        )
        bucket["shap_value"] += shap_val

    rows = []
    for item in grouped.values():
        abs_shap = abs(item["shap_value"])
        rows.append(
            {
                "feature": item["feature"],
                "readable_feature": item["readable_feature"],
                "clinical_value": item["clinical_value"],
                "display_value": format_clinical_value(
                    item["feature"],
                    item["clinical_value"],
                ),
                "shap_value": round(item["shap_value"], 4),
                "absolute_shap": round(abs_shap, 4),
                "direction": (
                    "Supports prediction"
                    if item["shap_value"] > 0
                    else "Opposes prediction"
                ),
            }
        )

    rows.sort(key=lambda row: row["absolute_shap"], reverse=True)
    max_abs = rows[0]["absolute_shap"] if rows else 0
    for row in rows:
        row["strength"] = influence_strength(row["absolute_shap"], max_abs)

    contributing = [r for r in rows if r["clinical_value"] is not None]
    not_recorded = [r for r in rows if r["clinical_value"] is None]
    top_features = contributing[:5]

    return {
        "probabilities": probabilities,
        "top_features": top_features,
        "contributing_features": contributing,
        "not_recorded_features": not_recorded,
        "full_explanation": rows,
        "global_features": load_global_explanation(disease_type),
        "predicted_class": int(predicted_class) if predicted_class is not None else None,
    }
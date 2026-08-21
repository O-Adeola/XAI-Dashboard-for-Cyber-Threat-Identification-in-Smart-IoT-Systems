# XAI-Dashboard.py
# ==================================
# EXPLAINABLE AI DASHBOARD
# FOR CYBER THREAT IDENTIFICATION
# IN SMART IOT SYSTEMS
# ==================================

from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import shap
import sys

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
)

# ------------------------------
# PAGE CONFIG
# ------------------------------
st.set_page_config(
    page_title="Explainable AI Cyber Threat Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ------------------------------
# CUSTOM STYLING
# ------------------------------
st.markdown("""
<style>

/* Background */
.stApp{
    background:#F4F7FB;
    color:#0B1220;
}

/* Main title */
h1{
    color:#0B5394;
    font-weight:700;
}

/* Section headings */
h2,h3{
    color:#1C4587;
    padding-top:10px;
}

/* Sidebar background */
section[data-testid="stSidebar"]{
    background:#1A40A3;
    width:250px !important;
}

/* Sidebar headings */
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p{
    color:white;
}

/* Selectbox text */
section[data-testid="stSidebar"] div[data-baseweb="select"]{
    color:black;
}

/* Selected option */
section[data-testid="stSidebar"] div[data-baseweb="select"] span{
    color:black !important;
}

div[data-testid="stToolbar"]{
    background:#1A40A3;
}

header[data-testid="stHeader"] * {
    color: white !important;
}

/*button[data-testid="stExpandSidebarButton"]{
    background:white !important;
}*/

/*button[data-testid="stBaseButton-headerNoPadding"]{
    background:white !important;
}

button[data-testid="stBaseButton-header"]{
    color:white;
}

button[data-testid="stMainMenuButton"]{
    color:white;
}*/

/* Dropdown menu */
ul[role="listbox"]{
    background:white;
    color:black;
}

ul[role="listbox"] li{
    color:black;
}

ul[role="listbox"] li:hover{
    background:#E3F2FD;
}

/* Metric cards */
div[data-testid="stMetric"]{
    background:
    linear-gradient(
    135deg,
    #1A40A3,
    #4196E1
    );
    # border-left:6px solid #1E88E5;
    border-radius:10px;
    padding:15px;
    box-shadow:0px 3px 8px rgba(0,0,0,0.15);

}

div[data-testid="stMetricValue"]{
    color:white;
    font-size:30px;
    font-weight:600;
}

div[data-testid="stMetric"] label{
    color:white !important;
    font-weight:600;
}

/* Success box */
div[data-baseweb="notification"]{
    border-radius:10px;
}

/* Buttons */
.stButton>button{
    background:#1E88E5;
    color:white;
    border-radius:8px;
    border:none;
    font-weight:600;
}

.stButton>button:hover{
    background:#1565C0;
}

/* Download button */
.stDownloadButton>button{
    background:#43A047;
    color:white;
    border-radius:8px;
}

.stDownloadButton>button:hover{
    background:#2E7D32;
}

/* Tables */
thead tr th{
    background:#4196E1 !important;
    color:#4196E1 !important;
    font-weight: bold;
}

/* Tabs */
button[data-baseweb="tab"]{
    font-size:16px;
    font-weight:600;
}

/* Expanders */
.streamlit-expanderHeader{
    font-weight:bold;
}

.hero{

background:
linear-gradient(
135deg,
#1A40A3,
#4196E1
);

padding:16px;
border-radius:24px;
margin-bottom:8px;
}

.hero h1{
color:white;
font-size:32px;
}

.hero p{
color:#CBD5E1;
font-size:14px;
}

/* =========================================================
   SIDEBAR MENU / COLLAPSE BUTTON
   ========================================================= */

/* Sidebar closed hamburger button */
[data-testid="stSidebarCollapsedControl"] {
    display: flex !important;
    visibility: visible !important;
    opacity: 1 !important;
    z-index: 999999 !important;
}


/* Sidebar open collapse button */
[data-testid="stSidebarCollapseButton"] {
    display: flex !important;
    visibility: visible !important;
    opacity: 1 !important;
    z-index: 999999 !important;
}


/* Button itself */
[data-testid="stSidebarCollapsedControl"] button,
[data-testid="stSidebarCollapseButton"] button {

    background: transparent !important;
    border: none !important;
    box-shadow: none !important;

}

span[data-testid="stIconMaterial"] {
    color: white !important;
}


/* Icon */
[data-testid="stSidebarCollapsedControl"] svg,
[data-testid="stSidebarCollapseButton"] svg {

    color: white !important;
    fill: white !important;
    stroke: white !important;

}

</style>
""", unsafe_allow_html=True)

# ------------------------------
# PATHS
# ------------------------------
BASE_DIR = Path(__file__).resolve().parent

# Use project folder structure
DATASET_DIR = BASE_DIR / "datasets"

# ------------------------------
# DATASET CONFIGURATION
# ------------------------------

DATASETS = {
    "TON_IoT": {
        "models_dir": DATASET_DIR / "TON_IoT" / "models",
        "artifacts_dir": DATASET_DIR / "TON_IoT" / "artifacts",
        "description":
            "TON_IoT dataset containing network and IoT-related observations "
            "used to identify different categories of network activity and cyber attacks.",

        "eda_intro":
            "This dataset is examined to understand its size, attack categories, "
            "data quality, feature behaviour and suitability for machine-learning "
            "based cyber-threat classification.",

        "feature_context":
            "The modelling process focuses on network-behaviour characteristics "
            "rather than specific host identifiers or message-content fields."
    },
    "EdgeIIoT": {
        "models_dir": DATASET_DIR / "EdgeIIoT" / "models",
        "artifacts_dir": DATASET_DIR / "EdgeIIoT" / "artifacts",
    "description":
            "EdgeIIoT dataset containing IoT and Industrial IoT network observations "
            "representing different types of normal and malicious activity.",

        "eda_intro":
            "This dataset is examined to understand its structure, attack categories, "
            "data quality, feature behaviour and suitability for machine-learning "
            "based attack classification.",

        "feature_context":
            "The modelling process removes unsuitable connection, host and payload "
            "fields and converts relevant categorical information into a numerical form."
    },
}

# Filenames only; selected dataset decides the folder
MODEL_FILES = {
    "Random Forest": "random_forest.pkl",
    "Decision Tree": "decision_tree.pkl",
    "LightGBM": "lightgbm.pkl",
    "XGBoost": "xgboost.pkl",
    "Logistic Regression": "logistic_regression.pkl",
}

# ------------------------------
# CACHED LOADERS
# ------------------------------
@st.cache_resource
def load_joblib_file(path: str):
    return joblib.load(path)

@st.cache_data
def load_csv_file(path: str):
    return pd.read_csv(path)

@st.cache_data
def load_results_table(path: str):
    return pd.read_csv(path)

# Dataset-aware artifact loader
@st.cache_resource
def load_artifacts(dataset_key: str):
    missing = []

    dataset_cfg = DATASETS[dataset_key]
    models_dir = dataset_cfg["models_dir"]
    artifacts_dir = dataset_cfg["artifacts_dir"]

    models = {}
    for name, filename in MODEL_FILES.items():
        path = models_dir / filename
        if path.exists():
            models[name] = load_joblib_file(str(path))
        else:
            missing.append(str(path))

    feature_encoder_path = artifacts_dir / "feature_encoder.pkl"
    target_encoder_path = artifacts_dir / "target_encoder.pkl"
    feature_names_path = artifacts_dir / "feature_names.pkl"
    cat_cols_path = artifacts_dir / "categorical_columns.pkl"
    drop_cols_path = artifacts_dir / "drop_columns.pkl"
    results_path = artifacts_dir / "results.csv"
    confusion_matrix_path = artifacts_dir / "confusion_matrices.pkl"
    classification_reports_path = artifacts_dir / "classification_reports.pkl"
    # eval_sample_path = artifacts_dir / "eval_sample.csv"
    # eval sample may be csv OR pkl
    eval_sample_csv = artifacts_dir / "eval_sample.csv"
    eval_sample_pkl = artifacts_dir / "eval_sample.pkl"
    eval_labels_path = artifacts_dir / "eval_labels.pkl"
    eda_summary_path = artifacts_dir / "eda_summary.pkl"

    eda_dir = artifacts_dir / "eda"

    eda_files = {
    
        "dataset_info": eda_dir / "dataset_info.pkl",
        "missing_values": eda_dir / "missing_values.pkl",
        "feature_statistics": eda_dir / "feature_statistics.pkl",
        "selected_features": eda_dir / "selected_features.pkl",
        "training_scores": eda_dir / "training_validation_scores.pkl",
        "numeric_features": eda_dir / "numeric_features.pkl",
    }
     
    for name, path in eda_files.items():
        if path.exists():
            eda_files[name] = load_joblib_file(
                str(path)
            )
        else:
            eda_files[name]=None

    required = [
        target_encoder_path,
        feature_names_path,
        cat_cols_path,
        drop_cols_path,
        results_path,
        eval_labels_path,
        confusion_matrix_path
    ]

    for p in required:
        if not p.exists():
            missing.append(str(p))
    
    if not eval_sample_csv.exists() and not eval_sample_pkl.exists():
        missing.append(
            str(artifacts_dir / "eval_sample.(csv or pkl)")
        )

    if missing:
        return None, missing

    feature_encoder = None
    if feature_encoder_path.exists():
        feature_encoder = load_joblib_file(
            str(feature_encoder_path)
        )
    
    target_encoder = load_joblib_file(
        str(target_encoder_path)
    )
    
    feature_names = load_joblib_file(
        str(feature_names_path)
    )
    
    cat_cols = load_joblib_file(
        str(cat_cols_path)
    )
    
    drop_cols = load_joblib_file(
        str(drop_cols_path)
    )
    
    results_df = load_results_table(
        str(results_path)
    )

    confusion_matrices = load_joblib_file(
        str(confusion_matrix_path)
    )

    classification_reports = load_joblib_file(
        str(classification_reports_path)
    )
    
    # eval sample may be csv or pkl
    if eval_sample_csv.exists():
        eval_sample = load_csv_file(
            str(eval_sample_csv)
        )
    else:
        eval_sample = load_joblib_file(
            str(eval_sample_pkl)
        )
    
    eval_labels = load_joblib_file(
        str(eval_labels_path)
    )

    eda_dir = artifacts_dir / "eda"

    eda_files = {}
    
    if eda_dir.exists():
    
        for file in eda_dir.glob("*.pkl"):
    
            eda_files[file.stem] = load_joblib_file(str(file))

    return {
        "models": models,
        "feature_encoder": feature_encoder,
        "target_encoder": target_encoder,
        "feature_names": feature_names,
        "cat_cols": cat_cols,
        "drop_cols": drop_cols,
        "results_df": results_df,
        "eval_sample": eval_sample,
        "eval_labels": eval_labels,
        "eda": eda_files,
        "confusion_matrices": confusion_matrices, 
        "classification_reports": classification_reports,
    }, []

# ------------------------------
# HELPERS
# ------------------------------
def preprocess_input(raw_df: pd.DataFrame, feature_names, cat_cols, feature_encoder, drop_cols):
    df = raw_df.copy()

    # Drop target columns if present
    for col in ["type", "label", "Attack_type", "attack_type"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Drop known unused columns
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    # Replace inf and missing values
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)

    # Encode categorical columns if they are present
    existing_cat_cols = [c for c in cat_cols if c in df.columns]
    if existing_cat_cols and feature_encoder is not None:
        try:
            df[existing_cat_cols] = feature_encoder.transform(df[existing_cat_cols].astype(str))
        except Exception:
            pass

    # Add any missing feature columns
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0

    # Drop extra columns and order exactly like training
    df = df[[c for c in feature_names if c in df.columns]]
    return df

def get_model_predictions(model, X):
    preds = model.predict(X)
    probs = model.predict_proba(X)
    return preds, probs

def get_model_importance(model_name: str, model, feature_names):
    if model_name == "Logistic Regression":
        coef = model.named_steps["lr"].coef_
        if coef.ndim == 2:
            # multiclass: use mean absolute coefficient across classes
            importance = np.mean(np.abs(coef), axis=0)
        else:
            importance = np.abs(coef)

        imp_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importance,
        }).sort_values(by="Importance", ascending=False)
    else:
        imp_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": model.feature_importances_,
        }).sort_values(by="Importance", ascending=False)

    return imp_df

def show_feature_importance_chart(imp_df, title):
    top = imp_df.head(20).sort_values(by="Importance", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(top["Feature"], top["Importance"])
    ax.set_title(title)
    ax.set_xlabel("Importance")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

def get_tree_shap(model, X_background, X_row):
    explainer = shap.TreeExplainer(model)
    shap_bg = explainer.shap_values(X_background)
    shap_row = explainer.shap_values(X_row)
    return explainer, shap_bg, shap_row

def plot_tree_shap_global(shap_values, background, title="SHAP Global Explanation"):
    st.subheader(title)

    try:
        fig = plt.figure(figsize=(13, 6))
        shap.summary_plot(shap_values, background, show=False)
        plt.tight_layout(pad=2)
        st.pyplot(fig)
        plt.close(fig)
    except Exception:
        if isinstance(shap_values, list):
            stacked = np.stack(shap_values, axis=0)
            mean_abs = np.mean(np.abs(stacked), axis=0)
            mean_abs = mean_abs.mean(axis=0)
        else:
            mean_abs = np.mean(np.abs(shap_values), axis=0)

        imp_df = pd.DataFrame({
            "Feature": background.columns,
            "Importance": mean_abs
        }).sort_values(by="Importance", ascending=False)

        show_feature_importance_chart(imp_df, "Mean |SHAP| Importance")

def plot_tree_shap_local(model, explainer, shap_row, X_row, target_encoder):
    pred = model.predict(X_row)[0]
    class_idx = int(np.asarray(pred).item()) if np.ndim(pred) == 0 else int(pred)

    if isinstance(shap_row, list):
        values = np.asarray(shap_row[class_idx][0]).reshape(-1)
        base_value = explainer.expected_value[class_idx]
    else:
        arr = np.asarray(shap_row)

        if arr.ndim == 3:
            values = arr[0, :, class_idx].reshape(-1)
            base_value = explainer.expected_value[class_idx]
        elif arr.ndim == 2:
            values = arr[:, class_idx].reshape(-1)
            base_value = explainer.expected_value[class_idx]
        else:
            values = arr.reshape(-1)
            base_value = explainer.expected_value

    base_value = float(np.asarray(base_value).reshape(-1)[0])

    explanation = shap.Explanation(
        values=values,
        base_values=base_value,
        data=X_row.iloc[0].values,
        feature_names=X_row.columns
    )

    fig = plt.figure(figsize=(13, 6))
    shap.plots.waterfall(explanation, max_display=15)
    plt.title(f"{selected_model_name} - SHAP Local Explanation Waterfall Plot")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    pred_label = target_encoder.inverse_transform([pred])[0]
    return pred_label

def plot_lr_reasoning(model, X_row, feature_names, target_encoder):
    lr = model.named_steps["lr"]
    pred = model.predict(X_row)[0]
    pred_label = target_encoder.inverse_transform([pred])[0]

    coef = lr.coef_
    if coef.ndim == 2:
        class_idx = int(pred)
        row_coef = coef[class_idx]
    else:
        row_coef = coef[0]

    signed = pd.DataFrame({
        "Feature": feature_names,
        "Coefficient": row_coef,
        "AbsCoefficient": np.abs(row_coef)
    }).sort_values(by="AbsCoefficient", ascending=False)

    st.dataframe(signed.head(20), use_container_width=True)

    top = signed.head(20).sort_values(by="AbsCoefficient", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(top["Feature"], top["AbsCoefficient"])
    ax.set_title("Logistic Regression Coefficient Importance")
    ax.set_xlabel("Absolute coefficient")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    return pred_label


# def plot_lr_waterfall(model, X_row, feature_names, target_encoder):
#     lr = model.named_steps["lr"]
#     # Prediction
#     pred = model.predict(X_row)[0]
#     pred_label = target_encoder.inverse_transform([pred])[0]

#     # -----------------------------
#     # Contribution table
#     # -----------------------------
#     coef = lr.coef_
#     class_idx = int(pred)
#     row_coef = coef[class_idx]

#     contribution_df = pd.DataFrame({
#         "Feature": feature_names,
#         "Coefficient": row_coef,
#         "Feature Value": x_row.iloc[0].values,
#         "Contribution": row_coef * x_row.iloc[0].values
#     })

#     contribution_df["Absolute Impact"] = (
#         contribution_df["Contribution"].abs()
#     )

#     contribution_df = (
#         contribution_df
#         .sort_values(
#             "Absolute Impact",
#             ascending=False
#         )
#     )

#     st.subheader("Feature contribution table")

#     st.dataframe(contribution_df.head(20), use_container_width=True)

#     # SHAP Linear Explainer
#     explainer = shap.LinearExplainer(lr, X_row, feature_perturbation="interventional")

#     shap_values = explainer.shap_values(X_row)

#     # Handle multiclass output
#     if len(shap_values.shape) == 3:
#         values = shap_values.values[0, :, class_idx]
#         base_value = shap_values.base_values[0, class_idx]
#     else:
#         values = shap_values.values[0]
#         base_value = shap_values.base_values[0]


#     explanation = shap.Explanation(
#         values=values,
#         base_values=base_value,
#         data=X_row.iloc[0].values,
#         feature_names=feature_names
#     )


#     fig = plt.figure(figsize=(12,7))

#     shap.plots.waterfall(
#         explanation,
#         max_display=15,
#         show=False
#     )

#     plt.tight_layout()
#     st.pyplot(fig)
#     plt.close(fig)

#     return pred_label



# def plot_lr_waterfall(model, X_row, feature_names, target_encoder):

#     lr = model.named_steps["lr"]

#     pred = model.predict(X_row)[0]

#     pred_label = target_encoder.inverse_transform(
#         [pred]
#     )[0]


#     background = X_row.copy()


#     explainer = shap.LinearExplainer(
#         lr,
#         background
#     )


#     shap_values = explainer.shap_values(
#         X_row
#     )


#     lr = model.named_steps["lr"]
    
#     background_transformed = model[:-1].transform(background)
    
#     explainer = shap.LinearExplainer(
#         lr,
#         background_transformed
#     )


#     explainer = shap.LinearExplainer(
#         model,
#         background_data
#     )
    
#     shap_values = explainer(x_row)


#     # multiclass handling
#     if isinstance(shap_values, list):
#         values = shap_values[pred][0]
#         base_value = explainer.expected_value[pred]

#     else:
#         shap_values = np.array(shap_values)


#         if shap_values.ndim == 3:
#             values = shap_values[0, :, pred]
#             base_value = explainer.expected_value[pred]

#         else:
#             values = shap_values[0]
#             base_value = explainer.expected_value


#     feature_names = np.asarray(feature_names).ravel()
#     values = np.asarray(values)
    
#     if values.ndim == 2:
#         values = values[0]
    
#     if values.ndim > 1:
#         values = values.ravel()



#     # Keep your table

#     reason_df = pd.DataFrame({
#         "Feature": feature_names,
#         "SHAP Value": values,
#         "Impact": np.abs(values)
#     })


#     reason_df = (
#         reason_df
#         .sort_values(
#             "Impact",
#             ascending=False
#         )
#     )


#     st.subheader(
#         "Feature contribution table"
#     )

#     st.dataframe(
#         reason_df.head(20),
#         use_container_width=True
#     )


#     # waterfall

#     explanation = shap.Explanation(
#         values=values,
#         base_values=float(
#             np.array(base_value).reshape(-1)[0]
#         ),
#         data=X_row.iloc[0].values,
#         feature_names=X_row.columns
#     )


#     st.subheader(
#         "SHAP Local Explanation"
#     )


#     fig = plt.figure(
#         figsize=(12,7)
#     )

#     shap.plots.waterfall(
#         explanation,
#         max_display=15,
#         show=False
#     )

#     plt.tight_layout()
#     st.pyplot(fig)
#     plt.close(fig)

#     return pred_label



# def plot_lr_waterfall(model, X_row, feature_names, target_encoder):
    
#     if len(X_row) > 1:
#         X_row = X_row.iloc[[0]]

#     # Get fitted scaler and LR model
#     scaler = model.named_steps["scaler"]
#     lr = model.named_steps["lr"]

#     # Scale the selected sample
#     X_scaled = scaler.transform(X_row)

#     # Prediction
#     pred = model.predict(X_row)[0]
#     pred_label = target_encoder.inverse_transform([pred])[0]

#     # Use the scaled sample as background
#     background = X_scaled

#     # st.write("X_row shape:", X_row.shape)
#     # st.write("X_scaled shape:", X_scaled.shape)

#     # Build SHAP explainer
#     explainer = shap.LinearExplainer(
#         lr,
#         background
#     )

#     # Explain only the selected instance
#     shap_values = explainer(X_scaled[0:1])

#     # st.write("SHAP values shape:", shap_values.values.shape)

#     # Predicted class
#     pred_class = int(pred)
    
#     # SHAP output shape:
#     # (samples, features, classes)
#     values = np.asarray(shap_values.values)[0, :, pred_class]
    
#     # Base value for predicted class
#     # base_value = np.asarray(shap_values.base_values)[0, pred_class]
    
    
#     # # Extract values
#     # values = np.asarray(shap_values.values)
    
#     # # Handle batch dimension
#     # if values.ndim == 2:
#     #     values = values[0]
    
#     # values = values.reshape(-1)

#     # Base value
#     base_value = float(np.asarray(shap_values.base_values).reshape(-1)[0])

#     # values = np.array(values).flatten()
    
#     if len(feature_names) != len(values):
#         st.warning(
#             f"Feature mismatch: {len(feature_names)} features but {len(values)} SHAP values"
#         )
    
#         if len(feature_names) != len(values):
#             raise ValueError(
#                 f"SHAP mismatch: {len(feature_names)} feature names but {len(values)} SHAP values"
#             )
    
#     reason_df = pd.DataFrame({
#         "Feature": feature_names,
#         "SHAP Value": values,
#         "Impact": np.abs(values)
#     })


#     reason_df = reason_df.sort_values(
#         "Impact",
#         ascending=False
#     )

#     st.subheader("Feature contribution table")
#     st.dataframe(
#         reason_df.head(20),
#         use_container_width=True,
#         hide_index=True
#     )

#     # Create SHAP Explanation
#     explanation = shap.Explanation(
#         values=values,
#         base_values=base_value,
#         data=X_scaled[0],
#         feature_names=feature_names
#     )

#     st.subheader("SHAP Local Explanation")

#     fig = plt.figure(figsize=(12,7))

#     shap.plots.waterfall(
#         explanation,
#         max_display=15,
#         show=False
#     )

#     plt.tight_layout()

#     st.pyplot(fig)

#     plt.close(fig)

#     return pred_label

    


def display_confusion_matrix(model_name, y_true, y_pred, class_names):
    fig, ax = plt.subplots(figsize=(12, 10))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=class_names,
        xticks_rotation=90,
        ax=ax,
        cmap="Blues"
    )
    ax.set_title(f"{model_name} - Confusion Matrix")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

def styled_table(df):
    return (
        df.style
        .set_table_styles([
            {
                "selector": "th",
                "props": [
                    ("background-color", "#4196E1"),
                    ("color", "#4196E1"),
                    ("font-weight", "bold"),
                    ("text-align", "center")
                ]
            }
        ])
    )


def pct(value):
    return f"{float(value) * 100:.2f}%"


def interpret_generalization(
    train_accuracy,
    validation_accuracy
):

    gap = (
        train_accuracy -
        validation_accuracy
    )

    # These are indicative dashboard flags,
    # not definitive statistical diagnoses.

    if gap >= 0.05:

        return (
            "Potential overfitting",
            "Training performance is noticeably higher than "
            "validation performance. This may indicate that "
            "the model has learned the training data too closely."
        )

    elif (
        train_accuracy < 0.85
        and validation_accuracy < 0.85
    ):

        return (
            "Possible underfitting",
            "Performance is relatively low on both training and "
            "validation data. The model may not be capturing "
            "enough of the underlying patterns in the dataset."
        )

    else:

        return (
            "No strong indication of overfitting",
            "Training and validation performance are reasonably "
            "close. This suggests relatively consistent behaviour "
            "on unseen data, although other evaluation metrics "
            "should also be considered."
        )


# ------------------------------
# SIDEBAR
# ------------------------------
st.sidebar.title("🛡️ Console")

# Dataset selection happens first
dataset = st.sidebar.selectbox(
    "Dataset",
    list(DATASETS.keys()),
    key="dataset_nav"
)

if st.session_state.get("active_dataset") != dataset:
    st.session_state["active_dataset"] = dataset
    st.session_state.pop("current_input_df", None)
    st.session_state.pop("current_row_index", None)

# show which dataset is active
st.sidebar.caption(f"Active dataset: {dataset}")
st.sidebar.caption(DATASETS[dataset]["description"])

# load only the selected dataset's artifacts
artifacts, missing = load_artifacts(dataset)

if artifacts is None:
    st.error(f"Some required model or artifact files are missing for {dataset}.")
    st.write("Missing files:")
    for item in missing:
        st.write(f"- {item}")
    st.stop()

models = artifacts["models"]
feature_encoder = artifacts["feature_encoder"]
target_encoder = artifacts["target_encoder"]
feature_names = artifacts["feature_names"]
cat_cols = artifacts["cat_cols"]
drop_cols = artifacts["drop_cols"]
results_df = artifacts["results_df"].copy()
confusion_matrices = artifacts["confusion_matrices"]
classification_reports = artifacts["classification_reports"]
eval_sample = artifacts["eval_sample"].copy()
eval_labels = np.array(artifacts["eval_labels"])
# eda = artifacts.get("eda", {})
eda = artifacts["eda"]

results_df = results_df.sort_values(by="Weighted_F1", ascending=False).reset_index(drop=True)

page = st.sidebar.selectbox(
    "Go to",
    [
        "About",
        "Overview",
        "Dataset Analysis",
        "Model Performance",
        "Prediction",
        "Explainability"
    ],
    key="page_nav"
)

selected_model_name = st.sidebar.selectbox(
    "Select model",
    list(models.keys()),
    key="selected_model"
)

selected_model = models[selected_model_name]

# ------------------------------
# HEADER
# ------------------------------
st.markdown("""
<div class="hero">

<h1>
🛡️ XAI Dashboard for Threat Detection in Smart IoT Systems
</h1>

<p>
Explore the dataset, compare machine-learning models, inspect predictions, and explain why a threat was flagged.
</p>

</div>

""", unsafe_allow_html=True)

# ------------------------------
# PAGE: OVERVIEW
# ------------------------------
if page == "Overview":

    overview_page_tabs = st.tabs(["Overview", "Higher-is-better Metrics", "Lower-is-better Metrics"])

    with overview_page_tabs[0]:
        st.subheader("Overview") 
        st.write("""
            This page presents a comparison of every trained machine learning model.
            Performance metrics allow users to identify the strongest performing model while
            comparing prediction accuracy, robustness and reliability across different algorithms.
            """)
        
        top_row = results_df.iloc[0]
    
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Best Model", top_row["Model"])
        c2.metric("Weighted F1", f'{top_row["Weighted_F1"]:.2%}')
        c3.metric("Accuracy", f'{top_row["Accuracy"]:.2%}')
        c4.metric("ROC AUC", f'{top_row["ROC_AUC"]:.2%}')
    
        # st.subheader("Dataset and model summary")
        # summary_cols = st.columns(3)
        # summary_cols[0].metric("Features", len(feature_names))
        # summary_cols[1].metric("Attack classes", len(target_encoder.classes_))
        # summary_cols[2].metric("Evaluation rows", len(eval_sample))


        st.subheader("Top model ranking")
        st.write("""
                Models are ranked according to their weighted F1-score. 
                The weighted F1-score considers both precision and recall while accounting for class
                imbalance, making it one of the most reliable metrics for multi-class intrusion detection.
                """)
        st.dataframe(
            styled_table(results_df),
            hide_index=True,
            use_container_width=True
        )

    with overview_page_tabs[1]:
        st.subheader("Higher-is-better metrics")
        st.write("""
                Higher metric values indicate better predictive performance.
                
                Accuracy measures overall correctness, Balanced Accuracy compensates for class imbalance,
                Precision measures false alarm reduction, Recall measures attack detection capability,
                F1-score balances Precision and Recall, Cohen's Kappa measures agreement beyond chance,
                and ROC-AUC measures overall discrimination ability.
                """)
        fig1, ax1 = plt.subplots(figsize=(13, 6))
        results_df.set_index("Model")[[
            "Accuracy",
            "Balanced_Accuracy",
            "Precision_Weighted",
            "Precision_Macro",
            "Recall_Weighted",
            "Recall_Macro",
            "Weighted_F1",
            "Macro_F1",
            "Cohen_Kappa",
            "ROC_AUC"
        ]].plot(kind="bar", ax=ax1)
        ax1.set_ylabel("Score")
        ax1.set_title("Model Performance Comparison")
        plt.xticks(rotation=45)
        plt.tight_layout(pad=2)
        st.pyplot(fig1)
        plt.close(fig1)

    with overview_page_tabs[2]:
        st.subheader("Lower-is-better metrics")
        st.write("""
                Lower values are desirable for these metrics.
                
                Hamming Loss measures the proportion of incorrect predictions, while Log Loss evaluates
                how confident the model's probability estimates are. Lower values indicate more reliable
                and better calibrated predictions.
                """)
        fig2, ax2 = plt.subplots(figsize=(13, 6))
        results_df.set_index("Model")[["Hamming_Loss", "Log_Loss"]].plot(kind="bar", ax=ax2)
        ax2.set_ylabel("Score")
        ax2.set_title("Loss-Based Comparison")
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

# ------------------------------
# PAGE: DATASET ANALYSIS
# ------------------------------
# elif page == "Dataset Analysis":

#     dataset_analysis_tabs = st.tabs([
#         "Exploratory Data Analysis (EDA)", "Performance Summary", "Missing Value Analysis"])

#     with dataset_analysis_tabs[0]:
#         st.subheader(
#             "Exploratory Data Analysis (EDA)"
#             )

#         st.write("""
#             This page presents exploratory data analysis (EDA) to help understand the characteristics of the dataset. 
#             It includes the number of records, features and attack classes. It also reports missing values, 
#             duplicate records and possible overfitting/underfitting, allowing users to quickly assess 
#             the quality of the data. 

#             The raw datasets were already processed and filtered for the use of evaluating traditional machine learning-based intrusion detection systems. 
#             Because of that, both datasets did not contain any missing values. 
#             However, this page helps users understand what is contained in the selected dataset
#             before looking at the machine-learning results.
            
#             It answers simple questions such as:
            
#             • What is in the dataset?
            
#             • What is the quality of the data? Does it contain missing or duplicate records?
            
#             • Which features are important?
            
#             • Does the data show any potential problems?
#             """)

#         eda = artifacts["eda"]
#         if eda["dataset_info"]:
#             info = eda["dataset_info"]

#             with dataset_analysis_tabs[1]:
#                 with st.container():
#                     st.subheader("Dataset Information")

#                     st.write("""
#                     This page answers "What is in this dataset?". These numbers give a quick overview of the size and structure of the dataset after cleaning.
                    
#                     The rows represents each observation in the dataset.
#                     Features are the individual characteristics of the network activity that the
#                     machine-learning models use to identify different types of behaviour. They are listed under the "Missing Values" tab.
#                     Duplicate rows are identical observations. They are checked because repeated observations can affect how the model learns from the data.
#                     Attack classes are the categories of cyber attacks the model is being trained to recognise.
                
#                     These values help the user understand the size and complexity of the problem
#                     before machine learning is applied.
#                     """)
                    
#                     c1,c2,c3,c4,c5 = st.columns(5)
            
#                     c1.metric(
#                         "Dataset Rows",
#                         info["rows"]
#                     )
            
#                     c2.metric(
#                         "Features",
#                         info["columns"]
#                     )
            
#                     c3.metric(
#                         "Duplicate Rows",
#                         info["duplicates"]
#                     )

#                     c4.metric(
#                         "Attack classes", 
#                         len(target_encoder.classes_)
#                     )
                    
#                     c5.metric(
#                         "Evaluation rows", 
#                         len(eval_sample)
#                     )
                        
#                     st.write(
#                     "Attack Classes:"
#                     )
            
#                     st.write(
#                         info["classes"]
#                     )
            
#         with dataset_analysis_tabs[2]:
#             st.subheader("Missing Value Analysis")
    
#             st.write("""
#                     This page answers "What is the quality of the data?". Before using the data for 
#                     machine learning, it is important to check for common data-quality issues.
                    
#                     Missing values represent incomplete information within the dataset.
#                     Datasets containing many missing values can reduce prediction performance and may
#                     require cleaning or imputation before training machine learning models. In this case, 
#                     the downloaded raw dataset were already cleaned and prepared for machine learning projects. 
#                     """)
        
#             missing = eda["missing_values"]
        
#             missing_df = (
#                 missing
#                 .reset_index()
#             )
        
#             missing_df.columns=[
#                 "Feature",
#                 "Missing Values"
#             ]

#             st.dataframe(
#                 missing_df,
#                 hide_index=True,
#                 use_container_width=True
#             )

#             st.subheader("Overfitting/Underfitting")

#             st.write("""
#                 Training performance shows how well the model performs on data it learned from.
#                 Testing performance shows how well it performs on data it did not see during training.
                
#                 A large difference between training and testing performance may indicate that
#                 the model has learned the training data too closely. A small difference suggests
#                 more consistent performance on unseen data. In this case, there is very little overfitting gap suggesting consistency. 
#                 """)




elif page == "Dataset Analysis":

    dataset_cfg = DATASETS[dataset]

    st.header("📊 Explore the Dataset")

    st.caption(
        dataset_cfg["description"]
    )

    dataset_analysis_tabs = st.tabs([
        "Exploratory Data Analysis (EDA)", "What is in the dataset?", "Is the data suitable for analysis", "What types of activity are in the dataset?", "Do some features behave similarly", "Does the model behave consistently on unseen data?", "What did we learn from this dataset?"])

    with dataset_analysis_tabs[0]:
        st.subheader(
            "Exploratory Data Analysis (EDA)"
            )

        # st.write(
        #     dataset_cfg["eda_intro"]
        #     )

        st.write("""
            This page presents exploratory data analysis (EDA) to help understand the characteristics of the dataset. 
            It includes the number of records, features and attack classes. It also reports missing values, 
            duplicate records and possible overfitting/underfitting, allowing users to quickly assess 
            the quality of the data. It answers simple questions such as:
            
            • What is in the dataset?
            
            • What is the quality of the data? 
            
            • Which features are important?
            
            • Does the data show any potential problems?
            """)

        eda = artifacts["eda"]
        if eda["dataset_info"]:
            info = eda["dataset_info"]


        with dataset_analysis_tabs[1]:
            with st.container():
                st.subheader("What is in the dataset?")
                st.caption("These figures give a basic overview of the size and structure of the selected dataset") 

                dataset_info = eda.get(
                    "dataset_info",
                    {}
                )
            
                quality = eda.get(
                    "dataset_quality_summary",
                    {}
                )
            
            
                rows = quality.get(
                    "clean_rows",
                    dataset_info.get("rows", 0)
                )
            
                columns = quality.get(
                    "clean_columns",
                    dataset_info.get("columns", 0)
                )
            
                classes = len(
                    dataset_info.get(
                        "classes",
                        []
                    )
                )
            
            
                numeric_features = eda.get(
                    "numeric_features"
                )
            
                numeric_count = (
                    len(numeric_features.columns)
                    if isinstance(
                        numeric_features,
                        pd.DataFrame
                    )
                    else 0
                )
            
            
                c1, c2, c3, c4 = st.columns(4)
            
                c1.metric(
                    "Records",
                    f"{rows:,}"
                )
            
                c2.metric(
                    "Features",
                    f"{columns:,}"
                )
            
                c3.metric(
                    "Attack Classes",
                    f"{classes:,}"
                )
            
                c4.metric(
                    "Evaluation rows", 
                    len(eval_sample)
                )

                st.write(
                    "Attack Classes:"
                )
            
                st.write(
                    info["classes"]
                )

                st.subheader("ℹ️ Why does this matter?")
        
                st.write(
                    """
                    A record represents one observation in the dataset, while features are the individual characteristics 
                    of the network activity that the machine-learning models use to identify different types of behaviour. 
                    Attack classes are the categories of cyber attacks the model is being trained to recognise. Evaluation rows 
                    are samples of the dataset selected to test and explain the machine-learning models

                    These values help the user understand the size and complexity of the problem before looking at the machine-learning results.
                    """
                )

            

        with dataset_analysis_tabs[2]:
            with st.container():
                st.subheader("Is the data suitable for analysis?")
            
                st.caption("Before machine learning is applied, the dataset is checked for common data-quality issues.")
            
            
                original_rows = quality.get(
                    "original_rows",
                    rows
                )
            
                clean_rows = quality.get(
                    "clean_rows",
                    rows
                )
            
                original_duplicates = quality.get(
                    "original_duplicate_rows",
                    0
                )
            
                clean_duplicates = quality.get(
                    "clean_duplicate_rows",
                    0
                )
            
                original_missing = quality.get(
                    "original_missing_values",
                    0
                )
            
                clean_missing = quality.get(
                    "clean_missing_values",
                    0
                )
            
            
                q1, q2, q3, q4 = st.columns(4)
            
                q1.metric(
                    "Original Records",
                    f"{original_rows:,}"
                )
            
                q2.metric(
                    "Duplicate Records Identified",
                    f"{original_duplicates:,}"
                )
            
                q3.metric(
                    "Original Missing Values",
                    f"{original_missing:,}"
                )
            
                q4.metric(
                    "Records After Preparation",
                    f"{clean_rows:,}"
                )
            
            
                st.subheader(
                    "Before and after data preparation"
                )
            
            
                quality_table = pd.DataFrame({
                    "Check": [
                        "Records",
                        "Missing values",
                        "Duplicate records"
                    ],
            
                    "Before preparation": [
                        original_rows,
                        original_missing,
                        original_duplicates
                    ],
            
                    "After preparation": [
                        clean_rows,
                        clean_missing,
                        clean_duplicates
                    ]
                })
            
            
                st.dataframe(
                    quality_table,
                    use_container_width=True,
                    hide_index=True
                )
    
                st.subheader("ℹ️ Why do we check these things?")
            
                st.write(
                    """
                    Missing values represent incomplete information within the dataset. Datasets containing many missing 
                    values can reduce prediction performance and may require cleaning or imputation before training 
                    machine learning models. In this case, the downloaded raw dataset were already cleaned and prepared 
                    for machine learning projects so contained no missing values.
            
                    Duplicate records are repeated observations containing the same recorded information. They are checked 
                    because repeated observations can affect how the model learns from the data. 
            
                    These checks help users understand whether the data needs cleaning
                    or preparation before it is given to the machine-learning models.
                    """
                    )
            

        

        with dataset_analysis_tabs[3]:
            with st.container():
                st.subheader("What types of activity are in the dataset?")
        
            st.caption(
                "This chart shows how many records belong to each traffic or "
                "attack category."
            )
        
        
            class_distribution = eda.get(
                "class_distribution"
            )
        
        
            if class_distribution is not None:
        
                if isinstance(
                    class_distribution,
                    pd.Series
                ):
        
                    class_dist = (
                        class_distribution
                        .sort_values(
                            ascending=False
                        )
                    )
        
                else:
        
                    class_dist = pd.Series(
                        class_distribution
                    ).sort_values(
                        ascending=False
                    )
        
        
                fig, ax = plt.subplots(
                    figsize=(12, 6)
                )
        
                class_dist.plot(
                    kind="bar",
                    ax=ax
                )
        
                ax.set_xlabel(
                    "Activity / Attack Category"
                )
        
                ax.set_ylabel(
                    "Number of Records"
                )
        
                ax.set_title(
                    f"{dataset} - Class Distribution"
                )
        
                plt.xticks(
                    rotation=45,
                    ha="right"
                )
        
                plt.tight_layout()
        
                st.pyplot(fig)
        
                plt.close(fig)
        
        
                st.subheader("ℹ️ How should I interpret this chart?")
                st.write(
                    """
                    Taller bars mean that the dataset contains more examples
                    of that category. Shorter bars mean that fewer examples
                    are available. If some categories are much larger than others, the dataset
                    is imbalanced. This matters because a model may find common
                    categories easier to recognise than less common categories.
        
                    This is one of the reasons why model evaluation considers measures
                    such as recall, F1 score and balanced accuracy rather than
                    relying only on overall accuracy.
                    """
                    )
        
            else:
        
                st.warning(
                    "Class distribution information is not available."
                )


        # with dataset_analysis_tabs[4]:
        #     with st.container():

        #         st.subheader(
        #             "4. What do the feature values look like?"
        #         )
            
        #         st.caption(
        #             "Features are the pieces of information the models use to "
        #             "distinguish different types of network behaviour."
        #         )
            
            
        #         feature_statistics = eda.get(
        #             "feature_statistics"
        #         )
            
            
        #         if isinstance(
        #             feature_statistics,
        #             pd.DataFrame
        #         ):
            
        #             st.dataframe(
        #                 styled_table(
        #                     feature_statistics
        #                 ),
        #                 use_container_width=True
        #             )
            
        #         st.subheader("ℹ️ What do these statistics mean?")
            
        #         st.write(
        #             """
        #             Mean: the average value.
            
        #             Median: the middle value when the observations are ordered.
            
        #             Standard deviation: how much the values vary around the average.
            
        #             Minimum and maximum: the smallest and largest recorded values.
            
        #             The 25% and 75% values help show where most observations fall.
        #             """
        #             )
            



        # with dataset_analysis_tabs[4]:
        #     with st.container():

        #         st.subheader(
        #             "5. Are there unusually high or low values?"
        #         )
            
        #         st.caption(
        #             "Select a numerical feature to investigate its distribution "
        #             "and potential outliers."
        #         )
            
            
        #         if isinstance(
        #             numeric_features,
        #             pd.DataFrame
        #         ) and not numeric_features.empty:
            
        #             numeric_columns = list(
        #                 numeric_features.columns
        #             )
            
        #             selected_feature = st.selectbox(
        #                 "Select a feature",
        #                 numeric_columns,
        #                 key=f"eda_feature_{dataset}"
        #             )
            
            
        #             fig, ax = plt.subplots(
        #                 figsize=(10, 5)
        #             )
            
        #             ax.boxplot(
        #                 numeric_features[
        #                     selected_feature
        #                 ].dropna()
        #             )
            
        #             ax.set_ylabel(
        #                 selected_feature
        #             )
            
        #             ax.set_title(
        #                 f"{selected_feature} - Value Distribution"
        #             )
            
        #             plt.tight_layout()
            
        #             st.pyplot(fig)
            
        #             plt.close(fig)
            
            
        #             outlier_summary = eda.get(
        #                 "outlier_summary",
        #                 {}
        #             )
            
        #             selected_outliers = (
        #                 outlier_summary
        #                 .get(
        #                     selected_feature,
        #                     {}
        #                 )
        #             )
            
            
        #             if selected_outliers:
            
        #                 st.metric(
        #                     "Potential Outliers",
        #                     f'{selected_outliers.get("Outlier_Count", 0):,}'
        #                 )
            
            
        #             st.subheader("ℹ️ How should I interpret a box plot?")
            
        #             st.write(
        #                 """
        #                 The central box represents the middle portion of the data.
        
        #                 The line inside the box represents the median.
            
        #                 The whiskers show the broader typical range.
            
        #                 Points or observations outside this range may be unusually
        #                 high or low compared with most observations.
            
        #                 Unusual values are not automatically errors or attacks.
        #                 They may represent legitimate but uncommon network behaviour.
        #                 """
        #                 )
            



        with dataset_analysis_tabs[4]:
            with st.container():
                st.subheader(
                    "Do some features behave similarly?"
                )
            
                st.caption(
                    "This analysis shows relationships between numerical features."
                )
            
            
                correlation_matrix = eda.get(
                    "correlation_matrix"
                )
            
            
                if isinstance(
                    correlation_matrix,
                    pd.DataFrame
                ) and not correlation_matrix.empty:
            
                    fig, ax = plt.subplots(
                        figsize=(12, 8)
                    )
            
                    im = ax.imshow(
                        correlation_matrix.values,
                        aspect="auto",
                        cmap="Blues",
                        vmin=-1,
                        vmax=1
                    )
            
                    ax.set_xticks(
                        range(
                            len(correlation_matrix.columns)
                        )
                    )
            
                    ax.set_xticklabels(
                        correlation_matrix.columns,
                        rotation=90,
                        fontsize=7
                    )
            
                    ax.set_yticks(
                        range(
                            len(correlation_matrix.index)
                        )
                    )
            
                    ax.set_yticklabels(
                        correlation_matrix.index,
                        fontsize=7
                    )
            
                    ax.set_title(
                        f"{dataset} - Correlation Between Numerical Features"
                    )
            
                    fig.colorbar(
                        im,
                        ax=ax,
                        label="Correlation"
                    )
            
                    plt.tight_layout()
            
                    st.pyplot(fig)
            
                    plt.close(fig)
            
            
                    st.subheader("ℹ️ How should I interpret correlation?")
            
                    st.write(
                        """
                        A correlation matrix was generated to explore the relationships between numerical features in the dataset. 
                        The matrix was used as part of the exploratory data analysis to improve understanding of the dataset and feature relationships.
                        Values closer to +1 indicate a strong positive relationship. Values closer to -1 indicate a strong negative 
                        relationship. Values near 0 indicate little or no clear linear relationship.
                        
                        Highly correlated variables may indicate redundant information, while weak correlations suggest more independent 
                        relationships between features. Strong relationships can mean that two features contain
                        overlapping information.
            
                        In simple terms, it shows which pieces of network information tend to change together. 
                        Stronger colours indicate a stronger relationship..
                        """
                        )





        # with dataset_analysis_tabs[5]:
        #     with st.container():
        #         st.subheader(
        #             "7. Which information is used by the models?"
        #         )
            
        #         st.caption(
        #             dataset_cfg["feature_context"]
        #         )
            
            
        #         selected_features = eda.get(
        #             "selected_features",
        #             []
        #         )
            
            
        #         st.write(
        #             f"**Final modelling features: {len(selected_features)}**"
        #         )
            
            
        #         st.subheader("View selected features")
            
        #         feature_df = pd.DataFrame({
        #             "Feature": selected_features
        #         })
            
        #         st.dataframe(
        #             feature_df,
        #             use_container_width=True,
        #             hide_index=True
        #         )


        with dataset_analysis_tabs[5]:
            with st.container():

                st.subheader(
                    "Does the model behave consistently on unseen data?"
                )
            
                st.caption(
                    "Training performance is compared with validation and testing "
                    "performance to look for signs of overfitting or underfitting."
                )
            
            
                generalization = eda.get(
                    "generalization_results",
                    {}
                )
            
            
                if generalization:
            
                    generalization_rows = []
            
            
                    for model_name, values in generalization.items():
            
                        status, explanation = (
                            interpret_generalization(
                                values["train_accuracy"],
                                values["validation_accuracy"]
                            )
                        )
            
            
                        generalization_rows.append({
            
                            "Model":
                                model_name,
            
                            "Training Accuracy":
                                pct(values["train_accuracy"]),
            
                            "Validation Accuracy":
                                pct(values["validation_accuracy"]),
            
                            "Test Accuracy":
                                pct(values["test_accuracy"]),
            
                            "Train-Test Gap":
                                pct(values["train_test_accuracy_gap"]),
            
                            "Assessment":
                                status
                        })
            
            
                    generalization_df = pd.DataFrame(
                        generalization_rows
                    )
            
            
                    st.dataframe(
                        generalization_df,
                        use_container_width=True,
                        hide_index=True
                    )
            
            
                    st.subheader("ℹ️ What do overfitting and underfitting mean?")
            
                    st.write(
                        """
                        Training performance shows how well the model performs on the data it learned from.
                        Validation and testing performance show how well the model performs on data it did not use to learn.
            
                        If training performance is much higher than validation or testing performance, the model may be 
                        overfitting. This means it may have learned the training examples too closely.
            
                        If both training and validation performance are relatively low, the model may be underfitting. 
                        This can happen when the model is not capturing enough of the useful patterns in the dataset. 

                        A small difference suggests more consistent performance on unseen data. In this case, there 
                        is very little overfitting gap suggesting consistency. 
            
                        These are indicative assessments rather than definitive diagnoses of model behaviour.
                        """
                    )



        with dataset_analysis_tabs[6]:
            with st.container():
                st.subheader("What did we learn from this dataset?")
            
            
                best_model = (
                    results_df
                    .sort_values(
                        "Weighted_F1",
                        ascending=False
                    )
                    .iloc[0]
                )
            
            
                st.write(
                    f"""
                    - The selected dataset contains {rows:,} records and {classes:,} activity or attack categories.
            
                    - The dataset was examined for missing values, duplicate records,
                    class distribution, feature behaviour and relationships between
                    numerical features.
            
                    - The final modelling features were prepared before the machine-learning
                    models were trained.
            
                    - Based on the saved model evaluation results, the highest weighted
                    F1 score was achieved by **{best_model["Model"]}**.
            
                    - The next step is to compare the models in more detail or use a trained
                    model to analyse an individual network record.
                    """
                )
                

                

        
        # with dataset_analysis_tabs[10]:
        #     with st.container():
        #         st.subheader("Dataset Information")

        #         st.write("""
        #             This page answers "What is in this dataset?". These numbers give a quick overview of the size and structure of the dataset after cleaning.
                    
        #             The rows represents each observation in the dataset.
        #             Features are the individual characteristics of the network activity that the
        #             machine-learning models use to identify different types of behaviour. They are listed under the "Missing Values" tab.
        #             Duplicate rows are identical observations. They are checked because repeated observations can affect how the model learns from the data.
        #             Attack classes are the categories of cyber attacks the model is being trained to recognise.
                
        #             These values help the user understand the size and complexity of the problem
        #             before machine learning is applied.
        #             """)
                    
        #         c1,c2,c3,c4,c5 = st.columns(5)
            
        #         c1.metric(
        #             "Dataset Rows",
        #             info["rows"]
        #         )
        
        #         c2.metric(
        #             "Features",
        #             info["columns"]
        #         )
        
        #         c3.metric(
        #             "Duplicate Rows",
        #             info["duplicates"]
        #         )

        #         c4.metric(
        #             "Attack classes", 
        #             len(target_encoder.classes_)
        #         )
                    
        #         c5.metric(
        #             "Evaluation rows", 
        #             len(eval_sample)
        #         )
                        
        #         st.write(
        #             "Attack Classes:"
        #         )
            
        #         st.write(
        #             info["classes"]
        #         )
            
        # with dataset_analysis_tabs[11]:
        #     st.subheader("Missing Value Analysis")
    
        #     st.write("""
        #             This page answers "What is the quality of the data?". Before using the data for 
        #             machine learning, it is important to check for common data-quality issues.
                    
        #             Missing values represent incomplete information within the dataset.
        #             Datasets containing many missing values can reduce prediction performance and may
        #             require cleaning or imputation before training machine learning models. In this case, 
        #             the downloaded raw dataset were already cleaned and prepared for machine learning projects. 
        #             """)
        
        #     missing = eda["missing_values"]
        
        #     missing_df = (
        #         missing
        #         .reset_index()
        #     )
        
        #     missing_df.columns=[
        #         "Feature",
        #         "Missing Values"
        #     ]

        #     st.dataframe(
        #         missing_df,
        #         hide_index=True,
        #         use_container_width=True
        #     )

        #     st.subheader("Overfitting/Underfitting")

        #     st.write("""
        #         Training performance shows how well the model performs on data it learned from.
        #         Testing performance shows how well it performs on data it did not see during training.
                
        #         A large difference between training and testing performance may indicate that
        #         the model has learned the training data too closely. A small difference suggests
        #         more consistent performance on unseen data. In this case, there is very little overfitting gap suggesting consistency. 
        #         """)





    # dataset_cfg = DATASETS[dataset]

    # st.header("📊 Explore the Dataset")

    # st.info(
    #     dataset_cfg["eda_intro"]
    # )

    # st.caption(
    #     dataset_cfg["description"]
    # )


    # # ==================================
    # # 1. DATASET OVERVIEW
    # # ==================================

    # st.divider()

    # st.subheader(
    #     "1. What is in this dataset?"
    # )

    # st.caption(
    #     "These figures give a basic overview of the size and structure "
    #     "of the selected dataset."
    # )


    # dataset_info = eda.get(
    #     "dataset_info",
    #     {}
    # )

    # quality = eda.get(
    #     "dataset_quality_summary",
    #     {}
    # )


    # rows = quality.get(
    #     "clean_rows",
    #     dataset_info.get("rows", 0)
    # )

    # columns = quality.get(
    #     "clean_columns",
    #     dataset_info.get("columns", 0)
    # )

    # classes = len(
    #     dataset_info.get(
    #         "classes",
    #         []
    #     )
    # )


    # numeric_features = eda.get(
    #     "numeric_features"
    # )

    # numeric_count = (
    #     len(numeric_features.columns)
    #     if isinstance(
    #         numeric_features,
    #         pd.DataFrame
    #     )
    #     else 0
    # )


    # c1, c2, c3, c4 = st.columns(4)

    # c1.metric(
    #     "Records",
    #     f"{rows:,}"
    # )

    # c2.metric(
    #     "Features",
    #     f"{columns:,}"
    # )

    # c3.metric(
    #     "Attack Classes",
    #     f"{classes:,}"
    # )

    # c4.metric(
    #     "Numerical Features",
    #     f"{numeric_count:,}"
    # )


    # with st.expander(
    #     "ℹ️ Why does this matter?"
    # ):

    #     st.write(
    #         """
    #         A record represents one observation in the dataset, while a
    #         feature represents one piece of information about that observation.

    #         These figures help us understand the size and complexity of the
    #         problem before looking at the machine-learning results.
    #         """
    #     )


    # # ==================================
    # # 2. DATA QUALITY
    # # ==================================

    # st.divider()

    # st.subheader(
    #     "2. Is the data suitable for analysis?"
    # )

    # st.caption(
    #     "Before machine learning is applied, the dataset is checked for "
    #     "common data-quality issues."
    # )


    # original_rows = quality.get(
    #     "original_rows",
    #     rows
    # )

    # clean_rows = quality.get(
    #     "clean_rows",
    #     rows
    # )

    # original_duplicates = quality.get(
    #     "original_duplicate_rows",
    #     0
    # )

    # clean_duplicates = quality.get(
    #     "clean_duplicate_rows",
    #     0
    # )

    # original_missing = quality.get(
    #     "original_missing_values",
    #     0
    # )

    # clean_missing = quality.get(
    #     "clean_missing_values",
    #     0
    # )


    # q1, q2, q3, q4 = st.columns(4)

    # q1.metric(
    #     "Original Records",
    #     f"{original_rows:,}"
    # )

    # q2.metric(
    #     "Duplicate Records Identified",
    #     f"{original_duplicates:,}"
    # )

    # q3.metric(
    #     "Original Missing Values",
    #     f"{original_missing:,}"
    # )

    # q4.metric(
    #     "Records After Preparation",
    #     f"{clean_rows:,}"
    # )


    # st.subheader(
    #     "Before and after data preparation"
    # )


    # quality_table = pd.DataFrame({
    #     "Check": [
    #         "Records",
    #         "Missing values",
    #         "Duplicate records"
    #     ],

    #     "Before preparation": [
    #         original_rows,
    #         original_missing,
    #         original_duplicates
    #     ],

    #     "After preparation": [
    #         clean_rows,
    #         clean_missing,
    #         clean_duplicates
    #     ]
    # })


    # st.dataframe(
    #     quality_table,
    #     use_container_width=True,
    #     hide_index=True
    # )


    # with st.expander(
    #     "ℹ️ Why do we check these things?"
    # ):

    #     st.write(
    #         """
    #         Missing values mean that some information was not recorded.

    #         Duplicate records are repeated observations containing the same
    #         recorded information.

    #         These checks help us understand whether the data needs cleaning
    #         or preparation before it is given to the machine-learning models.
    #         """
    #     )


    # # ==================================
    # # 3. CLASS DISTRIBUTION
    # # ==================================

    # st.divider()

    # st.subheader(
    #     "3. What types of activity are in the dataset?"
    # )

    # st.caption(
    #     "This chart shows how many records belong to each traffic or "
    #     "attack category."
    # )


    # class_distribution = eda.get(
    #     "class_distribution"
    # )


    # if class_distribution is not None:

    #     if isinstance(
    #         class_distribution,
    #         pd.Series
    #     ):

    #         class_dist = (
    #             class_distribution
    #             .sort_values(
    #                 ascending=False
    #             )
    #         )

    #     else:

    #         class_dist = pd.Series(
    #             class_distribution
    #         ).sort_values(
    #             ascending=False
    #         )


    #     fig, ax = plt.subplots(
    #         figsize=(12, 6)
    #     )

    #     class_dist.plot(
    #         kind="bar",
    #         ax=ax
    #     )

    #     ax.set_xlabel(
    #         "Activity / Attack Category"
    #     )

    #     ax.set_ylabel(
    #         "Number of Records"
    #     )

    #     ax.set_title(
    #         f"{dataset} - Class Distribution"
    #     )

    #     plt.xticks(
    #         rotation=45,
    #         ha="right"
    #     )

    #     plt.tight_layout()

    #     st.pyplot(fig)

    #     plt.close(fig)


    #     with st.expander(
    #         "ℹ️ How should I interpret this chart?"
    #     ):

    #         st.write(
    #             """
    #             Taller bars mean that the dataset contains more examples
    #             of that category. Shorter bars mean that fewer examples
    #             are available.

    #             If some categories are much larger than others, the dataset
    #             is imbalanced. This matters because a model may find common
    #             categories easier to recognise than less common categories.

    #             This is one reason why model evaluation considers measures
    #             such as recall, F1 score and balanced accuracy rather than
    #             relying only on overall accuracy.
    #             """
    #         )

    # else:

    #     st.warning(
    #         "Class distribution information is not available."
    #     )


    # # ==================================
    # # 4. FEATURE ANALYSIS
    # # ==================================

    # st.divider()

    # st.subheader(
    #     "4. What do the feature values look like?"
    # )

    # st.caption(
    #     "Features are the pieces of information the models use to "
    #     "distinguish different types of network behaviour."
    # )


    # feature_statistics = eda.get(
    #     "feature_statistics"
    # )


    # if isinstance(
    #     feature_statistics,
    #     pd.DataFrame
    # ):

    #     st.dataframe(
    #         styled_table(
    #             feature_statistics
    #         ),
    #         use_container_width=True
    #     )

    # with st.expander(
    #     "ℹ️ What do these statistics mean?"
    # ):

    #     st.write(
    #         """
    #         Mean: the average value.

    #         Median: the middle value when the observations are ordered.

    #         Standard deviation: how much the values vary around the average.

    #         Minimum and maximum: the smallest and largest recorded values.

    #         The 25% and 75% values help show where most observations fall.
    #         """
    #     )


    # # ==================================
    # # 5. BOX PLOT / OUTLIERS
    # # ==================================

    # st.subheader(
    #     "5. Are there unusually high or low values?"
    # )

    # st.caption(
    #     "Select a numerical feature to investigate its distribution "
    #     "and potential outliers."
    # )


    # if isinstance(
    #     numeric_features,
    #     pd.DataFrame
    # ) and not numeric_features.empty:

    #     numeric_columns = list(
    #         numeric_features.columns
    #     )

    #     selected_feature = st.selectbox(
    #         "Select a feature",
    #         numeric_columns,
    #         key=f"eda_feature_{dataset}"
    #     )


    #     fig, ax = plt.subplots(
    #         figsize=(10, 5)
    #     )

    #     ax.boxplot(
    #         numeric_features[
    #             selected_feature
    #         ].dropna()
    #     )

    #     ax.set_ylabel(
    #         selected_feature
    #     )

    #     ax.set_title(
    #         f"{selected_feature} - Value Distribution"
    #     )

    #     plt.tight_layout()

    #     st.pyplot(fig)

    #     plt.close(fig)


    #     outlier_summary = eda.get(
    #         "outlier_summary",
    #         {}
    #     )

    #     selected_outliers = (
    #         outlier_summary
    #         .get(
    #             selected_feature,
    #             {}
    #         )
    #     )


    #     if selected_outliers:

    #         st.metric(
    #             "Potential Outliers",
    #             f'{selected_outliers.get("Outlier_Count", 0):,}'
    #         )


    #     with st.expander(
    #         "ℹ️ How should I interpret a box plot?"
    #     ):

    #         st.write(
    #             """
    #             The central box represents the middle portion of the data.

    #             The line inside the box represents the median.

    #             The whiskers show the broader typical range.

    #             Points or observations outside this range may be unusually
    #             high or low compared with most observations.

    #             Unusual values are not automatically errors or attacks.
    #             They may represent legitimate but uncommon network behaviour.
    #             """
    #         )


    # # ==================================
    # # 6. CORRELATION
    # # ==================================

    # st.divider()

    # st.subheader(
    #     "6. Do some features behave similarly?"
    # )

    # st.caption(
    #     "This analysis shows relationships between numerical features."
    # )


    # correlation_matrix = eda.get(
    #     "correlation_matrix"
    # )


    # if isinstance(
    #     correlation_matrix,
    #     pd.DataFrame
    # ) and not correlation_matrix.empty:

    #     fig, ax = plt.subplots(
    #         figsize=(12, 8)
    #     )

    #     im = ax.imshow(
    #         correlation_matrix.values,
    #         aspect="auto",
    #         cmap="Blues",
    #         vmin=-1,
    #         vmax=1
    #     )

    #     ax.set_xticks(
    #         range(
    #             len(correlation_matrix.columns)
    #         )
    #     )

    #     ax.set_xticklabels(
    #         correlation_matrix.columns,
    #         rotation=90,
    #         fontsize=7
    #     )

    #     ax.set_yticks(
    #         range(
    #             len(correlation_matrix.index)
    #         )
    #     )

    #     ax.set_yticklabels(
    #         correlation_matrix.index,
    #         fontsize=7
    #     )

    #     ax.set_title(
    #         f"{dataset} - Correlation Between Numerical Features"
    #     )

    #     fig.colorbar(
    #         im,
    #         ax=ax,
    #         label="Correlation"
    #     )

    #     plt.tight_layout()

    #     st.pyplot(fig)

    #     plt.close(fig)


    #     with st.expander(
    #         "ℹ️ How should I interpret correlation?"
    #     ):

    #         st.write(
    #             """
    #             Values closer to +1 indicate a strong positive relationship.

    #             Values closer to -1 indicate a strong negative relationship.

    #             Values near 0 indicate little or no clear linear relationship.

    #             Strong relationships can mean that two features contain
    #             overlapping information.

    #             Correlation does not mean that one feature causes another.
    #             """
    #         )


    # # ==================================
    # # 7. FEATURES USED BY MODELS
    # # ==================================

    # st.divider()

    # st.subheader(
    #     "7. Which information is used by the models?"
    # )

    # st.caption(
    #     dataset_cfg["feature_context"]
    # )


    # selected_features = eda.get(
    #     "selected_features",
    #     []
    # )


    # st.write(
    #     f"**Final modelling features: {len(selected_features)}**"
    # )


    # with st.expander(
    #     "View selected features"
    # ):

    #     feature_df = pd.DataFrame({
    #         "Feature": selected_features
    #     })

    #     st.dataframe(
    #         feature_df,
    #         use_container_width=True,
    #         hide_index=True
    #     )


    # # ==================================
    # # 8. MODEL GENERALISATION
    # # ==================================

    # st.divider()

    # st.subheader(
    #     "8. Does the model behave consistently on unseen data?"
    # )

    # st.caption(
    #     "Training performance is compared with validation and testing "
    #     "performance to look for signs of overfitting or underfitting."
    # )


    # generalization = eda.get(
    #     "generalization_results",
    #     {}
    # )


    # if generalization:

    #     generalization_rows = []


    #     for model_name, values in generalization.items():

    #         status, explanation = (
    #             interpret_generalization(
    #                 values["train_accuracy"],
    #                 values["validation_accuracy"]
    #             )
    #         )


    #         generalization_rows.append({

    #             "Model":
    #                 model_name,

    #             "Training Accuracy":
    #                 pct(values["train_accuracy"]),

    #             "Validation Accuracy":
    #                 pct(values["validation_accuracy"]),

    #             "Test Accuracy":
    #                 pct(values["test_accuracy"]),

    #             "Train-Test Gap":
    #                 pct(values["train_test_accuracy_gap"]),

    #             "Assessment":
    #                 status
    #         })


    #     generalization_df = pd.DataFrame(
    #         generalization_rows
    #     )


    #     st.dataframe(
    #         generalization_df,
    #         use_container_width=True,
    #         hide_index=True
    #     )


    #     with st.expander(
    #         "ℹ️ What do overfitting and underfitting mean?"
    #     ):

    #         st.write(
    #             """
    #             Training performance shows how well the model performs on
    #             the data it learned from.

    #             Validation and testing performance show how well the model
    #             performs on data it did not use to learn.

    #             If training performance is much higher than validation or
    #             testing performance, the model may be overfitting. This means
    #             it may have learned the training examples too closely.

    #             If both training and validation performance are relatively
    #             low, the model may be underfitting. This can happen when the
    #             model is not capturing enough of the useful patterns in
    #             the dataset.

    #             These are indicative assessments rather than definitive
    #             diagnoses of model behaviour.
    #             """
    #         )


    # # ==================================
    # # 9. WHAT DID WE LEARN?
    # # ==================================

    # st.divider()

    # st.subheader(
    #     "9. What did we learn from this dataset?"
    # )


    # best_model = (
    #     results_df
    #     .sort_values(
    #         "Weighted_F1",
    #         ascending=False
    #     )
    #     .iloc[0]
    # )


    # st.info(
    #     f"""
    #     The selected dataset contains {rows:,} records and
    #     {classes:,} activity or attack categories.

    #     The dataset was examined for missing values, duplicate records,
    #     class distribution, feature behaviour and relationships between
    #     numerical features.

    #     The final modelling features were prepared before the machine-learning
    #     models were trained.

    #     Based on the saved model evaluation results, the highest weighted
    #     F1 score was achieved by **{best_model["Model"]}**.

    #     The next step is to compare the models in more detail or use a trained
    #     model to analyse an individual network record.
    #     """
    # )
    
# ------------------------------
# PAGE: MODEL PERFORMANCE
# ------------------------------
elif page == "Model Performance":

    model_comparison_tabs = st.tabs([
        "Selected Model", "Confusion Matrix", "Classification Report"])

    with model_comparison_tabs[0]:
        st.write("""
        This page presents detailed performance statistics, the confusion matrix and a
        classification report to help evaluate how well the model distinguishes between
        different cyber attack categories.

        Several machine-learning models were trained on the selected dataset.
        This page compares their performance to identify which models classify
        the different types of network activity most reliably.
        """)
        
        st.subheader(f"Selected model: {selected_model_name}")
    
        row = results_df[results_df["Model"] == selected_model_name]
        if not row.empty:
            row = row.iloc[0]
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Accuracy", f'{row["Accuracy"]:.2%}')
            m2.metric("Balanced Accuracy", f'{row["Balanced_Accuracy"]:.2%}')
            m3.metric("Weighted F1", f'{row["Weighted_F1"]:.2%}')
            m4.metric("Macro F1", f'{row["Macro_F1"]:.2%}')
    
        # numeric_cols = results_df.select_dtypes(include=np.number).columns
    
        # styled_df = (
        #     results_df.style
        #     .format({col: "{:.4f}" for col in numeric_cols})
        #     .highlight_max(
        #         subset=["Accuracy","Balanced_Accuracy", "Weighted_F1","Macro_F1", "ROC_AUC"],
        #         color="#d4edda"
        #     )
        # )
        
        st.dataframe(
            styled_table(results_df),
            hide_index=True,
            use_container_width=True
        )
    
        # n = min(
        #     len(eval_labels),
        #     len(y_pred_eval)
        # )
        
        # y_true = np.array(eval_labels)[:n]
        # y_pred = np.array(y_pred_eval)[:n]
        
    with model_comparison_tabs[1]:
        st.subheader("Confusion Matrix on saved evaluation sample")
    
        st.write("""
                The confusion matrix compares actual attack labels against predicted labels.
                
                Values along the diagonal represent correct classifications, while values outside
                the diagonal indicate misclassifications between attack types.
                """)

        cm = confusion_matrices[selected_model_name]
        
        fig, ax = plt.subplots(
            figsize=(12,10)
        )
        
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=target_encoder.classes_
        )
        
        disp.plot(
            ax=ax,
            cmap="Blues",
            xticks_rotation=90
        )
        
        ax.set_title(
            f"{selected_model_name} Confusion Matrix"
        )
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    
    with model_comparison_tabs[2]:
        st.subheader("Classification report")
    
        st.write("""
                The classification report provides Precision, Recall and F1-score for every attack class.
                
                These metrics help identify which attacks are detected reliably and which attack
                types remain challenging for the model.
            """)

        report_df = pd.DataFrame(
            classification_reports[selected_model_name]
        ).transpose()
    
        st.dataframe(
            styled_table(report_df),
            use_container_width=True
        )

# ------------------------------
# PAGE: ABOUT
# ------------------------------
elif page == "About":
    # st.header("🛡️ XAI Dashboard for Cyber Threat Identification in Smart IoT Systems")
    
    about_page_tabs = st.tabs([
        "About this Dashboard", "Machine Learning Models Used", "Understanding Model Evaluation Metrics", "Dashboard Pages", "What a User Gets From This System"])

    with about_page_tabs[0]:
        st.subheader("About this Dashboard")
        st.write(
            """
            This dashboard demonstrates how Explainable Artificial Intelligence (XAI) can
            improve cyber threat identification within Smart Internet of Things (IoT)
            environments.
            
            It uses five machine learning models trained on public IoT datasets to analyse IoT network activity, 
            and identify potentially malicious behaviour and explains their predictions. It allows users to explore 
            the data, check its quality, compare different machine-learning models, upload data for predictions, 
            and understand why the model made a particular decision using SHAP XAI technique.
            
            Unlike traditional intrusion detection systems that simply classify traffic, this dashboard provides 
            transparent, interpretable and trustworthy explanations so that security analysts can
            understand the reasoning behind every prediction.
            """
        )

        st.subheader("How the system works")
    
        st.write("""
            The system follows a simple process:
            
            1. The dataset is explored and checked for data-quality issues.
            2. Machine-learning models learn patterns from the network data.
            3. The models are evaluated to see how reliably they classify different types of network activity.
            4. A selected model can be used to analyse a network record for potential cyber attacks.
            5. Explainable AI shows which features influenced the prediction.
            """)

    with about_page_tabs[1]:
        st.subheader("Machine Learning Models Used")
    
        model_description = pd.DataFrame({
            "Model":[
                "Random Forest",
                "Decision Tree",
                "LightGBM",
                "XGBoost",
                "Logistic Regression"
            ],
    
            "Purpose":[
                "Combines many decision trees to improve accuracy and reduce errors.",
                "Creates decision rules that are easy to interpret.",
                "Fast gradient boosting model designed for high performance.",
                "Advanced boosting algorithm for complex attack detection.",
                "Linear classification model used as a baseline comparison."
            ],
    
            "Behaviour":[
                "Strong performance with complex IoT traffic patterns.",
                "Simple explanations but can overfit.",
                "Handles large datasets efficiently.",
                "Usually provides high predictive accuracy.",
                "Provides understandable feature influence."
            ]
        })
    
        st.dataframe(
            model_description,
            hide_index=True,
            use_container_width=True
        )

    with about_page_tabs[2]:
        st.subheader("Understanding Model Evaluation Metrics")
    
        metrics = pd.DataFrame({
    
            "Metric":[
                "Accuracy",
                "Precision",
                "Recall",
                "F1 Score",
                "ROC-AUC",
                "Log Loss",
                "Confusion Matrix"
            ],
    
            "Meaning":[
    
                "Percentage of all predictions that were correct.",
    
                "Shows how many predicted attacks were actually attacks.",
    
                "Shows how many real attacks were successfully detected.",
    
                "Balance between precision and recall. Useful when classes are imbalanced.",
    
                "Measures how well the model separates different attack categories.",
    
                "Measures confidence error. Lower values are better.",
    
                "Shows which attack types are correctly or incorrectly classified."
            ]
        })
    
        st.dataframe(
            metrics,
            hide_index=True,
            use_container_width=True
        )

    with about_page_tabs[3]:
        st.subheader("Dashboard Pages")
    
        pages = pd.DataFrame({
    
            "Page":[
                "About",
                "Dataset Analysis",
                "Overview",
                "Model Performance",
                "Prediction",
                "Explainability"
            ],
    
            "Purpose":[
    
                "Gives an understanding of the project and its elements.",
                
                "Shows dataset characteristics before modelling.",
    
                "Provides a complete summary of the trained ML system.",
    
                "Compares model performance using evaluation metrics.",
    
                "Allows users to classify new IoT traffic samples.",
    
                "Explains why the AI produced a prediction."
            ]
        })
    
        st.dataframe(
            pages,
            hide_index=True,
            use_container_width=True
        )

    with about_page_tabs[4]:
        st.subheader("What a User Gets From This System")
    
        st.write(
        """
        A security analyst can use this dashboard to:
    
        - Understand the cybersecurity dataset
    
        - Identify important network features
    
        - Check data quality problems
    
        - Compare different AI models
    
        - Detect possible cyber attacks
    
        - Understand why the AI selected a specific threat category
    
        - Make better security decisions using explainable AI
        """
        )

# ------------------------------
# PAGE: PREDICTION
# ------------------------------
elif page == "Prediction":
    st.write("""
        This page allows users to use a trained AI model to analyse a network record and identify the 
        type of activity.
        
        Users may upload a dataset or use the saved evaluation sample. The selected machine
        learning model predicts the attack category as well as its confidence score.
        """)
    
    st.subheader("Predict threats from a CSV or the saved demo sample")

    input_mode = st.radio(
        "Input source",
        ["Use saved evaluation sample", "Upload CSV"],
        horizontal=True,
        key="input_mode"
    )

    if input_mode == "Upload CSV":
        uploaded_file = st.file_uploader(
            "Upload a CSV file with the same feature columns as training",
            type=["csv"]
        )

        if uploaded_file is not None:
            raw_df = pd.read_csv(uploaded_file)
            prepared_df = preprocess_input(
                raw_df,
                feature_names=feature_names,
                cat_cols=cat_cols,
                feature_encoder=feature_encoder,
                drop_cols=drop_cols
            )

            prepared_df = prepared_df.reindex(
                columns=feature_names,
                fill_value=0
            )
                        
            st.success(f"Loaded {len(prepared_df)} rows from upload.")
            st.dataframe(prepared_df.head(), use_container_width=True)
        else:
            st.info("Upload a CSV to continue.")
            prepared_df = None
    else:
        prepared_df = (
            eval_sample
            .copy()
            .reindex(
                columns=feature_names,
                fill_value=0
            )
        )

        st.success(f"Using saved evaluation sample with {len(prepared_df)} rows.")
        st.dataframe(prepared_df.head(), use_container_width=True)

    if prepared_df is not None and len(prepared_df) > 0:
        row_index = st.number_input(
            "Row index to inspect",
            min_value=0,
            max_value=len(prepared_df) - 1,
            value=0,
            step=1
        )

        x_row = prepared_df.iloc[[int(row_index)]]

        st.session_state["current_input_df"] = prepared_df
        st.session_state["current_row_index"] = int(row_index)

        if st.button("Run prediction", type="primary"):
            pred, proba = get_model_predictions(selected_model, x_row)
            pred_label = target_encoder.inverse_transform(pred)[0]
            confidence = float(np.max(proba[0]))

            pred_label_lower = pred_label.lower()

            is_normal = pred_label_lower in [
                "normal",
                "benign",
                "safe"
            ]

            if is_normal:
                st.success(
                    f"""
                    ✅ Normal network activity
            
                    Confidence:
                    {confidence:.2%}
                    """
                )
            
            else:
                if confidence >= 0.80:
                    st.error(
                        f"""
                        🚨 High-risk threat detected
            
                        Category:
                        {pred_label}
            
                        Confidence:
                        {confidence:.2%}
                        """
                    )
            
                elif confidence >= 0.50:
                    st.warning(
                        f"""
                        ⚠ Potential threat detected
            
                        Category:
                        {pred_label}
            
                        Confidence:
                        {confidence:.2%}
                        """
                    )
            
                else:
                    st.info(
                        f"""
                        ℹ Low-confidence prediction
            
                        Possible category:
                        {pred_label}
            
                        Confidence:
                        {confidence:.2%}
                        """
                    )

            proba_df = pd.DataFrame(
                [proba[0]],
                columns=target_encoder.classes_
            ).T
            proba_df.columns = ["Probability"]
            proba_df = proba_df.sort_values(by="Probability", ascending=False)


            st.subheader("Class probabilities")
            st.dataframe(
                styled_table(proba_df),
                use_container_width=True
            )
            st.write("""
                    Rather than only predicting one class, the model estimates the probability of
                    every possible attack category.
                    
                    Higher probabilities indicate greater confidence in the prediction.
                    """)

            st.subheader("Input row")
            st.dataframe(x_row, use_container_width=True)

            download_df = x_row.copy()
            download_df["Prediction"] = pred_label
            download_df["Confidence"] = confidence

            csv_bytes = download_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download this prediction",
                data=csv_bytes,
                file_name="prediction_result.csv",
                mime="text/csv"
            )

# ------------------------------
# PAGE: EXPLAINABILITY
# ------------------------------
elif page == "Explainability":
    
    st.subheader(f"Explainability for {selected_model_name}")
    
    st.write("""
            Explainable Artificial Intelligence (XAI) helps users understand why the model
            made a particular decision rather than simply displaying the prediction.
            This improves transparency, trust and accountability when machine learning models
            are deployed for cyber security applications.
            """)

    if (
        "current_input_df" in st.session_state
        and len(st.session_state["current_input_df"]) > 0
        and list(st.session_state["current_input_df"].columns)
            == list(feature_names)
    ):
        input_df = st.session_state["current_input_df"]
        row_index = st.session_state.get("current_row_index", 0)
    else:
        input_df = (
            eval_sample
            .copy()
            .reindex(
                columns=feature_names,
                fill_value=0
            )
        )
        row_index = 0

    if len(input_df) == 0:
        st.warning("No input data available for explanation.")
        st.stop()

    row_index = min(max(int(row_index), 0), len(input_df) - 1)
    x_row = input_df.iloc[[row_index]]

    st.write("Selected row:")
    st.dataframe(x_row, use_container_width=True)

    model_type_tabs = st.tabs(["Local explanation", "Global importance", "Reasoning"])

    # 1) LOCAL EXPLANATION
    with model_type_tabs[0]:
        st.subheader("Local explanation")

        st.write("""
                Local explanations describe why the model classified one individual network record.
                SHAP values look at the selected prediction and shows which features
                influenced the model's decision.
                """)

        if selected_model_name == "Logistic Regression": 
            st.write("""
                The dashboard currently uses a coefficient/contribution-based explanation for Logistic Regression. Logistic Regression is a linear model, so its decisions can be explained directly using its learned coefficients and the values of the features. The other models use tree-based SHAP explanations.

                Each bar represents a feature from the network record. The bar shows how strongly that feature contributed to the model's decision. Longer bars represent stronger contributions. Features with values close to zero had little influence on this particular decision.
                
                The feature contribution table shows the individual features that had the greatest influence on the current prediction. The contribution is calculated from the feature's value and the Logistic Regression model's learned coefficient.
                
                Positive contributions support the predicted class, while negative contributions move the prediction away from it.
                """)
            # st.write("CALLER x_row shape:", x_row.shape)
            
            pred_label = plot_lr_reasoning(selected_model, x_row, feature_names, target_encoder)
            st.success(f"Predicted class: {pred_label}")
            st.info(
                "For Logistic Regression, the dashboard shows coefficient-based importance instead of Tree SHAP."
            )
        else:
            st.write("""
                The prediction is like a decision being built step by step. The model starts from a 
                baseline expectation and then each feature pushes the decision towards or away from a 
                particular attack class. The baseline is basically the model's starting point. Each 
                feature then pushes the decision in one direction or the other until the final prediction is reached.

                • Each bar on the waterfall plot represents a feature that influenced the prediction. 
                Larger bars indicate a stronger influence on this particular decision.
                
                • Colour: Red pushes the prediction higher while blue pushes the prediction lower.
            
                • Features pushing the prediction towards the predicted class are shown as postive 
                contributions and therefore increase the model's output, while features pushing 
                it away are shown as negative contributions and therefore reduce the model's output. 
            
                • E[f(X)] represents the model's baseline output which is what the model would expect 
                before considering the specific features of this network record. The individual feature 
                contributions are added to this baseline to arrive at f(x), the model output for the selected record.
            
                • The final output represents the model's decision for this record.
                """)
            
            background = eval_sample.sample(
                n=min(1000, len(eval_sample)),
                random_state=42
            )

            explainer, shap_bg, shap_row = get_tree_shap(selected_model, background, x_row)
            pred = selected_model.predict(x_row)[0]
            pred_label = target_encoder.inverse_transform([pred])[0]

            st.success(f"Predicted class: {pred_label}")
            plot_tree_shap_local(
                selected_model,
                explainer,
                shap_row,
                x_row,
                target_encoder
            )

    # 2) GLOBAL IMPORTANCE
    with model_type_tabs[1]:
        st.subheader("Global feature importance")

        st.write("""
                Global feature importance analysis looks across many network records and summarises which variables influenced the model most
                across the entire dataset. It shows which features the model generally relies on most when distinguishing
                between different types of network activity. 
                """)
        st.subheader("Feature Importance Table")
        st.write("""
                This table ranks the input features according to their overall influence on the model's predictions. 
                Features with higher importance values have a greater influence on the model's decisions, 
                while features with lower values have less influence.
                
                Feature importance is calculated across the model as a whole, rather than for one individual prediction. 
                It helps identify which characteristics of the IoT network data the model considers most useful for 
                distinguishing between normal and malicious activity.
                
                Important: A highly important feature does not necessarily cause a cyberattack. It only means that the 
                model found the feature useful when making predictions.
                """)

        imp_df = get_model_importance(
            selected_model_name,
            selected_model,
            feature_names
        )
        st.dataframe(
            styled_table(imp_df.head(20)),
            use_container_width=True
        )


        if selected_model_name == "Logistic Regression": 
            # background = eval_sample.sample(
            #     n=min(1000, len(eval_sample)),
            #     random_state=42
            # )
                    
            # lr = selected_model.named_steps["lr"]
                    
            # explainer = shap.LinearExplainer(
            #     lr,
            #     background
            # )
                    
            # shap_values = explainer.shap_values(background)
                    
            # plt.figure(figsize=(13,6))
        
            # plt.title(f"{selected_model_name} - SHAP Global Feature Importance")
                    
            # shap.summary_plot(
            #     shap_values,
            #     background,
            #     show=False
            # )
                    
            # st.pyplot(plt.gcf())
                    
            # plt.close()




            background = eval_sample.sample(
                n=min(1000, len(eval_sample)),
                random_state=42
            )

            expected_features = selected_model.feature_names_in_
            
            background = background.reindex(
                columns=expected_features,
                fill_value=0
            )
            
            x_row = x_row.reindex(
                columns=expected_features,
                fill_value=0
            )

            # Transform data exactly as Logistic Regression sees it
            transformed_background = selected_model[:-1].transform(
                background
            )

            lr = selected_model.named_steps["lr"]

            explainer = shap.LinearExplainer(lr, transformed_background)

            shap_values = explainer.shap_values(transformed_background)

            plt.figure(figsize=(13,6))

            st.subheader(f"{selected_model_name} - SHAP Global Feature Importance Beeswarm Plot")


            shap.summary_plot(
                shap_values,
                transformed_background,
                feature_names=background.columns.tolist(),
                show=False
            )


            st.pyplot(plt.gcf())
            plt.close()





            # # # Convert SHAP values to a numpy array
            # # shap_array = np.asarray(shap_values.values if hasattr(shap_values, "values") else shap_values)
            
            # # # Select the predicted class
            # # pred = selected_model.predict(x_row)
            # # class_idx = int(pred[0])
            
            # # # For multiclass SHAP: (samples, features, classes)
            # # if shap_array.ndim == 3:
            # #     shap_for_class = shap_array[:, :, class_idx]
            # # else:
            # #     shap_for_class = shap_array
            
            # # # Force numeric dtype
            # # shap_for_class = np.asarray(shap_for_class, dtype=np.float64)
            
            # # # Make sure feature data is numeric too
            # # X_plot = background.copy()
            
            # # for col in X_plot.columns:
            # #     X_plot[col] = pd.to_numeric(X_plot[col], errors="coerce")
            
            # # X_plot = X_plot.astype(np.float64)
            
            # # # Plot
            # # fig = plt.figure(figsize=(12, 8))
            
            # # shap.summary_plot(
            # #     shap_for_class,
            # #     X_plot,
            # #     feature_names=X_plot.columns,
            # #     show=False,
            # #     max_display=15
            # # )
            
            # # plt.tight_layout()
            # # st.pyplot(fig)
            # # plt.close(fig)




            # shap_array = np.asarray(
            #     shap_values.values if hasattr(shap_values, "values") else shap_values,
            #     dtype=np.float64
            # )
            
            # if shap_array.ndim == 3:
            #     # samples × features × classes
            #     global_importance = np.mean(
            #         np.abs(shap_array),
            #         axis=(0, 2)
            #     )
            # else:
            #     # samples × features
            #     global_importance = np.mean(
            #         np.abs(shap_array),
            #         axis=0
            #     )
            
            # # importance_df = pd.DataFrame({
            # #     "Feature": background.columns,
            # #     "Mean |SHAP|": global_importance
            # # }).sort_values(
            # #     "Mean |SHAP|",
            # #     ascending=False
            # # )
            
            # # st.dataframe(
            # #     importance_df.head(20),
            # #     use_container_width=True
            # # )



            # pred = selected_model.predict(x_row)
            # class_idx = int(pred[0])
            
            # shap_class = shap_array[:, :, class_idx]
            
            # shap.summary_plot(
            #     shap_class,
            #     background,
            #     feature_names=background.columns,
            #     max_display=15,
            #     show=False
            # )


        # if selected_model_name != "Logistic Regression":
        #     try:
        #         background = eval_sample.sample(
        #             n=min(1000, len(eval_sample)),
        #             random_state=42
        #         )
        #         explainer = shap.TreeExplainer(selected_model)
        #         shap_values_bg = explainer.shap_values(background)

        #         fig = plt.figure(figsize=(13, 6))
        #         shap.summary_plot(shap_values_bg, background, show=False)
        #         plt.tight_layout(pad=2)
        #         st.pyplot(fig)
        #         plt.close(fig)
        #     except Exception as e:
        #         st.warning(f"SHAP summary plot could not be rendered here: {e}")



        elif selected_model_name != "Logistic Regression":
            try:
                background = eval_sample.sample(
                    n=min(1000, len(eval_sample)),
                    random_state=42
                )
        
                explainer = shap.TreeExplainer(selected_model)
        
                shap_values_bg = explainer.shap_values(
                    background
                )
        
                plt.figure(figsize=(13, 6))

                st.subheader(f"{selected_model_name} - SHAP Global Feature Importance Beeswarm Plot")
        
                shap.summary_plot(
                    shap_values_bg,
                    background,
                    show=False
                )
        
                st.pyplot(
                    plt.gcf()
                )
        
                plt.close()
        
            except Exception as e:
                st.warning(
                    f"SHAP summary plot could not be rendered here: {e}"
                )


        st.write("""
                The beeswarm plot shows which features generally have the greatest influence
                on the model across many network records.
                
                • Each dot: Represents one network record. The position of the dot shows 
                whether that feature pushed the model's output higher or lower for that observation.
                
                • Vertical position: Features at the top have greater overall influence on the model, 
                while features near the bottom have less influence. 
                Each point represents the effect of a feature for an individual
                record. It means the model found these features useful when distinguishing between attack classes.
                
                • Horizontal position: shows the direction and strength of the feature's influence for individual records.
                A point further to the right indicates a stronger positive contribution to the model output being explained.
                A point further to the left indicates a stronger negative contribution.
                
                • Colour: Red represents relatively high feature values and blue represents
                  relatively low feature values.
                  """)

            
            
    # 3) REASONING
    with model_type_tabs[2]:

        st.write("""
                The reasoning section translates technical explainability outputs into an
                analyst-friendly explanation. It identifies the strongest contributing features and provides practical
                recommendations that can assist cyber security investigations.
                """)
        
        st.subheader("Why did the model make this decision?")

        pred, proba = get_model_predictions(selected_model, x_row)
        pred_label = target_encoder.inverse_transform(pred)[0]
        confidence = float(np.max(proba[0]))

        reasoning_cols = st.columns(2)
        reasoning_cols[0].metric("Predicted class", pred_label)
        reasoning_cols[1].metric("Confidence", f"{confidence:.2%}")

        # proba_df = pd.DataFrame(
        #     [proba[0]],
        #     columns=target_encoder.classes_
        # ).T
        # proba_df.columns = ["Probability"]
        # proba_df = proba_df.sort_values(by="Probability", ascending=False)

        # st.dataframe(proba_df, use_container_width=True)

        st.subheader("Top Reasons For The Prediction")

        st.write("""
            SHAP is used to show how different input features influenced the model's prediction. Each feature either 
            pushed the prediction towards or away from the predicted class. Larger contributions indicate a stronger 
            influence on the decision. These explanations help users understand which factors contributed most to that prediction.
            """)

        if selected_model_name == "Logistic Regression":
            lr = selected_model.named_steps["lr"]

            coef = lr.coef_

            if coef.ndim == 2:
                class_idx = int(pred[0])
                row_coef = coef[class_idx]
            else:
                row_coef = coef[0]

            reason_df = pd.DataFrame({
                "Feature": feature_names,
                "Coefficient": row_coef,
                "AbsCoefficient": np.abs(row_coef)
            }).sort_values(by="AbsCoefficient", ascending=False)

            st.subheader("Generated Explanation")
            
            top_reasons = reason_df.head(5)

            # st.write("Top coefficient-driven reasons:")
            # st.dataframe(top_reasons, use_container_width=True)
            

            explanation = (
                f"The model predicted **{pred_label}** with a confidence of "
                f"**{confidence:.2%}**. The strongest influencing features were: \n"
            )

            for _, row in top_reasons.iterrows():
                direction = (
                    "increased"
                    if row["Coefficient"] > 0
                    else "decreased"
                )

                explanation += (
                    f"\n• {row['Feature']} "
                    f"({direction} likelihood, "
                    f"SHAP={row['Coefficient']:.3f})\n"
                )
            


            st.write(explanation)

            st.subheader("Recommended Analyst Actions")
            
            st.info(
                        f"""
                        Note that explanations are approximations, and they support only support human judgement rather than replace them. 
                        """
                    )

            recommendations = []

            for _, row in top_reasons.iterrows():
                recommendations.append(
                    f"Investigate **{row['Feature']}**."
                )

            for rec in recommendations:
                st.write("•", rec)

        else:
            try:
                background = eval_sample.sample(
                    n=min(1000, len(eval_sample)),
                    random_state=42
                )

                explainer, shap_bg, shap_row = get_tree_shap(
                    selected_model,
                    background,
                    x_row
                )

                if isinstance(shap_row, list):
                    class_idx = int(pred[0])
                    local_values = shap_row[class_idx][0]

                elif len(shap_row.shape) == 3:
                    class_idx = int(pred[0])
                    local_values = shap_row[0, :, class_idx]

                else:
                    local_values = shap_row[0]

                reason_df = pd.DataFrame({
                    "Feature": x_row.columns,
                    "SHAP Value": local_values
                    # "Impact": np.abs(local_values)
                })

                reason_df = reason_df.sort_values(
                    # by="Impact",
                    by="SHAP Value",
                    ascending=False
                )

                st.dataframe(
                    reason_df.head(10),
                    use_container_width=True
                )

                st.subheader("Generated Explanation")

                top_features = reason_df.head(5)

                explanation = (
                    f"The model predicted '{pred_label}' "
                    f"with a confidence of {confidence:.2%}. "
                    f"This prediction was mainly influenced by: \n"
                )

                for _, row in top_features.iterrows():
                    direction = (
                        "increased"
                        if row["SHAP Value"] > 0
                        else "decreased"
                    )

                    explanation += (
                        f"\n• {row['Feature']} "
                        f"({direction} likelihood, "
                        f"SHAP={row['SHAP Value']:.3f})\n"
                    )

                st.write(explanation)

                st.subheader("Recommended Analyst Actions")

                st.info(
                        f"""
                        Note that explanations are approximations, and they support only support human judgement rather than replace them. 
                        """
                    )
                
                recommendations = []

                for feature in top_features["Feature"]:
                    recommendations.append(
                        f"Investigate **'{feature}'**."
                    )

                recommendations = recommendations[:5]

                for rec in recommendations:
                    st.write("•", rec)

            except Exception as e:
                st.warning(f"Could not generate reasoning: {e}")

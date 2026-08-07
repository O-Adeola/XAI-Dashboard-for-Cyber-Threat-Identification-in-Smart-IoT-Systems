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
    initial_sidebar_state="expanded"
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
font-size:12px;
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

/* =========================================================
   STREAMLIT HEADER TOOLBAR ICONS
   ========================================================= */

header[data-testid="stHeader"] svg {
    color: white !important;
    fill: white !important;
    stroke: white !important;
}

header[data-testid="stHeader"] svg path {
    fill: white !important;
    stroke: white !important;
}

header[data-testid="stHeader"] button {
    color: white !important;
}

header[data-testid="stHeader"] button:hover {
    background: rgba(255,255,255,0.15) !important;
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
        "description": "TON_IoT dataset for smart IoT threat detection",
    },
    "EdgeIIoT": {
        "models_dir": DATASET_DIR / "EdgeIIoT" / "models",
        "artifacts_dir": DATASET_DIR / "EdgeIIoT" / "artifacts",
        "description": "EdgeIIoT dataset for IoT/IIoT threat detection",
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
eda = artifacts.get("eda", {})

results_df = results_df.sort_values(by="Weighted_F1", ascending=False).reset_index(drop=True)

page = st.sidebar.selectbox(
    "Go to",
    [
        "Overview",
        "Dataset Analysis",
        "Model Comparison",
        "Prediction",
        "Explainability",
        "About"
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
Inspect predictions, and explain why a threat was flagged.
</p>

</div>

""", unsafe_allow_html=True)

# ------------------------------
# PAGE: OVERVIEW
# ------------------------------
if page == "Overview":

    overview_page_tabs = st.tabs(["Overview", "Top Model Ranking", "Higher-is-better Metrics", "Lower-is-better Metrics"])

    with overview_page_tabs[0]:
        st.subheader("Overview") 
        st.write("""
            This page presents an overall comparison of every trained machine learning model.
            Performance metrics allow users to identify the strongest performing model while
            comparing prediction accuracy, robustness and reliability across different algorithms.
            """)
        
        top_row = results_df.iloc[0]
    
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Best Model", top_row["Model"])
        c2.metric("Weighted F1", f'{top_row["Weighted_F1"]:.4f}')
        c3.metric("Accuracy", f'{top_row["Accuracy"]:.4f}')
        c4.metric("ROC AUC", f'{top_row["ROC_AUC"]:.4f}')
    
        st.subheader("Dataset and model summary")
        summary_cols = st.columns(3)
        summary_cols[0].metric("Features", len(feature_names))
        summary_cols[1].metric("Attack classes", len(target_encoder.classes_))
        summary_cols[2].metric("Evaluation rows", len(eval_sample))

    with overview_page_tabs[1]:
        st.subheader("Top model ranking")
        with st.expander("ℹ️ About this analysis"):
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

    with overview_page_tabs[2]:
        st.subheader("Higher-is-better metrics")
        with st.expander("ℹ️ About this analysis"):
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

    with overview_page_tabs[3]:
        st.subheader("Lower-is-better metrics")
        with st.expander("ℹ️ About this analysis"):
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
elif page == "Dataset Analysis":

    dataset_analysis_tabs = st.tabs([
        "Exploratory Data Analysis (EDA)", "Performance Summary", "Missing Value Analysis"])

    with dataset_analysis_tabs[0]:
        st.subheader(
            "Exploratory Data Analysis (EDA)"
            )

        st.write("""
            This page provides a high-level overview of the selected dataset before model training.
            
            It presents exploratory data analysis (EDA) to help understand the characteristics of the dataset. 
            It includes the number of records, features and attack classes. It also reports missing values, 
            duplicate records exist and possible overfitting/underfitting, allowing users to quickly assess 
            the quality of the data.
            """)

        eda = artifacts["eda"]
        if eda["dataset_info"]:
            info = eda["dataset_info"]

            with dataset_analysis_tabs[1]:
                with st.container():
                    st.subheader("Performance Summary")
                    c1,c2,c3 = st.columns(3)
            
                    c1.metric(
                        "Dataset Rows",
                        info["rows"]
                    )
            
                    c2.metric(
                        "Features",
                        info["columns"]
                    )
            
                    c3.metric(
                        "Duplicate Rows",
                        info["duplicates"]
                    )
                        
                    st.write(
                    "Attack Classes:"
                    )
            
                    st.write(
                        info["classes"]
                    )
            
        with dataset_analysis_tabs[2]:
            st.subheader("Missing Value Analysis")
    
            with st.expander("ℹ️ About this analysis"):
                st.write("""
                    Missing values represent incomplete information within the dataset.
                    
                    Datasets containing many missing values can reduce prediction performance and may
                    require cleaning or imputation before training machine learning models.
                    """)
        
            missing = eda["missing_values"]
        
            missing_df = (
                missing
                .reset_index()
            )
        
            missing_df.columns=[
                "Feature",
                "Missing Values"
            ]

            st.dataframe(
                missing_df,
                hide_index=True,
                use_container_width=True
            )
    
# ------------------------------
# PAGE: MODEL COMPARISON
# ------------------------------
elif page == "Model Comparison":

    st.write("""
        This page presents detailed performance statistics, the confusion matrix and a complete
        classification report to help evaluate how well the model distinguishes between
        different cyber attack categories.
        """)

    model_comparison_tabs = st.tabs([
        "Selected Model", "Confusion Matrix", "Classification Report"])

    with model_comparison_tabs[0]:
        st.subheader(f"Selected model: {selected_model_name}")
    
        row = results_df[results_df["Model"] == selected_model_name]
        if not row.empty:
            row = row.iloc[0]
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Accuracy", f'{row["Accuracy"]:.4f}')
            m2.metric("Balanced Accuracy", f'{row["Balanced_Accuracy"]:.4f}')
            m3.metric("Weighted F1", f'{row["Weighted_F1"]:.4f}')
            m4.metric("Macro F1", f'{row["Macro_F1"]:.4f}')
    
        numeric_cols = results_df.select_dtypes(include=np.number).columns
    
        styled_df = (
            results_df.style
            .format({col: "{:.4f}" for col in numeric_cols})
            .highlight_max(
                subset=["Accuracy","Balanced_Accuracy", "Weighted_F1","Macro_F1", "ROC_AUC"],
                color="#d4edda"
            )
        )
        
        st.dataframe(
            styled_df,
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
    
        with st.expander("ℹ️ About this analysis"):
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
    
        with st.expander("ℹ️ About this analysis"):
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
            
            It compares five machine learning models trained on public IoT datasets and explains their predictions 
            and decision-making process then uses SHAP XAI technique to explain why each prediction has been made.
            
            Unlike traditional intrusion detection systems that simply classify traffic, this dashboard provides 
            transparent, interpretable and trustworthy explanations so that security analysts can
            understand the reasoning behind every prediction.
            
            The system performs three main tasks:
            1. Understand the IoT dataset
            2. Identify cyber attacks using machine learning models
            3. Explain why the model made a particular decision
            """
        )

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
                "Model Comparison",
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
    
        st.success(
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
        This page allows users to classify new network traffic records.
        
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
            with st.expander("ℹ️ About this analysis"):
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
    with st.expander("ℹ️ About this analysis"):
        st.write("""
            Explainable Artificial Intelligence (XAI) helps users understand why the model
            made a particular decision rather than simply displaying the prediction.
            
            This improves transparency, trust and accountability when machine learning models
            are deployed for cyber security applications.
            """)
    st.subheader(f"Explainability for {selected_model_name}")

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

        with st.expander("ℹ️ About this analysis"):
            st.write("""
                Local explanations describe why the model classified one individual network record.
                
                SHAP values identify which features increased or decreased the likelihood of the
                predicted attack for that specific instance.
                """)

        if selected_model_name == "Logistic Regression":
            pred_label = plot_lr_reasoning(selected_model, x_row, feature_names, target_encoder)
            st.success(f"Predicted class: {pred_label}")
            st.info(
                "For Logistic Regression, the dashboard shows coefficient-based importance instead of Tree SHAP."
            )
        else:
            background = eval_sample.sample(
                n=min(1000, len(eval_sample)),
                random_state=42
            )

            explainer, shap_bg, shap_row = get_tree_shap(selected_model, background, x_row)
            pred = selected_model.predict(x_row)[0]
            pred_label = target_encoder.inverse_transform([pred])[0]

            plt.title(f"{selected_model_name} - SHAP Local Explanation Waterfall Plot")
        
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

        with st.expander("ℹ️ About this analysis"):
            st.write("""
                Global feature importance summarises which variables influenced the model most
                across the entire dataset.
                
                Frequently important features often represent the strongest indicators of malicious
                network behaviour.
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
            background = eval_sample.sample(
                n=min(1000, len(eval_sample)),
                random_state=42
            )
            
            lr = selected_model.named_steps["lr"]
            
            explainer = shap.LinearExplainer(
                lr,
                background
            )
            
            shap_values = explainer.shap_values(background)
            
            plt.figure(figsize=(13,6))

            # plt.title(f"{selected_model_name} - SHAP Global Feature Importance")

            st.subheader(f"{selected_model_name} - SHAP Global Feature Importance Beeswarm Plot")
            
            shap.summary_plot(
                shap_values,
                background,
                show=False
            )
            
            st.pyplot(plt.gcf())
            
            plt.close()

        if selected_model_name != "Logistic Regression":
            try:
                background = eval_sample.sample(n=min(1000, len(eval_sample)), random_state=42)
        
                explainer = shap.TreeExplainer(selected_model)
        
                shap_values_bg = explainer.shap_values(background)
        
                plt.figure(figsize=(13, 6))

                # plt.title(f"{selected_model_name} - SHAP Global Feature Importance")

                st.subheader(f"{selected_model_name} - SHAP Global Feature Importance Beeswarm Plot")
        
                shap.summary_plot(shap_values_bg, background, show=False)
        
                st.pyplot(plt.gcf())
        
                plt.close()
        
            except Exception as e:
                st.warning(f"SHAP summary plot could not be rendered here: {e}")

    # 3) REASONING
    with model_type_tabs[2]:

        with st.expander("ℹ️ About this analysis"):
            st.write("""
                The reasoning section translates technical explainability outputs into an
                analyst-friendly explanation.
                
                It identifies the strongest contributing features and provides practical
                recommendations that can assist cyber security investigations.
                """)
        
        st.subheader("Why did the model make this decision?")

        pred, proba = get_model_predictions(selected_model, x_row)
        pred_label = target_encoder.inverse_transform(pred)[0]
        confidence = float(np.max(proba[0]))

        reasoning_cols = st.columns(2)
        reasoning_cols[0].metric("Predicted class", pred_label)
        reasoning_cols[1].metric("Confidence", f"{confidence:.4f}")

        proba_df = pd.DataFrame(
            [proba[0]],
            columns=target_encoder.classes_
        ).T
        proba_df.columns = ["Probability"]
        proba_df = proba_df.sort_values(by="Probability", ascending=False)

        st.dataframe(proba_df, use_container_width=True)

        st.subheader("Top Reasons For The Prediction")

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

            top_reasons = reason_df.head(5)

            st.write("Top coefficient-driven reasons:")
            st.dataframe(top_reasons, use_container_width=True)

            explanation = (
                f"The model predicted **{pred_label}** with a confidence of "
                f"**{confidence:.2%}**. The strongest influencing features were: "
                f"{', '.join(top_reasons['Feature'].astype(str).tolist())}."
            )

            st.info(explanation)

            st.subheader("Recommended Analyst Actions")
            
            st.info(
                        f"""
                        Note that explanations are approximations, and they support only support human judgement rather than replace them. 
                        
                        ℹ Low-confidence prediction
            
                        Possible category:
                        {pred_label}
            
                        Confidence:
                        {confidence:.2%}
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
                    "SHAP Value": local_values,
                    "Impact": np.abs(local_values)
                })

                reason_df = reason_df.sort_values(
                    by="Impact",
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

                st.info(explanation)

                st.subheader("Recommended Analyst Actions")

                st.info(
                        f"""
                        Note that explanations are approximations, and they support only support human judgement rather than replace them. 
                        
                        ℹ Low-confidence prediction
            
                        Possible category:
                        {pred_label}
            
                        Confidence:
                        {confidence:.2%}
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

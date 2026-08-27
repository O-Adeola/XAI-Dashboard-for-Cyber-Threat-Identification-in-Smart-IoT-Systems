#!/usr/bin/env python
# coding: utf-8

# In[1]:


#===================
#IMPORTS 
#===================
import pandas as pd 
import numpy as np 

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, balanced_accuracy_score, f1_score, accuracy_score, precision_score, recall_score, ConfusionMatrixDisplay, cohen_kappa_score, hamming_loss, roc_auc_score, log_loss
from pathlib import Path


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

import shap
import joblib



PROJECT_DIR = Path.cwd()

DATASET_DIR = (
    PROJECT_DIR /
    "datasets" /
    "EdgeIIoT"
)

MODEL_DIR = DATASET_DIR / "models"
ARTIFACTS_DIR = DATASET_DIR / "artifacts"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


# In[2]:


#===================
#LOAD DATASETS 
#===================
raw_df = pd.read_csv("C:\\Users\\User\\Downloads\\ML-EdgeIIoT-dataset.csv\\ML-EdgeIIoT-dataset.csv", low_memory=False)

df = raw_df.copy()

df.head()


# In[3]:


#===================
#INSPECT DATASET 
#===================
print("Shape:", df.shape)
print(df.columns)
print(df.info())
print(df.nunique())
print(df.describe())
print(df["Attack_type"].value_counts())

print("Duplicates before cleaning:", df.duplicated().sum())


# In[4]:


# ==================================
# DATASET QUALITY
# ==================================

missing_before = df.isnull().sum().sum()


dataset_quality = {
    "rows": len(df),
    "columns": len(df.columns),
    "missing_values": int(missing_before),
}

print(missing_before)
print(dataset_quality)
print(df["Attack_type"].value_counts())


# In[5]:


#===================
#LABEL DISTRIBUTION
#===================
df['Attack_type'].value_counts() 

plt.figure(figsize=(12,6))

df['Attack_type'].value_counts().plot(
    kind='bar'
)

plt.title("Attack Class Distribution Before Cleaning")


plt.savefig(
    ARTIFACTS_DIR /
    "attack_distribution.png"
)

plt.show()


# In[6]:


#===================
#CLEAN AND PREPARE DATASET  
#===================
drop_columns = [
    "frame.time", "ip.src_host", "ip.dst_host", "arp.src.proto_ipv4",
    "arp.dst.proto_ipv4", "http.file_data", "http.request.full_uri",
    "icmp.transmit_timestamp", "http.request.uri.query", "tcp.options",
    "tcp.payload", "tcp.srcport", "tcp.dstport", "udp.port", "mqtt.msg"
]

drop_columns = [c for c in drop_columns if c in df.columns]
df = df.drop(columns=drop_columns)



duplicates_before = int(
    df.duplicated().sum()
)


df = df.drop_duplicates()


duplicates_after = int(
    df.duplicated().sum()
)

duplicates_removed = (
    duplicates_before -
    duplicates_after
)



df = shuffle(df, random_state=42).reset_index(drop=True)

print("Shape after cleaning:", df.shape)
print(df["Attack_type"].value_counts())



# In[7]:


#===================
#DATASET QUALITY/PREPROCESSING REPORT  
#===================
dataset_quality["features_after_preprocessing"] = len(df.columns)
dataset_quality["removed_columns"] = drop_columns


# In[8]:


# ==================================
# SAVE CLEAN DATASET FOR DASHBOARD
# ==================================

clean_df = df.copy()


# In[10]:


# ==================================
# DATASET QUALITY INFORMATION AFTER CLEANING
# ==================================

missing_after = df.isnull().sum().sum()

dataset_quality_after = {
    "rows": len(df),
    "columns": len(df.columns),
    "missing_values": int(missing_after),
}

print(missing_after)
print(dataset_quality_after)
print(df["Attack_type"].value_counts())


# In[12]:


#===================
#LABEL DISTRIBUTION
#===================
df['Attack_type'].value_counts() 

plt.figure(figsize=(12,6))

df['Attack_type'].value_counts().plot(
    kind='bar'
)

plt.title("Attack Class Distribution After Cleaning")


plt.savefig(
    ARTIFACTS_DIR /
    "attack_distribution.png"
)

plt.show()


# In[14]:


# ===================
# SPLIT FEATURES/TARGET 
# TRAIN-VALIDATE-TEST 
# ===================


import re

df.columns = df.columns.astype(str).str.replace(
    r"[^0-9a-zA-Z_]", "_", regex=True
)

X = df.drop(columns=["Attack_label", "Attack_type"])
y = df["Attack_type"]



#FIT ENCODER
target_encoder = LabelEncoder()
y_enc = target_encoder.fit_transform(y)

print(target_encoder.classes_)


print(np.unique(y))


# In[16]:


X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y_enc,
    test_size=0.4,
    random_state=42,
    stratify=y_enc
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.5,
    random_state=42,
    stratify=y_temp
)

print("Train:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)


# In[17]:


print(X_train.dtypes)



cat_cols = X_train.select_dtypes(
    include=["object", "string"]
).columns.tolist()

for col in cat_cols:
    print(
        col,
        "unique:", X_train[col].nunique(),
        "examples:", X_train[col].dropna().unique()[:10]
    )


# In[18]:


X_train = X_train.copy()
X_val = X_val.copy()
X_test = X_test.copy()


# In[20]:


# ==================================
# FEATURE (ORDINAL) ENCODING
# ==================================


encoder = OrdinalEncoder(
    handle_unknown="use_encoded_value",
    unknown_value=-1
)



# Fit ONLY on training data
X_train_encoded = encoder.fit_transform(X_train[cat_cols])

# Transform validation and test
X_val_encoded = encoder.transform(X_val[cat_cols])
X_test_encoded = encoder.transform(X_test[cat_cols])

# Get names for the new dummy columns
encoded_cols = encoder.get_feature_names_out(cat_cols)

# Convert encoded arrays into DataFrames
X_train_encoded = pd.DataFrame(
    X_train_encoded,
    columns=encoded_cols,
    index=X_train.index
)

X_val_encoded = pd.DataFrame(
    X_val_encoded,
    columns=encoded_cols,
    index=X_val.index
)

X_test_encoded = pd.DataFrame(
    X_test_encoded,
    columns=encoded_cols,
    index=X_test.index
)

# Remove original categorical columns
X_train = X_train.drop(columns=cat_cols)
X_val = X_val.drop(columns=cat_cols)
X_test = X_test.drop(columns=cat_cols)

# Add encoded columns
X_train = pd.concat([X_train, X_train_encoded], axis=1)
X_val = pd.concat([X_val, X_val_encoded], axis=1)
X_test = pd.concat([X_test, X_test_encoded], axis=1)

print("Train:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)


# In[21]:


import re

def clean_feature_names(df):
    df = df.copy()
    df.columns = (
        df.columns
        .astype(str)
        .str.replace(r"[^0-9a-zA-Z_]", "_", regex=True)
        .str.replace(r"_+", "_", regex=True)
        .str.strip("_")
    )
    return df


X_train = clean_feature_names(X_train)
X_val = clean_feature_names(X_val)
X_test = clean_feature_names(X_test)


# In[22]:


dataset_quality["categorical_features"] = list(cat_cols)
dataset_quality["selected_features"] = list(X_train.columns)


# In[23]:


# ==================================
# MODEL EVALUATION FUNCTION
# ==================================


def evaluate_model(model_name, model, X_data, y_true):

    y_pred = model.predict(X_data)

    # Probabilities required for ROC-AUC and Log Loss
    y_prob = model.predict_proba(X_data)

    accuracy = accuracy_score(y_true, y_pred)

    balanced_acc = balanced_accuracy_score(
        y_true,
        y_pred
    )

    precision_weighted = precision_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    precision_macro = precision_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    recall_weighted = recall_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    recall_macro = recall_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    weighted_f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    kappa = cohen_kappa_score(
        y_true,
        y_pred
    )

    hamming = hamming_loss(
        y_true,
        y_pred
    )

    roc_auc = roc_auc_score(
        y_true,
        y_prob,
        multi_class="ovr",
        average="weighted"
    )

    loss = log_loss(
        y_true,
        y_prob
    )

    print("\n" + "="*60)
    print(model_name)
    print("="*60)

    print(classification_report(
        y_true, 
        y_pred, 
        target_names=target_encoder.classes_, 
        zero_division=0
    ))

    print("Accuracy:", round(accuracy,4))
    print("Balanced Accuracy:", round(balanced_acc,4))
    print("Precision Weighted:", round(precision_weighted,4))
    print("Precision Macro:", round(precision_macro,4))
    print("Recall Weighted:", round(recall_weighted,4))
    print("Recall Macro:", round(recall_macro,4))
    print("Weighted F1:", round(weighted_f1,4))
    print("Macro F1:", round(macro_f1,4))
    print("Cohen Kappa:", round(kappa,4))
    print("Hamming Loss:", round(hamming,4))
    print("ROC AUC:", round(roc_auc,4))
    print("Log Loss:", round(loss,4))

    return [
        model_name,
        accuracy,
        balanced_acc,
        precision_weighted,
        precision_macro,
        recall_weighted,
        recall_macro,
        weighted_f1,
        macro_f1,
        kappa,
        hamming,
        roc_auc,
        loss
    ]


# In[24]:


# ==================================
# RESULTS STORAGE
# ==================================

results = []



# In[27]:


# ==================================
# BOXPLOT
# ==================================

numeric_columns = df.select_dtypes(include=np.number).columns

plt.figure(figsize=(14,6))

df[numeric_columns].iloc[:, :20].boxplot(
    rot=90
)

plt.tight_layout()
plt.show()

plt.savefig(
    ARTIFACTS_DIR /
    "boxplot.png"
)

plt.close()


# In[28]:


# ==================================
# CORRELATION HEATMAP
# ==================================

corr = df[numeric_columns].corr()

plt.figure(figsize=(12,10))

sns.heatmap(
    corr,
    cmap="coolwarm",
    center=0
)

plt.tight_layout()
plt.show()

plt.savefig(
    ARTIFACTS_DIR /
    "correlation_heatmap.png"
)

plt.close()


# In[29]:


#===================
#RANDOM FOREST 
#===================

rf = RandomForestClassifier(
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        n_estimators=300,
        max_depth=20,
        min_samples_leaf=1,
        max_features="sqrt"
)

rf.fit(X_train, y_train)



#VALIDATION EVALUATION
# ==================================
# RF VALIDATION PERFORMANCE
# ==================================

evaluate_model(
    "Random Forest Validation",
    rf,
    X_val,
    y_val
)


# ==================================
# RF FINAL TEST EVALUATION
# ==================================
results.append(
    evaluate_model(
        "Random Forest",
        rf,
        X_test,
        y_test
    )
)


# In[30]:


#===================
#DECISION TREE 
#===================

dt = DecisionTreeClassifier(
        class_weight="balanced",
        random_state=42,
        max_depth=20,
        min_samples_leaf=1,
        criterion="entropy"
)

dt.fit(X_train, y_train)


evaluate_model(
    "Decision Tree Validation",
    dt,
    X_val,
    y_val
)

results.append(
    evaluate_model(
        "Decision Tree",
        dt,
        X_test,
        y_test
    )
)


# In[31]:


#===================
#LIGHT GBM 
#===================

lgb = LGBMClassifier(
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        n_estimators=100,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8
)

lgb.fit(X_train, y_train)


evaluate_model(
    "LightGBM Validation",
    lgb,
    X_val,
    y_val
)

results.append(
    evaluate_model(
        "LightGBM",
        lgb,
        X_test,
        y_test
    )
)


# In[33]:


#===================
#XGBOOST 
#===================

xgb = XGBClassifier(
        tree_method="hist",
        eval_metric="mlogloss",
        random_state=42,
        n_estimators=100,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8
)

xgb.fit(X_train, y_train)




evaluate_model(
    "XGBoost Validation",
    xgb,
    X_val,
    y_val
)

results.append(
    evaluate_model(
        "XGBoost",
        xgb,
        X_test,
        y_test
    )
)


# In[34]:


#===================
#LOGISTIC REGRESSION  
#===================
lr_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LogisticRegression(
        max_iter=5000,
        class_weight="balanced",
        random_state=42
    ))
])

lr_param_grid = {
    "lr__C": [0.01, 0.1, 1, 10],
    "lr__solver": ["lbfgs", "saga"]
}

lr_grid = GridSearchCV(
    estimator=lr_pipeline,
    param_grid=lr_param_grid,
    scoring="f1_weighted",
    cv=3,
    n_jobs=-1,
    verbose=2
)

lr_grid.fit(X_train, y_train)

print(lr_grid.best_params_)
print(lr_grid.best_score_)


lr_pipeline = lr_grid.best_estimator_

evaluate_model(
    "Logistic Regression Validation",
    lr_pipeline,
    X_val,
    y_val
)

results.append(
    evaluate_model(
        "Logistic Regression",
        lr_pipeline,
        X_test,
        y_test
    )
)


# In[35]:


# ==================================
# CHECK FOR OVERFITTING
# ==================================

training_scores = {}

testing_scores = {}

for name, model in {
    "Random Forest": rf,
    "Decision Tree": dt,
    "LightGBM": lgb,
    "XGBoost": xgb,
    "Logistic Regression": lr_pipeline
}.items():

    train_acc = model.score(X_train, y_train)

    test_acc = accuracy_score(
        y_test,
        model.predict(X_test)
    )

    training_scores[name] = train_acc
    testing_scores[name] = test_acc

dataset_quality["training_accuracy"] = training_scores
dataset_quality["testing_accuracy"] = testing_scores

print(dataset_quality)

dataset_quality["overfitting"] = {
    model: (
        training_scores[model]
        - testing_scores[model]
    )
    for model in training_scores
}



print("\nModel Accuracy and Overfitting Gap:")
print("-" * 60)

for model in training_scores:
    train = training_scores[model]
    test = testing_scores[model]
    gap = train - test

    print(
        f"{model:<20} "
        f"Train: {train:.2%} | "
        f"Test: {test:.2%} | "
        f"Gap: {gap:.2%}"
    )


# In[36]:


# ==================================
# MODEL COMPARISON TABLE
# ==================================

results_df = pd.DataFrame(
    results,
    columns=[
        "Model",
        "Accuracy",
        "Balanced_Accuracy",
        "Precision_Weighted",
        "Precision_Macro",
        "Recall_Weighted",
        "Recall_Macro",
        "Weighted_F1",
        "Macro_F1",
        "Cohen_Kappa",
        "Hamming_Loss",
        "ROC_AUC",
        "Log_Loss"
    ]
)

results_df = results_df.sort_values(
    by="Weighted_F1",
    ascending=False
)

results_df


# In[37]:


# ==================================
# MODEL COMPARISON CHART (Higher is Better) 
# ==================================

results_df.set_index("Model")[
    [
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
    ]
].plot(
    kind="bar",
    figsize=(12,6)
)

plt.title("Model Performance Comparison")
plt.ylabel("Score")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()


# In[38]:


# ==================================
# MODEL COMPARISON CHART (Lower is Better) 
# ==================================

results_df.set_index("Model")[
    [
        "Hamming_Loss",
        "Log_Loss"
    ]
].plot(
    kind="bar",
    figsize=(12,6)
)

plt.title("Model Performance Comparison")
plt.ylabel("Score")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()


# In[39]:


# ==================================
# CONFUSION MATRIX FUNCTION
# ==================================

def show_confusion_matrix(model_name, model, X_data, y_true):

    y_pred = model.predict(X_data)

    fig, ax = plt.subplots(figsize=(12, 10))

    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        xticks_rotation=90,
        ax=ax,
        cmap="Blues"
    )

    plt.title(f"{model_name} - Confusion Matrix")
    plt.tight_layout()
    plt.show()


# In[40]:


top_models = {
    "Decision Tree": dt,
    "Random Forest": rf,
    "XGBoost": xgb,
    "LightGBM": lgb,
    "Logistic Regression": lr_pipeline
}

for name, model in top_models.items():
    show_confusion_matrix(name, model, X_test, y_test)


# In[41]:


# ==================================
# FEATURE IMPORTANCE FUNCTION
# ==================================

def plot_feature_importance(model, model_name, feature_names):

    if hasattr(model, "named_steps"):
        model = model.named_steps[list(model.named_steps.keys())[-1]]

    # -----------------------------
    # Tree-based models
    # -----------------------------
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_

    # -----------------------------
    # Logistic Regression
    # -----------------------------
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])

    else:
        raise ValueError(f"{model_name} does not support feature importance")

    # -----------------------------
    # Build dataframe
    # -----------------------------
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    # -----------------------------
    # Display top features
    # -----------------------------
    print(f"\nTop 20 Features - {model_name}")
    print(importance_df.head(20))

    # -----------------------------
    # Plot
    # -----------------------------
    plt.figure(figsize=(12, 8))
    plt.barh(
        importance_df.head(20)["Feature"],
        importance_df.head(20)["Importance"]
    )
    plt.gca().invert_yaxis()
    plt.title(f"{model_name} Feature Importance")
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.show()

    return importance_df


# In[42]:


rf_importance = plot_feature_importance(
    rf,
    "Random Forest",
    X_train.columns
)


# In[43]:


dt_importance = plot_feature_importance(
    dt,
    "Decision Tree",
    X_train.columns
)


# In[44]:


lgb_importance = plot_feature_importance(
    lgb,
    "LightGBM",
    X_train.columns
)


# In[45]:


xgb_importance = plot_feature_importance(
    xgb,
    "XGBoost",
    X_train.columns
)


# In[46]:


lr_importance = plot_feature_importance(
    lr_pipeline,
    "Logistic Regression",
    X_train.columns
)


# In[47]:


feature_importance = {
    "Random Forest": rf_importance,
    "Decision Tree": dt_importance,
    "LightGBM": lgb_importance,
    "XGBoost": xgb_importance,
    "Logistic Regression": lr_importance
}


# In[48]:


rf_importance.to_csv(
    ARTIFACTS_DIR /
    "rf_feature_importance.csv",
    index=False
)

dt_importance.to_csv(
    ARTIFACTS_DIR /
    "dt_feature_importance.csv",
    index=False
)

lgb_importance.to_csv(
    ARTIFACTS_DIR /
    "lgb_feature_importance.csv",
    index=False
)

xgb_importance.to_csv(
    ARTIFACTS_DIR /
    "xgb_feature_importance.csv",
    index=False
)

lr_importance.to_csv(
    ARTIFACTS_DIR /
    "lr_feature_importance.csv",
    index=False
)


# In[49]:


sample = X_test.sample(
    n=5000,
    random_state=42
)


# In[50]:


# ==================================
# RANDOM FOREST SHAP
# ==================================

rf_explainer = shap.TreeExplainer(rf)

rf_shap_values = rf_explainer.shap_values(
    sample
)


# In[51]:


#RF SHAP SUMMARY PLOT 
shap.summary_plot(
    rf_shap_values,
    sample
)


# In[52]:


#RF SHAP BAR PLOT
shap.summary_plot(
    rf_shap_values,
    sample,
    plot_type="bar"
)


# In[53]:


# ==================================
# DECISION TREE SHAP
# ==================================

dt_explainer = shap.TreeExplainer(dt)

dt_shap_values = dt_explainer.shap_values(
    sample
)


# In[54]:


#DT SHAP SUMMARY PLOT 
shap.summary_plot(
    dt_shap_values,
    sample
)


# In[55]:


#DT SHAP BAR PLOT
shap.summary_plot(
    dt_shap_values,
    sample,
    plot_type="bar"
)


# In[56]:


# ==================================
# LIGHTGBM SHAP
# ==================================

lgb_explainer = shap.TreeExplainer(lgb)

lgb_shap_values = lgb_explainer.shap_values(
    sample
)


# In[57]:


#LGB SHAP SUMMARY PLOT 
shap.summary_plot(
    lgb_shap_values,
    sample
)


# In[58]:


#LGB SHAP BAR PLOT
shap.summary_plot(
    lgb_shap_values,
    sample,
    plot_type="bar"
)


# In[59]:


# ==================================
# XGBOOST SHAP
# ==================================

xgb_explainer = shap.TreeExplainer(xgb)

xgb_shap_values = xgb_explainer.shap_values(
    sample
)


# In[60]:


#XGB SHAP SUMMARY PLOT
shap.summary_plot(
    xgb_shap_values,
    sample
)


# In[61]:


#XGB SHAP BAR PLOT
shap.summary_plot(
    xgb_shap_values,
    sample,
    plot_type="bar"
)


# In[62]:


# ==================================
# LOGISTIC REGRESSION SHAP
# ==================================

background = X_train.sample(n=min(5000, len(X_train)), random_state=42)
sample = X_test.sample(n=min(1000, len(X_test)), random_state=42)

# pull out LR + scaler from the pipeline
scaler = lr_pipeline.named_steps["scaler"]
lr_model = lr_pipeline.named_steps["lr"]

# scale both background and sample
background_scaled = scaler.transform(background)
sample_scaled = scaler.transform(sample)

# explicit masker: keep all background rows
masker = shap.maskers.Independent(background_scaled, max_samples=len(background_scaled))

# SHAP for logistic regression
lr_explainer = shap.LinearExplainer(lr_model, masker)
lr_shap_values = lr_explainer(sample_scaled)

# summary plot
shap.summary_plot(lr_shap_values.values, sample, feature_names=sample.columns)


# In[63]:


#LR SHAP SUMMARY PLOT
shap.summary_plot(
    lr_shap_values,
    sample
)


# In[64]:


#LR SHAP BAR PLOT
shap.summary_plot(
    lr_shap_values,
    sample,
    plot_type="bar"
)


# In[65]:


#EXPLAIN RF PREDICTION 
row_number = 0
class_idx = 0


# explainer
rf_explainer = shap.TreeExplainer(rf)

# shap values 
rf_shap_values = rf_explainer.shap_values(sample)

single_row = sample.iloc[[row_idx]].values

# FORCE PLOT 
shap.initjs()

shap.force_plot(
    rf_explainer.expected_value[class_idx],
    rf_shap_values[row_idx, :, class_idx],
    single_row,
    matplotlib=True
)


# In[66]:


#EXPLAIN DT PREDICTION 

row_idx = 0
class_idx = 0

single_row = sample.iloc[[row_idx]].values

# =========================
# EXPLAINER
# =========================
dt_explainer = shap.TreeExplainer(dt)
dt_shap_values = dt_explainer.shap_values(sample)

# =========================
# FORCE PLOT 
# =========================
shap.initjs()

shap.plots.force(
    dt_explainer.expected_value[class_idx],
    dt_shap_values[row_idx, :, class_idx],
    single_row
)


# In[67]:


#EXPLAIN LGB PREDICTION 

row_idx = 0
class_idx = 0

single_row = sample.iloc[[row_idx]].values

# =========================
# EXPLAINER
# =========================
lgb_explainer = shap.TreeExplainer(lgb)
lgb_shap_values = lgb_explainer.shap_values(sample)

# =========================
# FORCE PLOT 
# =========================
shap.initjs()

shap.plots.force(
    lgb_explainer.expected_value[class_idx],
    lgb_shap_values[row_idx, :, class_idx],
    single_row
)


# In[68]:


#EXPLAIN XGB PREDICTION 

row_idx = 0
class_idx = 0

single_row = sample.iloc[[row_idx]].values

# =========================
# EXPLAINER
# =========================
xgb_explainer = shap.TreeExplainer(xgb)
xgb_shap_values = xgb_explainer.shap_values(sample)

# =========================
# FORCE PLOT 
# =========================
shap.initjs()

shap.plots.force(
    xgb_explainer.expected_value[class_idx],
    xgb_shap_values[row_idx, :, class_idx],
    single_row
)


# In[69]:


#EXPLAIN LR PREDICTION 

background = X_train.sample(
    n=min(5000, len(X_train)),
    random_state=42
).astype(np.float64)

X_explain = X_test.sample(
    n=min(5000, len(X_test)),
    random_state=42
).astype(np.float64)

# -------------------------
# wrap the pipeline in a callable
# -------------------------
def lr_predict_proba(data):
    return lr_pipeline.predict_proba(data)

# -------------------------
# explicit masker so SHAP uses all background rows
# -------------------------
masker = shap.maskers.Independent(background, max_samples=len(background))

# -------------------------
# SHAP explainer for LR
# -------------------------
lr_explainer = shap.Explainer(
    lr_predict_proba,
    masker,
    algorithm="permutation"
)

lr_shap_values = lr_explainer(X_explain)

# -------------------------
# explain one prediction
# -------------------------
row_idx = 0
pred_class = int(np.argmax(lr_predict_proba(X_explain.iloc[[row_idx]])[0]))

# single prediction
shap.plots.waterfall(lr_shap_values[row_idx, :, pred_class])

# global view for that class
shap.plots.beeswarm(lr_shap_values[:, :, pred_class])


shap.plots.force(lr_shap_values[row_idx, :, pred_class])


# In[71]:


# ==========================
# ORIGINAL DATASET INFORMATION
# ==========================

original_rows = len(raw_df)
original_columns = len(raw_df.columns)

original_missing_values = int(
    raw_df.isnull().sum().sum()
)

original_duplicate_rows = int(
    raw_df.duplicated().sum()
)


# In[72]:


# ==========================
# PREPARED DATASET INFORMATION
# ==========================

prepared_rows = len(df)
prepared_columns = len(df.columns)

prepared_missing_values = int(
    df.isnull().sum().sum()
)

prepared_duplicate_rows = int(
    df.duplicated().sum()
)

rows_removed = (
    original_rows - prepared_rows
)

columns_removed = (
    original_columns - prepared_columns
)

duplicates_removed = (
    original_duplicate_rows -
    prepared_duplicate_rows
)


# In[73]:


# ==================================
# DATASET SUMMARY
# ==================================

dataset_summary = {

    "rows": len(clean_df),

    "columns": len(clean_df.columns),

    "missing_values": int(
        clean_df.isnull().sum().sum()
    ),

    "duplicate_rows": int(
        clean_df.duplicated().sum()
    ),

    "class_count": int(
        clean_df["Attack_type"].nunique()
    ),

    "numeric_features": len(
        clean_df.select_dtypes(include=np.number).columns
    ),

    "categorical_features": len(
        clean_df.select_dtypes(exclude=np.number).columns
    ),

    "class_distribution":
        clean_df["Attack_type"]
        .value_counts()
        .to_dict()
}


# In[74]:


from pathlib import Path
import joblib
from sklearn.metrics import classification_report
import pandas as pd

PROJECT_DIR = Path.cwd()

DATASET_DIR = (
    PROJECT_DIR /
    "datasets" /
    "EdgeIIoT"
)

MODEL_DIR = DATASET_DIR / "models"
ARTIFACTS_DIR = DATASET_DIR / "artifacts"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)



joblib.dump(rf, MODEL_DIR / "random_forest.pkl")
joblib.dump(dt, MODEL_DIR / "decision_tree.pkl")
joblib.dump(lgb, MODEL_DIR / "lightgbm.pkl")
joblib.dump(xgb, MODEL_DIR / "xgboost.pkl")
joblib.dump(lr_pipeline, MODEL_DIR / "logistic_regression.pkl")



joblib.dump(
    target_encoder,
    ARTIFACTS_DIR / "target_encoder.pkl"
)

joblib.dump(
    list(X_train.columns),
    ARTIFACTS_DIR / "feature_names.pkl"
)

joblib.dump(
    drop_columns,
    ARTIFACTS_DIR / "drop_columns.pkl"
)



joblib.dump(
    X_test,
    ARTIFACTS_DIR / "eval_sample.pkl"
)

joblib.dump(
    y_test,
    ARTIFACTS_DIR / "eval_labels.pkl"
)

joblib.dump(
    dataset_summary,
    ARTIFACTS_DIR / "dataset_summary.pkl"
)



results_df.to_csv(
    ARTIFACTS_DIR / "results.csv",
    index=False
)



from sklearn.metrics import confusion_matrix


# ==================================
# SAVE CONFUSION MATRICES
# ==================================

confusion_matrices = {}

for name, model in top_models.items():

    predictions = model.predict(X_test)

    cm = confusion_matrix(
        y_test,
        predictions
    )

    confusion_matrices[name] = cm


joblib.dump(
    confusion_matrices,
    ARTIFACTS_DIR / "confusion_matrices.pkl"
)



classification_reports = {}

for name, model in top_models.items():

    pred = model.predict(X_test)

    classification_reports[name] = (
        classification_report(
            y_test,
            pred,
            target_names=target_encoder.classes_,
            output_dict=True,
            zero_division=0
        )
    )

joblib.dump(
    classification_reports,
    ARTIFACTS_DIR /
    "classification_reports.pkl"
)


# In[75]:


joblib.dump(dataset_quality,
            ARTIFACTS_DIR / "dataset_quality.pkl")

joblib.dump(df,
            ARTIFACTS_DIR / "clean_dataset.pkl")

joblib.dump(df.describe(),
            ARTIFACTS_DIR / "dataset_summary.pkl")

joblib.dump(df.corr(numeric_only=True),
            ARTIFACTS_DIR / "correlation_matrix.pkl")

joblib.dump(df["Attack_type"].value_counts(),
            ARTIFACTS_DIR / "class_distribution.pkl")

joblib.dump(feature_importance,
            ARTIFACTS_DIR / "feature_importance.pkl")


# In[77]:


corr.to_csv(

    ARTIFACTS_DIR /

    "correlation.csv"

)


# In[78]:


# ==================================
# OVERFITTING METRICS
# ==================================

overfitting = {}

for name, model in top_models.items():

    train_acc = accuracy_score(
        y_train,
        model.predict(X_train)
    )

    test_acc = accuracy_score(
        y_test,
        model.predict(X_test)
    )

    overfitting[name] = {

        "Training Accuracy": train_acc,

        "Testing Accuracy": test_acc,

        "Difference": train_acc - test_acc

    }

joblib.dump(

    overfitting,

    ARTIFACTS_DIR /

    "overfitting.pkl"

)


# In[79]:


EDA_DIR = ARTIFACTS_DIR / "eda"

EDA_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# Missing values

missing_values = (
    df.isnull()
    .sum()
    .sort_values(
        ascending=False
    )
)


joblib.dump(
    missing_values,
    EDA_DIR / "missing_values.pkl"
)



# Dataset information

dataset_info = {

    "rows": df.shape[0],

    "columns": df.shape[1],

    "duplicates":
        df.duplicated().sum(),

    "classes":
        target_encoder.classes_.tolist()

}


joblib.dump(
    dataset_info,
    EDA_DIR / "dataset_info.pkl"
)



# Feature statistics

feature_statistics = (
    df.describe()
)

joblib.dump(
    feature_statistics,
    EDA_DIR / "feature_statistics.pkl"
)



# Final selected features

joblib.dump(
    list(X_train.columns),
    EDA_DIR / "selected_features.pkl"
)


# In[80]:


dataset_quality_summary = {

    "original_rows":
        original_rows,

    "original_columns":
        original_columns,

    "original_missing_values":
        original_missing_values,

    "original_duplicate_rows":
        original_duplicate_rows,

    "prepared_rows":
        prepared_rows,

    "prepared_columns":
        prepared_columns,

    "prepared_missing_values":
        prepared_missing_values,

    "prepared_duplicate_rows":
        prepared_duplicate_rows,

    "rows_removed":
        rows_removed,

    "columns_removed":
        columns_removed,

    "duplicates_removed":
        duplicates_removed,

}



joblib.dump(
    dataset_quality_summary,
    EDA_DIR /
    "dataset_quality_summary.pkl"
)


# In[81]:


training_scores={}

for name,model in top_models.items():

    train_pred=model.predict(X_train)

    val_pred=model.predict(X_val)


    training_scores[name]={

        "train_accuracy":
        accuracy_score(
            y_train,
            train_pred
        ),

        "validation_accuracy":
        accuracy_score(
            y_val,
            val_pred
        )

    }


joblib.dump(
    training_scores,
    EDA_DIR /
    "training_validation_scores.pkl"
)


# In[82]:


# ==================================
# MODEL GENERALISATION ANALYSIS
# ==================================

generalization_results = {}

for name, model in top_models.items():

    train_pred = model.predict(
        X_train
    )

    val_pred = model.predict(
        X_val
    )

    test_pred = model.predict(
        X_test
    )


    # Accuracy

    train_accuracy = accuracy_score(
        y_train,
        train_pred
    )

    validation_accuracy = accuracy_score(
        y_val,
        val_pred
    )

    test_accuracy = accuracy_score(
        y_test,
        test_pred
    )


    # Weighted F1

    train_f1 = f1_score(
        y_train,
        train_pred,
        average="weighted",
        zero_division=0
    )

    validation_f1 = f1_score(
        y_val,
        val_pred,
        average="weighted",
        zero_division=0
    )

    test_f1 = f1_score(
        y_test,
        test_pred,
        average="weighted",
        zero_division=0
    )


    generalization_results[name] = {

        "train_accuracy":
            train_accuracy,

        "validation_accuracy":
            validation_accuracy,

        "test_accuracy":
            test_accuracy,

        "train_validation_accuracy_gap":
            train_accuracy -
            validation_accuracy,

        "train_test_accuracy_gap":
            train_accuracy -
            test_accuracy,

        "train_weighted_f1":
            train_f1,

        "validation_weighted_f1":
            validation_f1,

        "test_weighted_f1":
            test_f1,

        "train_validation_f1_gap":
            train_f1 -
            validation_f1,

        "train_test_f1_gap":
            train_f1 -
            test_f1
    }


joblib.dump(
    generalization_results,
    EDA_DIR /
    "generalization_results.pkl"
)


# In[83]:


numeric_features=df.select_dtypes(
include=np.number
)


joblib.dump(
numeric_features,
EDA_DIR/"numeric_features.pkl"
)


# In[84]:


# ==================================
# CLASS DISTRIBUTION
# ==================================

class_distribution = (
    df["Attack_type"]
    .value_counts()
)

joblib.dump(
    class_distribution,
    EDA_DIR /
    "class_distribution.pkl"
)


# In[85]:


correlation_matrix = (
    df.corr(
        numeric_only=True
    )
)

joblib.dump(
    correlation_matrix,
    EDA_DIR /
    "correlation_matrix.pkl"
)


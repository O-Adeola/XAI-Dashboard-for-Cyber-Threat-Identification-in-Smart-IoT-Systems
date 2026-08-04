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
from sklearn.preprocessing import LabelEncoder, StandardScaler
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
df = pd.read_csv("C:\\Users\\User\\Downloads\\ML-EdgeIIoT-dataset.csv\\ML-EdgeIIoT-dataset.csv", low_memory=False)

df.head()


# In[3]:


#===================
#INSPECT DATASET 
#===================
print("Shape:", df.shape)
print(df.columns)
print(df.info())
print(df["Attack_type"].value_counts())


# In[4]:


#===================
#CLEAN DATASET  
#===================
drop_columns = [
    "frame.time", "ip.src_host", "ip.dst_host", "arp.src.proto_ipv4",
    "arp.dst.proto_ipv4", "http.file_data", "http.request.full_uri",
    "icmp.transmit_timestamp", "http.request.uri.query", "tcp.options",
    "tcp.payload", "tcp.srcport", "tcp.dstport", "udp.port", "mqtt.msg"
]

drop_columns = [c for c in drop_columns if c in df.columns]
df = df.drop(columns=drop_columns)


df = df.dropna(axis=0, how="any")
df = df.drop_duplicates()
df = shuffle(df, random_state=42).reset_index(drop=True)

print("Shape after cleaning:", df.shape)
print(df["Attack_type"].value_counts())






# df.isnull().sum() 

# df.isnull().sum().sum() 

# print(df["Attack_type"].isna().sum()) 

# df = df.dropna(subset=["Attack_type"])



# df.replace([np.inf, -np.inf], np.nan, inplace=True) 

# df.fillna(0, inplace=True)


# In[5]:


# ==================================
# SAVE CLEAN DATASET FOR DASHBOARD
# ==================================

clean_df = df.copy()


# In[88]:


# ==================================
# DATASET QUALITY
# ==================================

missing_values = df.isnull().sum()

missing_summary = pd.DataFrame({
    "Feature": missing_values.index,
    "Missing Values": missing_values.values,
    "Percent Missing":
        (missing_values.values / len(df)) * 100
})

print(missing_summary)


# In[7]:


#===================
#ENCODE CATEGORICAL COLUMNS 
#===================
def encode_text_dummy(df, name):
    if name in df.columns:
        dummies = pd.get_dummies(df[name], prefix=name)
        df = pd.concat([df.drop(columns=[name]), dummies], axis=1)
    return df

categorical_to_encode = [
    "http.request.method",
    "http.referer",
    "http.request.version",
    "dns.qry.name.len",
    "mqtt.conack.flags",
    "mqtt.protoname",
    "mqtt.topic"
]

for col in categorical_to_encode:
    df = encode_text_dummy(df, col)

print("Shape after dummy encoding:", df.shape)


# In[8]:


# target_col = "Attack_type"

# X = df.drop(columns=[target_col])
# y = df[target_col]#.astype(str)

# le = LabelEncoder()
# y_enc = le.fit_transform(y)

# print("Classes:", list(le.classes_))
# print("X shape:", X.shape)


# In[86]:


#===================
#LABEL DISTRIBUTION
#===================
df['Attack_type'].value_counts() 

plt.figure(figsize=(12,6))

df['Attack_type'].value_counts().plot(
    kind='bar'
)

plt.title("Attack Class Distribution")
# plt.show()


plt.savefig(
    ARTIFACTS_DIR /
    "attack_distribution.png"
)

plt.show()


# In[10]:


#DROP IP ADDRESSES 
# drop_cols = ["src_ip", 
#              "dst_ip", 
#              "dns_query", 
#              "ssl_subject", 
#              "ssl_issuer", 
#              "http_uri", 
#              "http_user_agent", 
#              "http_orig_mime_types", 
#              "http_resp_mime_types"
#             ]

# df = df.drop(columns=drop_cols, errors="ignore")


# In[11]:


# ===================
# SPLIT FEATURES/TARGET 
# TRAIN-VALIDATE-TEST 
# ===================

df = df.dropna(subset=["Attack_type"])

X = df.drop(columns=["Attack_type"])
y = df["Attack_type"]

target_encoder = LabelEncoder()
y_enc = target_encoder.fit_transform(y)

print(target_encoder.classes_)


print(np.unique(y))


# In[12]:


import re

df.columns = df.columns.astype(str).str.replace(
    r"[^0-9a-zA-Z_]", "_", regex=True
)


# In[13]:


X = df.drop(columns=["Attack_type"])
y = df["Attack_type"]


# In[14]:


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


# In[15]:


X_train = X_train.copy()
X_val = X_val.copy()
X_test = X_test.copy()


# In[16]:


# # ==================================
# # FEATURE (ORDINAL) ENCODING
# # ==================================

# cat_cols = X_train.select_dtypes(
#     include=["object", "string"]
# ).columns

# encoder = OrdinalEncoder(
#     handle_unknown="use_encoded_value",
#     unknown_value=-1
# )

# X_train[cat_cols] = encoder.fit_transform(
#     X_train[cat_cols]
# )

# X_val[cat_cols] = encoder.transform(
#     X_val[cat_cols]
# )

# X_test[cat_cols] = encoder.transform(
#     X_test[cat_cols]
# )


# In[17]:


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


# In[18]:


# ==================================
# RESULTS STORAGE
# ==================================

results = []


# In[19]:


# clean_df.to_csv(
#     "datasets/EdgeIIoT/artifacts/clean_dataset.csv",
#     index=False
# )


# In[20]:


# training_summary = pd.DataFrame({
#     "Train Accuracy":[train_accuracy],
#     "Test Accuracy":[test_accuracy],
#     "Overfitting":[train_accuracy-test_accuracy]
# })

# training_summary.to_csv(
#     "datasets/EdgeIIoT/artifacts/training_summary.csv",
#     index=False
# )


# In[21]:


# ==================================
# BOXPLOT
# ==================================

numeric_columns = df.select_dtypes(include=np.number).columns

plt.figure(figsize=(14,6))

df[numeric_columns].iloc[:, :20].boxplot(
    rot=90
)

plt.tight_layout()

plt.savefig(
    ARTIFACTS_DIR /
    "boxplot.png"
)

plt.close()


# In[22]:


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

plt.savefig(
    ARTIFACTS_DIR /
    "correlation_heatmap.png"
)

plt.close()


# In[23]:


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


# In[24]:


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


# In[25]:


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


# In[26]:


# # ===================
# # SPLIT FEATURES/TARGET 
# # TRAIN-VALIDATE-TEST 
# # ===================

# df = df.dropna(subset=["Attack_type"])

# X = df.drop(columns=["Attack_type"])
# y = df["Attack_type"]

# target_encoder = LabelEncoder()
# y_enc = target_encoder.fit_transform(y_train)

# print(target_encoder.classes_)


# print(np.unique(y))


# In[27]:


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


# In[28]:


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


# In[29]:


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


# In[30]:


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


# In[31]:


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


# In[32]:


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


# In[33]:


top_models = {
    "Decision Tree": dt,
    "Random Forest": rf,
    "XGBoost": xgb,
    "LightGBM": lgb,
    "Logistic Regression": lr_pipeline
}

for name, model in top_models.items():
    show_confusion_matrix(name, model, X_test, y_test)


# In[34]:


# ==================================
# FEATURE IMPORTANCE FUNCTION
# ==================================

def plot_feature_importance(model, model_name, feature_names):

    # -----------------------------
    # Extract actual estimator if pipeline
    # -----------------------------
    if hasattr(model, "named_steps"):
        model = model.named_steps[list(model.named_steps.keys())[-1]]

    # -----------------------------
    # Tree-based models
    # -----------------------------
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_

    # -----------------------------
    # Linear models (Logistic Regression, SVM linear)
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


# In[35]:


rf_importance = plot_feature_importance(
    rf,
    "Random Forest",
    X_train.columns
)


# In[36]:


dt_importance = plot_feature_importance(
    dt,
    "Decision Tree",
    X_train.columns
)


# In[37]:


lgb_importance = plot_feature_importance(
    lgb,
    "LightGBM",
    X_train.columns
)


# In[38]:


xgb_importance = plot_feature_importance(
    xgb,
    "XGBoost",
    X_train.columns
)


# In[39]:


lr_importance = plot_feature_importance(
    lr_pipeline,
    "Logistic Regression",
    X_train.columns
)


# In[40]:


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


# In[41]:


sample = X_test.sample(
    n=5000,
    random_state=42
)


# In[42]:


# ==================================
# RANDOM FOREST SHAP
# ==================================

rf_explainer = shap.TreeExplainer(rf)

rf_shap_values = rf_explainer.shap_values(
    sample
)


# In[43]:


#RF SHAP SUMMARY PLOT 
shap.summary_plot(
    rf_shap_values,
    sample
)


# In[44]:


#RF SHAP BAR PLOT
shap.summary_plot(
    rf_shap_values,
    sample,
    plot_type="bar"
)


# In[45]:


# ==================================
# DECISION TREE SHAP
# ==================================

dt_explainer = shap.TreeExplainer(dt)

dt_shap_values = dt_explainer.shap_values(
    sample
)


# In[46]:


#DT SHAP SUMMARY PLOT 
shap.summary_plot(
    dt_shap_values,
    sample
)


# In[47]:


#DT SHAP BAR PLOT
shap.summary_plot(
    dt_shap_values,
    sample,
    plot_type="bar"
)


# In[48]:


# ==================================
# LIGHTGBM SHAP
# ==================================

lgb_explainer = shap.TreeExplainer(lgb)

lgb_shap_values = lgb_explainer.shap_values(
    sample
)


# In[49]:


#LGB SHAP SUMMARY PLOT 
shap.summary_plot(
    lgb_shap_values,
    sample
)


# In[50]:


#LGB SHAP BAR PLOT
shap.summary_plot(
    lgb_shap_values,
    sample,
    plot_type="bar"
)


# In[51]:


# ==================================
# XGBOOST SHAP
# ==================================

xgb_explainer = shap.TreeExplainer(xgb)

xgb_shap_values = xgb_explainer.shap_values(
    sample
)


# In[52]:


#XGB SHAP SUMMARY PLOT
shap.summary_plot(
    xgb_shap_values,
    sample
)


# In[53]:


#XGB SHAP BAR PLOT
shap.summary_plot(
    xgb_shap_values,
    sample,
    plot_type="bar"
)


# In[54]:


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


# In[55]:


#LR SHAP SUMMARY PLOT
shap.summary_plot(
    lr_shap_values,
    sample
)


# In[56]:


#LR SHAP BAR PLOT
shap.summary_plot(
    lr_shap_values,
    sample,
    plot_type="bar"
)


# In[57]:


#EXPLAIN RF PREDICTION 
row_number = 0
class_idx = 0


# explainer
rf_explainer = shap.TreeExplainer(rf)

# shap values (your shape: samples, features, classes)
rf_shap_values = rf_explainer.shap_values(sample)

# pick instance + class
row_idx = 0
class_idx = 0

# IMPORTANT: make row 2D
single_row = sample.iloc[[row_idx]].values

# FORCE PLOT 
shap.initjs()

shap.force_plot(
    rf_explainer.expected_value[class_idx],
    rf_shap_values[row_idx, :, class_idx],
    single_row,
    matplotlib=True
)


# In[58]:


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
# FORCE PLOT (correct indexing)
# =========================
shap.initjs()

shap.plots.force(
    dt_explainer.expected_value[class_idx],
    dt_shap_values[row_idx, :, class_idx],
    single_row
)


# In[59]:


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


# In[60]:


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
# FORCE PLOT (correct indexing)
# =========================
shap.initjs()

shap.plots.force(
    xgb_explainer.expected_value[class_idx],
    xgb_shap_values[row_idx, :, class_idx],
    single_row
)


# In[62]:


#EXPLAIN LR PREDICTION 

# background = X_train.sample(n=min(5000, len(X_train)), random_state=42)
# X_explain = X_test.sample(n=min(5000, len(X_test)), random_state=42)


# background = background.astype(np.float64)
# X_explain = X_explain.astype(np.float64)


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


# In[63]:


# from pathlib import Path

# # ARTIFACTS_DIR = Path("artifacts/edgeiiot")
# # MODEL_DIR = Path("models/edgeiiot")
# # ARTIFACTS_DIR.mkdir(exist_ok=True)
# # MODEL_DIR.mkdir(exist_ok=True)


# PROJECT_DIR = Path.cwd().resolve()
# if PROJECT_DIR.name.lower() == "notebooks":
#     PROJECT_DIR = PROJECT_DIR.parent

# MODEL_DIR = PROJECT_DIR / "datasets" / "edgeiot" / "models"
# ARTIFACTS_DIR = PROJECT_DIR / "datasets" / "edgeiot" / "artifacts"

# MODEL_DIR.mkdir(parents=True, exist_ok=True)
# ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)



# # Save trained models
# joblib.dump(rf, MODEL_DIR / "random_forest.pkl")
# joblib.dump(dt, MODEL_DIR / "decision_tree.pkl")
# joblib.dump(lgb, MODEL_DIR / "lightgbm.pkl")
# joblib.dump(xgb, MODEL_DIR / "xgboost.pkl")
# joblib.dump(lr_pipeline, MODEL_DIR / "logistic_regression.pkl")

# # Save encoders and metadata
# # joblib.dump(encoder, ARTIFACTS_DIR / "feature_encoder.pkl")
# # joblib.dump(target_encoder, ARTIFACTS_DIR / "target_encoder.pkl")
# # joblib.dump(list(X_train.columns), ARTIFACTS_DIR / "feature_names.pkl")
# # joblib.dump(list(cat_cols), ARTIFACTS_DIR / "categorical_columns.pkl")
# # joblib.dump(drop_cols, ARTIFACTS_DIR / "drop_columns.pkl")

# joblib.dump(target_encoder, ARTIFACTS_DIR / "target_encoder.pkl")
# joblib.dump(list(X_train.columns), ARTIFACTS_DIR / "feature_names.pkl")
# joblib.dump(drop_columns, ARTIFACTS_DIR / "drop_columns.pkl")
# joblib.dump(categorical_to_encode, ARTIFACTS_DIR / "categorical_columns.pkl")


# # Save results table
# results_df.to_csv(ARTIFACTS_DIR / "results.csv", index=False)

# # Save evaluation sample for confusion matrices and SHAP
# # eval_sample = X_test.sample(n=min(5000, len(X_test)), random_state=42)
# # eval_labels = y_test.loc[eval_sample.index].to_numpy()


# # sample_idx = np.random.RandomState(42).choice(
# #     len(X_test),
# #     size=min(5000, len(X_test)),
# #     replace=False
# # )

# # eval_sample = X_test.iloc[sample_idx]
# # eval_labels = y_test[sample_idx]


# eval_sample = X_test.sample(
#     n=min(5000, len(X_test)),
#     random_state=42
# )

# joblib.dump(
#     eval_sample,
#     ARTIFACTS_DIR / "eval_sample.pkl"
# )

# joblib.dump(
#     y_test.loc[eval_sample.index]
#     if hasattr(y_test, "loc")
#     else y_test[:len(eval_sample)],
#     ARTIFACTS_DIR / "eval_labels.pkl"
# )


# # eval_sample.to_csv(ARTIFACTS_DIR / "eval_sample.csv", index=False)
# # joblib.dump(eval_labels, ARTIFACTS_DIR / "eval_labels.pkl")


# reports = {}

# for name, model in top_models.items():

#     preds = model.predict(X_test)

#     reports[name] = classification_report(
#         y_test,
#         preds,
#         target_names=target_encoder.classes_,
#         output_dict=True,
#         zero_division=0
#     )

# joblib.dump(
#     reports,
#     ARTIFACTS_DIR / "classification_reports.pkl"
# )


# In[64]:


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


# In[83]:


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
    list(categorical_to_encode),
    ARTIFACTS_DIR / "categorical_columns.pkl"
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


# In[66]:


missing_summary.to_csv(

    ARTIFACTS_DIR /

    "missing_values.csv",

    index=False

)


# In[67]:


corr.to_csv(

    ARTIFACTS_DIR /

    "correlation.csv"

)


# In[68]:


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


# In[69]:


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


# In[70]:


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


# In[71]:


numeric_features=df.select_dtypes(
include=np.number
)


joblib.dump(
numeric_features,
EDA_DIR/"numeric_features.pkl"
)


# In[ ]:





# In[ ]:





# In[72]:


#===================
#FEATURE IMPORTANCE (QUICK INSIGHT) 
#===================

plt.figure(figsize=(10,6))
plt.bar(range(len(xgb.feature_importances_)), xgb.feature_importances_)
plt.title("Feature Importance (XGBoost)")
plt.show()


# In[73]:


predictedValue = rf.predict(X_test)
print(predictedValue)


# In[74]:


rf_preds = rf.predict(X_test)

comparison_df = pd.DataFrame({
    "Actual": target_encoder.inverse_transform(y_test),
    "Predicted": target_encoder.inverse_transform(rf_preds)
})

comparison_df.head(20)


# In[ ]:





# In[ ]:





# In[75]:


import os

print(os.getcwd())


# In[76]:


import os

print(os.path.exists("models"))
print(os.path.exists("artifacts"))


# In[77]:


os.listdir("models")


# In[78]:


os.listdir("artifacts")


# In[ ]:





# In[79]:


import streamlit as st

st.sidebar.write("BASE_DIR:", BASE_DIR)
st.sidebar.write("ASSETS_DIR:", ASSETS_DIR)
st.sidebar.write(DATASETS[dataset]["models_dir"])
st.sidebar.write(DATASETS[dataset]["artifacts_dir"])


# In[ ]:





# In[ ]:


from pathlib import Path
import os

# Change this to the correct dataset folder if needed
BASE_DIR = Path.cwd()

print("=" * 80)
print("CURRENT WORKING DIRECTORY")
print("=" * 80)
print(BASE_DIR)

print()

# Search for every folder called artifacts or models
for root, dirs, files in os.walk(BASE_DIR):

    folder_name = Path(root).name.lower()

    if folder_name in ["artifacts", "models"]:

        print("=" * 80)
        print(f"FOLDER : {root}")
        print("=" * 80)

        if len(files) == 0:
            print("No files found.")

        for f in sorted(files):

            path = Path(root) / f

            try:
                size = round(path.stat().st_size / 1024, 2)
            except:
                size = "Unknown"

            print(f"{f:<40} {size} KB")

        print()


# In[ ]:


import pandas as pd
from pathlib import Path

BASE_DIR = Path.cwd()

results_files = list(BASE_DIR.rglob("results.csv"))

print("=" * 80)
print("RESULTS FILES FOUND")
print("=" * 80)

for file in results_files:

    print("\n")
    print(file)

    df = pd.read_csv(file)

    print("\nColumns:")

    for c in df.columns:
        print("-", c)

    print("\nFirst five rows:")

    display(df.head())


# In[ ]:


from pathlib import Path

BASE_DIR = Path.cwd()

print("=" * 80)
print("FEATURE FILES")
print("=" * 80)

keywords = [

    "feature",

    "importance",

    "selected",

    "ranking"

]

for file in BASE_DIR.rglob("*"):

    if file.is_file():

        name = file.name.lower()

        if any(k in name for k in keywords):

            print(file)


#!/usr/bin/env python
# coding: utf-8

# In[1]:


#===================
#IMPORTS 
#===================
import pandas as pd 
import numpy as np 

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, OrdinalEncoder
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
    "TON_IoT"
)

MODEL_DIR = DATASET_DIR / "models"
ARTIFACTS_DIR = DATASET_DIR / "artifacts"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


pd.set_option('display.max_columns', None)


# In[2]:


#===================
#LOAD DATASETS 
#===================
df = pd.read_csv("C:\\Users\\User\\Downloads\\Train_Test_Network_dataset\\Train_Test_Network_dataset\\train_test_network.csv")

df.head()


# In[3]:


#===================
#MISSING VALUES ANALYSIS 
#===================
df.isnull().sum() 

df.isnull().sum().sum() 

print(df["type"].isna().sum()) 

df = df.dropna(subset=["type"])



df.replace([np.inf, -np.inf], np.nan, inplace=True) 

df.fillna(0, inplace=True)


# In[4]:


# ==================================
# DATASET QUALITY INFORMATION
# ==================================

missing_before = df.isnull().sum().sum()

dataset_quality = {
    "rows": len(df),
    "columns": len(df.columns),
    "missing_values": int(missing_before),
}


# In[5]:


#===================
#LABEL DISTRIBUTION
#===================
df['type'].value_counts() 

plt.figure(figsize=(12,6))

df['type'].value_counts().plot(
    kind='bar'
)

plt.title("Attack Class Distribution")
plt.show()


# In[6]:


#DROP COLUMNS 
drop_cols = ["src_ip", 
             "dst_ip", 
             "dns_query", 
             "ssl_subject", 
             "ssl_issuer", 
             "http_uri", 
             "http_user_agent", 
             "http_orig_mime_types", 
             "http_resp_mime_types"
            ]

df = df.drop(columns=drop_cols, errors="ignore")


# In[7]:


dataset_quality["features_after_preprocessing"] = len(df.columns)
dataset_quality["removed_columns"] = drop_cols


# In[8]:


#===================
#SPLIT FEATURES/TARGET 
#TRAIN-VALIDATE-TEST 
#===================
X = df.drop(['label', 'type'], axis=1)
y = df['type'] 



target_encoder = LabelEncoder()
y = target_encoder.fit_transform(y)


print(np.unique(y))


X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.4,
    random_state=42,
    stratify=y
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


# In[9]:


X_train = X_train.copy()
X_val = X_val.copy()
X_test = X_test.copy()


# In[10]:


# ==================================
# FEATURE (ORDINAL) ENCODING
# ==================================

cat_cols = X_train.select_dtypes(
    include=["object", "string"]
).columns

encoder = OrdinalEncoder(
    handle_unknown="use_encoded_value",
    unknown_value=-1
)

X_train[cat_cols] = encoder.fit_transform(
    X_train[cat_cols]
)

X_val[cat_cols] = encoder.transform(
    X_val[cat_cols]
)

X_test[cat_cols] = encoder.transform(
    X_test[cat_cols]
)


# In[11]:


dataset_quality["categorical_features"] = list(cat_cols)
dataset_quality["selected_features"] = list(X_train.columns)


# In[12]:


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


# In[13]:


# ==================================
# RESULTS STORAGE
# ==================================

results = []


# In[14]:


# clean_df.to_csv(
#     "datasets/TON_IoT/artifacts/clean_dataset.csv",
#     index=False
# )


# In[15]:


# training_summary = pd.DataFrame({
#     "Train Accuracy":[train_accuracy],
#     "Test Accuracy":[test_accuracy],
#     "Overfitting":[train_accuracy-test_accuracy]
# })

# training_summary.to_csv(
#     "datasets/TON_IoT/artifacts/training_summary.csv",
#     index=False
# )


# In[16]:


#===================
#RANDOM FOREST 
#===================

rf = RandomForestClassifier(
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        n_estimators=200,
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


# In[17]:


#===================
#DECISION TREE 
#===================

dt = DecisionTreeClassifier(
        class_weight="balanced",
        random_state=42,
        max_depth=20,
        min_samples_leaf=1,
        criterion="gini"
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


# In[18]:


#===================
#LIGHT GBM 
#===================

lgb = LGBMClassifier(
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        max_depth=-1,
        n_estimators=500,
        learning_rate=0.01,
        num_leaves=64,
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


# In[19]:


#===================
#XGBOOST 
#===================

xgb = XGBClassifier(
        tree_method="hist",
        eval_metric="mlogloss",
        random_state=42,
        n_estimators=300,
        max_depth=8,
        learning_rate=0.05,
        subsample=1.0,
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


# In[20]:


#===================
#LOGISTIC REGRESSION  
#===================

lr_pipeline = Pipeline([
    (
        "scaler",
        StandardScaler()
    ),
    (
        "lr",
        LogisticRegression(
            C=10,
            solver="lbfgs",
            max_iter=5000,
            class_weight="balanced",
            random_state=42
        )
    )
])




lr_pipeline.fit(X_train, y_train)



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


# In[21]:


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

dataset_quality["overfitting"] = {
    model: (
        training_scores[model]
        - testing_scores[model]
    )
    for model in training_scores
}


# In[22]:


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


# In[23]:


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


# In[24]:


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


# In[25]:


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


# In[26]:


top_models = {
    "Decision Tree": dt,
    "Random Forest": rf,
    "XGBoost": xgb,
    "LightGBM": lgb,
    "Logistic Regression": lr_pipeline
}

for name, model in top_models.items():
    show_confusion_matrix(name, model, X_test, y_test)


# In[27]:


# ==================================
# FEATURE IMPORTANCE FUNCTION
# ==================================

# def plot_feature_importance(model, model_name):

#     importance_df = pd.DataFrame({
#         "Feature": X_train.columns,
#         "Importance": model.feature_importances_
#     })

#     importance_df = importance_df.sort_values(
#         by="Importance",
#         ascending=False
#     )




#     lr_importance = pd.DataFrame({
#         "Feature": X_train.columns,
#         "Importance": np.abs(lr_model.named_steps["lr"].coef_[0])
#     }).sort_values(by="Importance", ascending=False)

#     lr_importance.head(20)






#     print(f"\nTop 20 Features - {model_name}")
#     print(importance_df.head(20))

#     plt.figure(figsize=(12,8))

#     plt.barh(
#         importance_df.head(20)["Feature"],
#         importance_df.head(20)["Importance"]
#     )

#     plt.gca().invert_yaxis()

#     plt.title(
#         f"{model_name} Feature Importance"
#     )

#     plt.xlabel("Importance Score")

#     plt.tight_layout()
#     plt.show()

#     return importance_df






# def plot_feature_importance(model, model_name, feature_names):

#     # CASE 1: Tree-based models
#     if hasattr(model, "feature_importances_"):
#         importance = model.feature_importances_

#     # CASE 2: Linear models (Logistic Regression, SVM linear, etc.)
#     elif hasattr(model, "coef_"):
#         importance = np.abs(model.coef_).mean(axis=0)

#     # CASE 3: Pipeline (extract final estimator)
#     elif hasattr(model, "named_steps"):
#         # try common step names
#         final_estimator = list(model.named_steps.values())[-1]

#         if hasattr(final_estimator, "feature_importances_"):
#             importance = final_estimator.feature_importances_

#         elif hasattr(final_estimator, "coef_"):
#             importance = np.abs(final_estimator.coef_).mean(axis=0)

#         else:
#             raise ValueError("Model type not supported for feature importance")

#     else:
#         raise ValueError("Model does not support feature importance")

#     importance_df = pd.DataFrame({
#         "Feature": feature_names,
#         "Importance": importance
#     })

#     importance_df = importance_df.sort_values(by="Importance", ascending=False)

#     print(f"\nTop 20 Features - {model_name}")
#     print(importance_df.head(20))

#     plt.figure(figsize=(12, 8))

#     plt.barh(
#         importance_df.head(20)["Feature"],
#         importance_df.head(20)["Importance"]
#     )

#     plt.gca().invert_yaxis()
#     plt.title(f"{model_name} Feature Importance")
#     plt.xlabel("Importance Score")
#     plt.tight_layout()
#     plt.show()

#     return importance_df




# def plot_feature_importance(model, model_name, feature_names):

#     # extract model if it's a pipeline
#     if hasattr(model, "named_steps"):
#         model = model.named_steps[list(model.named_steps.keys())[-1]]

#     # check correct attribute
#     if not hasattr(model, "feature_importances_"):
#         raise ValueError(f"{model_name} does not support feature_importances_")

#     importance_df = pd.DataFrame({
#         "Feature": feature_names,
#         "Importance": model.feature_importances_
#     })

#     importance_df = importance_df.sort_values(by="Importance", ascending=False)

#     print(f"\nTop 20 Features - {model_name}")
#     print(importance_df.head(20))

#     plt.figure(figsize=(12, 8))
#     plt.barh(
#         importance_df.head(20)["Feature"],
#         importance_df.head(20)["Importance"]
#     )

#     plt.gca().invert_yaxis()
#     plt.title(f"{model_name} Feature Importance")
#     plt.xlabel("Importance Score")
#     plt.tight_layout()
#     plt.show()

#     return importance_df





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


# In[28]:


rf_importance = plot_feature_importance(
    rf,
    "Random Forest",
    X_train.columns
)


# In[29]:


dt_importance = plot_feature_importance(
    dt,
    "Decision Tree",
    X_train.columns
)


# In[30]:


lgb_importance = plot_feature_importance(
    lgb,
    "LightGBM",
    X_train.columns
)


# In[31]:


xgb_importance = plot_feature_importance(
    xgb,
    "XGBoost",
    X_train.columns
)


# In[32]:


lr_importance = plot_feature_importance(
    lr_pipeline,
    "Logistic Regression",
    X_train.columns
)


# In[33]:


feature_importance = {
    "Random Forest": rf_importance,
    "Decision Tree": dt_importance,
    "LightGBM": lgb_importance,
    "XGBoost": xgb_importance,
    "Logistic Regression": lr_importance
}


# In[ ]:





# In[34]:


# #===================
# #FEATURE IMPORTANCE (QUICK INSIGHT) 
# #===================

# plt.figure(figsize=(10,6))
# plt.bar(range(len(xgb.feature_importances_)), xgb.feature_importances_)
# plt.title("Feature Importance (XGBoost)")
# plt.show()


# In[35]:


# # ==================================
# # XGBOOST FEATURE IMPORTANCE
# # ==================================

# feature_importance = pd.DataFrame({
#     "Feature": X_train.columns,
#     "Importance": xgb.feature_importances_
# })

# feature_importance = feature_importance.sort_values(
#     by="Importance",
#     ascending=False
# )

# feature_importance.head(20)


# In[36]:


# # TOP 20 IMPORTANT FEATURES
# # ==================================

# top_features = feature_importance.head(20)

# plt.figure(figsize=(12,8))

# plt.barh(
#     top_features["Feature"],
#     top_features["Importance"]
# )

# plt.gca().invert_yaxis()

# plt.title("Top 20 Important Features - XGBoost")
# plt.xlabel("Importance Score")

# plt.tight_layout()
# plt.show()


# In[37]:


sample = X_test.sample(
    n=5000,
    random_state=42
)


# In[38]:


# ==================================
# RANDOM FOREST SHAP
# ==================================

rf_explainer = shap.TreeExplainer(rf)

rf_shap_values = rf_explainer.shap_values(
    sample
)


# In[39]:


#RF SHAP SUMMARY PLOT 
shap.summary_plot(
    rf_shap_values,
    sample
)


# In[40]:


#RF SHAP BAR PLOT
shap.summary_plot(
    rf_shap_values,
    sample,
    plot_type="bar"
)


# In[41]:


# ==================================
# DECISION TREE SHAP
# ==================================

dt_explainer = shap.TreeExplainer(dt)

dt_shap_values = dt_explainer.shap_values(
    sample
)


# In[42]:


#DT SHAP SUMMARY PLOT 
shap.summary_plot(
    dt_shap_values,
    sample
)


# In[43]:


#DT SHAP BAR PLOT
shap.summary_plot(
    dt_shap_values,
    sample,
    plot_type="bar"
)


# In[44]:


# ==================================
# LIGHTGBM SHAP
# ==================================

lgb_explainer = shap.TreeExplainer(lgb)

lgb_shap_values = lgb_explainer.shap_values(
    sample
)


# In[45]:


#LGB SHAP SUMMARY PLOT 
shap.summary_plot(
    lgb_shap_values,
    sample
)


# In[46]:


#LGB SHAP BAR PLOT
shap.summary_plot(
    lgb_shap_values,
    sample,
    plot_type="bar"
)


# In[47]:


# ==================================
# XGBOOST SHAP
# ==================================

xgb_explainer = shap.TreeExplainer(xgb)

xgb_shap_values = xgb_explainer.shap_values(
    sample
)


# In[48]:


#XGB SHAP SUMMARY PLOT
shap.summary_plot(
    xgb_shap_values,
    sample
)


# In[49]:


#XGB SHAP BAR PLOT
shap.summary_plot(
    xgb_shap_values,
    sample,
    plot_type="bar"
)


# In[50]:


# ==================================
# LOGISTIC REGRESSION SHAP
# ==================================

# lr_explainer = shap.Explainer(
#     lr_pipeline,
#     sample
# )

# lr_shap_values = lr_explainer(sample)




# shap.initjs()

# scaler = lr_pipeline.named_steps["scaler"]
# model = lr_pipeline.named_steps["lr"]

# X_sample = scaler.transform(sample)

# lr_explainer = shap.LinearExplainer(model, X_sample)
# lr_shap_values = lr_explainer.shap_values(X_sample)

# row = 0
# class_idx = 0  # pick the class you want to explain

# shap.plots.force(
#     lr_explainer.expected_value[class_idx],
#     lr_shap_values[class_idx][row]
# )




# # background used by SHAP
# background = X_train.sample(n=min(5000, len(X_train)), random_state=42)

# # force SHAP to keep all background rows
# masker = shap.maskers.Independent(background, max_samples=len(background))

# # logistic regression pipeline instead:
# model_fn = lr_pipeline.predict_proba

# lr_explainer = shap.Explainer(model_fn, masker)

# sample = X_test.sample(n=min(1000, len(X_test)), random_state=42)
# lr_shap_values = lr_explainer(sample)







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



# # If Tree model (RF / XGB / LGB), use this:
# explainer = shap.TreeExplainer(lr_pipeline.named_steps["lr"])
# shap_values = explainer.shap_values(sample)

# # =========================
# # PLOT ONE INSTANCE
# # =========================
# row = 0

# # multiclass-safe handling
# if isinstance(shap_values, list):
#     class_idx = 0
#     shap.plots.force(
#         explainer.expected_value[class_idx],
#         shap_values[class_idx][row]
#     )
# else:
#     shap.plots.force(
#         explainer.expected_value,
#         shap_values[row]
#     )


# In[51]:


#LR SHAP SUMMARY PLOT
shap.summary_plot(
    lr_shap_values,
    sample
)


# In[52]:


#LR SHAP BAR PLOT
shap.summary_plot(
    lr_shap_values,
    sample,
    plot_type="bar"
)


# In[53]:


print(type(rf_shap_values))
print(np.array(rf_shap_values).shape)

row_idx = 0
class_idx = 0

print(rf_shap_values[class_idx][row_idx].shape)
print(single_row.shape)
print(X_sample.shape)


# In[54]:


#EXPLAIN RF PREDICTION 
row_number = 0
class_idx = 0

# single_prediction = sample.iloc[row_number]

# shap.force_plot(
#     explainer.expected_value[class_idx],
#     lr_shap_values[class_idx][row_number],
#     single_prediction
# )



# single_prediction = sample_scaled[row_number]

# # build Explanation object (THIS is the fix)
# exp = shap.Explanation(
#     values=lr_shap_values[class_idx][row_number],
#     base_values=explainer.expected_value[class_idx],
#     data=single_prediction
# )

# shap.plots.force(exp)



# # MUST be 2D row (not 1D)
# single_prediction = sample_scaled[row_number].reshape(1, -1)

# # SHAP values must also match shape
# values = lr_shap_values[class_idx][row_number]

# # build proper Explanation
# exp = shap.Explanation(
#     values=values,
#     base_values=explainer.expected_value[class_idx],
#     data=single_prediction[0]   # flatten back for display
# )

# shap.plots.force(exp)




# shap.plots.waterfall(
#     shap.Explanation(
#         values=lr_shap_values[class_idx][row_number],
#         base_values=explainer.expected_value[class_idx],
#         data=sample.iloc[row_number]
#     )
# )




# # =========================
# # TREE EXPLAINER (Random Forest)
# # =========================
# rf_explainer = shap.TreeExplainer(rf)

# rf_shap_values = rf_explainer.shap_values(sample)

# # =========================
# # CLASS SELECTION (for multiclass)
# # =========================
# class_idx = 0
# row_idx = 0

# # =========================
# # FORCE PLOT
# # =========================
# shap.initjs()

# shap.force_plot(
#     rf_explainer.expected_value[class_idx],
#     rf_shap_values[class_idx][row_idx],
#     X_sample.iloc[row_idx],
#     matplotlib=True
# )

# # =========================
# # SUMMARY PLOT (always works)
# # =========================
# shap.summary_plot(
#     rf_shap_values[class_idx],
#     sample
# )




# # =========================
# # RF EXPLAINER
# # =========================
# rf_explainer = shap.TreeExplainer(rf)

# rf_shap_values = rf_explainer.shap_values(sample)

# # =========================
# # SINGLE ROW
# # =========================
# row_idx = 0
# class_idx = 0

# single_row = sample.iloc[row_idx]   # <-- FIX IS HERE

# # =========================
# # FORCE PLOT
# # =========================
# shap.initjs()

# shap.force_plot(
#     rf_explainer.expected_value[class_idx],
#     rf_shap_values[class_idx][row_idx],
#     single_row,
#     matplotlib=True
# )



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


# In[55]:


#EXPLAIN DT PREDICTION 
# row_number = 0
# class_idx = 0

# single_prediction = sample.iloc[row_number]

# shap.force_plot(
#     dt_explainer.expected_value[class_idx],
#     dt_shap_values[class_idx][row_number, :],
#     single_prediction
# )




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


# In[56]:


#EXPLAIN LGB PREDICTION 
# row_number = 0
# class_idx = 0

# single_prediction = sample.iloc[row_number]

# shap.force_plot(
#     lgb_explainer.expected_value[class_idx],
#     lgb_shap_values[class_idx][row_number, :],
#     single_prediction
# )




row_idx = 0
class_idx = 0

single_row = sample.iloc[[row_idx]].values

# =========================
# EXPLAINER
# =========================
lgb_explainer = shap.TreeExplainer(lgb)
lgb_shap_values = lgb_explainer.shap_values(sample)

# =========================
# FORCE PLOT (correct indexing)
# =========================
shap.initjs()

shap.plots.force(
    lgb_explainer.expected_value[class_idx],
    lgb_shap_values[row_idx, :, class_idx],
    single_row
)


# In[57]:


#EXPLAIN XGB PREDICTION 
# row_number = 0
# class_idx = 0

# single_prediction = sample.iloc[row_number]

# shap.force_plot(
#     xgb_explainer.expected_value[class_idx],
#     xgb_shap_values[class_idx][row_number, :],
#     single_prediction
# )




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


# In[58]:


print(type(lr_shap_values))
print(np.array(lr_shap_values).shape)


# In[59]:


#EXPLAIN LR PREDICTION 
# row_number = 0

# single_prediction = sample.iloc[[row_number]]

# shap.force_plot(
#     lr_explainer.expected_value,
#     lr_shap_values[row_number],
#     single_prediction
# )





# # =========================
# # DATA (use numpy, not pandas for SHAP)
# # =========================
# X_bg = sample.values  # background
# X_explain = sample.values[:100]  # small explain set

# # =========================
# # EXPLAINER (WORKS FOR LR PIPELINE)
# # =========================
# lr_model = lr_pipeline.named_steps["lr"]

# lr_explainer = shap.LinearExplainer(
#     lr_model,
#     masker=X_bg
# )

# # SHAP values
# shap_values = explainer.shap_values(X_explain)

# # =========================
# # PICK ONE SAMPLE
# # =========================
# i = 0

# # =========================
# # FORCE PLOT (binary or multi-class safe)
# # =========================
# shap.initjs()

# shap.plots.force(
#     lr_explainer.expected_value,
#     shap_values[i],
#     X_explain[i]
# )





# single_row = sample.iloc[[row_idx]].values

# lr_explainer = shap.Explainer(
#     lr_pipeline.predict,
#     single_row
# )

# lr_shap_values = lr_explainer(single_row)

# # Explain a single prediction
# row_idx = 0

# shap.plots.force(
#     lr_shap_values[row_idx]
# )





# # =========================
# # 1. Pick model + data
# # =========================

# model = lr_pipeline

# # ensure dataframe format (fixes feature-name warning)
# X_explain = sample.reset_index(drop=True)

# # =========================
# # 2. Build explainer (MODERN WAY)
# # =========================

# lr_explainer = shap.Explainer(
#     model,
#     X_explain,
#     algorithm="permutation"   # SAFE for ANY model (LR, RF, DT, XGB, LGB)
# )

# # =========================
# # 3. Compute SHAP values
# # =========================

# lr_shap_values = lr_explainer(X_explain)

# print(type(lr_shap_values))
# print(lr_shap_values.values.shape)   # (samples, features, classes) or (samples, features)

# # =========================
# # 4. Pick one sample + class
# # =========================

# i = 0
# class_idx = 0  # change if multi-class

# # =========================
# # 5. FORCE PLOT (correct SHAP v0.20+ style)
# # =========================

# shap.plots.force(
#     lr_shap_values[i, :, class_idx] if len(lr_shap_values.values.shape) == 3 else lr_shap_values[i, :]
# )





# # =========================
# # DATA (must be DataFrame)
# # =========================
# X_explain = sample.copy().reset_index(drop=True)

# # =========================
# # SAFE PREDICTION FUNCTION
# # (THIS FIXES YOUR ERROR)
# # =========================
# def model_predict(data):
#     return lr_pipeline.predict_proba(data)

# # =========================
# # SHAP EXPLAINER (WORKING FIX)
# # =========================
# explainer = shap.Explainer(
#     model_predict,
#     X_explain,
#     algorithm="permutation"
# )

# # =========================
# # COMPUTE SHAP VALUES
# # =========================
# shap_values = explainer(X_explain)

# # =========================
# # PICK ONE SAMPLE + CLASS
# # =========================
# i = 0
# class_idx = 0

# # =========================
# # FORCE PLOT (CORRECT FORMAT)
# # =========================
# shap.plots.force(
#     shap_values[i, :, class_idx]
# )




# # =========================
# # pick sample + class
# # =========================
# i = 0
# class_idx = 0

# # =========================
# # SAFE slicing
# # =========================
# single_explanation = shap_values[i, :, class_idx]

# # =========================
# # FORCE PLOT (modern API)
# # =========================
# shap.plots.force(single_explanation)








background = X_train.sample(n=min(5000, len(X_train)), random_state=42)
X_explain = X_test.sample(n=min(5000, len(X_test)), random_state=42)

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


# In[60]:


# from pathlib import Path

# # ARTIFACTS_DIR = Path("artifacts/ton_iot")
# # MODEL_DIR = Path("models/ton_iot")
# # ARTIFACTS_DIR.mkdir(exist_ok=True)
# # MODEL_DIR.mkdir(exist_ok=True)


# PROJECT_DIR = Path.cwd().resolve()
# if PROJECT_DIR.name.lower() == "notebooks":
#     PROJECT_DIR = PROJECT_DIR.parent

# MODEL_DIR = PROJECT_DIR / "datasets" / "TON_IoT" / "models" 
# ARTIFACTS_DIR = PROJECT_DIR / "datasets" / "TON_IoT" / "artifacts" 

# MODEL_DIR.mkdir(parents=True, exist_ok=True)
# ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


# # Save trained models
# joblib.dump(rf, MODEL_DIR / "random_forest.pkl")
# joblib.dump(dt, MODEL_DIR / "decision_tree.pkl")
# joblib.dump(lgb, MODEL_DIR / "lightgbm.pkl")
# joblib.dump(xgb, MODEL_DIR / "xgboost.pkl")
# joblib.dump(lr_pipeline, MODEL_DIR / "logistic_regression.pkl")

# # Save encoders and metadata
# joblib.dump(encoder, ARTIFACTS_DIR / "feature_encoder.pkl")
# joblib.dump(target_encoder, ARTIFACTS_DIR / "target_encoder.pkl")
# joblib.dump(list(X_train.columns), ARTIFACTS_DIR / "feature_names.pkl")
# joblib.dump(list(cat_cols), ARTIFACTS_DIR / "categorical_columns.pkl")
# joblib.dump(drop_cols, ARTIFACTS_DIR / "drop_columns.pkl")

# # Save results table
# results_df.to_csv(ARTIFACTS_DIR / "results.csv", index=False)

# # Save evaluation sample for confusion matrices and SHAP
# # eval_sample = X_test.sample(n=min(5000, len(X_test)), random_state=42)
# # eval_labels = y_test.loc[eval_sample.index].to_numpy()


# sample_idx = np.random.RandomState(42).choice(
#     len(X_test),
#     size=min(5000, len(X_test)),
#     replace=False
# )

# # eval_sample = X_test.iloc[sample_idx]
# # eval_labels = y_test[sample_idx]

# eval_sample = X_test.sample(n=min(5000, len(X_test)), random_state=42)
# eval_labels = y_test.loc[eval_sample.index].to_numpy()


# eval_sample.to_csv(ARTIFACTS_DIR / "eval_sample.csv", index=False)
# joblib.load(ARTIFACTS_DIR / "eval_sample.pkl")
# joblib.dump(eval_labels, ARTIFACTS_DIR / "eval_labels.pkl")


# In[74]:



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
    drop_cols,
    ARTIFACTS_DIR / "drop_columns.pkl"
)

joblib.dump(
    list(cat_cols),
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


# In[62]:


joblib.dump(dataset_quality,
            ARTIFACTS_DIR / "dataset_quality.pkl")

joblib.dump(df,
            ARTIFACTS_DIR / "clean_dataset.pkl")

joblib.dump(df.describe(),
            ARTIFACTS_DIR / "dataset_summary.pkl")

joblib.dump(df.corr(numeric_only=True),
            ARTIFACTS_DIR / "correlation_matrix.pkl")

joblib.dump(df["type"].value_counts(),
            ARTIFACTS_DIR / "class_distribution.pkl")

joblib.dump(feature_importance,
            ARTIFACTS_DIR / "feature_importance.pkl")


# In[63]:


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


# In[64]:


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


# In[65]:


numeric_features=df.select_dtypes(
include=np.number
)


joblib.dump(
numeric_features,
EDA_DIR/"numeric_features.pkl"
)


# In[ ]:





# In[ ]:





# In[66]:


predictedValue = rf.predict(X_test)
print(predictedValue)


# In[67]:


rf_preds = rf.predict(X_test)

comparison_df = pd.DataFrame({
    "Actual": target_encoder.inverse_transform(y_test),
    "Predicted": target_encoder.inverse_transform(rf_preds)
})

comparison_df.head(20)


# In[ ]:





# In[68]:


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


# In[69]:


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


# In[70]:


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


# Explainable AI Dashboard for Cyber Threat Identification in Smart IoT Systems

This project provides an interactive Streamlit dashboard for identifying cyber threats in Smart IoT Systems using machine learning and Explainable AI (SHAP).

## Datasets 

The system is trained and evaluated on two public IoT security datasets:

### TON_IoT

- **Description:** Network flow records from IoT devices with benign and multiple attack classes.
- **Source:** UNSW. Available at: https://ton.iot.unsw.edu.au/

### Edge‑IIoT

- **Description:** Smart‑home and industrial IoT traffic with diverse protocols and attack types.
- **Source:** Edge‑IIoT. Available at: https://github.com/EdgeIIoT/EdgeIIoT_dataset


## Features

- Data Preprocessing and Feature Engineering 
- EDA 
- Machine Learning model training 
- Threat prediction
- SHAP explanations
- Interactive dashboard
- Data visualisation

## System Architecture

The system consists of three main components:

1. **Data & Model Training (offline)**
   - Data preprocessing and feature engineering in Python (pandas, NumPy, scikit‑learn).
   - Training and evaluation of five ML models per dataset.
   - Generation and storage of artefacts: trained models, encoders, feature lists, SHAP background samples, metrics.

2. **Explainability Module**
   - SHAP (Tree SHAP) for local and global explanations.
   - Coefficient‑based reasoning for Logistic Regression.
   - Reasoning generator that converts SHAP values into plain‑language text.

3. **Streamlit Dashboard (online)**
   - Loads pre‑trained models and artefacts.
   - Provides interactive pages:
     - About
     - Overview
     - Dataset Analysis
     - Model Comparison
     - Prediction

## Installation (Local)

You can run the dashboard locally if you prefer not to use the deployed version.

### Requirements

- Python 3.14
- pip
- Git
- Command prompt 

### Clone the repository

git clone https://github.com/O-Adeola/XAI-Dashboard-for-Cyber-Threat-Identification-in-Smart-IoT-Systems.git
cd XAI-Dashboard-for-Cyber-Threat-Identification-in-Smart-IoT-Systems

### Install dependencies

pip install -r requirements.txt

### Run the dashboard

streamlit run app.py

The dashboard will open in your default browser (usually at http://localhost:8501).


## Usage

### 1. Select dataset and model

- Use the sidebar to choose:
  - **Dataset:** `TON_IoT` or `Edge‑IIoT`
  - **Model:** Decision Tree, Random Forest, LightGBM, XGBoost, or Logistic Regression

### 2. Explore performance

- **Overview:** See dataset statistics and best‑performing models.
- **Dataset Analysis:** Inspect class distributions, missing values and feature summaries.
- **Model Comparison:** Compare models using accuracy, F1‑score, ROC‑AUC, confusion matrices and classification reports.

### 3. Run a prediction

- Go to the **Prediction** page.
- Choose:
  - **Input source:** Saved evaluation sample or uploaded CSV.
  - **Row index** (for saved samples).
- Click **Predict**.
- View:
  - Predicted attack class
  - Confidence score
  - Class probabilities

### 4. Understand the prediction

- Go to the **Explainability** page.
- View:
  - **Local explanation:** SHAP waterfall / force plot for the selected record.
  - **Global feature importance:** Which features matter most across the dataset.
  - **Plain‑language reasoning:** A short text explaining why the model made this prediction and suggested analyst actions.


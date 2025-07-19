# Semi-Supervised-Learning-on-UCI-Heart-Disease

This project applies semi-supervised machine learning techniques to classify the severity of heart disease using both labeled and unlabeled data from the UCI Heart Disease dataset. The classification task targets a multiclass label (0–4) representing increasing severity.

## 📌 Objectives

- Predict heart disease severity from medical features using a hybrid learning approach.
- Leverage both labeled and unlabeled data to improve classification performance.
- Compare and optimize different clustering and classification pipelines.

---

## 🧾 Dataset Description

The dataset contains 14+ medical features including:
- Age, sex, chest pain type, resting blood pressure
- Serum cholesterol, fasting blood sugar, resting ECG
- Maximum heart rate, ST depression, number of major vessels, etc.
- Target variable: `num` (values 0–4)

---

## 🛠 Technologies & Libraries

- **Python 3.10+**
- **Pandas**, **NumPy**
- **Scikit-learn**, **CatBoost**, **XGBoost**
- **Matplotlib**, **Seaborn**
- **Optuna** (for hyperparameter tuning)

---

## 🧠 Modeling Pipeline

### 1. Preprocessing
- **Label Encoding:** Converted categorical features to numeric format using `LabelEncoder`.
- **Missing Value Imputation:** Applied `IterativeImputer` with `RandomForestRegressor` for robust multi-feature estimation.
- **Normalization:** Scaled numerical features with `StandardScaler` for model stability.
- **Dimensionality Reduction:** Used **PCA** to retain 95% of the variance and reduce overfitting risk.

### 2. Semi-Supervised Labeling
- Applied **SelfTrainingClassifier** with **LightGBM** as the base estimator.
- Automatically labeled unlabeled instances using a `k_best` criterion to improve model training capacity.

### 3. Model Training & Optimization
- Trained a **CatBoostClassifier** on the expanded labeled dataset.
- Performed **hyperparameter tuning** using **Optuna** to find the best combination of:
  - `iterations`, `depth`, `learning_rate`, `l2_leaf_reg`
- Evaluated the final model on a stratified test split using **accuracy** as the key metric.

### ✅ Final Model
- **Model:** CatBoost (Optuna-tuned)
- **Learning Strategy:** Semi-Supervised with Self-Training
- **Best Accuracy Achieved:** *72.46%*  

---

## 📊 Evaluation Metrics

- **Accuracy**: 72.46% (best with CatBoost + SelfTrainingClassifier)
- Comprehensive comparison across all models in terms of performance and robustness

---
## 📦 Output
Final predictions are exported to lightgbm_catboost_optuna_submission.csv, which includes an id column and predicted num values

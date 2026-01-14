# Credit Card Fraud Detection: Machine Learning Benchmarking

This project evaluates three distinct machine learning architectures — **LightGBM**, **XGBoost**, and a **Multi-Layer Perceptron (Neural Network)** — to detect fraudulent credit card transactions in an **extremely imbalanced dataset** (fraud rate of only **0.17%**).

---

##  Project Overview

The main objective of this project is to **minimize financial loss** by correctly identifying fraudulent transactions (high **Recall**) while also maintaining a good customer experience by limiting unnecessary transaction blocks (high **Precision**).

---

## Key Challenges

- **Extreme Class Imbalance**  
  Only **492 fraud cases** out of **284,807 transactions**.

- **Feature Anonymization**  
  Most features are PCA-transformed (`V1`–`V28`), which requires careful scaling rather than standard normalization.

---

## Tech Stack & Methodology

### Data Preprocessing
- Used **RobustScaler** via `ColumnTransformer`
- Chosen because it relies on the **median and Interquartile Range (IQR)**, making it robust to extreme outliers in the transaction **Amount** feature

---

### Modeling Pipeline
To prevent **data leakage**, we implemented an **imblearn Pipeline** consisting of:

- **SMOTE (Synthetic Minority Over-sampling Technique)**  
  Applied **only on training folds** to balance the classes
- **Model Training**  
  Benchmarked three different architectures
- **GridSearchCV**  
  Hyperparameter tuning optimized for **F1-score** using **3-fold cross-validation**

---

## Model Selection Rationale

- **LightGBM**  
  - Fast training
  - Native handling of class imbalance using `class_weight`

- **XGBoost**  
  - Industry standard for tabular data
  - Precise control over imbalance using `scale_pos_weight`

- **Neural Network (MLP)**  
  - Included to capture deep, non-linear relationships in PCA-transformed features

---

## Results & Evaluation

### Precision–Recall Analysis
Because accuracy is misleading for imbalanced datasets, we prioritized **AUPRC (Area Under the Precision–Recall Curve)**.

- **XGBoost (Winner)**  
  - Highest AUPRC: **0.821**
- **LightGBM**  
  - AUPRC: **0.788**
  - Slightly unstable curve at high-confidence thresholds due to `class_weight` sensitivity
- **MLP**  
  - Performance degraded faster after ~75% recall compared to tree-based models

---

### Real-World Performance (Confusion Matrix)

At the optimal threshold, test set results were:

| Metric | LightGBM | XGBoost | Neural Network |
|------|---------|---------|----------------|
| Caught Fraud (TP) | 77 | 77 | 71 |
| Missed Fraud (FN) | 18 | 18 | 24 |
| False Alarms (FP) | 20 | 24 | 22 |

---

## Conclusions

**XGBoost** is the recommended model for this use case.

Although LightGBM produced **4 fewer false alarms**, XGBoost’s superior **AUPRC** indicates it is the most robust model for distinguishing fraudulent transactions across different operational thresholds.

This makes XGBoost the best overall choice when balancing fraud detection performance and real-world deployment constraints.

## Dataset 

Since there is size limitation to upload .csv file, dataset is taken from https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

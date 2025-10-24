# 🏙️ LA Crime Prediction with Perceptron

**Machine learning project predicting crime severity in Los Angeles using open data (2020–Present) and a calibrated Perceptron model.**

---

## 📖 Overview

This project explores **spatial**, **temporal**, and **demographic** patterns in crime incidents across Los Angeles and builds a full **machine learning pipeline** to classify whether an incident falls under **Part 1 (serious)** or **Part 2 (non-serious)** crime categories.

It is designed as a **reproducible, portfolio-grade data science project** using scikit-learn pipelines, random search tuning, calibration, and permutation-based feature importance.

---

## 🧩 Features & Workflow

| Stage | Description |
|--------|--------------|
| **1. Data Loading** | Loads the LA City Open Crime Dataset (2020–Present) from Google Drive or Socrata API. |
| **2. Cleaning & Feature Engineering** | Parses dates/times, bins coordinates, handles missing values, encodes categories. |
| **3. Preprocessing Pipeline** | `ColumnTransformer` + `OneHotEncoder(min_frequency=50)` + `StandardScaler` |
| **4. Modeling** | Perceptron classifier (linear, calibrated for probabilities) tuned with RandomizedSearchCV |
| **5. Evaluation** | Accuracy, Precision, Recall, F1-macro, ROC-AUC, and Confusion Matrix |
| **6. Baselines** | Dummy “Most Frequent” and “Stratified Random” models for comparison |
| **7. Explainability** | Permutation importance (raw + one-hot expanded) and feature ranking |
| **8. Persistence** | Saves `model.joblib`, `metrics.json`, `schema.json`, and feature lists in `artifacts/` |
| **9. Visualization** | Time-of-day, weekday, spatial (map), and feature importance plots |

---

## 📊 Example Results

| Metric | Perceptron | Dummy (Majority) | Dummy (Random) |
|:-------|:-----------:|:----------------:|:---------------:|
| Accuracy | 0.84 | 0.65 | 0.51 |
| F1 (macro) | 0.82 | 0.40 | 0.48 |
| ROC-AUC | 0.87 | — | — |

*Results may vary slightly depending on dataset sampling and random seeds.*

---

## 🧠 Key Insights

- **Crime hotspots** cluster near downtown and along major highways.  
- **Peak crime hours:** evenings between 7 PM and 1 AM.  
- **Weapon type**, **premise**, and **time of day** are strong predictors of severity.  
- The calibrated perceptron outperforms dummy baselines by a large margin.

---

## 🧰 Installation & Environment

### 1️⃣ Clone the repo

```bash
git clone https://github.com/prans-code/la-crime-prediction-perceptron.git
cd la-crime-prediction-perceptron

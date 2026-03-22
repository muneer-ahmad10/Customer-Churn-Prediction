# 📊 Customer Churn Prediction (Machine Learning Project)

## 🚀 Overview

This project focuses on predicting customer churn using machine learning techniques on a large-scale dataset (100,000 records). The objective is to identify customers likely to leave and provide actionable insights for business retention strategies.

---

## 🎯 Objectives

* Perform data preprocessing and feature engineering
* Train and compare multiple machine learning models
* Improve churn detection using threshold tuning
* Identify key drivers of customer churn

---

## 🛠️ Tech Stack

* **Programming:** Python
* **Libraries:** Pandas, NumPy, Scikit-learn
* **Models:** Logistic Regression, Random Forest, Gradient Boosting

---

## 📂 Project Structure

```
customer-churn-prediction/
│── data/
│   └── synthetic_customer_churn_100k.csv
│── src/
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│── main.py
```

---

## 🔄 Workflow

1. Data Cleaning and Preprocessing
2. Feature Encoding using One-Hot Encoding
3. Train-Test Split
4. Model Training and Evaluation
5. Threshold Tuning for improving recall
6. Feature Importance Analysis

---

## 🤖 Models Performance

### 🔹 Logistic Regression

* Accuracy: **72%**
* Churn Recall: **46%**
* F1-score: **0.53**

---

### 🔹 Random Forest

* Accuracy: **73%**
* Churn Recall: **49%**
* F1-score: **0.55**

---

### 🔹 Gradient Boosting (Best Base Model)

* Accuracy: **76%**
* Churn Recall: **56%**
* F1-score: **0.61**

---

### 🔥 Threshold Tuned Model (> 0.35)

* Accuracy: **75%**
* Churn Recall: **60%** ✅
* F1-score: **0.62**
* Precision: **64%**

👉 Improved churn detection from **49% → 60%** using threshold tuning.

---

## 📊 Key Insights (Feature Importance)

Top factors influencing churn:

* **MonthlyCharges (0.32)** → Higher charges increase churn
* **Tenure (0.32)** → Longer tenure reduces churn
* **Contract Type (One year / Two year)** → Long-term contracts reduce churn significantly

👉 Business Insight:

* Customers on **month-to-month plans** are more likely to churn
* High-paying customers need retention strategies
* Loyalty (tenure) plays a major role

---

## ⚙️ How to Run

### 1. Clone Repository

```
git clone https://github.com/muneer-ahmad10/Customer-Churn-Prediction.git
cd customer-churn-prediction
```

### 2. Install Dependencies

### 3. Run Project

```
python main.py
```

---

## 📦 Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
```

---

## 🔥 Key Highlights

* Built an end-to-end machine learning pipeline
* Compared multiple models for performance evaluation
* Improved model performance using **threshold tuning**
* Identified key business drivers using feature importance
* Implemented modular and production-style code structure

---

## 📌 Future Improvements

* Apply advanced models like XGBoost
* Perform hyperparameter tuning (GridSearchCV)
* Deploy model using Flask or Streamlit

---

## 👤 Author

**Muneer Ahmad Dar**

* GitHub: https://github.com/muneer-ahmad10
* LinkedIn: https://linkedin.com/in/muneerahmad-826363267

---

## ⭐ If you found this useful

Give it a ⭐ on GitHub!

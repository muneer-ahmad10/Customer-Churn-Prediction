# 📉 Customer Churn Prediction — Machine Learning

> Predicting which customers will leave — before they do.

![Python](https://img.shields.io/badge/Python-3.9+-0f2027?style=flat-square&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat-square&logo=jupyter&logoColor=white)

---

## 🚩 Problem

Customer churn is costly — acquiring a new customer costs 5–7x more than retaining an existing one. Early prediction enables targeted retention campaigns before customers leave.

---

## ✅ Approach

Built and compared three classification models to identify churning customers from behavioral data, with a focus on maximizing **recall** (catching as many churners as possible).

---

## 📊 Results

| Model | Precision | Recall | F1-Score |
|---|---|---|---|
| Logistic Regression | 0.72 | 0.49 | 0.58 |
| Random Forest | 0.74 | 0.55 | 0.63 |
| Gradient Boosting | 0.71 | **0.60** | **0.65** |

> Recall improved from **0.49 → 0.60** using threshold tuning and class weighting on imbalanced data.

---

## 🛠️ Key techniques

- Handling imbalanced classes with `class_weight='balanced'`
- Threshold tuning to optimize recall vs precision trade-off
- Feature importance analysis with Random Forest
- Model evaluation: precision, recall, F1, confusion matrix, ROC-AUC

---

## 🛠️ Tech stack

`Python` · `Scikit-learn` · `Pandas` · `NumPy` · `Matplotlib` · `Seaborn`

---

## 🚀 Run locally

```bash
git clone https://github.com/muneer-ahmad10/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction
pip install -r requirements.txt
jupyter notebook
```

---

## 👨‍💻 Author

**Muneer Ahmad Dar** · [LinkedIn](https://linkedin.com/in/muneerahmad-826363267) · [GitHub](https://github.com/muneer-ahmad10)

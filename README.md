# Customer Churn Prediction — Machine Learning

## Overview

This project explores customer churn prediction using machine learning classification models in Python. The objective is to identify customers who may leave a service and investigate how class imbalance and probability-threshold adjustments affect model predictions.

The project uses a **synthetic dataset containing 100,000 customer records** and compares Logistic Regression, Random Forest, and Gradient Boosting classifiers.

## Objectives

- Build a machine learning pipeline for customer churn classification.
- Compare multiple classification algorithms.
- Explore class imbalance handling using class weights.
- Investigate the precision–recall trade-off through probability-threshold tuning.
- Identify influential features using model-based feature importance.

## Tech Stack

- **Language:** Python
- **Data processing:** Pandas, NumPy
- **Machine learning:** Scikit-learn
- **Visualization and analysis:** Matplotlib, Seaborn (if used in your analysis)
- **Development environment:** Jupyter Notebook, VS Code

## Machine Learning Workflow

1. **Data loading:** Load the synthetic customer churn dataset.
2. **Preprocessing:** Remove the customer identifier, separate features from the target, and convert categorical features into numerical variables using one-hot encoding.
3. **Train-test split:** Split the data into training and testing sets using an 80:20 ratio.
4. **Feature scaling:** Standardize features for Logistic Regression while using the original encoded features for Random Forest and Gradient Boosting.
5. **Model training:** Train and compare three classification models.
6. **Evaluation:** Generate classification reports to examine precision, recall, F1-score, and support.
7. **Threshold tuning:** Adjust the Gradient Boosting prediction threshold to explore how changing the decision boundary affects classification performance.
8. **Feature importance:** Examine the most influential features identified by the Gradient Boosting model.

## Models Implemented

| Model | Approach |
|---|---|
| Logistic Regression | Baseline classification model using standardized features |
| Random Forest | Ensemble classifier with balanced class weights |
| Gradient Boosting | Ensemble classifier with probability-threshold tuning |

## Handling Class Imbalance

The project explores two approaches:

- **Class weighting:** Random Forest uses `class_weight='balanced'` to account for differences in class frequencies.
- **Threshold tuning:** Gradient Boosting uses a custom probability threshold of `0.35` instead of the default classification threshold.

These approaches are applied to different models in the current implementation. Their results should therefore be evaluated separately rather than attributing all performance changes to one combined technique.

## Evaluation

The current implementation uses scikit-learn's `classification_report` to evaluate model predictions.

The evaluation focuses on:

- **Precision:** How many predicted churn cases are actual churn cases.
- **Recall:** How many actual churn cases the model identifies.
- **F1-score:** The balance between precision and recall.
- **Support:** The number of actual examples in each class.

The project also compares predictions from the default Gradient Boosting classifier with predictions made using the custom threshold.

*Note:* Numerical performance claims should be added after rerunning the current pipeline and verifying the results. A threshold-tuned result should be compared against the default threshold on the same held-out test set.

## Project Structure

```text
Customer-Churn-Prediction/
├── Data/
│   └── synthetic_customer_churn_100k.csv
├── src/
│   ├── preprocess.py
│   ├── train.py
│   └── evaluate.py
├── main.py
├── README.md
└── requirements.txt
```

## Installation and Usage

### 1. Clone the repository

```bash
git clone https://github.com/muneer-ahmad10/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
```

On macOS or Linux:

```bash
source venv/bin/activate
```

On Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install pandas numpy scikit-learn
```

If your analysis or notebooks use Matplotlib or Seaborn, install those packages as well.

Alternatively, if you maintain a complete `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 4. Check the dataset path

The current `main.py` expects the dataset at:

```text
data/synthetic_customer_churn_100k.csv
```

The repository structure currently uses `Data/` with an uppercase `D`. Make the directory name and the path in `main.py` consistent before running the project. For example, you can update the path in `main.py` to:

```python
X, y = load_and_preprocess(
    "Data/synthetic_customer_churn_100k.csv"
)
```

### 5. Run the pipeline

```bash
python main.py
```

The script trains the models, prints classification reports, evaluates Gradient Boosting with the custom probability threshold, and displays the model's top features.

## Key Learnings

- Building a modular machine learning workflow with separate preprocessing, training, and evaluation components.
- Comparing linear and ensemble classification models.
- Understanding why recall can be important in customer churn prediction.
- Exploring class weighting and decision-threshold adjustments for imbalanced classification.
- Interpreting precision, recall, and F1-score when evaluating model performance.
- Using feature importance to investigate which input variables influence model predictions.

## Limitations and Future Improvements

- Evaluate models using a consistent set of metrics, including a confusion matrix and ROC-AUC where appropriate.
- Compare default and tuned thresholds using the same test set, selecting the threshold on validation data to avoid tuning directly on the final test set.
- Add reproducible result tables and save evaluation outputs.
- Investigate cross-validation and hyperparameter tuning.
- Explore explainability techniques such as SHAP, if appropriate.
- Validate the approach on a real-world customer churn dataset before drawing conclusions about real customer behavior.

## Disclaimer

This project uses **synthetic customer data** for learning and experimentation. Its results should not be interpreted as evidence of performance on real customer populations.

## Author

**Muneer Ahmad Dar**

- GitHub: https://github.com/muneer-ahmad10
- LinkedIn: https://www.linkedin.com/in/muneerah-826363267/

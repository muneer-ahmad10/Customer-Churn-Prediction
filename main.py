from src.preprocess import load_and_preprocess
from src.train import train_models
from src.evaluate import evaluate_models, threshold_tuning, feature_importance

def main():
    # Load & preprocess
    X, y = load_and_preprocess('data/synthetic_customer_churn_100k.csv')

    # Train
    models, y_test = train_models(X, y)

    # Evaluate all models
    evaluate_models(models, y_test)

    # Threshold tuning (on Gradient Boosting)
    gb_model, X_test = models["gb"]
    threshold_tuning(gb_model, X_test, y_test, threshold=0.35)

    # Feature importance
    feature_importance(gb_model, X.columns)

if __name__ == "__main__":
    main()
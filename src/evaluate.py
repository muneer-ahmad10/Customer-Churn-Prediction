from sklearn.metrics import classification_report
import pandas as pd

def evaluate_models(models, y_test):
    for name, (model, X_test) in models.items():
        y_pred = model.predict(X_test)
        print(f"\n{name.upper()} MODEL:\n")
        print(classification_report(y_test, y_pred))

def threshold_tuning(model, X_test, y_test, threshold=0.35):
    y_probs = model.predict_proba(X_test)[:, 1]
    y_pred_custom = (y_probs > threshold).astype(int)

    print(f"\nTHRESHOLD TUNED (>{threshold}):\n")
    print(classification_report(y_test, y_pred_custom))

def feature_importance(model, feature_names):
    importances = model.feature_importances_
    features = pd.Series(importances, index=feature_names)
    
    print("\nTOP FEATURES:\n")
    print(features.sort_values(ascending=False).head(10))
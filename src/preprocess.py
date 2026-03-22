import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess(path):
    # Load data
    df = pd.read_csv(path)

    # Basic cleaning
    df.drop(['CustomerID'], axis=1, inplace=True)

    # Split features and target
    X = df.drop('Churn', axis=1)
    y = df['Churn']

    # Encoding features
    X = pd.get_dummies(X, drop_first=True)

    # Encoding target
    le = LabelEncoder()
    y = le.fit_transform(y)

    return X, y
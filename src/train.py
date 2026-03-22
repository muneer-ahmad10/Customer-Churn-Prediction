from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

def train_models(X, y):
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scaling (only for LR)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Models
    lr = LogisticRegression(random_state=42)
    rf = RandomForestClassifier(class_weight='balanced', random_state=42)
    gb = GradientBoostingClassifier(random_state=42)

    # Train
    lr.fit(X_train_scaled, y_train)
    rf.fit(X_train, y_train)
    gb.fit(X_train, y_train)

    return {
        "lr": (lr, X_test_scaled),
        "rf": (rf, X_test),
        "gb": (gb, X_test)
    }, y_test
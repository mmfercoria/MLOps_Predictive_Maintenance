from zenml import step
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.base import ClassifierMixin
import pandas as pd

@step
def evaluate_model(
    model: ClassifierMixin,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> None:
    y_pred = model.predict(X_test)
    y_probs = model.predict_proba(X_test)[:, 1]

    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print(f"AUC: {roc_auc_score(y_test, y_probs):.4f}")

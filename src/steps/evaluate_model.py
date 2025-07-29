from zenml import step
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.base import ClassifierMixin
import pandas as pd
import json
from pathlib import Path

@step
def evaluate_model(
    model: ClassifierMixin,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> None:
    # Predict
    y_pred = model.predict(X_test)
    y_probs = model.predict_proba(X_test)[:, 1]

    # Metrics
    cm = confusion_matrix(y_test, y_pred).tolist()
    report = classification_report(y_test, y_pred, output_dict=True)
    auc = roc_auc_score(y_test, y_probs)

    # Combine all results
    results = {
        "confusion_matrix": cm,
        "classification_report": report,
        "roc_auc": round(auc, 4)
    }

    # Save to JSON
    output_path = Path("src/models/metrics/evaluation_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Evaluation results saved to: {output_path}")

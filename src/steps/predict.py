from zenml import step
from sklearn.base import ClassifierMixin
import pandas as pd

@step
def predict(model: ClassifierMixin, features: pd.DataFrame) -> pd.Series:
    X = features.drop(columns=["machineID", "datetime"], errors="ignore")
    preds = model.predict(X)
    return pd.Series(preds)

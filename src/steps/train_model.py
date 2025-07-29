from zenml import step
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from typing import Tuple, Annotated
import pandas as pd

@step
def train_model(
    data: pd.DataFrame
) -> Tuple[
    Annotated[RandomForestClassifier, "model"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "y_test"]
]:
    
    X = data.drop(columns=["label"])
    y = data["label"]


    train_data = pd.concat([X, y], axis=1)
    class_0 = train_data[train_data.label == 0]
    class_1 = train_data[train_data.label == 1]

    class_0_downsampled = resample(class_0, replace=False, n_samples=len(class_1)*3, random_state=42)
    balanced = pd.concat([class_1, class_0_downsampled])

    X_bal = balanced.drop(columns=["label", "machineID"])
    y_bal = balanced["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X_bal, y_bal, test_size=0.2, random_state=42, stratify=y_bal
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    return model, X_test, y_test

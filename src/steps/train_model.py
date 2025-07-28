from zenml import step
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from typing import Tuple, Annotated
import pandas as pd

@step
def train_model(
    features: pd.DataFrame, 
    failures: pd.DataFrame
) -> Tuple[
    Annotated[RandomForestClassifier, "model"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "y_test"]
]:
    # Convertir datetime
    features["datetime"] = pd.to_datetime(features["datetime"])
    failures["datetime"] = pd.to_datetime(failures["datetime"])

    # Crear etiqueta de fallas futuras (24h antes del evento real)
    future_failures = failures.copy()
    future_failures["datetime"] = future_failures["datetime"] - pd.Timedelta("24h")
    future_failures["label"] = 1  # 👈 Etiqueta explícita

    # Crear labels por machineID y datetime (unión de features y fallas futuras)
    telemetry = features[["machineID", "datetime"]].drop_duplicates()
    labels = telemetry.merge(
        future_failures[["machineID", "datetime", "label"]],
        on=["machineID", "datetime"],
        how="left"
    )
    labels["label"] = labels["label"].fillna(0).astype(int)

    # Agregar etiquetas a los features
    dataset = pd.merge(
        features,
        labels[["machineID", "datetime", "label"]],
        on=["machineID", "datetime"],
        how="left"
    )
    dataset["label"] = dataset["label"].fillna(0).astype(int)
    dataset = dataset.dropna()

    # Dividir por fecha
    split_date = dataset["datetime"].quantile(0.8)
    train = dataset[dataset["datetime"] < split_date]
    test = dataset[dataset["datetime"] >= split_date]

    # Features y etiquetas
    X_train = train.drop(columns=["datetime", "label"])
    y_train = train["label"]
    X_test = test.drop(columns=["datetime", "label"])
    y_test = test["label"]

    # Balancear clases (undersampling)
    train_data = pd.concat([X_train, y_train], axis=1)
    class_0 = train_data[train_data.label == 0]
    class_1 = train_data[train_data.label == 1]
    class_0_under = class_0.sample(n=len(class_1)*3, random_state=42)
    balanced_train = pd.concat([class_0_under, class_1])

    X_train_bal = balanced_train.drop(columns="label")
    y_train_bal = balanced_train["label"]

    # Entrenar modelo
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight='balanced'
    )
    model.fit(X_train_bal, y_train_bal)

    return model, X_test, y_test

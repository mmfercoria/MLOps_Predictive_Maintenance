from zenml import step
import pandas as pd
from pathlib import Path
from typing import Tuple

@step
def preprocess_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    telemetry = pd.read_csv("data/PdM_telemetry.csv", parse_dates=["datetime"])
    errors = pd.read_csv("data/PdM_errors.csv", parse_dates=["datetime"])
    failures = pd.read_csv("data/PdM_failures.csv", parse_dates=["datetime"])
    machines = pd.read_csv("data/PdM_machines.csv")
    maint = pd.read_csv("data/PdM_maint.csv", parse_dates=["datetime"])
    
    # Feature Engineering
    telemetry = telemetry.sort_values(["machineID", "datetime"])

    # Estadísticas móviles (3h y 24h)
    rolling_3h = telemetry.set_index("datetime").groupby("machineID")[["volt", "rotate", "pressure", "vibration"]].rolling("3h", min_periods=1).agg(['mean', 'std']).reset_index()
    rolling_3h.columns = ["machineID", "datetime"] + [f"{col[0]}_{col[1]}_3h" for col in rolling_3h.columns[2:]]

    rolling_24h = telemetry.set_index("datetime").groupby("machineID")[["volt", "rotate", "pressure", "vibration"]].rolling("24h", min_periods=1).agg(['mean', 'std']).reset_index()
    rolling_24h.columns = ["machineID", "datetime"] + [f"{col[0]}_{col[1]}_24h" for col in rolling_24h.columns[2:]]

    # Unir features
    features = pd.merge(rolling_3h, rolling_24h, on=["machineID", "datetime"])

    # Agregación de errores
    error_counts = errors.copy()
    error_counts["count"] = 1
    error_counts = error_counts.set_index("datetime").groupby(["machineID", "errorID"]).rolling("24h", min_periods=1)["count"].sum().reset_index()
    error_pivot = error_counts.pivot_table(index=["machineID", "datetime"], columns="errorID", values="count").fillna(0).reset_index()

    features = pd.merge(features, error_pivot, on=["machineID", "datetime"], how="left").fillna(0)

    return features, failures

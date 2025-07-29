from zenml import step
import pandas as pd
from pathlib import Path
from typing import Tuple

@step
def preprocess_data() -> pd.DataFrame:
    telemetry = pd.read_csv("data/train_telemetry.csv", parse_dates=["datetime"])
    failures = pd.read_csv("data/train_failures.csv", parse_dates=["datetime"])
    machines = pd.read_csv("data/PdM_machines.csv")

    # Sort the telemetry data by machine and time
    telemetry = telemetry.sort_values(by=["machineID", "datetime"])

    # Create rolling statistics every 3 hours for each machine
    rolling = telemetry.set_index("datetime") \
        .groupby("machineID")[["volt", "rotate", "pressure", "vibration"]] \
        .rolling("3h", min_periods=1) \
        .agg(['mean', 'std']) \
        .reset_index()

    # Rename columns to something simpler
    rolling.columns = ["machineID", "datetime"] + [f"{col[0]}_{col[1]}_3h" for col in rolling.columns[2:]]

    # Mark failures with a label = 1
    failures["label"] = 1

    # Add the label to telemetry data if there's a failure within 24h for the same machine
    telemetry_labels = pd.merge_asof(
        rolling.sort_values("datetime"),
        failures.sort_values("datetime"),
        by="machineID",
        on="datetime",
        direction="forward",
        tolerance=pd.Timedelta("24h")
    )

    # Fill the rest with 0 (no failure)
    telemetry_labels["label"] = telemetry_labels["label"].fillna(0)

    # Add info about the machine
    data = pd.merge(telemetry_labels, machines, on="machineID", how="left")

    # Convert the machine model (text) into numbers using one-hot encoding
    data["model"] = data["model"].astype(str)
    data = pd.get_dummies(data, columns=["model"])

    # Remove the column "failure"
    if "failure" in data.columns:
        data = data.drop(columns=["failure"])

    # Remove the datetime column
    data = data.drop(columns=["datetime"])

    return data

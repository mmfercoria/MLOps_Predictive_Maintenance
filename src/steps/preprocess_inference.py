from zenml import step
import pandas as pd

@step
def preprocess_inference() -> pd.DataFrame:
    telemetry = pd.read_csv("data/inference_telemetry.csv", parse_dates=["datetime"])
    machines = pd.read_csv("data/PdM_machines.csv")

    # Sort the data by machine and time
    telemetry = telemetry.sort_values(by=["machineID", "datetime"])

    # Calculate 3-hour average and std for each sensor
    rolling = telemetry.set_index("datetime") \
        .groupby("machineID")[["volt", "rotate", "pressure", "vibration"]] \
        .rolling("3h", min_periods=1) \
        .agg(['mean', 'std']) \
        .reset_index()

    # Rename the columns
    rolling.columns = ["machineID", "datetime"] + [f"{col[0]}_{col[1]}_3h" for col in rolling.columns[2:]]

    # Add machine info to the data
    data = pd.merge(rolling, machines, on="machineID", how="left")

    # Convert machine model (text) into numbers
    data["model"] = data["model"].astype(str)
    data = pd.get_dummies(data, columns=["model"])

    data = data.drop(columns=["datetime"])

    return data

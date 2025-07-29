from zenml import step
from zenml.client import Client
import joblib
import pandas as pd

@step
def predict(data: pd.DataFrame) -> pd.Series:
    # Get the latest saved model
    model_artifact = Client().get_artifact_version("rf_model_artifact")
    model_path = model_artifact.load()

    # Load the model from the file
    model = joblib.load(model_path)

    # Make predictions
    predictions = model.predict(data)

    return pd.Series(predictions)

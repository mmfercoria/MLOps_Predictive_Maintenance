from zenml import step, register_artifact
from zenml.client import Client
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime

@step
def save_model(model: RandomForestClassifier) -> None:
    client = Client()

    # Get the artifact store base path
    artifact_store_prefix = client.active_stack.artifact_store.path

    # Add timestamp to make path unique per run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"rf_model_{timestamp}.joblib"
    model_path = os.path.join(artifact_store_prefix, "my_models", model_filename)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # Save the model
    joblib.dump(model, model_path)

    # Register artifact with unique name (optional)
    register_artifact(
        folder_or_file_uri=model_path,
        name=f"rf_model_artifact_{timestamp}"
    )

    print(f"Model saved and registered at: {model_path}")

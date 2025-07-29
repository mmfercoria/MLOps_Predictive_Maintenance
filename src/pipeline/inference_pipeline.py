from zenml import Model, pipeline

from src.steps.predict import predict
from src.steps.preprocess_inference import preprocess_inference


@pipeline(model=Model(name="predictive_maintenance_model"))
def inference_pipeline():
    # Step 1: Prepare the new data
    data = preprocess_inference()

    # Step 2: Make predictions
    predictions = predict(data)

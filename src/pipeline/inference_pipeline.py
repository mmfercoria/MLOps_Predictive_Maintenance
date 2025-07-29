from zenml import pipeline, Model
from src.steps.preprocess_inference import preprocess_inference
from src.steps.predict import predict

@pipeline(model=Model(name="predictive_maintenance_model"))
def inference_pipeline():
    # Step 1: Prepare the new data
    data = preprocess_inference()
    
    # Step 2: Make predictions
    predictions = predict(data)

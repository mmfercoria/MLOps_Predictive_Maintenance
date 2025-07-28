from zenml import pipeline, Model
from src.steps.preprocess_data import preprocess_data
from src.steps.predict import predict

@pipeline(model=Model(name="predictive_maintenance_model"))
def inference_pipeline():
    features, _ = preprocess_data()
    predictions = predict(features=features)

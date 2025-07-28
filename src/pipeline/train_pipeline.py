from zenml import pipeline
from src.steps.preprocess_data import preprocess_data

@pipeline
def training_pipeline():
    data_preproc = preprocess_data()

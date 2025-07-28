from zenml import Model, pipeline
from src.steps.preprocess_data import preprocess_data
from src.steps.train_model import train_model
from src.steps.evaluate_model import evaluate_model

@pipeline(
    model=Model(
        name="predictive_maintenance_model",
        description="Random Forest model to predict machine failure",
        tags=["sklearn", "maintenance", "classification"]
    )
)

def training_pipeline():
    features, failures = preprocess_data()
    model, X_test, y_test = train_model(features, failures)
    #evaluate_model(model=model, X_test=X_test, y_test=y_test)


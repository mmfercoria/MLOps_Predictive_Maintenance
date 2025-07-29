from zenml import Model, pipeline
from src.steps.preprocess_data import preprocess_data
from src.steps.train_model import train_model
from src.steps.evaluate_model import evaluate_model
from src.steps.save_model import save_model


@pipeline(
    model=Model(
        name="predictive_maintenance_model",
        description="Random Forest model to predict machine failure",
        tags=["sklearn", "maintenance", "classification"]
    )
)

def training_pipeline():
    # Step 1: Preprocess data
    data = preprocess_data()

    # Step 2: Train model
    model, X_test, y_test = train_model(data)

    # Step 3: Evaluate model
    evaluate_model(model=model, X_test=X_test, y_test=y_test)

    # Step 4: Save model
    save_model(model=model)
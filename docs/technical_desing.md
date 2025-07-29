## About the model

A Random Forest model is trained using sensor readings (volt, rotate, pressure, vibration), processed with 3-hour rolling windows to compute mean and standard deviation per machine. Machine metadata like model and age is added, and categorical columns are one-hot encoded.

The model uses binary classification, where each data point is labeled 1 if a failure occurs within the next 24 hours, and 0 otherwise. 

---
## ML pipeline automation

The pipelines in this project were built using ZenML, an MLOps orchestration platform that simplifies building, running, and versioning machine learning workflows.

ZenML makes it easy to define steps like preprocessing, training, evaluation, model registration, and inference within a clean and modular structure. Its web interface allows users to explore pipeline runs, review saved artifacts, and track model versions over time.

![ZenML Pipelines UI](images/pipelines.png)

🔗 [ZenML Pipelines Dashboard](https://mmfercoria-mlops-predictive-maintenance.hf.space/projects/default/pipelines)


### Training

![ZenML Training](images/train_pipeline.png)

#### Steps:

1. **`preprocess_data`**:  
   - Loads telemetry and failure data.
   - Calculates rolling 3-hour features.
   - Assigns binary label (1 if failure occurs within 24h, else 0).
   - One-hot encodes machine model.

2. **`train_model`**:  
   - Trains a `RandomForestClassifier` using preprocessed data.
   - Splits the data into training and test sets.

3. **`evaluate_model`**:  
   - Evaluates model performance using metrics like AUC, confusion matrix, etc.
   - Saves results in a JSON file.

4. **`save_model`**:  
   - Saves the trained model in the ZenML artifact store.

### Inference

![ZenML Training](images/inference_pipeline.png)

#### Steps:

1. **`preprocess_inference`**:  
    - Loads telemetry data for prediction.
    - Computes rolling 3-hour features.
    - Merges machine metadata and encodes model.

2. **`predict`**:  
    - Loads the latest saved model.
    - Runs predictions on the new data.
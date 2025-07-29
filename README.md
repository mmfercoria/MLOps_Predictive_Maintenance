# MLOps Predictive Maintenance

This project showcases a complete MLOps workflow for predictive maintenance, leveraging modern machine learning and automation practices to predict machinery failures before they happen.

## Objective

To proactively detect potential machine failures using historical telemetry, and failure data. The ultimate goal is to reduce unplanned downtime through accurate failure predictions integrated into automated ML pipelines.

---

## Tech Stack

- **ZenML** – MLOps pipeline orchestration  
- **scikit-learn** – Model training and evaluation  
- **pandas / numpy** – Data manipulation  
- **joblib** – Model serialization  
- **Hugging Face Spaces** – Optional deployment interface  

---

## Project Structure

```bash
MLOps_Predictive_Maintenance/
│
├── README.md                 # Project overview
├── .gitignore
├── pyproject.toml           # Dependency management with uv
├── uv.lock                  # Locked dependencies
├── notebooks/               # Data exploration and experiments
├── data/                    # Raw and inference data
├── src/                     # Source code
│   ├── models/              # Model training and evaluation
│   ├── pipelines/           # ZenML pipeline definitions
│   └── steps/               # Modular pipeline steps
└── tests/                   # Unit and integration tests
```

---

## MLOps pipelines

Training Pipeline:
    1. Preprocess and feature engineering
    2. Train model
    3. Evaluate performance
    4. Register model artifact

Inference Pipeline:
    1. Preprocess data
    2. Load latest registered model
    3. Generate predictions

---

## 🚀 How to Run the Project

### 1. Install dependencies

```bash
uv venv
uv pip install -r requirements.txt
```
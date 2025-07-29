# ⚙️ MLOps Predictive Maintenance

This project showcases a complete MLOps workflow for predictive maintenance, leveraging modern machine learning and automation practices to predict machinery failures before they happen.

## 🎯 Objective

Build a complete machine learning solution that predicts whether a machine is likely to fail in the near future, using the provided dataset. The main goal is to proactively detect potential machine failures using historical telemetry and failure data. By integrating accurate failure predictions into automated ML pipelines, this project aims to minimize unplanned downtime and improve maintenance strategies in industrial settings.

This solution is powered by a public predictive maintenance dataset from **Microsoft Azure**, available on Kaggle:

[Microsoft Azure Predictive Maintenance Dataset on Kaggle](https://www.kaggle.com/datasets/arnabbiswas1/microsoft-azure-predictive-maintenance/data)

---

## Tech Stack

- **ZenML** – MLOps pipeline orchestration
- **scikit-learn** – Model training and evaluation
- **pandas / numpy** – Data manipulation
- **joblib** – Model serialization
- **Hugging Face Spaces** – Optional deployment interface

---

## 📁 Project Structure

```bash
MLOps_Predictive_Maintenance/
│
├── README.md                 # Project overview
├── .gitignore
├── pyproject.toml           # Dependency management with uv
├── uv.lock                  # Locked dependencies
├── data/                    # Raw and inference data
├── docs/                      # Project documentation
│   ├── technical_design.md    # Architecture, data flow, tools and rationale
│   └── short_report.md        # Concise summary of the technical solution
├── src/                     # Source code
│   ├── models/              # Model training and evaluation
│   ├── pipelines/           # ZenML pipeline definitions
│   └── steps/               # Modular pipeline steps
└── .pre-commit-config.yaml  # Linting and formatting hooks
```

---

## 🔄 MLOps Pipelines

This project includes two fully modular pipelines built with ZenML: one for training and another for inference.

### Training Pipeline

Steps:
1. **Preprocess and engineer features**
2. **Train the model**
3. **Evaluate model performance**
4. **Register the model artifact**

### Inference Pipeline

Steps:
1. **Preprocess input data**
2. **Load the latest registered model**
3. **Generate predictions**

You can explore the full pipeline executions and visualizations here:
🔗 [ZenML Pipelines Dashboard](https://mmfercoria-mlops-predictive-maintenance.hf.space/projects/default/pipelines)

> **This project is deployed and supported by [Hugging Face Spaces](https://huggingface.co/spaces/mmfercoria/MLOps_Predictive_Maintenance)** – enabling live interaction and visualization of MLOps workflows.

---

## 🚀 How to Run the Project

### Sync Project Dependencies

```bash
uv sync
```

### Authenticate and Set ZenML Stack

```bash
zenml login https://mmfercoria-mlops-predictive-maintenance.hf.space
zenml stack set hf_stack
```

###  Run Training and Inference Pipelines

```bash
python main_train.py
python main_inf.py
```
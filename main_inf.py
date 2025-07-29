from zenml.client import Client

from src.pipeline.inference_pipeline import inference_pipeline

if __name__ == "__main__":
    client = Client()

    # Ejecutar el pipeline
    inference_pipeline()

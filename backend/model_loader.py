import os
import mlflow
import mlflow.pyfunc

MODEL_NAME = os.getenv("MODEL_NAME", "churn-model")
MODEL_STAGE = os.getenv("MODEL_STAGE", "Staging")

def load_model():
    # Configuration de l'accès à DagsHub
    tracking_uri = os.environ["MLFLOW_TRACKING_URI"]
    token = os.environ["MLFLOW_TRACKING_TOKEN"]
    mlflow.set_tracking_uri(tracking_uri)
    os.environ["MLFLOW_TRACKING_USERNAME"] = token
    os.environ["MLFLOW_TRACKING_PASSWORD"] = token
    
    # Chargement dynamique selon le stade (Staging ou Production)
    uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
    return mlflow.pyfunc.load_model(uri) 
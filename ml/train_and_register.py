import os
import mlflow
import mlflow.sklearn
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

MODEL_NAME = os.getenv("MODEL_NAME", "churn-model")

def main():
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    
    # Génération de données factices
    X, y = make_classification(n_samples=2000, n_features=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    model = LogisticRegression()
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    
    with mlflow.start_run() as run:
        mlflow.log_metric("accuracy", float(acc))
        mlflow.sklearn.log_model(model, artifact_path="model")
        
        # Enregistrement du modèle dans le registre
        model_uri = f"runs:/{run.info.run_id}/model"
        mlflow.register_model(model_uri=model_uri, name=MODEL_NAME)
        
        print(f'{{"run_id": "{run.info.run_id}", "accuracy": {acc}, "model_version": "latest"}}')

if __name__ == "__main__":
    main()
# --- FIX MLFLOW URI EN WINDOWS ---
from pathlib import Path
import os

def force_mlflow_uri(mlruns_dir: Path):
    mlruns_dir.mkdir(parents=True, exist_ok=True)
    uri = mlruns_dir.resolve().as_uri()  # file:///C:/...
    os.environ["MLFLOW_TRACKING_URI"] = uri
    os.environ["MLFLOW_REGISTRY_URI"] = uri
    return uri

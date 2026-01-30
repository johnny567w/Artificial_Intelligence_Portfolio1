# --- REENTRENAMIENTO DESDE NUEVA DATA (HUMAN + AUTO) ---
from pathlib import Path
from datetime import datetime
import json
import shutil

import mlflow
from ultralytics import YOLO

from src.utils.mlflow_fix import force_mlflow_uri
from src.retraining.ingest import list_pairs, ingest_pairs

def retrain_from_new_data(
    model_path: Path,
    data_yaml_path: Path,
    train_images_dir: Path,
    train_labels_dir: Path,
    auto_images_dir: Path,
    auto_labels_dir: Path,
    human_images_dir: Path,
    human_labels_dir: Path,
    runs_dir: Path,
    mlruns_dir: Path,
    artifacts_dir: Path,
    epochs: int,
    imgsz: int,
    batch: int,
    seed: int
):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1) scan
    human = list_pairs(human_images_dir, human_labels_dir)
    auto  = list_pairs(auto_images_dir, auto_labels_dir)

    if len(human) == 0 and len(auto) == 0:
        return {"status": "no_new_data"}

    # 2) ingest (HUMAN primero)
    human_ing = ingest_pairs(human, train_images_dir, train_labels_dir, prefix=f"human_{ts}")
    auto_ing  = ingest_pairs(auto,  train_images_dir, train_labels_dir, prefix=f"auto_{ts}")

    # 3) MLflow fix
    uri = force_mlflow_uri(mlruns_dir)
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment("coco2017_car_airplane_truck")

    # 4) train
    model = YOLO(str(model_path))
    run_name = f"app_retrain_{ts}"

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "stage": "app_retrain",
            "epochs": int(epochs),
            "imgsz": int(imgsz),
            "batch": int(batch),
            "human_ingested": int(human_ing),
            "auto_ingested": int(auto_ing),
        })

        model.train(
            data=str(data_yaml_path),
            epochs=int(epochs),
            imgsz=int(imgsz),
            batch=int(batch),
            seed=int(seed),
            device="cpu",
            workers=0,
            cache=False,
            project=str(runs_dir),
            name=run_name,
            pretrained=True,
            patience=3,
            verbose=True
        )

        save_dir = Path(model.trainer.save_dir)
        best = save_dir / "weights" / "best.pt"
        if not best.exists():
            raise FileNotFoundError(f"No se generó best.pt en {save_dir}")

        # backup + update model
        backup = model_path.parent / f"yolo_best_backup_{ts}.pt"
        shutil.copy2(model_path, backup)
        shutil.copy2(best, model_path)

        # val + metrics
        val = model.val(data=str(data_yaml_path), imgsz=int(imgsz), batch=int(batch), verbose=False)
        d = val.results_dict
        metrics = {
            "precision": float(d.get("metrics/precision(B)", 0)),
            "recall": float(d.get("metrics/recall(B)", 0)),
            "mAP50": float(d.get("metrics/mAP50(B)", 0)),
            "mAP50_95": float(d.get("metrics/mAP50-95(B)", 0)),
        }
        mlflow.log_metrics(metrics)

        # artifacts
        mlflow.log_artifact(str(model_path), artifact_path="model")
        mlflow.log_artifact(str(backup), artifact_path="model_backup")

    return {"status": "ok", "metrics": metrics}

# --- PATHS DEL PROYECTO ---
from pathlib import Path
import json

def get_paths():
    project_root = Path(__file__).resolve().parents[2]

    artifacts = project_root / "artifacts"
    cfg_path = artifacts / "config_snapshot.json"

    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    paths = {
        "project_root": project_root,
        "artifacts_dir": artifacts,
        "model_path": project_root / "models" / "yolo_best.pt",
        "data_yaml_path": project_root / "data" / "processed" / "yolo_dataset" / "data.yaml",

        "train_images_dir": project_root / "data" / "processed" / "yolo_dataset" / "images" / "train",
        "train_labels_dir": project_root / "data" / "processed" / "yolo_dataset" / "labels" / "train",

        "auto_images_dir": project_root / "data" / "new" / "auto" / "images",
        "auto_labels_dir": project_root / "data" / "new" / "auto" / "labels",

        "human_images_dir": project_root / "data" / "new" / "human" / "images",
        "human_labels_dir": project_root / "data" / "new" / "human" / "labels",

        "runs_dir": project_root / "runs",
        "mlruns_dir": project_root / "mlruns",

        "imgsz": cfg.get("img_size", 640),
        "batch": cfg.get("batch_size", 8),
        "seed": cfg.get("seed", 42),
    }

    for k in ["auto_images_dir","auto_labels_dir","human_images_dir","human_labels_dir","runs_dir","mlruns_dir"]:
        paths[k].mkdir(parents=True, exist_ok=True)

    return paths

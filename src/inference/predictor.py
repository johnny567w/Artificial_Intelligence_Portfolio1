# --- PREDICCIÓN YOLO + GUARDADO (AUTO/HUMAN) - ROBUSTO ---
import time
from pathlib import Path
from typing import Dict, List
import numpy as np
from PIL import Image
from ultralytics import YOLO

from src.inference.visualize import draw_boxes

IMG_QUALITY = 95

def _to_yolo_line(cls_id: int, xyxy, w: int, h: int) -> str:
    x1, y1, x2, y2 = [float(v) for v in xyxy]
    bw = x2 - x1
    bh = y2 - y1
    xc = x1 + bw / 2
    yc = y1 + bh / 2
    # normalizado [0,1]
    return f"{cls_id} {xc/w:.6f} {yc/h:.6f} {bw/w:.6f} {bh/h:.6f}"

def _extract_detections(result, names: Dict[int, str]) -> List[Dict]:
    """
    Extrae detecciones de forma robusta para distintas versiones de ultralytics.
    Devuelve: [{cls_id, name, conf, xyxy}, ...]
    """
    dets = []

    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return dets

    # xyxy / cls / conf pueden venir como tensors
    try:
        xyxy_t = boxes.xyxy
        cls_t  = boxes.cls
        conf_t = boxes.conf

        if xyxy_t is None or cls_t is None or conf_t is None:
            return dets

        xyxy = xyxy_t.cpu().numpy()
        cls  = cls_t.cpu().numpy().astype(int)
        conf = conf_t.cpu().numpy()

        n = min(len(conf), len(cls), len(xyxy))
        for i in range(n):
            cid = int(cls[i])
            dets.append({
                "cls_id": cid,
                "name": names.get(cid, str(cid)),
                "conf": float(conf[i]),
                "xyxy": [float(x) for x in xyxy[i].tolist()]
            })
        return dets

    except Exception:
        # Si algo raro pasa, no revientes la app
        return dets

def predict_and_save(
    pil_image: Image.Image,
    model_path: Path,
    conf: float,
    iou: float,
    auto_images_dir: Path,
    auto_labels_dir: Path,
    human_images_dir: Path,
    human_labels_dir: Path,
    send_to_human: bool
) -> Dict:

    model = YOLO(str(model_path))
    img_np = np.array(pil_image)
    h, w = img_np.shape[:2]

    preds = model.predict(img_np, conf=conf, iou=iou, verbose=False)
    r = preds[0]

    dets = _extract_detections(r, model.names)

    labels_present = sorted(list({d["name"] for d in dets}))
    annotated = draw_boxes(pil_image, dets)

    # --- si no hay detecciones, NO guardamos label ---
    if len(dets) == 0:
        return {
            "labels_present": [],
            "num_classes_present": 0,
            "num_detections": 0,
            "saved": False,
            "saved_where": None,
            "image_path": None,
            "label_path": None,
            "detections": [],
            "annotated_image": annotated
        }

    ts = str(int(time.time() * 1000))
    img_name = f"new_{ts}.jpg"
    lbl_name = f"new_{ts}.txt"

    if send_to_human:
        img_dir, lbl_dir = human_images_dir, human_labels_dir
        prefix = "human"
    else:
        img_dir, lbl_dir = auto_images_dir, auto_labels_dir
        prefix = "auto"

    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    img_path = img_dir / img_name
    lbl_path = lbl_dir / lbl_name

    pil_image.save(img_path, format="JPEG", quality=IMG_QUALITY)

    lines = [_to_yolo_line(d["cls_id"], d["xyxy"], w, h) for d in dets]
    lbl_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "labels_present": labels_present,
        "num_classes_present": len(labels_present),
        "num_detections": len(dets),
        "saved": True,
        "saved_where": prefix,
        "image_path": str(img_path),
        "label_path": str(lbl_path),
        "detections": dets,
        "annotated_image": annotated
    }

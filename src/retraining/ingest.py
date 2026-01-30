# --- INGESTA HUMAN + AUTO (HUMAN PRIMERO) ---
from pathlib import Path
import shutil

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

def list_pairs(images_dir: Path, labels_dir: Path):
    pairs = []
    for img in images_dir.iterdir():
        if not img.is_file() or img.suffix.lower() not in IMG_EXTS:
            continue
        lbl = labels_dir / f"{img.stem}.txt"
        if lbl.exists():
            pairs.append((img, lbl))
    return pairs

def ingest_pairs(pairs, train_images_dir: Path, train_labels_dir: Path, prefix: str):
    train_images_dir.mkdir(parents=True, exist_ok=True)
    train_labels_dir.mkdir(parents=True, exist_ok=True)
    ingested = 0
    for img, lbl in pairs:
        shutil.copy2(img, train_images_dir / f"{prefix}_{img.name}")
        shutil.copy2(lbl, train_labels_dir / f"{prefix}_{lbl.name}")
        ingested += 1
    return ingested

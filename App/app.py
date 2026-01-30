# --- APP LOCAL: PREDICT + RETRAIN (UI MEJORADA + BUGFIXES) ---
import sys
from pathlib import Path

# --- FIX IMPORTS: AÑADIR PROJECT ROOT AL PYTHONPATH ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
from PIL import Image
import pandas as pd

from src.utils.paths import get_paths
from src.inference.predictor import predict_and_save
from src.retraining.retrain import retrain_from_new_data

st.set_page_config(page_title="YOLO Detector", layout="wide")

paths = get_paths()

st.title("Detección de Objetos — car / airplane / truck")
st.caption("App local: predicción + reentrenamiento continuo (AUTO + HUMAN)")

# --- SIDEBAR / CONTROLES ---
with st.sidebar:
    st.header("Controles")

    conf = st.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05)
    iou  = st.slider("IoU threshold", 0.10, 0.95, 0.50, 0.05)

    st.divider()
    st.subheader("Guardado de nueva data")
    to_review = st.toggle("Marcar para revisión manual (HUMAN)", value=False)
    st.caption("AUTO: pseudo-labels de la predicción. HUMAN: datos para corregir/etiquetar manualmente.")

    st.divider()
    st.subheader("Reentrenamiento")
    epochs = st.number_input("Épocas", min_value=1, max_value=50, value=10, step=1)

    with st.expander("Rutas (debug)", expanded=False):
        st.write("Modelo:", str(paths["model_path"]))
        st.write("Dataset:", str(paths["data_yaml_path"]))
        st.write("AUTO images:", str(paths["auto_images_dir"]))
        st.write("AUTO labels:", str(paths["auto_labels_dir"]))
        st.write("HUMAN images:", str(paths["human_images_dir"]))
        st.write("HUMAN labels:", str(paths["human_labels_dir"]))


# --- LAYOUT PRINCIPAL ---
left, right = st.columns([1, 1.25], gap="large")

with left:
    st.subheader("Entrada")
    uploaded = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png", "webp"])

    c1, c2 = st.columns(2)
    with c1:
        predict_btn = st.button("Predict", use_container_width=True)
    with c2:
        retrain_btn = st.button("Retrain", use_container_width=True)

    st.info("CPU tip: si no detecta nada, baja confidence a 0.10–0.20 y usa imágenes nítidas.")

with right:
    st.subheader("Resultado")
    result_container = st.container()


# --- HELPERS ---
def summarize_detections(detections):
    """
    Retorna un DataFrame resumen por clase:
    class | count | avg_conf_% | max_conf_%
    """
    if not detections:
        return None

    df = pd.DataFrame([{
        "class": d["name"],
        "conf_%": round(float(d["conf"]) * 100, 2)
    } for d in detections])

    summary = (
        df.groupby("class")["conf_%"]
          .agg(["count", "mean", "max"])
          .reset_index()
          .rename(columns={"mean": "avg_conf_%", "max": "max_conf_%"})
    )
    summary = summary.sort_values(by=["count", "max_conf_%"], ascending=[False, False]).reset_index(drop=True)
    summary["avg_conf_%"] = summary["avg_conf_%"].round(2)
    summary["max_conf_%"] = summary["max_conf_%"].round(2)
    return summary

def build_top_detections_df(detections):
    """
    Retorna DataFrame ordenado por confidence_% desc:
    class | confidence_% | bbox_xyxy
    """
    if not detections:
        return None

    df = pd.DataFrame([{
        "class": d["name"],
        "confidence_%": round(float(d["conf"]) * 100, 2),
        "bbox_xyxy": [round(float(x), 1) for x in d["xyxy"]],
    } for d in detections])

    if df.empty or "confidence_%" not in df.columns:
        return None

    return df.sort_values("confidence_%", ascending=False).reset_index(drop=True)


# --- PREDICT ---
if predict_btn:
    if uploaded is None:
        st.warning("Sube una imagen primero.")
    else:
        img = Image.open(uploaded).convert("RGB")

        with right:
            st.image(img, caption="Imagen cargada", use_container_width=True)

        with st.spinner("Ejecutando predicción..."):
            out = predict_and_save(
                pil_image=img,
                model_path=paths["model_path"],
                conf=conf,
                iou=iou,
                auto_images_dir=paths["auto_images_dir"],
                auto_labels_dir=paths["auto_labels_dir"],
                human_images_dir=paths["human_images_dir"],
                human_labels_dir=paths["human_labels_dir"],
                send_to_human=to_review
            )

        dets = out.get("detections", []) or []  # <- la verdad para la UI
        summary_df = summarize_detections(dets)
        top_df = build_top_detections_df(dets)

        with result_container:
            # --- BUGFIX: decidir por dets, no por num_detections ---
            if not dets:
                st.error("No se detectaron objetos con esos umbrales.")
                st.caption("Prueba bajar confidence (0.10–0.20) o usar otra imagen (mejor iluminación/ángulo).")
                st.image(out["annotated_image"], caption="Resultado (sin detecciones)", use_container_width=True)
            else:
                # KPIs
                st.success(f"Detectó {out.get('num_classes_present', 0)} clase(s) y {len(dets)} detección(es).")
                k1, k2, k3 = st.columns(3)
                k1.metric("Clases", out.get("num_classes_present", 0))
                k2.metric("Detecciones", len(dets))
                k3.metric("Destino", out.get("saved_where", "auto" if not to_review else "human"))

                st.write("**Clases detectadas:**", ", ".join(out.get("labels_present", [])))

                # Imagen anotada
                st.image(out["annotated_image"], caption="Resultado (con cajas y etiqueta)", use_container_width=True)

                # Resumen por clase
                st.markdown("### Resumen por clase")
                if summary_df is not None and not summary_df.empty:
                    st.dataframe(summary_df, use_container_width=True, hide_index=True)
                else:
                    st.info("No hay datos suficientes para el resumen.")

                # Top detecciones
                st.markdown("### Top detecciones (ordenadas por confianza)")
                if top_df is not None and not top_df.empty:
                    st.dataframe(top_df.head(10), use_container_width=True, hide_index=True)
                else:
                    st.info("No hay detecciones para mostrar.")

                # Detalles de guardado (solo si saved=True)
                if out.get("saved", False):
                    with st.expander("Detalles de guardado", expanded=False):
                        st.write("Destino:", out.get("saved_where"))
                        st.write("Imagen:", out.get("image_path"))
                        st.write("Label:", out.get("label_path"))


# --- RETRAIN ---
if retrain_btn:
    with st.spinner("Reentrenando... (en CPU puede tardar bastante)"):
        result = retrain_from_new_data(
            model_path=paths["model_path"],
            data_yaml_path=paths["data_yaml_path"],
            train_images_dir=paths["train_images_dir"],
            train_labels_dir=paths["train_labels_dir"],
            auto_images_dir=paths["auto_images_dir"],
            auto_labels_dir=paths["auto_labels_dir"],
            human_images_dir=paths["human_images_dir"],
            human_labels_dir=paths["human_labels_dir"],
            runs_dir=paths["runs_dir"],
            mlruns_dir=paths["mlruns_dir"],
            artifacts_dir=paths["artifacts_dir"],
            epochs=int(epochs),
            imgsz=int(paths["imgsz"]),
            batch=int(paths["batch"]),
            seed=int(paths["seed"])
        )

    if result["status"] == "no_new_data":
        st.info("No hay nueva data válida en AUTO/HUMAN para reentrenar.")
    else:
        st.success("Reentrenamiento completado. Modelo actualizado.")
        m = result["metrics"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Precision", f"{m['precision']:.3f}")
        c2.metric("Recall", f"{m['recall']:.3f}")
        c3.metric("mAP50", f"{m['mAP50']:.3f}")
        c4.metric("mAP50-95", f"{m['mAP50_95']:.3f}")
        st.caption(f"Modelo actualizado: {paths['model_path']}")

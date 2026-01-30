# --- DIBUJO DE BOXES CON ETIQUETA + % ---
from PIL import Image, ImageDraw, ImageFont

def draw_boxes(pil_image, dets):
    img = pil_image.copy()
    draw = ImageDraw.Draw(img)

    # Fuente por defecto (evita depender de archivos)
    try:
        font = ImageFont.load_default()
    except:
        font = None

    for d in dets:
        x1, y1, x2, y2 = d["xyxy"]
        x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)

        label = f'{d["name"]} {d["conf"]*100:.1f}%'

        # Caja
        draw.rectangle([x1, y1, x2, y2], width=3)

        # Fondo del texto (para que no se pierda)
        if font is not None:
            text_w, text_h = draw.textbbox((0, 0), label, font=font)[2:]
        else:
            text_w, text_h = (len(label) * 6, 12)

        pad = 3
        tx1 = x1
        ty1 = max(0, y1 - (text_h + 2*pad))
        tx2 = x1 + text_w + 2*pad
        ty2 = ty1 + text_h + 2*pad

        draw.rectangle([tx1, ty1, tx2, ty2], fill="black")
        draw.text((tx1 + pad, ty1 + pad), label, fill="white", font=font)

    return img

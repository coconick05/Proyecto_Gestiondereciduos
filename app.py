import base64
import io
import os
import threading
import unicodedata
import uuid

import numpy as np
import paho.mqtt.client as mqtt
import streamlit as st
from keras.models import load_model
from PIL import Image, ImageOps

# ================== CONFIGURACIÓN ==================
BROKER = "broker.mqttdashboard.com"
PUERTO = 1883
PREFIJO = "nicolemp7391/tribin"  # debe ser IGUAL al del sketch del ESP32
TOPIC_ABRIR = f"{PREFIJO}/abrir"
TOPIC_ESTADO = f"{PREFIJO}/estado"
UMBRAL = 0.99                    # 0.99 = "100%" en Teachable Machine
ESPERA_LISTO_S = 20              # segundos máximos esperando que el ESP32 cierre
IMAGEN_HEADER = "header.png"     # la ilustración (misma carpeta que este archivo)
# ===================================================

st.set_page_config(page_title="TriBin", page_icon="♻️", layout="centered")

# ================== ESTILO (papel recortado) ==================
# Paleta tomada de la ilustración:
#   crema #F7F5F1 · papel #FFFDF8 · hoja #5E9B3F · bosque #3E7B3A
#   globo #A9D1C6 · tejado #D9783F · sol #F2B84B · tierra #8A5A3B
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fredoka:wght@500;600;700&family=Nunito:wght@400;600;700&display=swap');

:root{
  --crema:#F7F5F1; --papel:#FFFDF8; --hoja:#5E9B3F; --bosque:#3E7B3A;
  --globo:#A9D1C6; --tejado:#D9783F; --sol:#F2B84B; --tierra:#8A5A3B;
  --borde:#E4D9BD; --texto:#3B3A30;
}

/* Fondo crema con grano de papel */
.stApp{
  background-color:var(--crema);
  background-image:url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='220' height='220'><filter id='n'><feTurbulence type='fractalNoise' baseFrequency='0.8' numOctaves='2' stitchTiles='stitch'/><feColorMatrix values='0 0 0 0 0.55  0 0 0 0 0.45  0 0 0 0 0.25  0 0 0 0.07 0'/></filter><rect width='100%' height='100%' filter='url(%23n)'/></svg>");
  color:var(--texto);
}
html, body, .stApp, .stMarkdown, p, label, input, textarea, button{ font-family:'Nunito',sans-serif; }

/* Los íconos de Streamlit necesitan su propia fuente */
[data-testid="stIconMaterial"],
.material-symbols-rounded,
span[class*="material"]{
  font-family:'Material Symbols Rounded' !important;
  font-weight:400 !important;
}

/* Quitar elementos de Streamlit que ensucian */
#MainMenu, footer, header[data-testid="stHeader"]{ visibility:hidden; height:0; }
.block-container{ max-width:680px; padding-top:1.5rem; padding-bottom:3rem; }

/* Título */
h1{
  font-family:'Fredoka',sans-serif !important; font-weight:700 !important;
  color:var(--bosque) !important; text-align:center; letter-spacing:.01em;
  padding:0 !important; margin-bottom:.1rem !important;
}
.subtitulo{ text-align:center; color:var(--tierra); font-weight:600; margin:0 0 1.2rem; }

/* Ilustración principal */
.hero{ display:block; margin:0 auto .4rem; width:min(300px,70%); height:auto;
       mix-blend-mode:multiply; }

/* Pestañas como etiquetas de papel */
div[data-baseweb="tab-list"]{ gap:10px; border-bottom:none; justify-content:center; }
button[data-baseweb="tab"]{
  font-family:'Fredoka',sans-serif; font-size:1.05rem; font-weight:600;
  background:#F1E7CC; color:var(--tierra);
  border-radius:18px 16px 4px 4px; padding:.55rem 1.4rem; height:auto;
  box-shadow:0 -2px 0 var(--borde) inset;
}
button[data-baseweb="tab"][aria-selected="true"]{ background:var(--papel); color:var(--bosque); }
div[data-baseweb="tab-highlight"]{ background-color:var(--tejado); height:4px; border-radius:4px; }
div[data-baseweb="tab-border"]{ display:none; }

/* Hoja de papel que contiene cada pestaña */
div[data-baseweb="tab-panel"]{
  background:var(--papel); padding:1.4rem 1.4rem 1.6rem; margin-top:-2px;
  border-radius:6px 6px 30px 24px;
  box-shadow:0 2px 0 var(--borde), 0 10px 0 -4px #EFE6CF, 0 14px 22px rgba(90,70,30,.10);
}

/* Botones */
.stButton > button, [data-testid="stBaseButton-primary"], [data-testid="stBaseButton-secondary"]{
  font-family:'Fredoka',sans-serif; font-weight:600; font-size:1.05rem;
  border-radius:20px 16px 22px 14px; border:none; padding:.6rem 1.6rem;
  transition:transform .12s ease, box-shadow .12s ease;
}
.stButton > button[kind="primary"], [data-testid="stBaseButton-primary"]{
  background:var(--hoja); color:#fff; box-shadow:0 4px 0 var(--bosque);
}
.stButton > button[kind="primary"]:hover, [data-testid="stBaseButton-primary"]:hover{
  background:#6BAA4A; color:#fff; transform:translateY(-1px); box-shadow:0 5px 0 var(--bosque);
}
.stButton > button[kind="primary"]:active, [data-testid="stBaseButton-primary"]:active{
  transform:translateY(3px); box-shadow:0 1px 0 var(--bosque);
}
[data-testid="stBaseButton-secondary"]{
  background:var(--sol); color:#5A3E10; box-shadow:0 4px 0 #C98F22;
}
.stButton > button:focus-visible{ outline:3px solid var(--tejado); outline-offset:2px; }

/* Botones internos de subir archivo y cámara (más pequeños) */
[data-testid="stFileUploaderDropzone"] [data-testid="stBaseButton-secondary"],
[data-testid="stCameraInput"] [data-testid="stBaseButton-secondary"],
[data-testid="stFileUploaderDropzone"] button,
[data-testid="stCameraInput"] button{
  font-family:'Nunito',sans-serif; font-size:.95rem; font-weight:700;
  padding:.4rem 1rem; border-radius:14px 12px 16px 10px;
  background:var(--sol); color:#5A3E10; box-shadow:0 3px 0 #C98F22; border:none;
}

/* Campos */
.stTextInput input{
  background:#FFFBEF; border:2px solid #D9C9A0; border-radius:16px; padding:.7rem 1rem;
  font-size:1.05rem; color:var(--texto);
}
.stTextInput input:focus{ border-color:var(--hoja); box-shadow:0 0 0 3px rgba(94,155,63,.2); }
[data-testid="stFileUploaderDropzone"]{
  background:#FBF4E1; border:2px dashed #C9B27A; border-radius:20px;
}
[data-testid="stCameraInput"] video, [data-testid="stCameraInput"] img{ border-radius:20px; }
[data-testid="stCameraInput"] > div{ border-radius:20px; }

/* Barras de probabilidad */
[data-testid="stProgress"] > div > div{ background:#EFE6CF; border-radius:999px; height:12px; }
[data-testid="stProgress"] div[role="progressbar"] > div{ background:var(--hoja); border-radius:999px; }

/* Avisos */
[data-testid="stAlert"]{ border-radius:20px 16px 22px 14px; border:none; }

/* Foto analizada como foto pegada */
.foto{
  display:block; width:250px; max-width:100%; height:auto; margin:.6rem 0 1.2rem;
  border:8px solid var(--papel); border-radius:10px;
  box-shadow:0 6px 16px rgba(90,70,30,.18); transform:rotate(-1.5deg);
}

/* Etiqueta de resultado */
.resultado{
  display:inline-block; padding:14px 30px; margin:.4rem 0 1rem; color:#fff;
  font-family:'Fredoka',sans-serif; font-size:1.9rem; font-weight:700;
  border-radius:26px 18px 30px 16px; transform:rotate(-1deg);
  box-shadow:0 5px 0 rgba(0,0,0,.18), 0 12px 20px rgba(60,50,20,.15);
}

@media (prefers-reduced-motion: reduce){ *{ transition:none !important; } }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def hero_html() -> str:
    """Ilustración incrustada; si no está el archivo, simplemente no se muestra."""
    if not os.path.exists(IMAGEN_HEADER):
        return ""
    with open(IMAGEN_HEADER, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f"<img class='hero' alt='Planeta con símbolo de reciclaje' src='data:image/png;base64,{b64}'/>"


def foto_html(img: Image.Image) -> str:
    """Foto del residuo incrustada en la página (no depende del servidor de medios)."""
    img = img.convert("RGB")
    img.thumbnail((600, 600))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"<img class='foto' alt='Residuo analizado' src='data:image/jpeg;base64,{b64}'/>"


CATEGORIAS = {
    "reciclable": ("♻️ RECICLABLE", "#4F8FB5"),
    "organico":   ("🍌 ORGÁNICO",   "#5E9B3F"),
    "ordinario":  ("🗑️ ORDINARIO",  "#6E6659"),
}

PALABRAS = {
    "organico": ["banano", "banana", "platano", "manzana", "naranja", "cascara", "fruta",
                 "verdura", "comida", "sobras", "huevo", "cafe", "borra", "hoja", "pan",
                 "arroz", "carne", "hueso", "lechuga", "tomate", "papa", "pasto", "flor"],
    "reciclable": ["botella", "plastico", "papel", "carton", "lata", "vidrio", "periodico",
                   "revista", "cuaderno", "caja", "aluminio", "tetrapak", "envase", "tarro"],
    "ordinario": ["servilleta", "icopor", "paquete de papas", "empaque", "chicle", "colilla",
                  "panal", "papel higienico", "bolsa sucia", "envoltura", "cabello",
                  "tapabocas", "ceramica", "espejo"],
}


def norm(s: str) -> str:
    s = unicodedata.normalize("NFD", s.lower())
    return "".join(c for c in s if unicodedata.category(c) != "Mn").strip()


# ---------------- Modelo ----------------
@st.cache_resource
def cargar_modelo():
    modelo = load_model("keras_model.h5", compile=False)
    with open("labels.txt", "r", encoding="utf-8") as f:
        # cada línea es "0 reciclable" -> nos quedamos con el nombre
        etiquetas = [norm(l.split(" ", 1)[1]) for l in f.read().splitlines() if l.strip()]
    return modelo, etiquetas


def clasificar_imagen(img: Image.Image):
    modelo, etiquetas = cargar_modelo()
    img = ImageOps.fit(img.convert("RGB"), (224, 224), Image.Resampling.LANCZOS)
    arr = np.asarray(img).astype(np.float32) / 127.5 - 1
    data = np.expand_dims(arr, axis=0)
    pred = modelo.predict(data, verbose=0)[0]
    probs = {etiquetas[i]: float(pred[i]) for i in range(len(etiquetas))}
    top = max(probs, key=probs.get)
    return top, probs[top], probs


def clasificar_texto(texto: str):
    t = norm(texto)
    return [c for c, ps in PALABRAS.items() if any(norm(p) in t for p in ps)]


# ---------------- MQTT ----------------
def abrir_y_esperar(categoria: str):
    """Publica el comando y espera 'abierta' y 'listo' del ESP32."""
    abierta, listo, suscrito = threading.Event(), threading.Event(), threading.Event()

    def on_message(client, userdata, msg):
        m = msg.payload.decode().strip().lower()
        if m == "abierta":
            abierta.set()
        elif m == "listo":
            listo.set()

    def on_subscribe(client, userdata, mid, reason_codes, properties):
        suscrito.set()

    c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2,
                    client_id=f"streamlit-tribin-{uuid.uuid4().hex[:8]}")
    c.on_message = on_message
    c.on_subscribe = on_subscribe
    c.connect(BROKER, PUERTO, keepalive=30)
    c.loop_start()
    try:
        c.subscribe(TOPIC_ESTADO)
        suscrito.wait(5)                      # suscribirse ANTES de publicar
        c.publish(TOPIC_ABRIR, categoria, qos=1)
        abierta.wait(8)
        listo.wait(ESPERA_LISTO_S)
    finally:
        c.loop_stop()
        c.disconnect()
    return abierta.is_set(), listo.is_set()


# ---------------- Estado de pantalla ----------------
if "pantalla" not in st.session_state:
    st.session_state.pantalla = "entrada"
    st.session_state.n = 0          # para reiniciar los widgets


def reiniciar():
    st.session_state.pantalla = "entrada"
    st.session_state.n += 1


if st.session_state.pantalla == "entrada":
    st.markdown(hero_html(), unsafe_allow_html=True)
st.title("¿Dónde va?")

# =============== PANTALLA 1: ENTRADA ===============
if st.session_state.pantalla == "entrada":
    st.markdown("<p class='subtitulo'>Toma una foto o escribe qué residuo tienes "
                "y te abro la caneca correcta.</p>", unsafe_allow_html=True)
    n = st.session_state.n
    tab_foto, tab_texto = st.tabs(["📷 Foto", "⌨️ Texto"])
    categoria, imagen = None, None

    with tab_foto:
        foto = st.camera_input("Toma una foto del residuo", key=f"cam{n}")
        subida = st.file_uploader("…o sube una foto", type=["jpg", "jpeg", "png"], key=f"up{n}")
        archivo = foto or subida
        if archivo and st.button("Analizar foto", type="primary", key=f"bf{n}"):
            imagen = Image.open(archivo).convert("RGB")
            top, conf, probs = clasificar_imagen(imagen)
            for nombre, p in sorted(probs.items(), key=lambda x: -x[1]):
                st.progress(p, text=f"{nombre}: {p*100:.1f}%")
            if top in CATEGORIAS and conf >= UMBRAL:
                categoria = top
            else:
                st.warning(f"Solo estoy {conf*100:.1f}% seguro. "
                           "La caneca NO se abrió. Intenta con otra foto o descríbelo con texto.")

    with tab_texto:
        txt = st.text_input("Describe tu residuo", placeholder="ej: cáscara de banano", key=f"tx{n}")
        if st.button("Analizar texto", type="primary", key=f"bt{n}"):
            hallados = clasificar_texto(txt)
            if len(hallados) == 1:
                categoria = hallados[0]
            elif len(hallados) > 1:
                st.warning("Tu descripción mezcla varios tipos de residuo. Analiza uno por uno.")
            else:
                st.warning("No reconocí ese residuo. La caneca NO se abrió. Prueba con una foto.")

    if categoria:
        st.session_state.categoria = categoria
        st.session_state.imagen = imagen
        st.session_state.pantalla = "resultado"
        st.rerun()

# =============== PANTALLA 2: RESULTADO ===============
elif st.session_state.pantalla == "resultado":
    cat = st.session_state.categoria
    etiqueta, color = CATEGORIAS[cat]
    if st.session_state.get("imagen") is not None:
        st.markdown(foto_html(st.session_state.imagen), unsafe_allow_html=True)
    st.write("Va en la caneca:")
    st.markdown(f"<div class='resultado' style='background:{color}'>{etiqueta}</div>",
                unsafe_allow_html=True)

    with st.spinner("Abriendo caneca… deposita tu residuo"):
        try:
            abierta, listo = abrir_y_esperar(cat)
        except Exception as e:
            abierta, listo = False, False
            st.error(f"No pude conectar al broker MQTT: {e}")

    if listo:
        st.session_state.pantalla = "listo"
        st.rerun()
    else:
        st.error("⚠️ El ESP32 no respondió. ¿Está corriendo la simulación en Wokwi?")
        st.button("Volver", on_click=reiniciar)

# =============== PANTALLA 3: LISTO ===============
elif st.session_state.pantalla == "listo":
    st.markdown(hero_html(), unsafe_allow_html=True)
    st.success("✅ Ya quedó listo. La caneca se cerró. ¡Gracias por reciclar bien!")
    st.button("Analizar otro residuo", type="primary", on_click=reiniciar)

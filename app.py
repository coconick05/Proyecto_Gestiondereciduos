import threading
import unicodedata
import uuid

import numpy as np
import paho.mqtt.client as mqtt
import streamlit as st
from keras.models import load_model
from PIL import Image, ImageOps

# ================== CONFIGURACIÓN ==================
BROKER = "broker.hivemq.com"
PUERTO = 1883
PREFIJO = "nicolemp/tribin"      # debe ser IGUAL al del sketch del ESP32
TOPIC_ABRIR = f"{PREFIJO}/abrir"
TOPIC_ESTADO = f"{PREFIJO}/estado"
UMBRAL = 0.99                    # 0.99 = "100%" en Teachable Machine
ESPERA_LISTO_S = 20              # segundos máximos esperando que el ESP32 cierre
# ===================================================

st.set_page_config(page_title="TriBin", page_icon="♻️", layout="centered")

CATEGORIAS = {
    "reciclable": ("♻️ RECICLABLE", "#1e63d6"),
    "organico":   ("🍌 ORGÁNICO",   "#2e9e4f"),
    "ordinario":  ("🗑️ ORDINARIO",  "#374151"),
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


st.title("♻️ ¿Dónde va?")

# =============== PANTALLA 1: ENTRADA ===============
if st.session_state.pantalla == "entrada":
    n = st.session_state.n
    tab_foto, tab_texto = st.tabs(["📷 Foto", "⌨️ Texto"])
    categoria, imagen = None, None

    with tab_foto:
        foto = st.camera_input("Toma una foto del residuo", key=f"cam{n}")
        subida = st.file_uploader("…o sube una foto", type=["jpg", "jpeg", "png"], key=f"up{n}")
        archivo = foto or subida
        if archivo and st.button("Analizar foto", type="primary", key=f"bf{n}"):
            imagen = Image.open(archivo)
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
        st.image(st.session_state.imagen, width=250)
    st.write("Va en la caneca:")
    st.markdown(
        f"<div style='display:inline-block;padding:12px 24px;border-radius:12px;"
        f"background:{color};color:white;font-size:26px;font-weight:700'>{etiqueta}</div>",
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
    st.success("✅ Ya quedó listo. La caneca se cerró. ¡Gracias por reciclar bien!")
    st.button("Analizar otro residuo", type="primary", on_click=reiniciar)

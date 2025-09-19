# =================================================================================
#                         STREAMLIT SCRIPT CONSOLIDADO
# =================================================================================

# ---------- IMPORTACIONES NECESARIAS ------------------
import streamlit as st
import pandas as pd
import joblib
import os
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Deshabilita las opciones de optimización de OneDNN para compatibilidad
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# ================================================================================
#                             1. CONFIGURACIÓN INICIAL
# ================================================================================
st.set_page_config(layout="wide", page_title="Dashboard de Proyectos de IA")

# ==============================
# CARGA DE MODELOS CON @st.cache_resource
# ==============================
@st.cache_resource
def load_vision_model_tflite():
    """Carga el modelo de visión por computadora en formato TFLite."""
    try:
        interpreter = tf.lite.Interpreter(model_path="model_unquant.tflite")
        interpreter.allocate_tensors()
        return interpreter
    except FileNotFoundError:
        st.error("❌ Error: 'model_unquant.tflite' no se encontró. Asegúrate de que esté en la misma carpeta.")
        return None
    except Exception as e:
        st.error(f"❌ Error al cargar el modelo de visión: {e}")
        return None

@st.cache_resource
def load_beer_model():
    """Carga el modelo de predicción de cerveza."""
    try:
        modelo = joblib.load("modelo_beer.joblib")
        return modelo
    except FileNotFoundError:
        st.error("❌ Error: 'modelo_beer.joblib' no se encontró.")
        return None
    except Exception as e:
        st.error(f"❌ Error al cargar el modelo de cerveza: {e}")
        return None

@st.cache_resource
def load_rf_model():
    """Carga el modelo de Random Forest para el Titanic."""
    try:
        modelo = joblib.load("rf_model.joblib")
        return modelo
    except FileNotFoundError:
        st.error("❌ Error: 'rf_model.joblib' no se encontró. Asegúrate de que el archivo esté en la misma carpeta.")
        return None
    except Exception as e:
        st.error(f"❌ Error al cargar el modelo de Random Forest: {e}")
        return None

# Cargar los modelos al inicio
vision_model_tflite = load_vision_model_tflite()
beer_model = load_beer_model()
rf_model = load_rf_model()

# ==============================
# OTRAS CONFIGURACIONES
# ==============================
class_labels = ["Apple___Podrido", "Apple___Saludable", "Apple___Sarnoso"]
temp_dir = "temp"
os.makedirs(temp_dir, exist_ok=True)

# ================================================================================
#                             2. FUNCIONES DE PROCESAMIENTO
# ================================================================================
def classify_image_tflite(image_path: str, interpreter):
    """Procesa y clasifica una imagen con el modelo TFLite."""
    img = Image.open(image_path).convert("RGB")
    img_resized = ImageOps.fit(img, (224, 224), Image.Resampling.LANCZOS)
    img_array_resized = np.asarray(img_resized)
    normalized_image_array = (img_array_resized.astype(np.float32) / 127.5) - 1
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]['shape']
    data = np.ndarray(shape=input_shape, dtype=np.float32)
    data[0] = normalized_image_array

    interpreter.set_tensor(input_details[0]['index'], data)
    interpreter.invoke()
    pred = interpreter.get_tensor(output_details[0]['index'])
    
    return pred[0]

# ================================================================================
#                             3. INTERFAZ DE STREAMLIT
# ================================================================================
st.sidebar.title("Menú")
option = st.sidebar.radio(
    "Selecciona una Opción:",
    ("Predicción de Cerveza", "Visión por computadora", "Random Forest")
)

# ---
### Sección 1: Predicción de Cerveza
if option == "Predicción de Cerveza":
    if beer_model is None:
        st.stop()
    st.title("🍺 Predicción de Producción de Cerveza")
    st.write("Introduce los valores para generar una predicción:")

    with st.form("beer_form"):
        year = st.number_input("Año", min_value=1956, max_value=2100, value=2025, step=1)
        month = st.selectbox("Mes", list(range(1, 13)), index=0)
        lag_1 = st.number_input("Producción del mes anterior", min_value=0.0, value=100.0)
        lag_12 = st.number_input("Producción hace 12 meses", min_value=0.0, value=110.0)
        lag_24 = st.number_input("Producción hace 24 meses", min_value=0.0, value=115.0)
        rolling_mean_12 = st.number_input("Media móvil 12 meses", min_value=0.0, value=105.0)
        submitted = st.form_submit_button("🔮 Predecir")

    if submitted:
        X_new = pd.DataFrame({
            "month": [month], "year": [year], "lag_1": [lag_1],
            "lag_12": [lag_12], "lag_24": [lag_24], "rolling_mean_12": [rolling_mean_12]
        })
        pred = beer_model.predict(X_new)[0]
        st.success(f"Predicción de producción: **{pred:.2f}**")

# ---
### Sección 2: Visión por Computadora
elif option == "Visión por computadora":
    if vision_model_tflite is None:
        st.stop()
        
    st.title("🖼️ Visión por Computadora")
    st.write("Sube una imagen para clasificarla.")
    uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        temp_path = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())
        
        st.image(uploaded_file, caption="Imagen seleccionada", use_container_width=True)

        try:
            pred = classify_image_tflite(temp_path, vision_model_tflite)
            predicted_class = np.argmax(pred)
            predicted_probability = pred[predicted_class]

            color = "red" if "Podrido" in class_labels[predicted_class] or "Sarnoso" in class_labels[predicted_class] else "green"
            message = f'<p style="color: {color}; font-size: 24px;">La imagen es <b>{class_labels[predicted_class]}</b> con una probabilidad de {predicted_probability:.3f}</p>'
            st.markdown(message, unsafe_allow_html=True)
        finally:
            os.remove(temp_path)

# ---
### Sección 3: Random Forest
elif option == "Random Forest":
    if rf_model is None:
        st.stop()
        
    st.title("🚢 Predicción de Supervivencia en el Titanic")
    st.write("Introduce los datos de un pasajero para predecir si sobrevivió.")
    
    with st.form("titanic_form"):
        pclass = st.selectbox("Clase del billete", options=[1, 2, 3], index=0)
        sex = st.selectbox("Sexo", options=["male", "female"], index=0)
        age = st.slider("Edad", min_value=0, max_value=100, value=30, step=1)
        sibsp = st.number_input("Número de hermanos/cónyuges a bordo ", min_value=0, value=0)
        parch = st.number_input("Número de padres/hijos a bordo ", min_value=0, value=0)
        fare = st.number_input("Tarifa del billete (7.25,71.2833,146.5208,262.375)", min_value=0.0, value=50.0)
        embarked = st.selectbox("Puerto de embarque ", options=["S", "C", "Q"], index=0)
        
        submitted = st.form_submit_button("🔮 Predecir")

        if submitted:
            input_df = pd.DataFrame([{
                'Pclass': pclass,
                'Sex': sex,
                'Age': age,
                'SibSp': sibsp,
                'Parch': parch,
                'Fare': fare,
                'Embarked': embarked
            }])
            
            input_df['Sex'] = input_df['Sex'].map({'male': 1, 'female': 0})
            input_df['Embarked'] = input_df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

            prediction = rf_model.predict(input_df)
            
            if prediction[0] == 1:
                st.success("¡El pasajero **sobrevivió**! 🎉")
            else:
                st.error("El pasajero **no sobrevivió**. 😔")
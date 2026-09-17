import streamlit as st
import os
import gc

# Importar funciones de los módulos de negocio (asegúrate de tenerlos en la misma carpeta)
from gatuno_forecaster import obtener_modelo_optimizado
from gatuno_data import cargar_datos_recientes

# Configuración inicial de la página para dispositivos móviles
st.set_page_config(
    page_title="Gatuno V15.1.1 - Pronósticos",
    page_icon="🐾",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Estilo CSS ligero para mejorar la legibilidad en pantallas táctiles
st.markdown("""
    <style>
    .stButton>button { width: 100%; border-radius: 8px; font-weight: bold; }
    .metric-card { background-color: #f0f2f6; padding: 10px; border-radius: 6px; margin-bottom: 8px; }
    </style>
""", unsafe_allow_html=True)

st.title("🐾 Sistema Gatuno V15.1.1")
st.caption("Motor de análisis y pronósticos optimizado para móvil.")

# Carga inicial optimizada del modelo con caché en disco
@st.cache_resource(show_spinner=False)
def cargar_motor():
    with st.spinner("Cargando motor de pronósticos..."):
        modelo = obtener_modelo_optimizado()
    return modelo

try:
    motor_activo = cargar_motor()
except Exception as e:
    st.error(f"Error al inicializar el motor: {e}")
    st.stop()

# Navegación por pestañas ligeras
pestana_principal, pestana_auditoria = st.tabs(["📊 Pronósticos Activos", "🔍 Auditoría y Historial"])

# Fragmento 1: Panel principal (Aislado para no recargar toda la app)
@st.fragment
def renderizar_panel_principal():
    st.subheader("Partidos y Tendencias")
    
    if st.button("🔄 Actualizar Datos Rápidos"):
        with st.spinner("Sincronizando fuentes..."):
            # Lógica ligera de actualización
            datos = cargar_datos_recientes(fast_mode=True)
            st.success("¡Datos sincronizados correctamente!")
            
    st.info("Selecciona una opción del panel para evaluar los niveles de confianza (Verde/Amarillo/Rojo).")
    
    # Liberación explícita de memoria para evitar saturación en RAM móvil
    gc.collect()

with pestana_principal:
    renderizar_panel_principal()

# Fragmento 2: Auditoría histórica independiente
@st.fragment
def renderizar_auditoria():
    st.subheader("Registro de Aciertos")
    st.write("Historial optimizado de rendimiento del motor de inversión y pronóstico.")
    
    if st.button("📊 Ejecutar Auditoría por Lotes"):
        st.write("Procesando auditoría acotada...")
        # Aquí llamarías a tu función de auditoría con límite de eventos

with pestana_auditoria:
    renderizar_auditoria()

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.integrate import odeint


# ===================================================================
#                    MODELO DE ESPACIO DE ESTADOS (Backend)
# ===================================================================

# ---------- PARÁMETROS FIJOS (Sistema SI) ----------
# Estos son los valores por defecto. En la interfaz se pueden modificar.
# Pero las funciones de simulación los usan como argumentos.
# No definimos constantes globales acá, las pasamos como parámetros.

# ---------- AEs (Ecuaciones Algebraicas/Constitutivas) ----------
def AEs(L, T, x, params, t=0):
    """
    Calcula las variables algebraicas (Y) a partir del estado X y parámetros.
    """
    # Parámetros base
    F0_base = params['F0']
    Cv = params['Cv']
    rho = params['rho']
    g = params['g']
    A = params['A']
    Cp = params['Cp']
    T0 = params['T0']
    Tv = params['Tv']
    Wa = params['Wa']
    UAs = params['UAs']
    
    # Perturbación en F0 (aumento para t >= 0)
    if t >= 0 and params.get('factor_F0', 1.0) > 1.0:
        F0 = F0_base * params['factor_F0']
    else:
        F0 = F0_base

    # Caudal de salida (modelo gravitatorio)
    L_segura = max(L, 0.001)
    F = Cv * x * np.sqrt(rho * g * L_segura)
    
    # Calor transferido por el serpentin
    Q = UAs * (Tv - T)
    
    return {
        'A': A,
        'F0': F0,
        'F': F,
        'rho': rho,
        'Cp': Cp,
        'T0': T0,
        'Q': Q,
        'Wa': Wa
    }

# ---------- ODEs (Ecuaciones Diferenciales) ----------
def ODEs(X, t, x, params):
    """
    Devuelve las derivadas de las variables de estado.
    """
    # Recuperar variables de estado
    L = X[0]
    T = X[1]
    
    # Calcular variables algebraicas (pasando t)
    Y = AEs(L, T, x, params, t)
    
    # Ecuaciones diferenciales
    dL = (Y['F0'] - Y['F']) / Y['A']
    dT = (Y['F0'] * Y['rho'] * Y['Cp'] * (Y['T0'] - T) + Y['Q'] + Y['Wa']) / (Y['A'] * L * Y['rho'] * Y['Cp'])
    
    return [dL, dT]

# ---------- Característica de la Válvula ----------
def f_apertura(x, tipo, R=50):
    """
    Calcula el flujo normalizado f(x) según el tipo de válvula.
    """
    x = max(0.0, min(1.0, x))
    
    if tipo == "Lineal":
        return x
    elif tipo == "Igual porcentaje (isoporcentual)":
        return R**(x-1)
    elif tipo == "Apertura rápida (quick opening)":
        return 1 - (1-x)**2
    else:
        return x

# ---------- Inicialización ----------
def inicializar(params_por_defecto):
    """
    Inicializa la simulación con los parámetros dados.
    Devuelve las condiciones iniciales y las leyendas.
    """
    # Condiciones iniciales
    L0 = params_por_defecto['L0']
    T0 = params_por_defecto['T0_initial']
    Xini = [L0, T0]
    
    # Leyendas (para las gráficas)
    LX = ['L', 'T']   # Variables de estado
    LY = ['A', 'F0', 'F', 'rho', 'Cp', 'T0', 'Q', 'Wa']  # Variables algebraicas
    
    return Xini, LX, LY

# ---------- Simulación ----------
def simulacion(tfin, dt, Xini, x, params):
    """
    Realiza la simulación dinámica.
    """
    # Vector de tiempo
    nts = int(np.ceil(tfin / dt)) + 1
    tpts = np.linspace(0, tfin, nts)
    
    # Resolver ODEs
    sol = odeint(ODEs, Xini, tpts, args=(x, params))
    L = sol[:, 0]
    T = sol[:, 1]
    
    # Calcular variables dependientes (Y) en cada instante
    Y = {}
    Y['A'] = np.full_like(tpts, params['A'])
    Y['F0'] = np.full_like(tpts, params['F0'])
    Y['rho'] = np.full_like(tpts, params['rho'])
    Y['Cp'] = np.full_like(tpts, params['Cp'])
    Y['T0'] = np.full_like(tpts, params['T0'])
    Y['Wa'] = np.full_like(tpts, params['Wa'])
    
    F = np.zeros_like(tpts)
    Q = np.zeros_like(tpts)
    F0_historico = np.zeros_like(tpts)
    
    for i in range(len(tpts)):
        Y_local = AEs(L[i], T[i], x, params, tpts[i])
        F[i] = Y_local['F']
        Q[i] = Y_local['Q']
        F0_historico[i] = Y_local['F0']
    Y['F'] = F
    Y['Q'] = Q
    Y['F0_historico'] = F0_historico
    
    return tpts, L, T, Y

# ---------- Análisis Estacionario ----------
def nivel_estacionario(F0, Cv, x, rho, g):
    """
    Calcula el nivel estacionario para una apertura x dada.
    """
    if x <= 0 or Cv <= 0:
        return np.inf
    return (F0 / (Cv * x))**2 / (rho * g)

def temperatura_estacionaria(F0, rho, Cp, T0, UAs, Tv, Wa):
    """
    Calcula la temperatura estacionaria.
    """
    return (F0 * rho * Cp * T0 + UAs * Tv + Wa) / (F0 * rho * Cp + UAs)

# ---------- Cálculo de Cv ----------
def calcular_Cv(F0, x0, rho, g, L0):
    """
    Calcula Cv a partir del punto de operación inicial.
    """
    if x0 <= 0 or L0 <= 0:
        return 0
    return F0 / (x0 * np.sqrt(rho * g * L0))

# ---------- Cálculo de UAs ----------
def calcular_UAs(F0, rho, Cp, T0, T_initial, Tv, Wa):
    """
    Calcula UAs a partir del punto de operación inicial.
    """
    if Tv == T_initial:
        return 0
    return -(F0 * rho * Cp * (T0 - T_initial) + Wa) / (Tv - T_initial)


# ===================================================================
#                    INTERFAZ DE USUARIO (Frontend)
# ===================================================================

# ---------- Configuración de la página ----------
st.set_page_config(
    page_title="Simulador de Tanque Calefaccionado", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- INYECCIÓN DE CSS PERSONALIZADO ----------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    footer {visibility: hidden;}
    /* #MainMenu visible para que el selector de temas funcione */
    
    .css-1d391kg, .css-1lcbmhc {
        background-color: #f5f6f7 !important;
        border-right: 1px solid #d0d0d5 !important;
        padding-top: 2rem !important;
    }
    
    .main .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2rem !important;
        max-width: 1400px !important;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-weight: 600 !important;
        letter-spacing: -0.02em !important;
        font-family: 'Inter', sans-serif !important;
    }
    
    h1 { font-size: 2rem !important; }
    h2 { font-size: 1.4rem !important; margin-top: 1.5rem !important; margin-bottom: 0.8rem !important; }
    h3 { font-size: 1.1rem !important; }
    
    .stButton > button {
        background-color: #e8e8ea !important;
        color: #1b1b32 !important;
        border: 1px solid #d0d0d5 !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        padding: 0.6rem 1.8rem !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        font-family: 'Inter', sans-serif !important;
        letter-spacing: 0.01em !important;
    }
    
    .stButton > button:hover {
        background-color: #1b1b32 !important;
        color: #ffffff !important;
        border-color: #1b1b32 !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.12) !important;
    }
    
    .stButton > button:active { transform: scale(0.98) !important; }
    
    .stNumberInput > div > div > input,
    .stSelectbox > div > div {
        border-radius: 8px !important;
        border: 1px solid #d0d0d5 !important;
        font-family: 'Inter', sans-serif !important;
    }
    
    .stNumberInput > div > div > input:focus,
    .stSelectbox > div > div:focus {
        border-color: #1b1b32 !important;
        box-shadow: 0 0 0 2px rgba(27, 27, 50, 0.1) !important;
    }
    
    .streamlit-expanderHeader {
        font-weight: 500 !important;
        background-color: #f5f6f7 !important;
        border-radius: 8px !important;
        border: 1px solid #e8e8ee !important;
        font-family: 'Inter', sans-serif !important;
    }
    
    .streamlit-expanderHeader:hover { background-color: #e8e8ea !important; }
    
    /* ============================================================
       TARJETAS MÉTRICAS - TAMAÑO MÁS PEQUEÑO
       ============================================================ */
    div[data-testid="metric-container"] {
        border: 1px solid #d0d0d5 !important;
        border-radius: 12px !important;
        padding: 0.8rem !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04) !important;
        transition: all 0.3s ease !important;
        background: transparent !important;
    }
    
    div[data-testid="metric-container"]:hover {
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.08) !important;
        transform: translateY(-2px) !important;
    }
    
    div[data-testid="metric-container"] > label {
        font-size: 0.7rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
        font-weight: 600 !important;
    }
    
    /* Ajustar el tamaño del valor numérico de las tarjetas */
    div[data-testid="stMetricValue"] {
        font-size: 1.4rem !important;
        font-weight: 600 !important;
    }
    
    .stAlert {
        border-radius: 10px !important;
        border-left: 4px solid #1b1b32 !important;
    }
    
    .stAlert > div {
        font-family: 'Inter', sans-serif !important;
        font-weight: 500 !important;
    }
    
    .stCaption, .caption {
        font-size: 0.75rem !important;
        font-weight: 400 !important;
    }
    
    @media (max-width: 768px) {
        .main .block-container { padding: 0.5rem 0.8rem !important; }
        .stButton > button { font-size: 0.8rem !important; padding: 0.5rem 1.2rem !important; }
        div[data-testid="metric-container"] { padding: 0.5rem !important; }
        div[data-testid="stMetricValue"] {
            font-size: 1.1rem !important;
        }
    }
    
    ::-webkit-scrollbar { width: 8px; height: 8px; }
    ::-webkit-scrollbar-track { background: #f5f6f7; }
    ::-webkit-scrollbar-thumb { background: #d0d0d5; border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: #b0b0b8; }
</style>
""", unsafe_allow_html=True)

# ---------- Título de la aplicación ----------
st.title("Simulador de Tanque Calefaccionado con Serpentín")

# ---------- Imagen ----------
st.markdown(
    f'''
    <div style="text-align: center; margin: 20px 0;">
       <img src="https://raw.githubusercontent.com/ffedezn-cloud/tanque-calefaccionado/main/assets/images/diagrama_calefaccionado.png"
             alt="Esquema del tanque calefaccionado" 
             style="width: 60%; max-width: 500px; border: 1px solid #ddd; border-radius: 8px;">
        <p style="margin-top: 8px;">Esquema del tanque con serpentín y descarga gravitatoria</p>
    </div>
    ''',
    unsafe_allow_html=True
)
st.markdown("---")

# ---------- Barra lateral (Entrada de datos) ----------
with st.sidebar:
    # Información del desarrollador
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; padding: 10px 0;">
            <p style="font-size: 17px; font-weight: 600; margin-bottom: 2px;">Federico Franco</p>
            <p style="font-size: 16px; color: #888; margin-bottom: 2px;">Ingeniería Química</p>
            <a href="mailto:ffede.zn@gmail.com" style="font-size: 16px; color: #888; text-decoration: none;">
                ffede.zn@gmail.com
            </a>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.subheader("Datos Geométricos del Tanque")
    D = st.number_input("Diámetro del tanque D (m)", value=1.0, min_value=0.3, max_value=5.0, step=0.05)
    A = np.pi * (D/2)**2
    st.caption(f"Area calculada: {A:.4f} m²")
    
    L0 = st.number_input("Nivel inicial L0 (m)", value=1.0, min_value=0.0, max_value=5.0, step=0.05)
    L_max = st.number_input("Nivel máximo (rebalse) L_max (m)", value=2.0, min_value=0.5, max_value=10.0, step=0.1)
    
    st.subheader("Datos de Operación")
    F0 = st.number_input("Caudal de entrada F0 (m³/s)", value=0.002, min_value=0.0001, max_value=0.1, format="%.5f")
    x0 = st.slider("Apertura inicial de válvula x0", 0.0, 1.0, 0.5, 0.01)
    xf = st.slider("Apertura final de válvula xf", 0.0, 1.0, 0.25, 0.01)
    
    st.subheader("Perturbaciones")
    perturbacion_F0 = st.checkbox("Aumentar F0 en el tiempo", value=False)
    porcentaje_aumento = st.slider("Porcentaje de aumento de F0 (%)", 0, 200, 65, 5) if perturbacion_F0 else 0
    factor_F0 = 1 + porcentaje_aumento / 100 if perturbacion_F0 else 1.0
    
    st.subheader("Datos del Fluido")
    rho = st.number_input("Densidad del fluido ρ (kg/m³)", value=1000.0, min_value=500.0, max_value=2000.0, step=10.0)
    Cp = st.number_input("Calor específico Cp (J/kg°C)", value=4187.0, min_value=1000.0, max_value=8000.0, step=100.0)
    T0 = st.number_input("Temperatura de entrada T0 (°C)", value=25.0, min_value=0.0, max_value=100.0, step=1.0)
    g = 9.81
    st.caption(f"Gravedad fija: g = {g} m/s²")
    
    st.subheader("Datos del Serpentín y Agitador")
    Tv = st.number_input("Temperatura del vapor Tv (°C)", value=132.0, min_value=50.0, max_value=250.0, step=1.0)
    Wa = st.number_input("Potencia del agitador Wa (W)", value=2000.0, min_value=0.0, max_value=10000.0, step=100.0)
    T_initial = st.number_input("Temperatura inicial T (°C)", value=60.0, min_value=0.0, max_value=150.0, step=1.0)
    T_max_seguridad = st.number_input("Temperatura máxima de seguridad (°C)", value=80.0, min_value=30.0, max_value=150.0, step=1.0)
    
    st.subheader("Característica de la Válvula")
    tipo_valvula = st.selectbox(
        "Tipo de característica",
        ["Lineal", "Igual porcentaje (isoporcentual)", "Apertura rápida (quick opening)"]
    )
    
    R = 50
    if tipo_valvula == "Igual porcentaje (isoporcentual)":
        R = st.slider("Relación de rango R (típico 20-50)", 20, 100, 50, 5)
    
    st.subheader("Parámetros de Simulación")
    t_final = st.slider("Tiempo de simulación (s)", 100, 100000, 1100, 100)

# ---------- Parámetros del modelo ----------
params = {
    'A': A,
    'F0': F0,
    'rho': rho,
    'g': g,
    'L0': L0,
    'Cp': Cp,
    'T0': T0,
    'Tv': Tv,
    'Wa': Wa,
    'T0_initial': T_initial,
    'factor_F0': factor_F0 
}

# Calcular Cv a partir del punto de operación inicial
Cv = calcular_Cv(F0, x0, rho, g, L0)
params['Cv'] = Cv

# Calcular UAs a partir del punto de operación inicial
UAs = calcular_UAs(F0, rho, Cp, T0, T_initial, Tv, Wa)
params['UAs'] = UAs

# Verificar que Cv no sea cero (para evitar errores)
if Cv == 0:
    st.error("Error: Cv es cero. Verifique que la apertura inicial x0 > 0 y L0 > 0.")
    st.stop()

# Verificar que UAs no sea cero
if UAs == 0:
    st.warning("Advertencia: UAs es cero. Verifique que Tv != T_initial.")

# Calcular variables estacionarias
L_ss_inicial = nivel_estacionario(F0, Cv, x0, rho, g)
L_ss_final = nivel_estacionario(F0, Cv, xf, rho, g)
x_min = F0 / (Cv * np.sqrt(rho * g * L_max)) if Cv > 0 and L_max > 0 else np.inf
T_ss = temperatura_estacionaria(F0, rho, Cp, T0, UAs, Tv, Wa)

# ---------- Mostrar parámetros calculados en tarjetas ----------
st.subheader("Parámetros del Sistema")

col_a, col_b, col_c = st.columns(3)
with col_a:
    st.metric("Area del tanque A", f"{A:.4f} m²")
    st.metric("Cv (valvula)", f"{Cv:.4e}")
with col_b:
    st.metric("Caudal de entrada F0", f"{F0:.5f} m³/s")
    st.metric("UA (serpentín)", f"{UAs:.2f} W/°C")
with col_c:
    st.metric("Nivel inicial L0", f"{L0:.2f} m")
    st.metric("Temperatura inicial", f"{T_initial:.1f} °C")


# ===================================================================
#                    SECCIÓN 1: ANÁLISIS DEL MODELO ESTACIONARIO
# ===================================================================

st.subheader("Análisis del Modelo Estacionario")

col_ss1, col_ss2, col_ss3, col_ss4 = st.columns(4)

with col_ss1:
    st.metric(
        label="Coeficiente Cv",
        value=f"{Cv:.4e}"
    )

with col_ss2:
    if np.isfinite(L_ss_final):
        st.metric(
            label=f"Nivel con xf = {xf:.3f}",
            value=f"{L_ss_final:.2f} m"
        )
    else:
        st.metric(
            label=f"Nivel con xf = {xf:.3f}",
            value="No disponible"
        )

with col_ss3:
    if np.isfinite(x_min):
        st.metric(
            label="Apertura mínima para no rebalsar",
            value=f"{x_min:.4f}"
        )
    else:
        st.metric(
            label="Apertura mínima para no rebalsar",
            value="No disponible"
        )

with col_ss4:
    st.metric(
        label="Temperatura estacionaria",
        value=f"{T_ss:.1f} °C"
    )

# Advertencias
if np.isfinite(L_ss_final):
    if L_ss_final > L_max:
        st.error(f"REBALSE DETECTADO\n\nEl tanque alcanza un nivel final de {L_ss_final:.2f} m, superando el nivel máximo de {L_max:.1f} m.")
    else:
        st.success(f"Sin rebalse\n\nEl tanque alcanza un nivel final de {L_ss_final:.2f} m, dentro del límite de {L_max:.1f} m.")

if T_ss > T_max_seguridad:
    st.error(f"TEMPERATURA EXCESIVA\n\nLa temperatura estacionaria ({T_ss:.1f} °C) supera el límite de seguridad ({T_max_seguridad} °C).")
else:
    st.success(f"Temperatura segura\n\nTemperatura estacionaria ({T_ss:.1f} °C) dentro del límite de seguridad ({T_max_seguridad} °C).")


# ===================================================================
#                    SECCIÓN 2: ANÁLISIS DEL MODELO DINÁMICO
# ===================================================================

st.subheader("Análisis del Modelo Dinámico")

# Parámetros de simulación
dt = 10

# Ejecutar simulación
Xini, LX, LY = inicializar(params)
tpts, L, T, Y = simulacion(t_final, dt, Xini, xf, params)

# Calcular variables
F = Y['F']
Q = Y['Q']
F0_historico = Y['F0_historico']

# Detectar rebalse
tiempo_rebalse = None
temperatura_rebalse = None
for i, nivel in enumerate(L):
    if nivel >= L_max:
        tiempo_rebalse = tpts[i]
        temperatura_rebalse = T[i]
        break

# Mostrar resultados en tarjetas
col_r1, col_r2, col_r3, col_r4 = st.columns(4)

with col_r1:
    st.metric(
        label="Nivel final",
        value=f"{L[-1]:.3f} m"
    )

with col_r2:
    st.metric(
        label="Temperatura final",
        value=f"{T[-1]:.1f} °C"
    )

with col_r3:
    st.metric(
        label="Caudal salida final",
        value=f"{F[-1]:.5f} m³/s"
    )

with col_r4:
    st.metric(
        label="Calor transferido final",
        value=f"{Q[-1]:.1f} W"
    )

# Alertas de estado
if tiempo_rebalse:
    st.error(f"REBALSE DURANTE LA SIMULACIÓN\n\nEl tanque alcanza el nivel máximo de {L_max} m a los {tiempo_rebalse:.1f} segundos.")
else:
    st.success(f"Sin rebalse durante la simulación\n\nNivel máximo alcanzado: {max(L):.3f} m (límite: {L_max} m)")

if T[-1] > T_max_seguridad:
    st.error(f"TEMPERATURA FINAL EXCESIVA\n\nTemperatura final ({T[-1]:.1f} °C) supera el límite de seguridad ({T_max_seguridad} °C).")
else:
    st.success(f"Temperatura final segura\n\nTemperatura final ({T[-1]:.1f} °C) dentro del límite de seguridad.")


# ---------- GRÁFICAS CON PESTAÑAS ----------
st.subheader("Gráficas de la Simulación")

# Detectar tema del navegador
tema_oscuro = st.get_option("theme.base") == "dark"

if tema_oscuro:
    bg_color = 'rgba(30,30,30,0.95)'
    text_color = 'white'
    grid_color = 'rgba(255,255,255,0.15)'
    legend_bg = 'rgba(0,0,0,0.6)'
    template = "plotly_dark"
    colors = {
        'primary': '#4dabf7',
        'success': '#51cf66',
        'danger': '#ff6b6b',
        'secondary': '#868e96'
    }
else:
    bg_color = 'white'
    text_color = 'black'
    grid_color = 'rgba(0,0,0,0.1)'
    legend_bg = 'rgba(255,255,255,0.8)'
    template = "plotly_white"
    colors = {
        'primary': '#1f77b4',
        'success': '#2ca02c',
        'danger': '#d62728',
        'secondary': '#7f7f7f'
    }

# Configuración común para todas las gráficas (CON BARRA DE HERRAMIENTAS)
config_plotly = {
    'scrollZoom': True,
    'displayModeBar': True,
    'responsive': True,
    'modeBarButtonsToRemove': ['toImage', 'sendDataToCloud'],
    'displaylogo': False
}

# Crear pestañas para organizar las gráficas
tab1, tab2, tab3, tab4 = st.tabs(["Nivel del Tanque", "Temperatura", "Caudales", "Características de Válvula"])

with tab1:
    with st.container(border=True):
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=tpts, y=L, mode='lines', name='Nivel L(t)', line=dict(color=colors['primary'], width=2.5)))
        
        fig1.add_hline(y=L_max, line=dict(color=colors['danger'], width=1.5, dash='dash'))
        fig1.add_hline(y=L0, line=dict(color=colors['secondary'], width=1, dash='dot'))
        if np.isfinite(L_ss_final) and L_ss_final <= L_max:
            fig1.add_hline(y=L_ss_final, line=dict(color=colors['success'], width=1, dash='dash'))
        if tiempo_rebalse:
            fig1.add_vline(x=tiempo_rebalse, line=dict(color=colors['danger'], width=1.5, dash='dot'))
        
        fig1.add_annotation(
            x=50, y=L_max+0.05, 
            text=f'L_max = {L_max} m', 
            showarrow=False,
            font=dict(color=text_color, size=12),
            bgcolor=legend_bg,
            bordercolor=colors['danger'],
            borderwidth=1
        )
        if tiempo_rebalse:
            fig1.add_annotation(
                x=tiempo_rebalse+20, y=L_max-0.15, 
                text=f'Rebalse t={tiempo_rebalse:.1f}s', 
                showarrow=True,
                arrowhead=1,
                ax=30, ay=-30,
                font=dict(color=text_color, size=12),
                bgcolor=legend_bg,
                bordercolor=colors['danger'],
                borderwidth=1
            )
        
        fig1.update_layout(
            template=template,
            dragmode=False,
            title=dict(text='Nivel del tanque', font=dict(color=text_color, size=16)),
            xaxis=dict(
                title='Tiempo (s)',
                title_font=dict(color=text_color, size=14),
                tickfont=dict(color=text_color, size=12),
                gridcolor=grid_color,
                showgrid=True,
                zeroline=True,
                zerolinecolor=grid_color,
                zerolinewidth=1,
                showline=True,
                linecolor=grid_color,
                linewidth=1
            ),
            yaxis=dict(
                title='Nivel L (m)',
                title_font=dict(color=text_color, size=14),
                tickfont=dict(color=text_color, size=12),
                gridcolor=grid_color,
                showgrid=True,
                zeroline=True,
                zerolinecolor=grid_color,
                zerolinewidth=1,
                showline=True,
                linecolor=grid_color,
                linewidth=1
            ),
            height=350,
            hovermode='x unified',
            plot_bgcolor=bg_color,
            paper_bgcolor=bg_color,
            font=dict(color=text_color, size=12),
            legend=dict(
                font=dict(color=text_color, size=11),
                bgcolor=legend_bg,
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='center',
                x=0.5
            ),
            margin=dict(l=40, r=10, t=50, b=50)
        )
        st.plotly_chart(fig1, use_container_width=True, config=config_plotly)

with tab2:
    with st.container(border=True):
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=tpts, y=T, mode='lines', name='Temperatura T(t)', line=dict(color=colors['danger'], width=2.5)))
        fig2.add_hline(y=T_max_seguridad, line=dict(color='#ff7f0e', width=1.5, dash='dash'))
        fig2.add_hline(y=T_initial, line=dict(color=colors['secondary'], width=1, dash='dot'))
        fig2.add_hline(y=T_ss, line=dict(color=colors['success'], width=1, dash='dash'))
        if tiempo_rebalse:
            fig2.add_vline(x=tiempo_rebalse, line=dict(color=colors['danger'], width=1.5, dash='dot'))
        
        fig2.add_annotation(
            x=50, y=T_max_seguridad+1, 
            text=f'T_max_seg = {T_max_seguridad} °C', 
            showarrow=False,
            font=dict(color=text_color, size=12),
            bgcolor=legend_bg,
            bordercolor='#ff7f0e',
            borderwidth=1
        )
        if tiempo_rebalse:
            fig2.add_annotation(
                x=tiempo_rebalse+20, y=T_max_seguridad-2, 
                text=f'Rebalse t={tiempo_rebalse:.1f}s', 
                showarrow=True,
                arrowhead=1,
                ax=30, ay=-30,
                font=dict(color=text_color, size=12),
                bgcolor=legend_bg,
                bordercolor=colors['danger'],
                borderwidth=1
            )
        
        fig2.update_layout(
            template=template,
            dragmode=False,
            title=dict(text='Temperatura del tanque', font=dict(color=text_color, size=16)),
            xaxis=dict(
                title='Tiempo (s)',
                title_font=dict(color=text_color, size=14),
                tickfont=dict(color=text_color, size=12),
                gridcolor=grid_color,
                showgrid=True,
                zeroline=True,
                zerolinecolor=grid_color,
                zerolinewidth=1,
                showline=True,
                linecolor=grid_color,
                linewidth=1
            ),
            yaxis=dict(
                title='Temperatura (°C)',
                title_font=dict(color=text_color, size=14),
                tickfont=dict(color=text_color, size=12),
                gridcolor=grid_color,
                showgrid=True,
                zeroline=True,
                zerolinecolor=grid_color,
                zerolinewidth=1,
                showline=True,
                linecolor=grid_color,
                linewidth=1
            ),
            height=350,
            hovermode='x unified',
            plot_bgcolor=bg_color,
            paper_bgcolor=bg_color,
            font=dict(color=text_color, size=12),
            legend=dict(
                font=dict(color=text_color, size=11),
                bgcolor=legend_bg,
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='center',
                x=0.5
            ),
            margin=dict(l=40, r=10, t=50, b=50)
        )
        st.plotly_chart(fig2, use_container_width=True, config=config_plotly)

with tab3:
    with st.container(border=True):
        fig3 = go.Figure()
        
        F0_inicial = params['F0']
        F0_final = F0_historico[0] if len(F0_historico) > 0 else F0_inicial
        
        fig3.add_trace(go.Scatter(x=tpts, y=[F0_final]*len(tpts), mode='lines', name='F0 (entrada)', line=dict(color=colors['success'], width=2.5, dash='dash')))
        fig3.add_trace(go.Scatter(x=tpts, y=F, mode='lines', name='F (salida)', line=dict(color=colors['danger'], width=2.5)))
        
        if tiempo_rebalse:
            fig3.add_vline(x=tiempo_rebalse, line=dict(color=colors['danger'], width=1.5, dash='dot'))
            fig3.add_annotation(
                x=tiempo_rebalse+20, y=max(F)*0.8, 
                text=f'Rebalse t={tiempo_rebalse:.1f}s', 
                showarrow=True,
                arrowhead=1,
                ax=30, ay=-30,
                font=dict(color=text_color, size=12),
                bgcolor=legend_bg,
                bordercolor=colors['danger'],
                borderwidth=1
            )
        
        fig3.update_layout(
            template=template,
            dragmode=False,
            title=dict(text='Caudales de entrada y salida', font=dict(color=text_color, size=16)),
            xaxis=dict(
                title='Tiempo (s)',
                title_font=dict(color=text_color, size=14),
                tickfont=dict(color=text_color, size=12),
                gridcolor=grid_color,
                showgrid=True,
                zeroline=True,
                zerolinecolor=grid_color,
                zerolinewidth=1,
                showline=True,
                linecolor=grid_color,
                linewidth=1
            ),
            yaxis=dict(
                title='Caudal (m³/s)',
                title_font=dict(color=text_color, size=14),
                tickfont=dict(color=text_color, size=12),
                gridcolor=grid_color,
                showgrid=True,
                zeroline=True,
                zerolinecolor=grid_color,
                zerolinewidth=1,
                showline=True,
                linecolor=grid_color,
                linewidth=1
            ),
            height=350,
            hovermode='x unified',
            plot_bgcolor=bg_color,
            paper_bgcolor=bg_color,
            font=dict(color=text_color, size=12),
            legend=dict(
                font=dict(color=text_color, size=11),
                bgcolor=legend_bg,
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='center',
                x=0.5
            ),
            margin=dict(l=40, r=10, t=50, b=50)
        )
        st.plotly_chart(fig3, use_container_width=True, config=config_plotly)

with tab4:
    with st.container(border=True):
        x_vals = np.linspace(0, 1, 200)
        f_lineal = [f_apertura(x, "Lineal", R) for x in x_vals]
        f_isoporc = [f_apertura(x, "Igual porcentaje (isoporcentual)", R) for x in x_vals]
        f_rapida = [f_apertura(x, "Apertura rápida (quick opening)", R) for x in x_vals]
        
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(x=x_vals, y=f_lineal, mode='lines', name='Lineal', line=dict(color='#1f77b4', width=3, dash='solid')))
        fig4.add_trace(go.Scatter(x=x_vals, y=f_isoporc, mode='lines', name=f'Isoporcentual (R={R})', line=dict(color='#d62728', width=3, dash='dash')))
        fig4.add_trace(go.Scatter(x=x_vals, y=f_rapida, mode='lines', name='Apertura rápida', line=dict(color='#2ca02c', width=3, dash='dot')))
        
        fig4.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Referencia y=x', line=dict(color='gray', width=1, dash='dash'), opacity=0.5))
        
        fig4.add_trace(go.Scatter(x=[x0], y=[f_apertura(x0, tipo_valvula, R)], mode='markers', name=f'x₀ = {x0:.2f}', marker=dict(color='#ff7f0e', size=14, symbol='circle', line=dict(color='white', width=2))))
        fig4.add_trace(go.Scatter(x=[xf], y=[f_apertura(xf, tipo_valvula, R)], mode='markers', name=f'x_f = {xf:.2f}', marker=dict(color='#9467bd', size=14, symbol='diamond', line=dict(color='white', width=2))))
        
        if np.isfinite(x_min):
            fig4.add_vline(x=x_min, line=dict(color='#ff7f0e', width=2, dash='dash'))
            fig4.add_annotation(
                x=x_min+0.03, y=0.9, 
                text=f'x_min = {x_min:.3f}', 
                showarrow=False,
                font=dict(color=text_color, size=12),
                bgcolor=legend_bg,
                bordercolor='#ff7f0e',
                borderwidth=1,
                borderpad=4
            )
        
        fig4.update_layout(
            template=template,
            dragmode=False,
            title=dict(text='Características de válvulas de control', font=dict(color=text_color, size=16)),
            xaxis=dict(
                title='Apertura x',
                title_font=dict(color=text_color, size=14),
                tickfont=dict(color=text_color, size=12),
                gridcolor=grid_color,
                showgrid=True,
                zeroline=True,
                zerolinecolor=grid_color,
                zerolinewidth=1,
                showline=True,
                linecolor=grid_color,
                linewidth=1,
                range=[-0.05, 1.05]
            ),
            yaxis=dict(
                title='Flujo normalizado f(x)',
                title_font=dict(color=text_color, size=14),
                tickfont=dict(color=text_color, size=12),
                gridcolor=grid_color,
                showgrid=True,
                zeroline=True,
                zerolinecolor=grid_color,
                zerolinewidth=1,
                showline=True,
                linecolor=grid_color,
                linewidth=1,
                range=[-0.05, 1.05]
            ),
            height=350,
            hovermode='x unified',
            plot_bgcolor=bg_color,
            paper_bgcolor=bg_color,
            font=dict(color=text_color, size=12),
            legend=dict(
                font=dict(color=text_color, size=11),
                bgcolor=legend_bg,
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='center',
                x=0.5
            ),
            margin=dict(l=40, r=10, t=60, b=50)
        )
        st.plotly_chart(fig4, use_container_width=True, config=config_plotly)


# ===================================================================
#                    SECCIÓN 3: DOCUMENTACIÓN DEL MODELO
# ===================================================================

st.markdown("---")
st.subheader("Documentación del Modelo")

with st.expander("Modelo Conceptual"):
    pdf_url = "https://raw.githubusercontent.com/ffedezn-cloud/tanque-calefaccionado/main/assets/docs/modelo_conceptual.pdf"
    viewer_url = f"https://docs.google.com/viewer?url={pdf_url}&embedded=true"
    
    st.markdown(
        f'''
        <iframe src="{viewer_url}" 
                width="100%" 
                height="700px" 
                style="border: 1px solid #ddd; border-radius: 4px;">
        </iframe>
        ''',
        unsafe_allow_html=True
    )

with st.expander("Código en Octave"):
    st.markdown("""
    Código autocontenido para simular el tanque calefaccionado en Octave.
    Para utilizarlo:
    1. Copiar el código
    2. Guardarlo en un archivo con extensión .m
    3. Ejecutarlo en Octave
    """)
    
    codigo_octave = '''% Tanque calefaccionado
% En X están las variables de estado.
% En Y deben ir las variables que se requieren en las ODEs o que se quieren graficar.

clear all; close all; clc;

%=============== Modelo =================

% ODEs
function dX = ODEs(t,X)
  % En dX devuelve el vector columna de derivadas

  % Recupera variables X
  [L T] = num2cell(X'){1,:};

  % Recupera variables Y
  Y = AEs(t,X);
  [A, F0, F, rho, Cp, T0, Q, Wa] = num2cell(Y){1,:};

  % Ecuaciones diferenciales
  dL = (F0 - F)/A;
  dT = (F0*rho*Cp*(T0-T)+Q+Wa)/(A*L*rho*Cp);

  dX = [dL dT]'; % vector columna
endfunction % ODEs
%---------------------------------------

% AEs
function Y = AEs(t,X)
  % En Y devuelve el vector fila de variables requeridas por ODEs o a graficar.

  % Recupera variables X
  [L T] = num2cell(X'){1,:};

  % Parámetros
  F0 = 2E-3; A = 0.785; Cv = 4.039E-5; rho = 1000; g = 9.81;
  Cp = 4.187E3; UAs = 4.04E3; T0 = 25; Tv = 132; Wa = 2000;  % Sistema SI

  % Ecuaciones algebraicas
  if t < 0                 %Apertura de la válvula
    x = 0.5;
  else
    x = 0.25;
  endif

  F = Cv*x*sqrt(rho*g*L);      %Caudal de salida
  Q = UAs*(Tv-T);

  Y = [A, F0, F, rho, Cp, T0, Q, Wa];
endfunction % AEs
%---------------------------------------

% Inicialización
function [tfin dt Xini LX LY] = inicializacion
  % Inicializa la simulación

  % Parámetros de simulación
  tfin = 1100; % tiempo final
  dt = 10; % paso temporal

  % Inicialización
  Lini = 1;  % m
  Tini = 60; %°C
  Xini = [Lini Tini]; % Inicializa la variable de estado

  % Leyendas
  LX = {'L' 'T'}; % Leyendas de las variables X
  LY = {'A' 'F0' 'F' 'rho' 'Cp' 'T0' 'Q' 'Wa'}; % Leyendas de las variables Y
endfunction % inicializar
%---------------------------------------

% Análisis
function analizar(LX,LY,tpts,X,Y)
  % Análisis de resultados. Funciones disponibles:
  % graficar({leyendas}, 'título', 'rótulo x', 'rótulo y', [limitesy])
  % vector(leyenda)

  % Solo graficar, sin exportar
  graficar({'L'}, 'Nivel vs. tiempo', 's', 'm', [0 3]);
  graficar({'F0' 'F'}, 'Caudales vs. tiempo', 's', 'm^3/s', [0 4E-3]);
  graficar({'T'}, 'Temperatura vs. tiempo', 's', '°C', [0 120]);
  
  % Control de rebalse
  Lmax = 2; % m Altura del tanque
  Lt = vector('L'); % Recupera el vector de niveles.
  if Lt(end) <= Lmax % Verifica el último nivel porque es el mayor.
    disp('El tanque no rebalsó.');
  else
    tr = interp1(Lt, tpts, Lmax); % Se puede usar interp1 porque Lt es creciente.
    disp(['El tanque rebalsó en el tiempo igual a ' num2str(tr) ' s.']);
    Tt = vector('T'); % Recupera el vector de temperaturas.
    Tr = interp1(tpts, Tt, tr); % Interpola la temperatura para tr.
    disp(['La temperatura del tanque en ese tiempo fue ' num2str(Tr) ' °C.']);
  endif

endfunction % analizar
%=======================================

%=============== Resolvedor (integrado) =================

function v = vector(leyenda)
  % Devuelve el vector columna correspondiente a la variable leyenda.
  global LX LY tpts X Y
  indicex = find(strcmp(LX, leyenda)); % Índice del elemento
  if length(indicex) == 1
    v = X(:,indicex);
  else
    indicey = find(strcmp(LY, leyenda)); % Índice del elemento
    if length(indicey) == 1
      v = Y(:,indicey);
    else
      disp(['Error: Variable "' leyenda '" no encontrada.']);
      error('Código de error: %d - Descripción del error', 1); % Detener con un mensaje
    endif
  endif
endfunction % vector
%---------------------------------------

function graficar(LV, titulo, rotulox, rotuloy, limitesy)
  % Crea una figura
  % LV: Arreglo de celdas fila que contiene los textos para las leyendas de las variables a graficar.
  % titulo: Título de la figura.
  % rotulox: Rótulo para la abscisa.
  % rotuloy: Rótulo para la ordenada.
  % limitesy: Vector fila con el límite inferior y el superior para la ordenada. Es opcional.
  global tpts

  colores = ['r' 'g' 'b' 'c' 'm' 'y' 'k'];

  figure;

  % Variables
  hold on; % Mantiene la figura para superponer la siguiente gráfica.
  for i = 1:length(LV)
    plot(tpts, vector(LV{i}), colores(mod(i-1,length(LV)) + 1), 'LineWidth', 2); % Línea con espesor 2
  endfor

  % Título del gráfico
  title(titulo);

  % Configurar los ejes
  xlabel(rotulox); % Título del eje x
  ylabel(rotuloy); % Título del eje y

  % Verificar y asignar valores predeterminados
  if nargin == 5
    ylim(limitesy); % Rango del eje y
  endif

  % Mostrar la cuadrícula
  grid on;

  % Añadir la leyenda
  legend(LV, 'Location', 'northeast'); % Leyenda en la esquina superior derecha

endfunction % graficar
%---------------------------------------

function [tpts X Y] = simulacion(tfin,dt,Xini)
  % Realiza la simulación.

  % Resolución
  nts = ceil(tfin/dt + 1); % redondea por exceso
  tpts = linspace(0, tfin, nts)';
  [tpts X] = ode45(@ODEs, tpts, Xini);

  % Cálculo de las variables dependientes
  for i = 1:size(tpts,1)
    Y(i,:) = AEs(tpts(i),X(i,:)');
  endfor

endfunction % simulacion

%=============== Simulación =================
clc;
disp('Resolvedor v01, 2025 (versión todo en uno del archivo del Dr. Tarifa)');
disp('');
disp('Resolviendo el modelo...');

global LX LY tpts X Y

% Inicialización
[tfin dt Xini LX LY] = inicializacion;

% Resolución
[tpts X Y] = simulacion(tfin,dt,Xini);

% Análisis (solo gráficos, sin exportar)
analizar(LX,LY,tpts,X,Y);

disp('');
disp('Simulación finalizada.');
'''
    
    st.code(codigo_octave, language="octave")
    
    st.download_button(
        label="Descargar modelo_tanque_calefaccionado.m",
        data=codigo_octave,
        file_name="modelo_tanque_calefaccionado.m",
        mime="text/plain"
    )

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #888; font-size: 14px; padding: 10px 0;">
        Simulador desplegado con Streamlit por Federico Franco
    </div>
    """,
    unsafe_allow_html=True
)

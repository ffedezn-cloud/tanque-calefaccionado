import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

st.set_page_config(page_title="Simulador Tanque Calefaccionado", layout="wide")

st.title("Simulador de Tanque Calefaccionado con Serpentin")
st.markdown("---")

with st.sidebar:
    st.header("Datos Geometricos del Tanque")
    D = st.number_input("Diametro del tanque D (m)", value=1.0, min_value=0.3, max_value=5.0, step=0.05)
    A = np.pi * (D/2)**2
    st.caption(f"Area calculada: {A:.4f} m²")
    
    L0 = st.number_input("Nivel inicial L0 (m)", value=1.0, min_value=0.0, max_value=5.0, step=0.05)
    L_max = st.number_input("Nivel maximo (rebalse) L_max (m)", value=2.0, min_value=0.5, max_value=10.0, step=0.1)
    
    st.header("Datos de Operacion")
    F0 = st.number_input("Caudal de entrada F0 (m³/s)", value=0.002, min_value=0.0001, max_value=0.1, format="%.5f")
    x0 = st.slider("Apertura inicial de valvula x0", 0.0, 1.0, 0.5, 0.01)
    xf = st.slider("Apertura final de valvula xf", 0.0, 1.0, 0.25, 0.01)
    
    st.header("Datos del Fluido")
    rho = st.number_input("Densidad del fluido rho (kg/m³)", value=1000.0, min_value=500.0, max_value=2000.0, step=10.0)
    Cp = st.number_input("Calor especifico Cp (J/kg°C)", value=4187.0, min_value=1000.0, max_value=8000.0, step=100.0)
    T0 = st.number_input("Temperatura de entrada T0 (°C)", value=25.0, min_value=0.0, max_value=100.0, step=1.0)
    
    st.header("Datos del Serpentin")
    Tv = st.number_input("Temperatura del vapor Tv (°C)", value=132.0, min_value=50.0, max_value=250.0, step=1.0)
    Wa = st.number_input("Potencia del agitador Wa (W)", value=2000.0, min_value=0.0, max_value=10000.0, step=100.0)
    T_initial = st.number_input("Temperatura inicial T0 (°C)", value=60.0, min_value=0.0, max_value=150.0, step=1.0)
    T_max_seguridad = st.number_input("Temperatura maxima de seguridad (°C)", value=80.0, min_value=30.0, max_value=150.0, step=1.0)
    
    st.header("Caracteristica de la Valvula")
    tipo_valvula = st.selectbox(
        "Tipo de caracteristica",
        ["Lineal", "Igual porcentaje (isoporcentual)", "Apertura rapida (quick opening)"]
    )
    
    R = 50
    if tipo_valvula == "Igual porcentaje (isoporcentual)":
        R = st.slider("Relacion de rango R (tipico 20-50)", 20, 100, 50, 5)
    
    st.header("Parametros de Simulacion")
    t_final = st.slider("Tiempo de simulacion (s)", 100, 2000, 1100, 100)

g = 9.81

# =============== FUNCION DE APERTURA DE VALVULA ===============
def f_apertura(x, tipo, R):
    x = max(0.0, min(1.0, x))
    if tipo == "Lineal":
        return x
    elif tipo == "Igual porcentaje (isoporcentual)":
        return R**(x-1)
    elif tipo == "Apertura rapida (quick opening)":
        return 1 - (1-x)**2
    return x

# =============== CALCULO DE PARAMETROS ===============
f0 = f_apertura(x0, tipo_valvula, R)
Cv_max = F0 / (f0 * np.sqrt(rho * g * L0))

# Calculo de UA a partir de condiciones estacionarias iniciales
UA = -(F0 * rho * Cp * (T0 - T_initial) + Wa) / (Tv - T_initial)

st.subheader("Parametros Calculados")
col_a, col_b = st.columns(2)
with col_a:
    st.metric("Area del tanque A", f"{A:.4f} m²")
    st.metric("Cv maximo", f"{Cv_max:.3e}")
    st.metric("UA (serpentin)", f"{UA:.2f} W/°C")
with col_b:
    st.metric("Caudal de entrada F0", f"{F0:.5f} m³/s = {F0*1000:.2f} L/s")
    st.metric("Nivel inicial L0", f"{L0:.2f} m")
    st.metric("Temperatura inicial T0", f"{T_initial:.1f} °C")

# =============== MODELO MATEMATICO ===============
def caudal_salida(L, x):
    f = f_apertura(x, tipo_valvula, R)
    return Cv_max * f * np.sqrt(rho * g * np.maximum(L, 0.001))

def modelo_dinamico(X, t, F0, A, x):
    L = X[0]
    T = X[1]
    
    F = caudal_salida(L, x)
    Q = UA * (Tv - T)
    
    dLdt = (F0 - F) / A
    dTdt = (F0 * rho * Cp * (T0 - T) + Q + Wa) / (A * L * rho * Cp)
    
    return [dLdt, dTdt]

# =============== ANALISIS ESTACIONARIO ===============
f_final = f_apertura(xf, tipo_valvula, R)

L_ss = (F0 / (Cv_max * f_final))**2 / (rho * g)
T_ss = (F0 * rho * Cp * T0 + UA * Tv + Wa) / (F0 * rho * Cp + UA)

st.subheader("Analisis Estacionario")

col1, col2 = st.columns(2)
with col1:
    st.info("Estado Estacionario Final (xf)")
    st.write(f"Nivel: **{L_ss:.2f} m**")
    st.write(f"{'Rebalsa' if L_ss > L_max else 'No rebalsa'}")
    st.write(f"Temperatura: **{T_ss:.1f} °C**")
    st.write(f"{'Peligro' if T_ss > T_max_seguridad else 'Seguro'}")

with col2:
    st.info("Apertura Minima para No Rebalsar")
    x_min = F0 / (Cv_max * np.sqrt(rho * g * L_max))
    st.metric("x_min", f"{x_min:.4f}", delta=f"Actual: {xf:.4f}")
    
    if xf < x_min:
        st.warning("La apertura final es menor que x_min. El tanque rebalsara.")
    else:
        st.success("La apertura final es mayor que x_min. Nivel seguro.")

# Temperatura de rebalse (si aplica)
if L_ss > L_max:
    T_rebalse = T_initial + (T_ss - T_initial) * (L_max - L0) / (L_ss - L0)
    st.warning(f"Temperatura estimada de rebalse: **{T_rebalse:.1f} °C**")

# =============== SIMULACION DINAMICA ===============
st.subheader("Simulacion Dinamica")

t = np.linspace(0, t_final, 1000)
X0 = [L0, T_initial]
sol = odeint(modelo_dinamico, X0, t, args=(F0, A, xf))

L = sol[:, 0]
T = sol[:, 1]
F = caudal_salida(L, xf)
Q = UA * (Tv - T)

# =============== GRAFICOS ===============
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Nivel
ax1 = axes[0, 0]
ax1.plot(t, L, 'b-', linewidth=2)
ax1.axhline(y=L_max, color='r', linestyle='--', label=f'L_max = {L_max} m')
ax1.axhline(y=L0, color='gray', linestyle=':', alpha=0.5, label=f'L0 = {L0} m')
ax1.set_xlabel("Tiempo (s)")
ax1.set_ylabel("Nivel L (m)")
ax1.set_title("Evolucion del nivel")
ax1.grid(True, alpha=0.3)
ax1.legend()

# Temperatura
ax2 = axes[0, 1]
ax2.plot(t, T, 'r-', linewidth=2)
ax2.axhline(y=T_max_seguridad, color='orange', linestyle='--', label=f'T_max_seg = {T_max_seguridad} °C')
ax2.axhline(y=T_initial, color='gray', linestyle=':', alpha=0.5, label=f'T0 = {T_initial} °C')
ax2.axhline(y=T_ss, color='g', linestyle='-.', alpha=0.5, label=f'T_ss = {T_ss:.1f} °C')
ax2.set_xlabel("Tiempo (s)")
ax2.set_ylabel("Temperatura (°C)")
ax2.set_title("Evolucion de la temperatura")
ax2.grid(True, alpha=0.3)
ax2.legend()

# Caudales
ax3 = axes[1, 0]
ax3.plot(t, np.ones_like(t)*F0, 'g--', linewidth=2, label=f'F0 = {F0:.5f} m³/s')
ax3.plot(t, F, 'r-', linewidth=2, label='F (salida)')
ax3.set_xlabel("Tiempo (s)")
ax3.set_ylabel("Caudal (m³/s)")
ax3.set_title("Caudales de entrada y salida")
ax3.grid(True, alpha=0.3)
ax3.legend()

# Calor
ax4 = axes[1, 1]
ax4.plot(t, Q, 'm-', linewidth=2, label='Q (serpentin)')
ax4.axhline(y=0, color='k', linestyle='-', alpha=0.2)
ax4.set_xlabel("Tiempo (s)")
ax4.set_ylabel("Calor (W)")
ax4.set_title("Transferencia de calor del serpentin")
ax4.grid(True, alpha=0.3)
ax4.legend()

plt.tight_layout()
st.pyplot(fig)

# =============== INFORME PARA EL OPERARIO ===============
st.subheader("Informe para el Operario")

L_final = L[-1]
T_final = T[-1]

tiempo_rebalse = None
temperatura_rebalse = None
for i, nivel in enumerate(L):
    if nivel >= L_max:
        tiempo_rebalse = t[i]
        temperatura_rebalse = T[i]
        break

col_r1, col_r2, col_r3, col_r4 = st.columns(4)
with col_r1:
    st.metric("Nivel final", f"{L_final:.3f} m", delta=f"{L_final - L0:.3f} m")
with col_r2:
    st.metric("Temperatura final", f"{T_final:.1f} °C", delta=f"{T_final - T_initial:.1f} °C")
with col_r3:
    st.metric("Caudal salida final", f"{F[-1]:.5f} m³/s")
with col_r4:
    st.metric("Calor final", f"{Q[-1]:.1f} W")

st.markdown("### Alertas de Seguridad")

col_alert1, col_alert2 = st.columns(2)

with col_alert1:
    if tiempo_rebalse:
        st.error(f"Rebalse detectado en t = {tiempo_rebalse:.1f} s")
        st.error(f"Temperatura de rebalse: {temperatura_rebalse:.1f} °C")
    else:
        st.success(f"Sin rebalse. Nivel maximo: {max(L):.3f} m")

with col_alert2:
    if T_final > T_max_seguridad:
        st.error(f"Temperatura final ({T_final:.1f} °C) supera el limite de seguridad ({T_max_seguridad} °C)")
    else:
        st.success(f"Temperatura segura. Maximo: {max(T):.1f} °C")

st.subheader("Recomendaciones")

recomendaciones = []

if xf < x_min:
    recomendaciones.append(f"Abrir la valvula al menos a x = {x_min:.4f} para evitar el rebalse.")

if T_final > T_max_seguridad:
    recomendaciones.append(f"Reducir la temperatura del vapor o aumentar la circulacion de agua. Temperatura actual: {T_final:.1f} °C")

if L_final > L_max * 0.9:
    recomendaciones.append(f"Nivel critico ({L_final:.2f} m). Monitorear constantemente.")

if T_final > T_max_seguridad * 0.8:
    recomendaciones.append(f"Temperatura elevada ({T_final:.1f} °C). Revisar sistemas de seguridad.")

if recomendaciones:
    for rec in recomendaciones:
        st.warning(rec)
else:
    st.success("Operacion segura. Todos los parametros dentro de limites aceptables.")

# =============== DOCUMENTACION ===============
st.markdown("---")
st.subheader("Documentacion del Modelo")

with st.expander("Modelo Conceptual"):
    st.markdown("### Modelo Dinamico")
    st.latex(r"A \frac{dL}{dt} = F_0 - F")
    st.latex(r"A L \rho Cp \frac{dT}{dt} = F_0 \rho Cp (T_0 - T) + Q + W_a")
    st.latex(r"F = C_v \cdot f(x) \cdot \sqrt{\rho g L}")
    st.latex(r"Q = UA \cdot (T_v - T)")
    
    st.markdown("### Modelo Estacionario")
    st.latex(r"L_{ss} = \frac{1}{\rho g} \left( \frac{F_0}{C_v \cdot f(x)} \right)^2")
    st.latex(r"T_{ss} = \frac{F_0 \rho Cp T_0 + UA \cdot T_v + W_a}{F_0 \rho Cp + UA}")

with st.expander("Variables del Modelo"):
    st.markdown("""
    | Simbolo | Descripcion | Unidad |
    |---------|-------------|--------|
    | A | Area del tanque | m² |
    | L | Nivel del liquido | m |
    | T | Temperatura del liquido | °C |
    | F0 | Caudal de entrada | m³/s |
    | F | Caudal de salida | m³/s |
    | Cv | Coeficiente de valvula | - |
    | f(x) | Caracteristica de valvula | - |
    | rho | Densidad del fluido | kg/m³ |
    | Cp | Calor especifico | J/kg°C |
    | T0 | Temperatura de entrada | °C |
    | Tv | Temperatura del vapor | °C |
    | UA | Coeficiente global de transferencia | W/°C |
    | Q | Calor transferido | W |
    | Wa | Potencia del agitador | W |
    | g | Gravedad | m/s² |
    """)

with st.expander("Codigo Octave (descargable)"):
    codigo_octave = """% Tanque calefaccionado con serpentin
% Modelo unificado para Octave

clear all; close all; clc;

disp('Simulador de Tanque calefaccionado');

% =============== PARAMETROS ===============
F0 = 2e-3;          % Caudal entrada (m³/s)
A = 0.785;          % Area (m²)
Cv = 4.039e-5;      % Coef. valvula
rho = 1000;         % Densidad (kg/m³)
g = 9.81;           % Gravedad (m/s²)
Cp = 4187;          % Calor especifico (J/kg°C)
T0 = 25;            % Temp. entrada (°C)
Tv = 132;           % Temp. vapor (°C)
Wa = 2000;          % Potencia agitador (W)
T_initial = 60;     % Temp. inicial (°C)
L0 = 1;             % Nivel inicial (m)
L_max = 2;          % Nivel maximo (m)
x0 = 0.5;           % Apertura inicial
xf = 0.25;          % Apertura final

% Calculo de UA
UA = -(F0*rho*Cp*(T0-T_initial) + Wa)/(Tv-T_initial);

% Funciones
function f = f_apertura(x)
    f = x;  % Lineal
endfunction

function F = caudal(L, x, Cv, rho, g)
    F = Cv * x * sqrt(rho * g * max(L, 0.001));
endfunction

function dX = modelo(t, X)
    L = X(1);
    T = X(2);
    
    F = caudal(L, xf, Cv, rho, g);
    Q = UA * (Tv - T);
    
    dL = (F0 - F) / A;
    dT = (F0*rho*Cp*(T0-T) + Q + Wa) / (A*L*rho*Cp);
    
    dX = [dL; dT];
endfunction

% Simulacion
t = linspace(0, 1100, 1000)';
X0 = [L0; T_initial];
[t, X] = ode45(@modelo, t, X0);

L = X(:,1);
T = X(:,2);

% Graficos
figure('Position', [100, 100, 1000, 800]);

subplot(2,2,1);
plot(t, L, 'b-', 'LineWidth', 2);
hold on;
yline(L_max, 'r--');
xlabel('Tiempo (s)');
ylabel('Nivel (m)');
title('Evolucion del nivel');
grid on;

subplot(2,2,2);
plot(t, T, 'r-', 'LineWidth', 2);
xlabel('Tiempo (s)');
ylabel('Temperatura (°C)');
title('Evolucion de la temperatura');
grid on;

subplot(2,2,3);
F = arrayfun(@(L) caudal(L, xf, Cv, rho, g), L);
plot(t, F, 'r-', 'LineWidth', 2);
hold on;
yline(F0, 'g--');
xlabel('Tiempo (s)');
ylabel('Caudal (m³/s)');
title('Caudales');
grid on;

subplot(2,2,4);
Q = UA * (Tv - T);
plot(t, Q, 'm-', 'LineWidth', 2);
xlabel('Tiempo (s)');
ylabel('Calor (W)');
title('Transferencia de calor');
grid on;

% Resultados
fprintf('\\n========== RESULTADOS ==========\\n');
fprintf('Nivel final: %.3f m\\n', L(end));
fprintf('Temperatura final: %.1f °C\\n', T(end));
fprintf('Temperatura estacionaria: %.1f °C\\n', (F0*rho*Cp*T0 + UA*Tv + Wa)/(F0*rho*Cp + UA));

disp('Simulacion finalizada.');
"""
    
    st.code(codigo_octave, language="octave")
    
    st.download_button(
        label="Descargar modelo_octave.m",
        data=codigo_octave,
        file_name="modelo_tanque_calefaccionado.m",
        mime="text/plain"
    )

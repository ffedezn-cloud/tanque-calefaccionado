
# Simulador de Tanque Calefaccionado con Serpentín

---

## Qué hace este simulador

Simula un tanque calefaccionado con descarga gravitatoria. Permite analizar la respuesta dinámica del sistema ante cambios en la apertura de la válvula de salida y perturbaciones en el caudal de entrada, con gráficas interactivas que muestran la evolución del nivel y la temperatura.

Incluye:

- Análisis con el modelo estacionario: cálculo de Cv experimental, determinar apertura mínima antes que rebalse y a qué altura rebalsa
- Análisis con el modelo dinámico: saber en qué momento va a rebalsar

> Documentación técnica: [modelo_conceptual.pdf](assets/docs/modelo_conceptual.pdf)

---

## Modelo de espacio de estados

El modelo matemático completo se encuentra desarrollado en el documento técnico. La estrategia de implementación fue la siguiente:

- Implementar en Octave el modelo de espacio de estados
- Pasar al lenguaje Python la aplicación
- Desplegar la aplicación en Streamlit a través de GitHub
- Usar IDEs: Geany y VSCodium, según conveniencia, con IA generativa para detectar errores de código, para indentación automática y sugerencias para mejorar experiencia frontend

---

## Tecnologías Utilizadas

- SO: AntiX Linux
- IDE: Geany / Geany Copilot - VSCodium / API DeepSeek

| Tecnología | Propósito |
|------------|-----------|
| Octave | Modelado inicial y validación |
| Python 3.8+ | Lenguaje necesario para desplegar en Streamlit |
| Streamlit | Interfaz web interactiva |
| Plotly | Gráficas interactivas |
| SciPy | Resolución de ecuaciones diferenciales |
| NumPy | Operaciones numéricas |

---

## Cómo usar el simulador

### Opción 1: En línea (recomendado)

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://tanque-calefaccionado.streamlit.app)

### Opción 2: Localmente

Clonar el repositorio:

    git clone https://github.com/ffedezn-cloud/tanque-calefaccionado.git
    cd tanque-calefaccionado

Instalar dependencias:

    pip install -r requirements.txt

Ejecutar la aplicación:

    streamlit run app.py

---

## Bibliografía

- Tarifa, E. (2025). Apuntes Simulación y Optimización de Procesos. UNJu - FI.
- Ingham, J. (1994). Chemical Engineering Dynamics. Editorial VCH.
- Documentación de Streamlit: https://docs.streamlit.io
- Documentación de SciPy: https://docs.scipy.org

---

## Créditos

| Rol | Nombre |
|-----|--------|
| Autor | Federico Franco |
| Carrera | Ingeniería Química |
| Año | 2026 |

---

## Licencia

Distribuido bajo licencia MIT. Ver el archivo LICENSE para más información.

---

## Contacto

Federico Franco
ffede.zn@gmail.com

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=flat-square&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/fede-franco-70a301418/)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/ffedezn-cloud)



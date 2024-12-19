import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import io

st.set_page_config(page_title="Proyecto Final", layout="wide")  # nombre que aparece arriba
col1, col2 = st.columns([7, 1])
with col2:
    st.image("logoutn.jpg", width=500)  # imagen utn

st.title("Simulador de Generación Fotovoltaica")
"""Esta aplicación fue desarrollada para analizar datos climatológicos de la ciudad de Santa Fe con el objetivo de estimar la potencia generada 
por un generador fotovoltaico, basado en el modelo matemático descrito en el proyecto."""

st.markdown("### Marco teórico")


st.markdown("""
La creciente demanda energética y el impacto ambiental asociado a las fuentes convencionales de energía han impulsado el desarrollo de soluciones sostenibles, 
como la energía fotovoltaica. Los **sistemas de generación fotovoltaica (GFV)** destacan por su capacidad de transformar la energía solar en electricidad 
de manera limpia y eficiente. Este marco teórico describe los fundamentos, modelos matemáticos y características de los sistemas fotovoltaicos, 
aplicados al proyecto de simulación desarrollado con Streamlit.
""")

# Fundamentos de la Energía Fotovoltaica
st.header("1. Fundamentos de la Energía Fotovoltaica")
st.markdown("""
Un **generador fotovoltaico (GFV)** convierte la radiación solar en electricidad mediante módulos solares, configurados en serie o paralelo según las necesidades.
El sistema se complementa con un controlador-inversor que estabiliza la energía y la convierte de corriente continua (CC) a corriente alterna (CA), adecuada para
su uso en redes eléctricas. El siguiente esquema describe el sistema:
""")
# Modelo Matemático
st.header("2. Modelo Matemático para el Cálculo de Potencia")
st.markdown("""
La potencia eléctrica generada por un GFV puede calcularse mediante la siguiente ecuación:
""")
st.latex(r"""
P \, [\text{kW}] = N \cdot \frac{G}{G_{std}} \cdot P_{pico} \cdot \left[ 1 + k_p \cdot (T_c - T_r) \right] \cdot \eta \cdot 10^{-3}
""")

st.markdown("**Donde:**")

"""
- \( G \): Irradiancia incidente (\(\text{W/m}^2\)).
- \( G_{std} \): Irradiancia estándar (\(1000 \, \text{W/m}^2\)).
- \( T_c \): Temperatura de la celda (\(°\text{C}\)).
- \( T_r \): Temperatura de referencia (\(25 \, °\text{C}\)).
- \( P_{pico} \): Potencia pico del módulo (\(240 \, \text{W}\)).
- \( N \): Número de módulos.
- \( k_p \): Coeficiente de temperatura-potencia (\(-0.0044 \, °\text{C}^{-1}\)).
- \( \eta \): Rendimiento global del sistema (\(0.97\)).
"""

# Estimación de la Temperatura de la Celda
st.header("3. Estimación de la Temperatura de la Celda")
st.markdown("""
La temperatura de las celdas fotovoltaicas (\(T_c\)) se puede estimar a partir de la temperatura ambiente (\(T_a\)) y la irradiancia incidente (\(G\)), 
según la siguiente fórmula:
""")
st.latex(r"""
T_c = T_a + 0.031 \cdot G
""")

st.markdown("""
Este modelo simplificado es válido en condiciones de viento nulo y asume que la radiación incrementa la temperatura de los módulos de manera proporcional.
""")

# Límites de Generación
st.header("4. Límites de Generación del GFV")
st.markdown("""
La potencia generada por el GFV está restringida por límites superiores e inferiores. Estos límites se expresan como:
""")

st.subheader("4.1 Límite Inferior")
st.markdown("""
La potencia generada debe superar un umbral mínimo para que el inversor funcione correctamente:
""")
st.latex(r"""
P_{\text{mín}} = \frac{\mu}{100} \cdot P_{\text{inv}}
""")

st.subheader("4.2 Límite Superior")
st.markdown("""
La potencia máxima está limitada por la capacidad nominal del inversor. La potencia entregada (\(P_r\)) se calcula como:
""")
st.latex(r"""
P_r =
\begin{cases}
0 & \text{si } P \leq P_{\text{mín}} \\
P & \text{si } P_{\text{mín}} < P \leq P_{\text{inv}} \\
P_{\text{inv}} & \text{si } P > P_{\text{inv}}
\end{cases}
""")

# Características del GFV de la UTN Santa Fe
st.header("5. Características del GFV de la UTN Santa Fe")
st.markdown("""
El generador fotovoltaico de la Facultad Regional Santa Fe está compuesto por los siguientes elementos:
- **Módulos fotovoltaicos**: \(N = 12\), con una potencia pico de \(240 \, \text{W}\) cada uno.
- **Inversor monofásico**: Marca **SMA**, modelo **SB2.5-1VL-40**, con potencia nominal de \(2.5 \, \text{kW}\).
- **Rendimiento global del sistema**: \(0.97\).

El GFV compensa parcialmente el consumo eléctrico de la facultad y, en circunstancias de baja demanda, puede inyectar excedentes a la red eléctrica.
""")

# Datos Climatológicos
st.header("6. Datos Climatológicos")
st.markdown("""
Los datos utilizados en este proyecto provienen de registros del Centro de Información Meteorológica (CIM) de la Universidad Nacional del Litoral (UNL). 
Estos datos incluyen:
- **Temperatura ambiente (\(T_a\))**.
- **Irradiancia solar (\(G\))**.

Las mediciones se realizaron a intervalos de 10 minutos durante todo el año 2019, proporcionando información detallada para la simulación.
""")

# Relevancia del Proyecto
st.header("7. Relevancia del Proyecto")
st.markdown("""
Este marco teórico sirve como base para el desarrollo de una herramienta interactiva que utiliza **Streamlit** para simular la generación fotovoltaica. 
La interfaz permite:
- Visualizar tablas dinámicas con datos climatológicos.
- Analizar gráficos de la potencia generada y su relación con variables como la irradiancia y la temperatura.
- Configurar parámetros del GFV de forma interactiva, como el número de paneles y la eficiencia del sistema.

Esta simulación es útil para entender el comportamiento de un GFV y evaluar su rendimiento en diferentes condiciones meteorológicas.
""")


st.latex(r"P \, (kW) = N \cdot \frac{G}{G_{std}} \cdot P_{pico} \cdot \left[ 1 + k_p \cdot (T_c - T_r) \right] \cdot \eta \cdot 10^{-3}")
st.latex(r"E \, [kWh] = \int P(t) \, dt")
st.latex(r"\eta_{sistema} = \eta_{panel} \cdot \eta_{inversor} \cdot \eta_{cables}")
st.latex(r"T_c = T_{a} + \frac{G}{G_{std}} \cdot \Delta T")
st.latex(r"I = \frac{P}{V} \quad \text{(Intensidad de corriente)}")
st.latex(r"V = \frac{P}{I} \quad \text{(Voltaje)}")
st.latex(r"PR = \frac{E_{real}}{E_{teorica}} \cdot 100 \% \quad \text{(Performance Ratio)}")

### BARRA LATERAL

st.sidebar.title("Configuración del Generador Fotovoltaico")

# Parámetros del generador
N = st.sidebar.number_input("Número de paneles (N)",
                            min_value=0.0,
                            value=12.0,
                            step=1.0)
Gstd = st.sidebar.number_input("Irradiancia (G) [W/m²]", 
                            min_value=0.0, 
                            max_value=1100.0, 
                            value=1000.0, 
                            step=10.0)
Ppico = st.sidebar.number_input("Potencia pico por módulo (Ppico) [W]",
                            min_value=100.0,
                            max_value=600.0,
                            value=240.0,
                            step=10.0)
Pinv = st.sidebar.slider("Potencia nominal del inversor (Pinv) [kW]",
                            min_value=0.0,
                            max_value=10.0,
                            value=2.5,
                            step=0.5)
kp = st.sidebar.number_input("Coeficiente temperatura-potencia (kp) [°C⁻¹]",
                            min_value=-1.0,
                            max_value=1.0,
                            value=-0.0044,
                            step=0.0001,
                            format="%.4f")
Tr = st.sidebar.number_input("Temperatura de referencia o ambiente (Tr) [°C]",
                            min_value=-50.0,
                            max_value=50.0,
                            value=25.0,
                            step=1.0)
Tc = st.sidebar.number_input("Temperatura de la celda (Tc) [°C]",
                            min_value=-50.0,
                            value=25.0,
                            step=1.0)
eta = st.sidebar.slider("Rendimiento global (η)",
                            min_value=0.0,
                            max_value=1.0,
                            value=0.97,
                            step=0.01)
mu = st.sidebar.slider("Porcentaje de umbral mínimo %",
                            min_value=1.0,
                            max_value=100.0,
                            value=25.0,
                            step=1.0)

Pmin = mu / 100 * Pinv
st.sidebar.text(f"Potencia del Inversor Mínima: {Pmin}(kW)")

# Cálculo de la potencia generada
P = N * (Gstd / 1000) * Ppico * (1 + kp * (Tc - Tr)) * eta * 1e-3
st.sidebar.markdown(f"""
    <div style='font-size: 24px; color: #ff6347; font-weight: bold; 
                padding: 10px; border: 2px solid #ff6347; 
                border-radius: 10px; background-color: #fff8f0;'>
        Potencia generada: {P:.2f} kW
    </div>
""", unsafe_allow_html=True)


# elección de qué archivo se usa
opcion = st.radio(
    "Selecciona una opción:",
    ("Usar archivo preestablecido", "Cargar archivo distinto")
)

if opcion == "Usar archivo preestablecido": #seleccionador de archivo
    archivo = pd.read_excel("Datos_climatologicos_Santa_Fe_2019.xlsx")
    st.success("Usando el archivo preestablecido.")
else:
    # Mostrar widget para subir archivo
    archivo_cargado = st.file_uploader("Sube un archivo Excel", type=["xlsx", "xls"])
    if archivo_cargado is not None:
        # Leer el archivo cargado
        archivo = pd.read_excel(archivo_cargado)
        st.success("Archivo cargado con éxito!")
    else:
        archivo = None
        st.warning("Por favor, suba un archivo para continuar.")

archivo['Tc (°C)'] = archivo['Temperatura (°C)'] + 0.031 * archivo['Irradiancia (W/m²)']

# Cálculo de potencia generada (kW)
archivo['Potencia (kW)'] = (N * (archivo['Irradiancia (W/m²)'] / Gstd) * 
                            Ppico * (1 + kp * (archivo['Tc (°C)'] - Tr)) * eta * 1e-3)

# Aplicar límites a la potencia generada
archivo['Potencia (kW)'] = np.where(
    archivo['Potencia (kW)'] <= Pmin, 0,  # Si P <= Pmin -> 0
    np.where(archivo['Potencia (kW)'] <= Pinv, archivo['Potencia (kW)'], Pinv)
)

st.write("Tabla con valores de Tc y Potencia Calculados:")
st.dataframe(archivo, width=1400, hide_index=True)


st.markdown("## Gráficas")
### GRÁFICAS

# Asegúrate de que la columna 'Fecha' esté en formato datetime
archivo['Fecha'] = pd.to_datetime(archivo['Fecha'])

# Extraer el mes de la fecha
archivo['Mes'] = archivo['Fecha'].dt.month

# Función para graficar la temperatura
def grafica_temperatura(opcion):
    if opcion == 'Por Día':
        # Agrupar por día y calcular el promedio de la temperatura
        archivo['Fecha_Solo'] = archivo['Fecha'].dt.date
        promedio_por_dia = archivo.groupby('Fecha_Solo')['Temperatura (°C)'].mean().reset_index()
        promedio_por_dia.columns = ['Fecha', 'Promedio Temperatura (°C)']
        
        # Crear la gráfica de temperatura
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(promedio_por_dia['Fecha'], promedio_por_dia['Promedio Temperatura (°C)'], 
                color='blue', label="Temperatura", marker="o", markersize=3)
        ax.set_title('Promedio de Temperatura por Día', fontsize=16)
        ax.set_xlabel('Fecha', fontsize=14)
        ax.set_ylabel('Promedio Temperatura (°C)', fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()
        return fig
    
    elif opcion == 'Por Mes':
        # Agrupar por mes y calcular el promedio de la temperatura
        archivo['Mes'] = archivo['Fecha'].dt.month
        promedio_por_mes = archivo.groupby('Mes')['Temperatura (°C)'].mean().reset_index()
        
        # Crear la gráfica de temperatura por mes
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(promedio_por_mes['Mes'], promedio_por_mes['Temperatura (°C)'], 
                color='blue', label="Temperatura Promedio", marker="o", markersize=5)
        ax.set_title('Promedio de Temperatura por Mes', fontsize=16)
        ax.set_xlabel('Mes', fontsize=14)
        ax.set_ylabel('Temperatura Promedio (°C)', fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(fontsize=12)
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic'], rotation=45)
        plt.tight_layout()
        return fig

# Función para graficar la potencia total generada
def grafica_potencia(opcion):
    if opcion == 'Por Día':
        # Agrupar por día y calcular la potencia total generada
        archivo['Fecha_Solo'] = archivo['Fecha'].dt.date
        potencia_total_dia = archivo.groupby('Fecha_Solo')['Potencia (kW)'].sum().reset_index()
        
        # Crear la gráfica de potencia total por día
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(potencia_total_dia['Fecha_Solo'], potencia_total_dia['Potencia (kW)'], 
                color='orange', label="Potencia Total", marker="o", markersize=5)
        ax.set_title('Potencia Total Generada por Día', fontsize=16)
        ax.set_xlabel('Fecha', fontsize=14)
        ax.set_ylabel('Potencia Total (kW)', fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()
        return fig
    
    elif opcion == 'Por Mes':
        # Agrupar por mes y calcular la potencia total generada
        potencia_total_mes = archivo.groupby('Mes')['Potencia (kW)'].sum().reset_index()
        
        # Asegurarnos de que la columna 'Mes' sea un entero
        potencia_total_mes['Mes'] = potencia_total_mes['Mes'].astype(int)

        # Crear la gráfica de barras de potencia por mes
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(potencia_total_mes['Mes'], potencia_total_mes['Potencia (kW)'], color='orange', edgecolor='black')
        
        # Configurar el título y etiquetas
        ax.set_title('Potencia Total Generada Mensualmente (kW)', fontsize=16)
        ax.set_xlabel('Mes', fontsize=14)
        ax.set_ylabel('Potencia Total (kW)', fontsize=14)

        # Ajustar las etiquetas del eje X para los meses
        ax.set_xticks(range(1, 13))  # Posiciones en el eje X para 12 meses
        ax.set_xticklabels(['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic'], rotation=45)

        ax.grid(axis='y', linestyle='--', alpha=0.6)
        plt.tight_layout()
        return fig

# Crear columnas para mostrar las gráficas una al lado de la otra
col1, col2 = st.columns(2)

# Crear botones para seleccionar cómo mostrar la temperatura
with col1:
    opcion_temperatura = st.radio("Selecciona cómo mostrar la temperatura", ("Por Día", "Por Mes"))
    fig_temperatura = grafica_temperatura(opcion_temperatura)
    st.pyplot(fig_temperatura)

# Crear botones para seleccionar cómo mostrar la potencia
with col2:
    opcion_potencia = st.radio("Selecciona cómo mostrar la potencia", ("Por Día", "Por Mes"))
    fig_potencia = grafica_potencia(opcion_potencia)
    st.pyplot(fig_potencia)

# Función para convertir la figura en un objeto de imagen para la descarga
def fig_to_image(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    return buf

# Función para crear un archivo descargable
def crear_descarga(fig, nombre_archivo):
    img = fig_to_image(fig)
    return st.download_button(label=f"Descargar {nombre_archivo}", data=img, file_name=nombre_archivo, mime="image/png")


# Función para convertir la figura en un objeto de imagen para la descarga
def fig_to_image(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    return buf

# Función para crear un archivo descargable
def crear_descarga(fig, nombre_archivo):
    img = fig_to_image(fig)
    return st.download_button(label=f"Descargar {nombre_archivo}", data=img, file_name=nombre_archivo, mime="image/png")

with col1:
    # Botón para descargar la gráfica de temperatura
    if st.button("Descargar Gráfica de Temperatura"):
        crear_descarga(fig_temperatura, "grafica_temperatura.png")

with col2:
    # Botón para descargar la gráfica de potencia
    if st.button("Descargar Gráfica de Potencia"):
        crear_descarga(fig_potencia, "grafica_potencia.png")

st.markdown("\n")
st.markdown("## Búsqueda por filtros")
### FILTROS

if archivo is not None:
    archivo['Fecha'] = pd.to_datetime(archivo['Fecha'])  # Convertir columna 'Fecha' a formato datetime
    
    # Filtros de fecha
    fecha_inicio = st.date_input("Selecciona la fecha de inicio", archivo['Fecha'].min().date())
    fecha_fin = st.date_input("Selecciona la fecha de fin", archivo['Fecha'].max().date())
    
    # Filtro de temperatura
    temp_min = archivo['Temperatura (°C)'].min()
    temp_max = archivo['Temperatura (°C)'].max()
    temperatura_filtrada = st.slider("Selecciona el rango de temperatura", 
                                    min_value=float(temp_min), max_value=float(temp_max), 
                                    value=(float(temp_min), float(temp_max)))
    
    # Filtro de irradiancia
    irrad_min = archivo['Irradiancia (W/m²)'].min()
    irrad_max = archivo['Irradiancia (W/m²)'].max()
    irradiancia_filtrada = st.slider("Selecciona el rango de irradiancia", 
                                     min_value=float(irrad_min), max_value=float(irrad_max), 
                                     value=(float(irrad_min), float(irrad_max)))
    
    # Filtrar los datos según las selecciones de fecha, temperatura e irradiancia
    archivo_filtrado = archivo[
        (archivo['Fecha'].dt.date >= fecha_inicio) & 
        (archivo['Fecha'].dt.date <= fecha_fin) & 
        (archivo['Temperatura (°C)'] >= temperatura_filtrada[0]) &
        (archivo['Temperatura (°C)'] <= temperatura_filtrada[1]) &
        (archivo['Irradiancia (W/m²)'] >= irradiancia_filtrada[0]) &
        (archivo['Irradiancia (W/m²)'] <= irradiancia_filtrada[1])
    ]
    
    # Mostrar los datos filtrados
    st.write(f"Datos filtrados de {fecha_inicio} a {fecha_fin}")
    st.dataframe(archivo_filtrado, width=1500, hide_index=True)
    
    # Crear opción para graficar según la selección de filtros
    if not archivo_filtrado.empty:
        # Función para graficar la temperatura
        def grafica_temperatura(opcion):
            if opcion == 'Por Día':
                # Agrupar por día y calcular el promedio de la temperatura
                archivo_filtrado['Fecha_Solo'] = archivo_filtrado['Fecha'].dt.date
                promedio_por_dia = archivo_filtrado.groupby('Fecha_Solo')['Temperatura (°C)'].mean().reset_index()
                promedio_por_dia.columns = ['Fecha', 'Promedio Temperatura (°C)']
                
                # Crear la gráfica de temperatura
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(promedio_por_dia['Fecha'], promedio_por_dia['Promedio Temperatura (°C)'], 
                        color='blue', label="Temperatura", marker="o", markersize=3)
                ax.set_title('Promedio de Temperatura por Día', fontsize=16)
                ax.set_xlabel('Fecha', fontsize=14)
                ax.set_ylabel('Promedio Temperatura (°C)', fontsize=14)
                ax.grid(True, linestyle='--', alpha=0.6)
                ax.legend(fontsize=12)
                plt.xticks(rotation=45)
                plt.tight_layout()
                return fig
            
            elif opcion == 'Por Mes':
                # Agrupar por mes y calcular el promedio de la temperatura
                archivo_filtrado['Mes'] = archivo_filtrado['Fecha'].dt.month
                promedio_por_mes = archivo_filtrado.groupby('Mes')['Temperatura (°C)'].mean().reset_index()
                
                # Crear la gráfica de temperatura por mes
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(promedio_por_mes['Mes'], promedio_por_mes['Temperatura (°C)'], 
                        color='blue', label="Temperatura Promedio", marker="o", markersize=5)
                ax.set_title('Promedio de Temperatura por Mes', fontsize=16)
                ax.set_xlabel('Mes', fontsize=14)
                ax.set_ylabel('Temperatura Promedio (°C)', fontsize=14)
                ax.grid(True, linestyle='--', alpha=0.6)
                ax.legend(fontsize=12)
                ax.set_xticks(range(1, 13))
                ax.set_xticklabels(['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic'], rotation=45)
                plt.tight_layout()
                return fig

        # Función para graficar la potencia total generada
        def grafica_potencia(opcion):
            if opcion == 'Por Día':
                # Agrupar por día y calcular la potencia total generada
                archivo_filtrado['Fecha_Solo'] = archivo_filtrado['Fecha'].dt.date
                potencia_total_dia = archivo_filtrado.groupby('Fecha_Solo')['Potencia (kW)'].sum().reset_index()
                
                # Crear la gráfica de potencia total por día
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(potencia_total_dia['Fecha_Solo'], potencia_total_dia['Potencia (kW)'], 
                        color='orange', label="Potencia Total", marker="o", markersize=5)
                ax.set_title('Potencia Total Generada por Día', fontsize=16)
                ax.set_xlabel('Fecha', fontsize=14)
                ax.set_ylabel('Potencia Total [kW]', fontsize=14)
                ax.grid(True, linestyle='--', alpha=0.6)
                ax.legend(fontsize=12)
                plt.xticks(rotation=45)
                plt.tight_layout()
                return fig
            
            elif opcion == 'Por Mes':
                # Agrupar por mes y calcular la potencia total generada
                potencia_total_mes = archivo_filtrado.groupby('Mes')['Potencia [kW]'].sum().reset_index()
                
                # Asegurarnos de que la columna 'Mes' sea un entero
                potencia_total_mes['Mes'] = potencia_total_mes['Mes'].astype(int)

                # Crear la gráfica de barras de potencia por mes
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.bar(potencia_total_mes['Mes'], potencia_total_mes['Potencia [kW]'], color='orange', edgecolor='black')
                
                # Configurar el título y etiquetas
                ax.set_title('Potencia Total Generada Mensualmente [kW]', fontsize=16)
                ax.set_xlabel('Mes', fontsize=14)
                ax.set_ylabel('Potencia Total [kW]', fontsize=14)

                # Ajustar las etiquetas del eje X para los meses
                ax.set_xticks(range(1, 13))  # Posiciones en el eje X para 12 meses
                ax.set_xticklabels(['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic'], rotation=45)

                ax.grid(axis='y', linestyle='--', alpha=0.6)
                plt.tight_layout()
                return fig

        # Mostrar las gráficas una al lado de la otra en columnas
        col1, col2 = st.columns(2)

        with col1:
            # Mostrar la opción para seleccionar la visualización de la temperatura
            opcion_temperatura = st.radio("Selecciona cómo mostrar la temperatura", ("Por Día", "Por Mes"), key="temperatura_radio")
            fig_temperatura = grafica_temperatura(opcion_temperatura)
            st.pyplot(fig_temperatura)

            # Botón de descarga para la gráfica de temperatura
            buf = io.BytesIO()
            fig_temperatura.savefig(buf, format="png")
            buf.seek(0)
            st.download_button("Descargar gráfica de Temperatura", buf, "grafica_temperatura.png", "image/png")

        with col2:
            # Mostrar la opción para seleccionar la visualización de la potencia
            opcion_potencia = st.radio("Selecciona cómo mostrar la potencia", ("Por Día", "Por Mes"), key="potencia_radio")
            fig_potencia = grafica_potencia(opcion_potencia)
            st.pyplot(fig_potencia)

            # Botón de descarga para la gráfica de potencia
            buf = io.BytesIO()
            fig_potencia.savefig(buf, format="png")
            buf.seek(0)
            st.download_button("Descargar gráfica de Potencia", buf, "grafica_potencia.png", "image/png")

            
# Estado de la sesión para manejar la visibilidad de la información
if 'mostrar_info' not in st.session_state:
    st.session_state.mostrar_info = False

# Botón para mostrar/ocultar la información
if st.button("Más información de Nosotros"):
    st.session_state.mostrar_info = not st.session_state.mostrar_info

# Mostrar la información adicional si el estado 'mostrar_info' es verdadero
if st.session_state.mostrar_info:
    st.markdown(
        """
        <div style="padding: 10px; background-color: #f4f4f9; border-radius: 5px; border: 1px solid #ccc;">
            <h4 style="color: #333;">Información adicional:</h4>
            <p style="color: #333;"><strong>Desarrolladores:</strong> Cisera Santino y Stangafero Eric</p>
            <p style="color: #333;"><strong>Correo de contacto:</strong> <a href="mailto:santinociseraa@gmail.com" style="color: #1e90ff;">santinociseraa@gmail.com</a> / <a href="mailto:edstangafe@gmail.com" style="color: #1e90ff;">edstangafe@gmail.com</a></p>
            <p style="color: #333;"><strong>Número de contacto:</strong> 3425328666 / 3425950884</p>
        </div>
        """, 
        unsafe_allow_html=True
    )

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Datos
N = 12                   # Número de paneles
Ppico = 240              # Potencia pico [W]
kp = -0.0044             # Coeficiente de temperatura de potencia [1/°C]
Pinv = 2.5               # Potencia nominal del inversor [kW]
eta = 0.97               # Eficiencia
Tr = 25                  # Temperatura de referencia [°C]
Gstd = 1000              # Irradiancia estándar [W/m²]
mu = 25                   # Porcentaje de umbral mínimo
Pmin = mu / 100 * Pinv   # Potencia mínima

# Cargar los datos desde Excel
tabla = pd.read_excel('Datos_climatologicos_Santa_Fe_2019.xlsx')

# Asegurarse de que 'Fecha' sea de tipo datetime
tabla['Fecha'] = pd.to_datetime(tabla['Fecha'])

# Cálculo de temperatura de las celdas [°C]
tabla['Tc'] = tabla['Temperatura (°C)'] + 0.031 * tabla['Irradiancia (W/m²)']

# Cálculo de potencia generada [kW]
tabla['Potencia [kW]'] = (N * (tabla['Irradiancia (W/m²)'] / Gstd) * 
                          Ppico * (1 + kp * (tabla['Tc'] - Tr)) * eta * 1e-3)

# Aplicar límites a la potencia generada
tabla['Potencia [kW]'] = np.where(
    tabla['Potencia [kW]'] <= Pmin, 0,  # Si P <= Pmin -> 0
    np.where(tabla['Potencia [kW]'] <= Pinv, tabla['Potencia [kW]'], Pinv)
)

# Agrupar por mes y calcular la potencia total generada
tabla['Mes'] = tabla['Fecha'].dt.month  # Extraer el mes
potencia_total_mensual = tabla.groupby('Mes')['Potencia [kW]'].sum()

# Graficar la potencia total mensual
plt.figure(figsize=(10, 6))
potencia_total_mensual.plot(kind='bar', color='orange', edgecolor='black')
plt.title('Potencia Total Generada Mensualmente [kW]')
plt.xlabel('Mes')
plt.ylabel('Potencia Total [kW]')
plt.xticks(ticks=range(0, 12), labels=['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 
                                       'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic'], rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

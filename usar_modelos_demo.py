import pandas as pd
import pickle

print("=== DEMOSTRACION: USANDO EL MODELO GANADOR ===")

import os

# Obtener la ruta base donde se encuentra este script
base_dir = os.path.dirname(os.path.abspath(__file__))

# 1. Cargar datos de prueba (simulando un "nuevo envio" que acaba de llegar a la empresa)
# Tomamos el primer registro del dataset pero le borramos la respuesta real
datos_nuevos = pd.read_csv(os.path.join(base_dir, "data/05_model_input/model_input_clasificacion.csv"), encoding="latin-1").head(1)
respuesta_real = datos_nuevos.pop("tiene_incidencia").iloc[0]  # Quitamos la respuesta

# 2. Cargar el modelo final ganador que Kedro guardó
ruta_modelo = os.path.join(base_dir, "data/06_models/modelo_final_clasificacion.pkl")
with open(ruta_modelo, "rb") as f:
    modelo_ganador = pickle.load(f)

print("\nModelo cargado exitosamente:", type(modelo_ganador.named_steps['modelo']).__name__)

# 3. Mostrar los detalles del envio
print("\n--- DATOS DEL NUEVO ENVIO RECIBIDO ---")
# Mostrar algunas columnas importantes (ignorando columnas de ID si existen)
columnas_mostrar = [col for col in datos_nuevos.columns if 'id' not in col.lower()][:5] # Mostrar las primeras 5 columnas relevantes
for col in columnas_mostrar:
    print(f"- {col.replace('_', ' ').title()}: {datos_nuevos[col].iloc[0]}")
print("...")

# 4. Hacer la predicción
prediccion = modelo_ganador.predict(datos_nuevos)[0]
probabilidad = modelo_ganador.predict_proba(datos_nuevos)[0][1] * 100

print("\n--- RESULTADOS DE LA PREDICCION ---")
if prediccion == 1:
    print(f"[!] ALERTA: El modelo predice que este envio VA A TENER INCIDENCIA.")
    print(f"   - Probabilidad de incidencia: {probabilidad:.1f}%")
    print(f"   - Accion recomendada: Revisar ruta, contactar conductor, o tomar medidas preventivas.")
else:
    print(f"[OK] EXITO: El modelo predice que este envio llegara SIN PROBLEMAS.")
    print(f"   - Riesgo de incidencia: {probabilidad:.1f}%")
    print(f"   - Accion recomendada: Proceder con normalidad.")

print(f"\n--- VERIFICACION EN LA REALIDAD ---")
print(f"Lo que realmente paso con este envio fue: {'[!] Tuvo incidencia' if respuesta_real == 1 else '[OK] Todo salio bien'}")

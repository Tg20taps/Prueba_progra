# 🎤 GUION DE PRESENTACIÓN — Matías Retamal
**Materia:** SCY1101 — Programación para la Ciencia de Datos
**Duración:** 5 a 7 minutos

---

## ⚙️ ANTES DE EMPEZAR — Tener listo:
- ✅ Terminal con un `kedro run` ya ejecutado (que se vea el log exitoso)
- ✅ Jupyter Lab con los 6 notebooks abiertos en pestañas, **todos ya ejecutados** (con gráficos y tablas visibles)
- ✅ Orden de las pestañas: NB1 → NB2 → NB3 → NB4 → NB5 → NB6

---

## [00:00 – 01:00] FASE 1 — Arquitectura y Datos
📌 *Muestra la terminal con el log de kedro run. Luego cambia al Notebook 1 (`01_eda_exploratorio`) y scrollea lento por los gráficos.*

**Di esto:**
"Como ven en la terminal, el pipeline de Kedro procesó todo exitosamente. El proyecto tiene 8 pipelines encadenados que van desde la ingesta de datos hasta el entrenamiento de modelos.

En el Notebook 1, analicé los datos crudos. Encontré problemas serios: un 10% de nulos en columnas clave, distancias imposibles de más de 18.000 km, y fuga de información que contaminaría los modelos.

Lo resolví con imputación por mediana, recorte de outliers con IQR, y eliminé 4 columnas con data leakage. El resultado: 866 registros limpios para clasificación y 790 para regresión, ambos con cero nulos."

---

## [01:00 – 02:30] FASE 2 — Clustering (No Supervisado)
📌 *Cambia al Notebook 5 (`05_unsupervised_learning`). Muestra el gráfico del codo y el gráfico de los 5 clusters.*

**Di esto:**
"En aprendizaje no supervisado probé 3 técnicas: PCA, DBSCAN y KMeans.

PCA me permitió visualizar los datos en 2 dimensiones. DBSCAN falló porque en logística los datos son muy dispersos y los marcó casi todos como ruido.

La solución ganadora fue KMeans con K=5 clusters. El método del codo me indicó ese número óptimo. El resultado es útil para el negocio: el Cluster 4, con 344 envíos, concentra la tasa más alta de incidencias, el 18.3%, lo que lo convierte en el grupo prioritario de monitoreo."

---

## [02:30 – 04:00] FASE 3 — Clasificación (Supervisado)
📌 *Cambia al Notebook 2 (`02_supervised_modeling`). Muestra la tabla con el ranking de los 15 modelos de clasificación.*

**Di esto:**
"Para clasificación, el objetivo es predecir si un envío va a tener una incidencia. Entrené y comparé 15 algoritmos con validación cruzada StratifiedKFold de 5 folds.

Incluí modelos avanzados como Random Forest, Gradient Boosting y SVM. Sin embargo, el problema principal era el desbalanceo de clases: solo el 17% de los envíos tenían incidencia. Por eso, mi métrica principal fue el Recall, no el Accuracy.

El ganador fue GaussianNB. Tras optimizarlo con GridSearch, alcanzó un Recall del 96.6%, es decir, detecta 97 de cada 100 incidencias reales. Lo elegí porque en logística es preferible revisar un envío innecesariamente que dejar pasar uno que va a fallar."

---

## [04:00 – 05:30] FASE 4 — Regresión (Supervisado)
📌 *Cambia al Notebook 3 (`03_model_evaluation`). Muestra el ranking de los 15 modelos de regresión y las métricas del KNN.*

**Di esto:**
"Para regresión, el objetivo es estimar los días de tránsito de un envío. Igual que en clasificación, comparé 15 modelos con validación cruzada KFold de 5 folds.

El hallazgo honesto aquí es que los datos disponibles tienen un poder predictivo limitado: el mejor modelo mejora al baseline solo por 0.04 días. Esto se debe a que faltan variables externas como clima o tráfico.

Aún así, el ganador fue KNeighborsRegressor con K=9, con un MAE de 1.44 días. Esto significa que el modelo puede darle al cliente una ventana de entrega con un margen de día y medio de error, que es lo máximo que estos datos permiten."

---

## [05:30 – 06:30] FASE 5 — Optimización
📌 *Cambia al Notebook 4 (`04_hyperparameter_optimization`). Muestra los resultados de GridSearch y RandomizedSearch.*

**Di esto:**
"Apliqué los dos métodos de optimización requeridos.

RandomizedSearchCV hizo una exploración amplia con 12 iteraciones por modelo para encontrar la zona prometedora del espacio de hiperparámetros. Luego, GridSearchCV hizo la búsqueda fina dentro de esa zona.

En clasificación, el GaussianNB optimizado mejoró el Recall de 93% a 96.6%. En regresión, el KNN optimizado duplicó el R² de 0.062 a 0.115. Ambas optimizaciones tuvieron impacto real y medible."

---

## [06:30 – 07:00] CIERRE
📌 *Cambia al Notebook 6 (`06_final_analysis`) o muestra el informe PDF.*

**Di esto:**
"Para cerrar: partimos de datos con errores graves y construimos un pipeline reproducible en Kedro con 15 modelos de clasificación, 15 de regresión, y clustering con 3 técnicas.

Las recomendaciones concretas son: usar GaussianNB como sistema de alerta antes del despacho, KNN para estimar tiempos de entrega, y priorizar el monitoreo del Cluster 4.

El código está documentado y es 100% reproducible. Muchas gracias."

---

## 📋 RESUMEN RÁPIDO — Qué notebook abrir en cada fase

| Fase | Notebook | Qué mostrar |
|------|----------|-------------|
| FASE 1 | `01_eda_exploratorio` | Gráficos de nulos y outliers |
| FASE 2 | `05_unsupervised_learning` | Gráfico del codo + 5 clusters |
| FASE 3 | `02_supervised_modeling` | Tabla con 15 modelos de clasificación |
| FASE 4 | `03_model_evaluation` | Tabla con 15 modelos de regresión |
| FASE 5 | `04_hyperparameter_optimization` | Resultados GridSearch y RandomizedSearch |
| CIERRE | `06_final_analysis` | Resumen final |

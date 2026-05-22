# SCY1101 - Evaluacion Parcial 2

# README Maestro del Proyecto Kedro de Machine Learning

## Caso: Logistica y Transporte

Ultima actualizacion: 2026-05-11

---

## 1. Proposito

Este README es la fuente de verdad del proyecto `database` para la Evaluacion Parcial 2 de SCY1101.

Regla central:

> No entrenar modelos sobre datos mal preparados. Primero se audita, corrige y construye un dataset confiable para machine learning.

La historia de negocio se mantiene en todo el trabajo:

> Transformar datos logisticos con problemas reales de calidad en una solucion reproducible de machine learning para anticipar incidencias, estimar dias de transito y segmentar operaciones.

---

## 2. Contexto academico

- Asignatura: SCY1101 - Programacion para la Ciencia de Datos.
- Evaluacion: Parcial 2.
- Modalidad real de este proyecto: trabajo individual.
- Caso: Logistica y Transporte.
- Datasets base: `envios.csv`, `rutas.csv`, `vehiculos.csv`, `incidencias.csv`.

La evaluacion exige modelos supervisados, modelos no supervisados, validacion cruzada, multiples metricas, optimizacion con `GridSearchCV` y `RandomizedSearchCV`, documentacion, notebooks/informe y defensa individual.

---

## 3. Estado actual real

Ubicacion local:

```txt
C:\Users\Tg20taps\Desktop\Correcion prueba 2\database
```

El proyecto Kedro ejecuta correctamente.

Comando usado para evitar problemas de encoding en Windows:

```powershell
$env:PYTHONIOENCODING='utf-8'
$env:PYTHONUTF8='1'
kedro-env\Scripts\kedro.exe run
```

Resultado verificado:

```txt
Pipeline execution completed successfully in 29.6 sec.
Completed 26 out of 26 tasks.
```

Nota importante: `master_envios` sigue reportando errores de validacion porque conserva problemas reales del dataset maestro. Eso queda documentado como evidencia de calidad de datos. Los modelos NO se entrenan directamente sobre `master_envios`; se entrenan sobre datasets filtrados en `data/05_model_input`.

---

## 4. Pipelines existentes

Pipelines Kedro registrados por `find_pipelines()`:

```txt
data_ingestion
data_cleaning
data_transformation
data_validation
model_input
model_training
hyperparameter_tuning
unsupervised_learning
```

Flujo actual:

```txt
ingesta -> limpieza -> transformacion -> validacion -> model_input -> clustering -> entrenamiento -> tuning
```

---

## 5. Preparacion de datos ML

Se crearon datasets especificos de machine learning:

```txt
data/05_model_input/model_input_clasificacion.csv
data/05_model_input/model_input_regresion.csv
```

Estado validado:

```txt
model_input_clasificacion: 866 filas, 26 columnas
model_input_regresion:     790 filas, 26 columnas
nulos totales:             0
leakage presente:          0 columnas
rangos invalidos ML:       0
```

Se excluyeron 74 filas por rangos no aptos para ML en:

```txt
distancia_km
peso_kg
eficiencia_peso
```

Columnas excluidas por data leakage:

```txt
total_incidencias
costo_impacto_total
tipos_incidencia
costo_impacto_total_norm
```

Columnas post-evento excluidas para clasificacion:

```txt
fecha_entrega
dias_en_transito
```

---

## 6. Modelado supervisado

### Clasificacion

Pregunta de negocio:

> Podemos anticipar si un envio tendra una incidencia?

Target:

```txt
tiene_incidencia
```

Se entrenaron 15 candidatos con validacion cruzada. La metrica principal se corrigio a `f1` de la clase positiva, porque `f1_weighted` favorecia demasiado a la clase mayoritaria.

Top 3 por CV:

```txt
1. GaussianNB                 f1=0.2757 | recall=0.8418
2. LogisticRegression         f1=0.2532 | recall=0.4170
3. SGDClassifier              f1=0.2457 | recall=0.4310
```

Modelo final optimizado:

```txt
GaussianNB con GridSearchCV
f1=0.2963
recall=0.9655
precision=0.1750
roc_auc=0.5063
```

Interpretacion: el modelo detecta casi todas las incidencias, pero genera muchos falsos positivos. Sirve como herramienta inicial de screening de riesgo operativo, no como modelo final productivo sin mas datos o mejores features.

### Regresion

Pregunta de negocio:

> Podemos estimar los dias de transito de un envio?

Target:

```txt
dias_en_transito
```

Se entrenaron 15 candidatos con validacion cruzada. Metrica principal: `MAE`.

Top 3 por CV:

```txt
1. KNeighborsRegressor  MAE=1.4408
2. DummyRegressor       MAE=1.4827
3. Lasso                MAE=1.4895
```

Modelo final optimizado:

```txt
KNeighborsRegressor con GridSearchCV
MAE=1.4374 dias
RMSE=1.6201 dias
R2=0.1153
```

Interpretacion: el modelo mejora levemente el baseline en error absoluto, pero el poder explicativo sigue siendo bajo. Esto debe defenderse como hallazgo honesto: los datos actuales permiten una estimacion inicial, no una prediccion robusta de alta precision.

---

## 7. Optimizacion obligatoria

Ya se aplicaron ambos metodos exigidos:

```txt
RandomizedSearchCV
GridSearchCV
```

Salidas generadas:

```txt
data/07_model_output/resultados_randomizedsearch.csv
data/07_model_output/resultados_gridsearch.csv
data/07_model_output/comparacion_optimizacion.csv
data/07_model_output/metricas_modelo_final.csv
data/06_models/modelo_final_clasificacion.pkl
data/06_models/modelo_final_regresion.pkl
```

---

## 8. Aprendizaje no supervisado

Se implemento:

```txt
KMeans
DBSCAN
PCA
```

Mejor KMeans:

```txt
k=5
silhouette_score=0.0816
```

DBSCAN con la configuracion actual dejo todos los registros como ruido, por lo que se documenta como tecnica exploratoria no adecuada con los parametros actuales.

Salidas:

```txt
data/07_model_output/perfil_clusters.csv
data/07_model_output/metricas_clustering.csv
data/07_model_output/clusters_pca.csv
```

---

## 9. Archivos clave

Codigo:

```txt
src/database/pipelines/model_input/
src/database/pipelines/model_training/
src/database/pipelines/hyperparameter_tuning/
src/database/pipelines/unsupervised_learning/
```

Configuracion:

```txt
conf/base/catalog.yml
conf/base/parameters.yml
```

Documentacion de continuidad:

```txt
docs/AVANCE_EP2.md
```

---

## 10. Checklist de avance

### Kedro

- [x] `kedro run` ejecuta correctamente.
- [x] `catalog.yml` actualizado.
- [x] `parameters.yml` actualizado.
- [x] Pipelines registrados automaticamente.
- [x] Datasets de modelado creados.
- [x] Outputs principales guardados.
- [x] README maestro actualizado.

### Datos

- [x] Nulos criticos resueltos en datasets ML.
- [x] Rangos invalidos filtrados en datasets ML.
- [x] Tipos y variables revisadas para modelado.
- [x] Leakage eliminado.
- [x] Dataset de clasificacion creado.
- [x] Dataset de regresion creado.
- [x] Documentar la integridad referencial como limitacion en informe tecnico.

### Modelos

- [x] 15 candidatos de clasificacion entrenados.
- [x] 15 candidatos de regresion entrenados.
- [x] Baseline incluido.
- [x] Validacion cruzada aplicada.
- [x] Metricas calculadas.
- [x] Ranking generado.
- [x] Finalistas seleccionados.
- [x] RandomizedSearchCV aplicado.
- [x] GridSearchCV aplicado.
- [x] Modelos finales guardados.

### No supervisado

- [x] KMeans aplicado.
- [x] DBSCAN aplicado.
- [x] PCA aplicado.
- [x] Metricas de clustering calculadas.
- [x] Perfil de clusters generado.
- [x] Mejorar visualizaciones e interpretacion en informe tecnico.

### Documentacion y presentacion

- [x] Pulir notebooks.
- [x] Crear notebooks separados para EDA, modelado, evaluacion, optimizacion, clustering y analisis final.
- [x] Reforzar storytelling, explicaciones de graficos y ejemplos de negocio en notebooks.
- [x] Crear informe tecnico en docs/INFORME_TECNICO.md.
- [ ] Preparar presentacion individual.
- [x] Redactar conclusiones y recomendaciones finales en informe tecnico.

---

## 11. Proximos pasos recomendados

1. Revisar visualmente los notebooks ejecutados en JupyterLab.
2. Convertir las salidas CSV y graficos de notebooks en informe tecnico.
3. Preparar presentacion individual con el guion de `06_final_analysis.ipynb`.
4. Explicar claramente que el dataset maestro tiene errores, pero los datasets ML fueron filtrados.
5. Defender el resultado de clasificacion como detector sensible de riesgo, con muchos falsos positivos.
6. Defender el resultado de regresion como modelo inicial que apenas supera baseline y requiere mejores variables.

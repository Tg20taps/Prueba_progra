# AVANCE EP2 - Continuidad para otra IA

Ultima actualizacion: 2026-05-11

## Resumen corto

El proyecto Kedro `database` ya corre completo con 26 nodos. Se implementaron y validaron las fases tecnicas principales: `model_input`, `model_training`, `hyperparameter_tuning` y `unsupervised_learning`.

Comando validado en Windows:

```powershell
$env:PYTHONIOENCODING='utf-8'
$env:PYTHONUTF8='1'
kedro-env\Scripts\kedro.exe run
```

Resultado:

```txt
Pipeline execution completed successfully in 29.6 sec.
Completed 26 out of 26 tasks.
```

## Cambios realizados

- Se reemplazo el README generico de Kedro por un README maestro del proyecto.
- Se corrigio `model_training`:
  - `AdaBoostClassifier` ya no usa `algorithm`, porque la version instalada de scikit-learn no lo acepta.
  - El ranking de clasificacion ahora usa `f1` de la clase positiva, no `f1_weighted`.
  - `QuadraticDiscriminantAnalysis` usa `reg_param=0.01` para evitar problemas de matriz singular.
- Se reforzo `model_input`:
  - Se filtran registros fuera de rango antes del entrenamiento.
  - Se documentan 74 filas excluidas por rangos no aptos para ML.
- Se agrego pipeline `hyperparameter_tuning`:
  - Selecciona finalistas desde los rankings.
  - Ejecuta `RandomizedSearchCV`.
  - Ejecuta `GridSearchCV`.
  - Guarda comparacion y modelos finales.
- Se agrego pipeline `unsupervised_learning`:
  - Ejecuta KMeans.
  - Ejecuta DBSCAN.
  - Ejecuta PCA.
  - Guarda perfil de clusters, metricas y coordenadas PCA.
- Se crearon y ejecutaron notebooks profesionales de entrega:
  - EDA exploratorio.
  - Modelado supervisado.
  - Evaluacion de modelos.
  - Optimizacion de hiperparametros.
  - Aprendizaje no supervisado.
  - Analisis final y defensa.
- Se reforzo el storytelling de los notebooks:
  - Se agregaron historias de negocio al inicio de cada notebook.
  - Se agregaron secciones "Como leer este grafico".
  - Se cambiaron etiquetas genericas por nombres mas claros.
  - Se agregaron ejemplos concretos para defensa oral.
  - Se explicaron falsos positivos, falsos negativos, MAE en dias, baseline y clusters.

## Archivos creados o modificados

```txt
README.md
docs/AVANCE_EP2.md
conf/base/catalog.yml
conf/base/parameters.yml
src/database/pipelines/model_input/nodes.py
src/database/pipelines/model_training/nodes.py
src/database/pipelines/hyperparameter_tuning/__init__.py
src/database/pipelines/hyperparameter_tuning/pipeline.py
src/database/pipelines/hyperparameter_tuning/nodes.py
src/database/pipelines/unsupervised_learning/__init__.py
src/database/pipelines/unsupervised_learning/pipeline.py
src/database/pipelines/unsupervised_learning/nodes.py
notebooks/01_eda_exploratorio.ipynb
notebooks/02_supervised_modeling.ipynb
notebooks/03_model_evaluation.ipynb
notebooks/04_hyperparameter_optimization.ipynb
notebooks/05_unsupervised_learning.ipynb
notebooks/06_final_analysis.ipynb
```

## Estado de datasets ML

Validacion directa posterior al ultimo `kedro run`:

```txt
model_input_clasificacion: 866 filas, 26 columnas
model_input_regresion:     790 filas, 26 columnas
nulos totales:             0
leakage presente:          0 columnas
rangos invalidos:          0
```

Rangos auditados:

```txt
peso_kg:           13.6 a 18466.0
distancia_km:      29.6 a 1903.4
eficiencia_peso:   0.00421 a 9.9362
dias_en_transito:  4.0 a 10.0
```

## Salidas generadas

Datasets ML:

```txt
data/05_model_input/model_input_clasificacion.csv
data/05_model_input/model_input_regresion.csv
```

Modelado y optimizacion:

```txt
data/07_model_output/ranking_modelos_clasificacion.csv
data/07_model_output/ranking_modelos_regresion.csv
data/07_model_output/resultados_randomizedsearch.csv
data/07_model_output/resultados_gridsearch.csv
data/07_model_output/comparacion_optimizacion.csv
data/07_model_output/metricas_modelo_final.csv
data/06_models/modelo_final_clasificacion.pkl
data/06_models/modelo_final_regresion.pkl
```

No supervisado:

```txt
data/07_model_output/perfil_clusters.csv
data/07_model_output/metricas_clustering.csv
data/07_model_output/clusters_pca.csv
```

## Resultados actuales

Clasificacion:

```txt
Modelo final: GaussianNB
Optimizacion: GridSearchCV
f1:           0.2963
recall:       0.9655
precision:    0.1750
roc_auc:      0.5063
```

Interpretacion para defensa: alta sensibilidad para detectar riesgos, pero muchos falsos positivos. Se recomienda presentarlo como screening inicial de riesgo operativo.

Regresion:

```txt
Modelo final: KNeighborsRegressor
Optimizacion: GridSearchCV
MAE:          1.4374 dias
RMSE:         1.6201 dias
R2:           0.1153
```

Interpretacion para defensa: mejora levemente el baseline, pero el poder explicativo sigue bajo. Es un resultado honesto y permite recomendar mejores variables operativas.

Clustering:

```txt
Mejor KMeans: k=5
silhouette:   0.0816
DBSCAN:       todos los registros quedaron como ruido con eps=2.5
PCA 2D:       varianza explicada aproximada 0.2008
```

Interpretacion para defensa: hay segmentos exploratorios, pero la separacion no es fuerte. Usar como analisis descriptivo, no como segmentacion definitiva.

## Pendientes importantes

- Revisar visualmente los notebooks en JupyterLab antes de entregar, por formato final.
- Preparar informe tecnico con tablas desde `data/07_model_output`.
- Preparar presentacion individual.
- Explicar que `master_envios` conserva 12 errores de validacion, pero que los datasets ML fueron corregidos/filtrados.
- Documentar integridad referencial como limitacion: rutas y vehiculos invalidos existen en el maestro original.

## Notebooks ejecutados

Validacion posterior a ejecucion:

```txt
01_eda_exploratorio.ipynb            21 celdas | 8 codigo | 7 con outputs
02_supervised_modeling.ipynb         15 celdas | 6 codigo | 5 con outputs
03_model_evaluation.ipynb            14 celdas | 7 codigo | 5 con outputs
04_hyperparameter_optimization.ipynb 13 celdas | 5 codigo | 4 con outputs
05_unsupervised_learning.ipynb       14 celdas | 6 codigo | 5 con outputs
06_final_analysis.ipynb              16 celdas | 7 codigo | 6 con outputs
```

Mejora posterior de storytelling:

```txt
01_eda_exploratorio.ipynb            24 celdas | 16 markdown | 8 codigo
02_supervised_modeling.ipynb         19 celdas | 13 markdown | 6 codigo
03_model_evaluation.ipynb            17 celdas | 10 markdown | 7 codigo
04_hyperparameter_optimization.ipynb 16 celdas | 11 markdown | 5 codigo
05_unsupervised_learning.ipynb       17 celdas | 11 markdown | 6 codigo
06_final_analysis.ipynb              19 celdas | 12 markdown | 7 codigo
```

## Mensaje para continuar

Si otra IA retoma: no rehacer desde cero. Leer primero:

```txt
README.md
docs/AVANCE_EP2.md
data/08_reporting/model_input_report.csv
data/07_model_output/metricas_modelo_final.csv
data/07_model_output/perfil_clusters.csv
```

Luego continuar con Fase 2 y Fase 7: notebooks, informe, visualizaciones, conclusiones y presentacion.

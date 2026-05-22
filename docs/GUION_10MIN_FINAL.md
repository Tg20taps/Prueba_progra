# Guion de Presentación Visual (Duración: 5 - 7 minutos)
**Materia:** SCY1101 — Programación para la Ciencia de Datos
**Estudiante:** Matías Retamal

*Este guion está diseñado para que hables MIENTRAS cambias de pestaña mostrando tus Notebooks ya ejecutados. Es dinámico y directo al grano.*

### 0. PREPARACIÓN ANTES DE PRESENTAR (Lo que debes tener abierto)
1. **VS Code / Jupyter:** Debes tener los 6 Notebooks abiertos en pestañas. **TODOS deben estar ejecutados previamente** (todas las celdas deben mostrar los gráficos y tablas ya cargadas).
2. **Terminal:** Tener una terminal abierta que muestre el final de un `kedro run` exitoso, para probar que el código completo corre.

---

### [00:00 - 01:00] Fase 1: Arquitectura y Datos Crudos (Notebook 1)
*(Acción visual: Muestra rápidamente tu terminal con el log de Kedro, luego cambia a la pestaña del **Notebook 1** y scrollea por los gráficos de nulos/outliers).*

> "Hola, el profesor me indicó que debo tener todo ejecutado. Como ven aquí en mi terminal, el pipeline de Kedro ya procesó toda la ingesta y el entrenamiento de los modelos exitosamente. 
> 
> Todo este trabajo experimental lo consolidé en una secuencia del Notebook 1 al 6. Si miramos el **Notebook 1**, al analizar los datos crudos, detecté problemas severos: un 10% de nulos en variables clave, distancias imposibles de más de 18,000 km, y fuga de información (data leakage). 
> Resolví esto aplicando imputación por mediana, recorte de outliers con el método IQR y eliminando las variables contaminadas, lo que me garantizó un dataset 100% limpio para modelar."

### [01:00 - 03:00] Fase 2: Modelado No Supervisado (Notebook 2 y 3)
*(Acción visual: Cambia al **Notebook de Clustering**. Muestra el gráfico de codo/silhouette, y luego el dataframe o gráfico con los 5 clusters).*

> "Pasando a la fase No Supervisada, probamos 3 algoritmos para entender nuestros datos operativos: PCA, DBSCAN y KMeans.
> 
> Primero usamos **PCA (Análisis de Componentes Principales)** para intentar aplastar nuestras 25 variables en solo 2 dimensiones y graficarlas. Luego probamos **DBSCAN**, que agrupa datos por densidad (como pintando con un spray zonas muy juntas). Sin embargo, falló porque en logística los datos son muy dispersos y marcó casi todo como ruido.
> 
> La solución ganadora fue **KMeans**. Le pedimos al algoritmo agrupar los datos por similitud. El método del codo nos llevó a usar **K=5 clusters**. Aunque matemáticamente se solapan un poco, a nivel de negocio es súper útil: logró aislar en el 'Cluster 4' a los envíos de larga distancia que concentran la mayor tasa histórica de incidencias, dándonos un foco claro de dónde intervenir."

### [02:30 - 04:30] Fase 3: Clasificación y los Top 5 Modelos (Notebook 4)
*(Acción visual: Cambia al **Notebook de Clasificación**. Muestra la celda donde está el ranking de modelos o la matriz de confusión del ganador).*

> "Ahora entramos al aprendizaje Supervisado, empezando por Clasificación para predecir si un envío tendrá incidencias. 
> Como ven en la tabla del Notebook, evalué rigurosamente **15 algoritmos distintos** con validación cruzada para asegurar una evaluación exhaustiva. Aquí en pantalla les muestro los resultados ordenados.
> 
> *(Acción visual: Detente en la tabla donde se ven los 15 algoritmos. Apunta con el mouse a modelos como Random Forest o SVM)*
> 
> Para que vean la profundidad técnica, evaluamos algoritmos avanzados como **Random Forest** (que funciona creando 100 árboles de decisión que votan para evitar sobreajuste), **Gradient Boosting** (que aprende secuencialmente de sus propios errores iteración tras iteración) y **Support Vector Machines (SVM)** (que eleva los datos a múltiples dimensiones para encontrar la frontera matemática perfecta de separación).
> 
> Sin embargo, el problema principal era el desbalanceo: casi no había incidencias registradas. Por eso, mi métrica clave no fue el Accuracy, sino el **Recall**. 
> 
> Mi modelo ganador fue **GaussianNB (Naïve Bayes)**. Como pueden ver en sus métricas tras optimizarlo, logró un **Recall del 96.6%**. Decidí elegirlo porque en nuestro negocio es preferible revisar un camión por falsa alarma, que dejar pasar un camión que realmente va a quedar varado. Este modelo funciona como un radar preventivo excelente."

### [04:30 - 06:30] Fase 4: Regresión y los Top 5 Modelos (Notebook 5)
*(Acción visual: Cambia al **Notebook de Regresión**. Muestra el ranking de los 5 modelos de regresión y las métricas de KNN).*

> "Finalmente, apliqué Regresión para estimar los 'Días en tránsito'. Al igual que en clasificación, entrené y comparé **15 modelos de regresión** distintos. Aquí en el Notebook pueden ver la tabla completa con el ranking de resultados.
> 
> La problemática aquí fue que ningún modelo superaba con gran margen al promedio básico, porque nos falta data externa como clima o tráfico. 
> 
> Aún así, tras la optimización de hiperparámetros, el ganador fue **KNeighborsRegressor (KNN)** con K=9. Lo elegí porque entregó el error más bajo constante, con un **MAE de 1.44 días**. Esto significa que podemos darle al cliente una ventana de entrega de un día y medio de precisión, que es el máximo potencial que estos datos pueden darnos."

### [06:30 - 07:00] Fase 5: Conclusión (Notebook 6)
*(Acción visual: Cambia al **Notebook 6** o muestra el PDF generado).*

> "Para cerrar, en el Notebook 6 consolido cómo pasamos de un dataset desastroso a un pipeline funcional con métricas claras. Elegimos KMeans para segmentar, GaussianNB para predecir riesgos y KNN para estimar tiempos, todo respaldado por la validación cruzada.
> 
> El código está 100% probado en Kedro y documentado. Muchas gracias y quedo atento a sus dudas."

# Guion de Presentación Individual — EP2 Logística y Transporte

**Duración estimada:** 8–10 minutos
**Formato:** Video grabado con apoyo de diapositivas
**Tono:** Profesional, técnico pero accesible, con ejemplos de negocio concretos

---

## Estructura General (10 diapositivas)

```
1. Portada  (30 seg)
2. El problema de negocio  (1 min)
3. Arquitectura Kedro  (1 min)
4. Calidad de datos: el reto real  (1.5 min)
5. Modelado: Clasificación  (1.5 min)
6. Modelado: Regresión  (1 min)
7. Optimización  (1 min)
8. Clustering  (1 min)
9. Conclusiones y recomendaciones  (1 min)
10. Cierre  (30 seg)
```

---

## Diapositiva 1 — Portada (30 seg)

**Texto en slide:**
```
Pipeline de Machine Learning para Logística y Transporte
SCY1101 — Programación para la Ciencia de Datos
Evaluación Parcial 2
[Tu Nombre]
Mayo 2026
```

**Guion:**
> "Hola, mi nombre es [Tu Nombre] y les presento mi trabajo de la Evaluación Parcial 2 de SCY1101. El proyecto consiste en un pipeline completo de machine learning sobre datos reales de logística y transporte, construido con el framework Kedro."

---

## Diapositiva 2 — El Problema de Negocio (1 min)

**Texto en slide:**
```
3 problemas, 3 preguntas:

🔴 Clasificación  →  ¿Podemos anticipar si un envío tendrá una incidencia?
📊 Regresión      →  ¿Podemos estimar los días de tránsito?
🔵 Segmentación   →  ¿Existen patrones operativos que agrupen envíos similares?
```

**Guion:**
> "El proyecto aborda tres problemas. Primero, clasificación: queremos anticipar si un envío va a tener una incidencia, antes de que ocurra. Segundo, regresión: estimar cuántos días va a tardar un envío en llegar. Y tercero, segmentación no supervisada: identificar patrones operativos en los envíos.
>
> Los datos de entrada son 4 tablas: envíos, rutas, vehículos e incidencias. En total, 1030 registros de envíos con información de rutas, pesos, tiempos, y 206 incidencias registradas."

---

## Diapositiva 3 — Arquitectura Kedro (1 min)

**Texto en slide:**
```
8 pipelines | 26 nodos | ~30 segundos de ejecución

Ingesta → Limpieza → Transformación → Validación
  → Dataset ML → Clustering → Entrenamiento → Tuning

Framework: Kedro 1.3.1
Reproducibilidad garantizada
Capas de datos separadas (raw → intermediate → primary → model_input)
```

**Guion:**
> "El proyecto está construido sobre Kedro 1.3.1, un framework que organiza pipelines de datos de manera reproducible. Tiene 8 pipelines encadenados que suman 26 nodos de procesamiento, y se ejecuta completo en aproximadamente 30 segundos.
>
> La regla central del proyecto es: no entrenar modelos sobre datos mal preparados. Por eso, antes de llegar a machine learning, los datos pasan por ingesta, limpieza, transformación y validación."

---

## Diapositiva 4 — Calidad de Datos: El Reto Real (1.5 min)

**Texto en slide:**
```
Problemas encontrados en datos raw:
┌──────────────────────┬────────────┐
│ Nulos col. críticas  │ 4.5–10.1%  │
│ Fuera de rango       │ 7.8–29.7%  │
│ Integridad referencial│ 28 rutas   │
│                      │ 77 vehíc.  │
│ Leakage eliminado    │ 4 columnas │
└──────────────────────┴────────────┘

Resultado:
✅ 866 filas para clasificación | 0 nulos | 0 leakage
✅ 790 filas para regresión     | 0 nulos | 0 leakage
```

**Guion:**
> "Esta es probablemente la parte más importante del proyecto. Los datos originales tenían problemas serios: entre un 5 y un 10 por ciento de nulos en columnas críticas, valores fuera de rango como distancias de más de 18 mil kilómetros, y problemas de integridad referencial donde rutas y vehículos no existían en sus tablas maestras.
>
> Se aplicaron técnicas de imputación por moda y mediana, filtrado IQR con un multiplicador conservador de 2.0, y se eliminaron 4 columnas con data leakage, como por ejemplo 'total_incidencias' que contiene información derivada del evento que queremos predecir.
>
> Al final, obtuvimos dos datasets limpios: 866 envíos para clasificación y 790 para regresión, ambos con cero nulos y cero leakage."

---

## Diapositiva 5 — Clasificación (1.5 min)

**Texto en slide:**
```
Target: ¿El envío tuvo incidencia? (Sí/No)
15 candidatos con validación cruzada (StratifiedKFold, 5 folds)

🏆 TOP 3 del ranking CV:
┌──────────────────────┬───────┬─────────┐
│ Modelo               │ F1    │ Recall  │
├──────────────────────┼───────┼─────────┤
│🥇 GaussianNB         │ 0.276 │ 0.842   │
│🥈 LogisticRegression │ 0.253 │ 0.417   │
│🥉 SGDClassifier      │ 0.246 │ 0.431   │
└──────────────────────┴───────┴─────────┘

Modelo final optimizado: GaussianNB
Recall = 96.6% │ F1 = 0.296 │ Precisión = 17.5%
```

**Guion:**
> "Para clasificación, el target es si un envío tuvo o no una incidencia. Entrenamos 15 modelos candidatos con validación cruzada de 5 folds. La métrica principal es F1 de la clase positiva, porque nos interesa detectar incidencias aunque la clase sea minoritaria —solo el 17% de los envíos tienen incidencia.
>
> El ganador fue GaussianNB, un clasificador Bayesiano ingenuo. ¿Por qué ganó un modelo tan simple? Porque prioriza la sensibilidad: detecta el 84% de las incidencias en validación cruzada.
>
> Después de optimizar con GridSearchCV, el modelo final alcanzó un recall del 96.6% —es decir, detecta 97 de cada 100 incidencias reales. La contraparte: solo el 17.5% de las alertas que genera son correctas, el resto son falsos positivos.
>
> Esto lo convierte en una herramienta de screening de riesgo operativo: nos ayuda a filtrar qué envíos revisar, pero no para tomar decisiones automáticas."

---

## Diapositiva 6 — Regresión (1 min)

**Texto en slide:**
```
Target: Días en tránsito
15 candidatos con validación cruzada (KFold, 5 folds)

🏆 TOP 3 del ranking CV:
┌──────────────────────┬──────┬───────┐
│ Modelo               │ MAE  │ R²    │
├──────────────────────┼──────┼───────┤
│🥇 KNeighborsRegr.    │ 1.44 │ 0.080 │
│🥈 DummyRegressor     │ 1.48 │ -0.02 │ ← baseline
│🥉 Lasso              │ 1.49 │ 0.206 │
└──────────────────────┴──────┴───────┘

Modelo final: KNeighborsRegressor (k=9, weights=uniform)
MAE = 1.44 días │ R² = 0.115
```

**Guion:**
> "Para regresión, el target son los días de tránsito. También 15 modelos candidatos. El ganador fue KNeighborsRegressor con un MAE de 1.44 días.
>
> Es importante notar que el baseline —que siempre predice el promedio— tiene un MAE de 1.48 días. Nuestro modelo mejora el baseline solo por 0.04 días, aproximadamente una hora. El R² es de 0.115, lo que significa que las variables actuales explican solo el 11.5% de la variabilidad.
>
> Esto es un hallazgo honesto: los datos disponibles permiten una estimación inicial, pero factores como clima, tráfico o tipo de conductor no están capturados. El modelo sirve como referencia, no como predicción de alta precisión."

---

## Diapositiva 7 — Optimización (1 min)

**Texto en slide:**
```
RandomizedSearchCV (exploración amplia)
  → 12 iteraciones por modelo
GridSearchCV (búsqueda fina)
  → Todas las combinaciones de la cuadrícula

Clasificación:
  Mejores params: GaussianNB → var_smoothing = 1e-11
  Recall mejoró de 93.1% → 96.6%

Regresión:
  Mejores params: KNN → n_neighbors=9, p=2, weights='uniform'
  R² mejoró de 0.062 → 0.115
```

**Guion:**
> "Aplicamos los dos métodos de optimización exigidos: RandomizedSearchCV para exploración amplia con 12 iteraciones por modelo, y GridSearchCV para la búsqueda fina. En clasificación, el mejor GaussianNB usa un var_smoothing de 1e-11, y mejoramos el recall de 93 a 96 por ciento. En regresión, el KNeighbors optimizado con 9 vecinos y peso uniforme duplicó el R² de 0.062 a 0.115."

---

## Diapositiva 8 — Clustering (1 min)

**Texto en slide:**
```
3 técnicas aplicadas:
  KMeans (k=3,4,5) → Mejor: k=5 | Silhouette=0.0816
  DBSCAN → Todos los registros como ruido
  PCA → 2 componentes | 20.1% varianza explicada

Perfil de los 5 clusters:
┌───┬────────┬──────────┬──────────┐
│ C │ n Env. │ Incid.   │ Dist. km │
├───┼────────┼──────────┼──────────┤
│ 0 │   30   │  16.7%   │    30    │ ← rutas cortas
│ 1 │   63   │  15.9%   │  1,027   │ ← larga dist.
│ 2 │  143   │  17.5%   │   947    │ ← baja efic.
│ 3 │  286   │  15.0%   │  1,041   │ ← estándar
│ 4 │  344   │  18.3%   │   954    │ ← mayor riesgo
└───┴────────┴──────────┴──────────┘
```

**Guion:**
> "En aprendizaje no supervisado aplicamos KMeans con 3, 4 y 5 clusters, DBSCAN y PCA. El mejor KMeans fue con k=5, aunque el silhouette score de 0.08 indica una separación débil —es normal en datos logísticos reales donde las fronteras son difusas.
>
> Los 5 clusters tienen sentido de negocio. Por ejemplo, el cluster 0 son rutas muy cortas de solo 30 km promedio, mientras el cluster 4 es el más grande con 344 envíos y la tasa de incidencia más alta, 18.3%. Esto permite priorizar intervenciones.
>
> DBSCAN clasificó todos los registros como ruido, lo que documentamos como hallazgo exploratorio: con estos parámetros, la densidad no forma clusters separables. PCA retiene solo el 20% de la varianza en 2 componentes."

---

## Diapositiva 9 — Conclusiones y Recomendaciones (1 min)

**Texto en slide:**
```
✅ Pipeline Kedro completo y ejecutable
✅ 15 modelos de clasificación + 15 de regresión
✅ RandomizedSearchCV + GridSearchCV aplicados
✅ KMeans, DBSCAN, PCA ejecutados
✅ Informe técnico generado

Recomendaciones:
  • Clasificador como screening pre-despacho
  • Regresión como referencia, no predicción exacta
  • Cluster 4 → prioridad de monitoreo
  • Mejora futura: incorporar datos climáticos y de tráfico
```

**Guion:**
> "En resumen, el proyecto cumple con todos los objetivos: pipeline Kedro funcional, 15 modelos de clasificación y 15 de regresión con validación cruzada, optimización con RandomizedSearchCV y GridSearchCV, clustering completo, informe técnico generado.
>
> La recomendación principal: usar el clasificador como sistema de alerta temprana antes del despacho. Los envíos marcados como riesgosos pasan a revisión manual. Para regresión, usar la estimación de 1.44 días de margen como ventana de entrega, no como valor exacto.
>
> A futuro, incorporar variables climáticas y de tráfico mejoraría significativamente ambos modelos."

---

## Diapositiva 10 — Cierre (30 seg)

**Texto en slide:**
```
"Transformar datos logísticos con problemas reales
de calidad en una solución reproducible de machine
learning"

📧 [Tu Correo]
🔗 [Tu LinkedIn/GitHub — opcional]

¡Gracias!
```

**Guion:**
> "Para cerrar, cito la historia de negocio que guió este trabajo: transformar datos logísticos con problemas reales de calidad en una solución reproducible de machine learning. Eso es exactamente lo que logramos: partimos de datos con errores y llegamos a modelos funcionales, documentados y listos para usar.
>
> Muchas gracias por su atención. Estoy abierto a preguntas."

---

## Tips para la Grabación del Video

1. **Fondo y luz:** Busca un fondo limpio y buena iluminación frontal.
2. **Ritmo:** Habla pausado (~130 palabras/minuto). El guion total son ~1200 palabras → ~9 minutos.
3. **Pantalla:** Si grabas con slides, asegúrate de que el texto se vea nítido. Usa fuente ≥24pt.
4. **Transiciones entre slides:** Haz pausas de 1-2 segundos al cambiar de tema.
5. **Confianza:** Si te trabas, repite la frase. Puedes editar después.
6. **Demo opcional:** Si quieres impresionar, muestra un `kedro run` ejecutándose en terminal (~5 segundos).

---

## Cómo Crear las Diapositivas Rápidamente

| Herramienta | Cómo hacerlo |
|-------------|--------------|
| **PowerPoint** | Copia el contenido de cada diapositiva de este guion a slides nuevas |
| **Google Slides** | Entra a slides.google.com → plantilla en blanco → pega contenido |
| **Canva** | Busca plantilla "presentación profesional" → reemplaza texto |
| **Markdown → PDF** | Usa `pandoc -t beamer` si tienes experiencia con LaTeX |

**Recomendación:** PowerPoint o Google Slides. Son las más rápidas para este formato de 10 slides.

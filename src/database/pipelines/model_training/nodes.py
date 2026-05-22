"""
Nodos del pipeline de Model Training (Fase 3 — EP2 SCY1101).

Responsabilidad:
    - Entrenar 15 modelos candidatos de clasificacion (target: tiene_incidencia).
    - Entrenar 15 modelos candidatos de regresion (target: dias_en_transito).
    - Evaluar cada modelo con validacion cruzada (cv=5).
    - Calcular multiples metricas por modelo.
    - Generar rankings comparativos ordenados por metrica principal.
    - Guardar resultados en data/07_model_output/.

Estrategia de 15 modelos (README maestro seccion 11-13):
    - Se incluye baseline (Dummy) para verificar que los modelos reales superan
      una referencia minima.
    - La metrica principal para clasificacion es f1 de la clase positiva.
    - La metrica principal para regresion es MAE (interpretable en dias).
    - No se optimizan hiperparametros aqui (eso es Fase 5: GridSearchCV/RandomizedSearchCV).

Frase guia del proyecto:
    'No elegimos el modelo por intuicion. Probamos 15 candidatos, comparamos
    metricas mediante validacion cruzada, descartamos los menos adecuados,
    optimizamos los finalistas y seleccionamos el modelo que mejor responde
    al problema logistico.'
"""

from __future__ import annotations

import logging
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    AdaBoostClassifier,
    AdaBoostRegressor,
    BaggingClassifier,
    BaggingRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import (
    ElasticNet,
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
    SGDClassifier,
    SGDRegressor,
)
from sklearn.model_selection import cross_validate, StratifiedKFold, KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Definicion de features (sin target, sin id_envio como predictor)
# ---------------------------------------------------------------------------

FEATURES_NUMERICAS = [
    "peso_kg", "volumen_m3", "distancia_km", "tiempo_estimado_hrs",
    "peaje_total", "capacidad_kg", "capacidad_m3", "km_recorridos",
    "eficiencia_peso", "eficiencia_volumen", "costo_por_km",
    "id_ruta", "id_vehiculo",
    "mes_envio", "dia_semana_envio", "trimestre_envio",
]

FEATURES_CATEGORICAS = [
    "estado", "tipo_carga_norm", "origen", "destino",
    "tipo_via", "tipo", "estado_vehiculo",
]

TARGET_CLF = "tiene_incidencia"
TARGET_REG = "dias_en_transito"


# ---------------------------------------------------------------------------
# Helpers privados
# ---------------------------------------------------------------------------

def _construir_preprocesador(
    features_num: list[str],
    features_cat: list[str],
) -> ColumnTransformer:
    """
    Construye el preprocesador ColumnTransformer para el pipeline de sklearn.

    Transformaciones aplicadas:
        - Numericas: SimpleImputer(median) + StandardScaler
        - Categoricas: SimpleImputer(mode) + OneHotEncoder(handle_unknown='ignore')

    StandardScaler es necesario para modelos sensibles a escala:
        LogisticRegression, SVC, KNN, SGD.
    Para modelos basados en arboles, StandardScaler no afecta el resultado
    pero tampoco lo perjudica.
    """
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, features_num),
            ("cat", categorical_pipe, features_cat),
        ],
        remainder="drop",
    )


def _get_features_presentes(df: pd.DataFrame, cols: list[str]) -> list[str]:
    """Retorna solo las features que existen en el DataFrame."""
    return [c for c in cols if c in df.columns]


def _entrenar_modelo_con_cv(
    nombre: str,
    modelo,
    X: pd.DataFrame,
    y: pd.Series,
    cv,
    scoring: dict,
) -> dict:
    """
    Entrena un modelo con validacion cruzada y retorna sus metricas.

    Usa cross_validate() para multiples metricas en un solo paso.
    Mide el tiempo de entrenamiento para comparacion de eficiencia.
    """
    t_inicio = time.time()
    try:
        resultados = cross_validate(
            modelo, X, y,
            cv=cv,
            scoring=scoring,
            return_train_score=False,
            n_jobs=1,
            error_score="raise",
        )
        t_fin = time.time()
        fila = {"modelo": nombre, "error": None, "tiempo_seg": round(t_fin - t_inicio, 2)}
        for metrica, valores in resultados.items():
            if metrica.startswith("test_"):
                nombre_metrica = metrica.replace("test_", "")
                fila[nombre_metrica] = round(float(np.mean(valores)), 4)
                fila[f"{nombre_metrica}_std"] = round(float(np.std(valores)), 4)
        logger.info("[model_training] %-35s | %.2f seg", nombre, fila["tiempo_seg"])
    except Exception as exc:
        t_fin = time.time()
        fila = {
            "modelo": nombre,
            "error": str(exc)[:120],
            "tiempo_seg": round(t_fin - t_inicio, 2),
        }
        logger.warning("[model_training] %-35s | ERROR: %s", nombre, str(exc)[:80])
    return fila


# ---------------------------------------------------------------------------
# Nodos publicos
# ---------------------------------------------------------------------------

def entrenar_candidatos_clasificacion(
    clf_data: pd.DataFrame,
    parameters: dict,
) -> pd.DataFrame:
    """
    Nodo 1: Entrena 15 modelos candidatos de clasificacion binaria.

    Target: tiene_incidencia (0=sin incidencia, 1=con incidencia)
    Metrica principal: f1 de la clase positiva (incidencia = 1)
    Validacion: StratifiedKFold(n_splits=5, shuffle=True)

    Los 15 modelos incluyen un baseline (DummyClassifier) para verificar
    que los modelos reales superan una referencia minima trivial.
    Esto es practica estandar en ML serio.

    Parametros
    ----------
    clf_data   : Dataset de clasificacion (05_model_input/model_input_clasificacion.csv).
    parameters : Parametros del proyecto.

    Retorna
    -------
    pd.DataFrame ranking guardado en 07_model_output/ranking_modelos_clasificacion.csv.
    """
    random_state = parameters.get("model_input", {}).get("random_state", RANDOM_STATE)

    # Preparar X, y
    features_num = _get_features_presentes(clf_data, FEATURES_NUMERICAS)
    features_cat = _get_features_presentes(clf_data, FEATURES_CATEGORICAS)
    X = clf_data[features_num + features_cat]
    y = clf_data[TARGET_CLF]

    logger.info(
        "[model_training] Clasificacion — X: %s | y distribucion: %s",
        X.shape, dict(y.value_counts()),
    )

    # Preprocesador
    preprocesador = _construir_preprocesador(features_num, features_cat)

    # 15 modelos candidatos (README maestro seccion 11)
    candidatos = [
        ("01_DummyClassifier",          DummyClassifier(strategy="most_frequent", random_state=random_state)),
        ("02_LogisticRegression",        LogisticRegression(max_iter=1000, random_state=random_state, class_weight="balanced")),
        ("03_KNeighborsClassifier",      KNeighborsClassifier(n_neighbors=5)),
        ("04_DecisionTreeClassifier",    DecisionTreeClassifier(random_state=random_state, class_weight="balanced")),
        ("05_RandomForestClassifier",    RandomForestClassifier(n_estimators=100, random_state=random_state, class_weight="balanced")),
        ("06_ExtraTreesClassifier",      ExtraTreesClassifier(n_estimators=100, random_state=random_state, class_weight="balanced")),
        ("07_GradientBoostingClassifier",GradientBoostingClassifier(n_estimators=100, random_state=random_state)),
        ("08_HistGradientBoosting",      HistGradientBoostingClassifier(random_state=random_state, class_weight="balanced")),
        ("09_AdaBoostClassifier",        AdaBoostClassifier(n_estimators=100, random_state=random_state)),
        ("10_SVC",                       SVC(kernel="rbf", probability=True, random_state=random_state, class_weight="balanced")),
        ("11_GaussianNB",                GaussianNB()),
        ("12_LinearDiscriminantAnalysis",LinearDiscriminantAnalysis()),
        ("13_QuadraticDiscriminantAnalysis", QuadraticDiscriminantAnalysis(reg_param=0.01)),
        ("14_BaggingClassifier",         BaggingClassifier(n_estimators=50, random_state=random_state)),
        ("15_SGDClassifier",             SGDClassifier(max_iter=1000, random_state=random_state, class_weight="balanced")),
    ]

    # Metricas (README maestro seccion 10.1)
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    # Entrenamiento de los 15 candidatos
    logger.info("[model_training] Iniciando entrenamiento de 15 candidatos de clasificacion...")
    resultados = []
    for nombre, modelo in candidatos:
        pipe = Pipeline([
            ("preprocesador", preprocesador),
            ("modelo", modelo),
        ])
        fila = _entrenar_modelo_con_cv(nombre, pipe, X, y, cv, scoring)
        resultados.append(fila)

    # Construir ranking ordenado por f1 de la clase positiva (metrica principal)
    ranking = pd.DataFrame(resultados)
    if "f1" in ranking.columns:
        ranking = ranking.sort_values("f1", ascending=False).reset_index(drop=True)
        ranking.insert(0, "posicion", ranking.index + 1)

    logger.info(
        "[model_training] Clasificacion completada. Top 3:\n%s",
        ranking[["modelo", "f1", "recall", "roc_auc"]].head(3).to_string() if "f1" in ranking.columns else "N/A",
    )
    return ranking


def entrenar_candidatos_regresion(
    reg_data: pd.DataFrame,
    parameters: dict,
) -> pd.DataFrame:
    """
    Nodo 2: Entrena 15 modelos candidatos de regresion.

    Target: dias_en_transito
    Metrica principal: neg_mean_absolute_error (MAE — interpretable en dias)
    Validacion: KFold(n_splits=5, shuffle=True)

    El baseline (DummyRegressor) predice siempre la media del target.
    Los modelos reales deben superar este baseline para ser utiles.

    Parametros
    ----------
    reg_data   : Dataset de regresion (05_model_input/model_input_regresion.csv).
    parameters : Parametros del proyecto.

    Retorna
    -------
    pd.DataFrame ranking guardado en 07_model_output/ranking_modelos_regresion.csv.
    """
    random_state = parameters.get("model_input", {}).get("random_state", RANDOM_STATE)

    # Preparar X, y
    features_num = _get_features_presentes(reg_data, FEATURES_NUMERICAS)
    features_cat = _get_features_presentes(reg_data, FEATURES_CATEGORICAS)
    X = reg_data[features_num + features_cat]
    y = reg_data[TARGET_REG]

    logger.info(
        "[model_training] Regresion — X: %s | y: media=%.2f dias, std=%.2f",
        X.shape, float(y.mean()), float(y.std()),
    )

    # Preprocesador
    preprocesador = _construir_preprocesador(features_num, features_cat)

    # 15 modelos candidatos (README maestro seccion 12)
    candidatos = [
        ("01_DummyRegressor",            DummyRegressor(strategy="mean")),
        ("02_LinearRegression",          LinearRegression()),
        ("03_Ridge",                     Ridge(alpha=1.0)),
        ("04_Lasso",                     Lasso(alpha=0.1, max_iter=2000)),
        ("05_ElasticNet",                ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=2000)),
        ("06_KNeighborsRegressor",       KNeighborsRegressor(n_neighbors=5)),
        ("07_DecisionTreeRegressor",     DecisionTreeRegressor(random_state=random_state)),
        ("08_RandomForestRegressor",     RandomForestRegressor(n_estimators=100, random_state=random_state)),
        ("09_ExtraTreesRegressor",       ExtraTreesRegressor(n_estimators=100, random_state=random_state)),
        ("10_GradientBoostingRegressor", GradientBoostingRegressor(n_estimators=100, random_state=random_state)),
        ("11_HistGradientBoosting",      HistGradientBoostingRegressor(random_state=random_state)),
        ("12_AdaBoostRegressor",         AdaBoostRegressor(n_estimators=100, random_state=random_state)),
        ("13_SVR",                       SVR(kernel="rbf")),
        ("14_BaggingRegressor",          BaggingRegressor(n_estimators=50, random_state=random_state)),
        ("15_SGDRegressor",              SGDRegressor(max_iter=1000, random_state=random_state)),
    ]

    # Metricas (README maestro seccion 10.2)
    scoring = {
        "mae":   "neg_mean_absolute_error",
        "mse":   "neg_mean_squared_error",
        "rmse":  "neg_root_mean_squared_error",
        "r2":    "r2",
    }

    cv = KFold(n_splits=5, shuffle=True, random_state=random_state)

    # Entrenamiento de los 15 candidatos
    logger.info("[model_training] Iniciando entrenamiento de 15 candidatos de regresion...")
    resultados = []
    for nombre, modelo in candidatos:
        pipe = Pipeline([
            ("preprocesador", preprocesador),
            ("modelo", modelo),
        ])
        fila = _entrenar_modelo_con_cv(nombre, pipe, X, y, cv, scoring)
        # Convertir MAE a positivo para facilitar interpretacion
        for col in ["mae", "mse", "rmse"]:
            if col in fila and fila[col] is not None and fila.get("error") is None:
                try:
                    fila[col] = abs(fila[col])
                except Exception:
                    pass
        resultados.append(fila)

    # Construir ranking ordenado por MAE (menor es mejor)
    ranking = pd.DataFrame(resultados)
    if "mae" in ranking.columns:
        ranking = ranking.sort_values("mae", ascending=True).reset_index(drop=True)
        ranking.insert(0, "posicion", ranking.index + 1)

    logger.info(
        "[model_training] Regresion completada. Top 3:\n%s",
        ranking[["modelo", "mae", "r2"]].head(3).to_string() if "mae" in ranking.columns else "N/A",
    )
    return ranking

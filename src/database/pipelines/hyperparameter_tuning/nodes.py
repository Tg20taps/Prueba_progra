"""
Nodos de Hyperparameter Tuning (Fase 5 - EP2 SCY1101).

Responsabilidad:
    - Seleccionar finalistas desde los rankings de 15 modelos.
    - Aplicar RandomizedSearchCV para exploracion amplia.
    - Aplicar GridSearchCV para busqueda fina.
    - Comparar modelo base vs modelos optimizados.
    - Guardar los modelos finales de clasificacion y regresion.

El pipeline conserva la regla central del README maestro: los modelos se
entrenan solo sobre los datasets limpios de data/05_model_input.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, LogisticRegression, Ridge
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

logger = logging.getLogger(__name__)

RANDOM_STATE = 42
TARGET_CLF = "tiene_incidencia"
TARGET_REG = "dias_en_transito"

FEATURES_NUMERICAS = [
    "peso_kg",
    "volumen_m3",
    "distancia_km",
    "tiempo_estimado_hrs",
    "peaje_total",
    "capacidad_kg",
    "capacidad_m3",
    "km_recorridos",
    "eficiencia_peso",
    "eficiencia_volumen",
    "costo_por_km",
    "id_ruta",
    "id_vehiculo",
    "mes_envio",
    "dia_semana_envio",
    "trimestre_envio",
]

FEATURES_CATEGORICAS = [
    "estado",
    "tipo_carga_norm",
    "origen",
    "destino",
    "tipo_via",
    "tipo",
    "estado_vehiculo",
]


def _features_presentes(df: pd.DataFrame, cols: list[str]) -> list[str]:
    """Retorna solo las columnas existentes en el DataFrame."""
    return [c for c in cols if c in df.columns]


def _preprocesador(df: pd.DataFrame) -> tuple[ColumnTransformer, list[str]]:
    """Crea ColumnTransformer reproducible para numericas y categoricas."""
    features_num = _features_presentes(df, FEATURES_NUMERICAS)
    features_cat = _features_presentes(df, FEATURES_CATEGORICAS)

    numeric_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    transformer = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, features_num),
            ("cat", categorical_pipe, features_cat),
        ],
        remainder="drop",
    )
    return transformer, features_num + features_cat


def _json_seguro(obj: Any) -> str:
    """Serializa parametros sklearn a JSON legible para CSV."""
    return json.dumps(obj, ensure_ascii=False, default=str, sort_keys=True)


def _contar_combinaciones(params: dict[str, list[Any]]) -> int:
    """Cuenta combinaciones posibles de una grilla/distribucion discreta."""
    total = 1
    for valores in params.values():
        total *= len(valores)
    return total


def _seleccionar_finalistas(
    ranking: pd.DataFrame,
    soportados: dict[str, dict[str, Any]],
    metrica: str,
    max_modelos: int = 3,
    menor_es_mejor: bool = False,
) -> list[str]:
    """Selecciona modelos soportados segun ranking, excluyendo baselines Dummy."""
    ranking_ok = ranking.copy()
    if "error" in ranking_ok.columns:
        ranking_ok = ranking_ok[ranking_ok["error"].isna()]
    ranking_ok = ranking_ok[~ranking_ok["modelo"].str.contains("Dummy", na=False)]

    if metrica in ranking_ok.columns:
        ranking_ok = ranking_ok.sort_values(metrica, ascending=menor_es_mejor)

    seleccionados: list[str] = []
    for modelo in ranking_ok["modelo"].tolist():
        if modelo in soportados and modelo not in seleccionados:
            seleccionados.append(modelo)
        if len(seleccionados) == max_modelos:
            break

    if len(seleccionados) < max_modelos:
        for modelo in soportados:
            if modelo not in seleccionados:
                seleccionados.append(modelo)
            if len(seleccionados) == max_modelos:
                break

    return seleccionados


def _espacios_clasificacion(random_state: int) -> dict[str, dict[str, Any]]:
    """Modelos finalistas posibles para clasificacion."""
    return {
        "02_LogisticRegression": {
            "estimator": LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=random_state,
            ),
            "random": {
                "modelo__C": [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
                "modelo__penalty": ["l2"],
                "modelo__solver": ["liblinear", "lbfgs"],
            },
            "grid": {
                "modelo__C": [0.1, 0.5, 1.0, 2.0],
                "modelo__penalty": ["l2"],
                "modelo__solver": ["liblinear"],
            },
        },
        "04_DecisionTreeClassifier": {
            "estimator": DecisionTreeClassifier(
                class_weight="balanced",
                random_state=random_state,
            ),
            "random": {
                "modelo__max_depth": [3, 4, 5, 6, 8, 10, None],
                "modelo__min_samples_split": [2, 5, 10, 20],
                "modelo__min_samples_leaf": [1, 2, 5, 10, 20],
                "modelo__criterion": ["gini", "entropy"],
            },
            "grid": {
                "modelo__max_depth": [3, 5, 8, None],
                "modelo__min_samples_leaf": [5, 10, 20],
                "modelo__criterion": ["gini", "entropy"],
            },
        },
        "05_RandomForestClassifier": {
            "estimator": RandomForestClassifier(
                class_weight="balanced",
                random_state=random_state,
            ),
            "random": {
                "modelo__n_estimators": [80, 120, 160],
                "modelo__max_depth": [4, 6, 8, None],
                "modelo__min_samples_leaf": [1, 3, 5, 10],
                "modelo__max_features": ["sqrt", "log2", None],
            },
            "grid": {
                "modelo__n_estimators": [100, 160],
                "modelo__max_depth": [6, None],
                "modelo__min_samples_leaf": [3, 5],
                "modelo__max_features": ["sqrt"],
            },
        },
        "08_HistGradientBoosting": {
            "estimator": HistGradientBoostingClassifier(
                class_weight="balanced",
                random_state=random_state,
            ),
            "random": {
                "modelo__learning_rate": [0.03, 0.05, 0.08, 0.1, 0.15],
                "modelo__max_iter": [80, 120, 160, 220],
                "modelo__max_leaf_nodes": [15, 31, 63],
                "modelo__min_samples_leaf": [10, 20, 40, 60],
                "modelo__l2_regularization": [0.0, 0.01, 0.1, 1.0],
            },
            "grid": {
                "modelo__learning_rate": [0.05, 0.08, 0.1],
                "modelo__max_iter": [120, 160],
                "modelo__max_leaf_nodes": [15, 31],
                "modelo__min_samples_leaf": [20, 40],
            },
        },
        "10_SVC": {
            "estimator": SVC(
                class_weight="balanced",
                probability=True,
                random_state=random_state,
            ),
            "random": {
                "modelo__C": [0.1, 0.5, 1.0, 2.0, 5.0],
                "modelo__gamma": ["scale", "auto", 0.01, 0.1],
                "modelo__kernel": ["rbf"],
            },
            "grid": {
                "modelo__C": [0.5, 1.0, 2.0],
                "modelo__gamma": ["scale", 0.01, 0.1],
                "modelo__kernel": ["rbf"],
            },
        },
        "11_GaussianNB": {
            "estimator": GaussianNB(),
            "random": {
                "modelo__var_smoothing": [
                    1e-12,
                    1e-11,
                    1e-10,
                    1e-9,
                    1e-8,
                    1e-7,
                    1e-6,
                ],
            },
            "grid": {
                "modelo__var_smoothing": [1e-11, 1e-10, 1e-9, 1e-8, 1e-7],
            },
        },
        "12_LinearDiscriminantAnalysis": {
            "estimator": LinearDiscriminantAnalysis(),
            "random": {
                "modelo__solver": ["lsqr", "eigen"],
                "modelo__shrinkage": ["auto", 0.0, 0.1, 0.5, 0.9],
            },
            "grid": {
                "modelo__solver": ["lsqr"],
                "modelo__shrinkage": ["auto", 0.1, 0.5],
            },
        },
        "15_SGDClassifier": {
            "estimator": SGDClassifier(
                class_weight="balanced",
                max_iter=2000,
                random_state=random_state,
            ),
            "random": {
                "modelo__loss": ["hinge", "log_loss", "modified_huber"],
                "modelo__alpha": [1e-5, 1e-4, 1e-3, 1e-2],
                "modelo__penalty": ["l2", "l1", "elasticnet"],
            },
            "grid": {
                "modelo__loss": ["log_loss", "modified_huber"],
                "modelo__alpha": [1e-4, 1e-3, 1e-2],
                "modelo__penalty": ["l2", "elasticnet"],
            },
        },
    }


def _espacios_regresion(random_state: int) -> dict[str, dict[str, Any]]:
    """Modelos finalistas posibles para regresion."""
    return {
        "03_Ridge": {
            "estimator": Ridge(),
            "random": {"modelo__alpha": [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]},
            "grid": {"modelo__alpha": [0.1, 0.5, 1.0, 2.0, 5.0]},
        },
        "04_Lasso": {
            "estimator": Lasso(max_iter=5000),
            "random": {"modelo__alpha": [0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]},
            "grid": {"modelo__alpha": [0.01, 0.05, 0.1, 0.2]},
        },
        "05_ElasticNet": {
            "estimator": ElasticNet(max_iter=5000),
            "random": {
                "modelo__alpha": [0.001, 0.01, 0.05, 0.1, 0.2, 0.5],
                "modelo__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
            },
            "grid": {
                "modelo__alpha": [0.01, 0.05, 0.1, 0.2],
                "modelo__l1_ratio": [0.3, 0.5, 0.7],
            },
        },
        "06_KNeighborsRegressor": {
            "estimator": KNeighborsRegressor(),
            "random": {
                "modelo__n_neighbors": [3, 5, 7, 9, 11, 15, 21],
                "modelo__weights": ["uniform", "distance"],
                "modelo__p": [1, 2],
            },
            "grid": {
                "modelo__n_neighbors": [3, 5, 7, 9, 11],
                "modelo__weights": ["uniform", "distance"],
                "modelo__p": [1, 2],
            },
        },
        "10_GradientBoostingRegressor": {
            "estimator": GradientBoostingRegressor(random_state=random_state),
            "random": {
                "modelo__n_estimators": [80, 120, 160, 220],
                "modelo__learning_rate": [0.03, 0.05, 0.08, 0.1],
                "modelo__max_depth": [2, 3, 4],
                "modelo__min_samples_leaf": [1, 3, 5, 10],
            },
            "grid": {
                "modelo__n_estimators": [100, 160],
                "modelo__learning_rate": [0.05, 0.08],
                "modelo__max_depth": [2, 3],
                "modelo__min_samples_leaf": [3, 5],
            },
        },
    }


def _evaluar_clasificacion(modelo: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, Any]:
    """Calcula metricas de clasificacion en holdout."""
    y_pred = modelo.predict(X_test)
    if hasattr(modelo, "predict_proba"):
        y_score = modelo.predict_proba(X_test)[:, 1]
    else:
        y_score = None

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    metricas = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_score) if y_score is not None else np.nan,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    return {k: round(float(v), 4) if isinstance(v, (float, np.floating)) else v for k, v in metricas.items()}


def _evaluar_regresion(modelo: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, Any]:
    """Calcula metricas de regresion en holdout."""
    y_pred = modelo.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    metricas = {
        "mae": mean_absolute_error(y_test, y_pred),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "r2": r2_score(y_test, y_pred),
    }
    return {k: round(float(v), 4) for k, v in metricas.items()}


def _registrar_comparacion(
    filas: list[dict[str, Any]],
    tipo: str,
    modelo: str,
    etapa: str,
    metricas: dict[str, Any],
    cv_score: float | None = None,
) -> None:
    """Agrega una fila estandar al reporte de comparacion."""
    fila = {
        "tipo_problema": tipo,
        "modelo": modelo,
        "etapa": etapa,
        "cv_score": round(float(cv_score), 4) if cv_score is not None else None,
    }
    fila.update(metricas)
    filas.append(fila)


def _ejecutar_busquedas(
    tipo: str,
    data: pd.DataFrame,
    target: str,
    ranking: pd.DataFrame,
    espacios: dict[str, dict[str, Any]],
    metrica_ranking: str,
    scoring: str,
    cv,
    random_state: int,
    test_size: float,
    n_iter: int,
    menor_es_mejor: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], Pipeline, dict[str, Any]]:
    """Ejecuta RandomizedSearchCV y GridSearchCV para un problema."""
    preprocesador, features = _preprocesador(data)
    X = data[features]
    y = data[target]

    stratify = y if tipo == "clasificacion" else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    finalistas = _seleccionar_finalistas(
        ranking=ranking,
        soportados=espacios,
        metrica=metrica_ranking,
        max_modelos=3,
        menor_es_mejor=menor_es_mejor,
    )
    logger.info("[%s] Finalistas seleccionados: %s", tipo, finalistas)

    filas_random: list[dict[str, Any]] = []
    filas_grid: list[dict[str, Any]] = []
    filas_comparacion: list[dict[str, Any]] = []
    modelos_entrenados: list[tuple[str, str, Pipeline, dict[str, Any], float]] = []

    for nombre in finalistas:
        config = espacios[nombre]
        base = Pipeline(
            [
                ("preprocesador", preprocesador),
                ("modelo", config["estimator"]),
            ]
        )

        base.fit(X_train, y_train)
        metricas_base = (
            _evaluar_clasificacion(base, X_test, y_test)
            if tipo == "clasificacion"
            else _evaluar_regresion(base, X_test, y_test)
        )
        _registrar_comparacion(
            filas_comparacion,
            tipo,
            nombre,
            "base_sin_optimizar",
            metricas_base,
        )

        n_iter_real = min(n_iter, _contar_combinaciones(config["random"]))
        randomized = RandomizedSearchCV(
            estimator=base,
            param_distributions=config["random"],
            n_iter=n_iter_real,
            scoring=scoring,
            cv=cv,
            random_state=random_state,
            n_jobs=1,
            error_score=np.nan,
        )
        randomized.fit(X_train, y_train)

        metricas_random = (
            _evaluar_clasificacion(randomized.best_estimator_, X_test, y_test)
            if tipo == "clasificacion"
            else _evaluar_regresion(randomized.best_estimator_, X_test, y_test)
        )
        filas_random.append(
            {
                "tipo_problema": tipo,
                "modelo": nombre,
                "search": "RandomizedSearchCV",
                "scoring": scoring,
                "n_iter": n_iter_real,
                "best_score_cv": round(float(randomized.best_score_), 4),
                "best_params": _json_seguro(randomized.best_params_),
            }
        )
        _registrar_comparacion(
            filas_comparacion,
            tipo,
            nombre,
            "randomized_search",
            metricas_random,
            randomized.best_score_,
        )
        modelos_entrenados.append(
            (nombre, "randomized_search", randomized.best_estimator_, metricas_random, float(randomized.best_score_))
        )

        grid = GridSearchCV(
            estimator=base,
            param_grid=config["grid"],
            scoring=scoring,
            cv=cv,
            n_jobs=1,
            error_score=np.nan,
        )
        grid.fit(X_train, y_train)

        metricas_grid = (
            _evaluar_clasificacion(grid.best_estimator_, X_test, y_test)
            if tipo == "clasificacion"
            else _evaluar_regresion(grid.best_estimator_, X_test, y_test)
        )
        filas_grid.append(
            {
                "tipo_problema": tipo,
                "modelo": nombre,
                "search": "GridSearchCV",
                "scoring": scoring,
                "n_combinaciones": _contar_combinaciones(config["grid"]),
                "best_score_cv": round(float(grid.best_score_), 4),
                "best_params": _json_seguro(grid.best_params_),
            }
        )
        _registrar_comparacion(
            filas_comparacion,
            tipo,
            nombre,
            "grid_search",
            metricas_grid,
            grid.best_score_,
        )
        modelos_entrenados.append(
            (nombre, "grid_search", grid.best_estimator_, metricas_grid, float(grid.best_score_))
        )

    if tipo == "clasificacion":
        mejor = max(modelos_entrenados, key=lambda item: item[3]["f1"])
    else:
        mejor = min(modelos_entrenados, key=lambda item: item[3]["mae"])

    nombre, etapa, modelo_final, metricas_finales, cv_score = mejor
    resumen_final = {
        "tipo_problema": tipo,
        "modelo_final": nombre,
        "etapa": etapa,
        "cv_score": round(float(cv_score), 4),
        "criterio_seleccion": "max_f1" if tipo == "clasificacion" else "min_mae",
    }
    resumen_final.update(metricas_finales)

    return filas_random, filas_grid, filas_comparacion, modelo_final, resumen_final


def optimizar_modelos_finalistas(
    clf_data: pd.DataFrame,
    reg_data: pd.DataFrame,
    ranking_clf: pd.DataFrame,
    ranking_reg: pd.DataFrame,
    parameters: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Pipeline, Pipeline]:
    """
    Nodo principal de Fase 5.

    Retorna:
        resultados_randomizedsearch, resultados_gridsearch,
        comparacion_optimizacion, metricas_modelo_final,
        modelo_final_clasificacion, modelo_final_regresion.
    """
    cfg = parameters.get("hyperparameter_tuning", {})
    random_state = int(cfg.get("random_state", RANDOM_STATE))
    test_size = float(cfg.get("test_size", 0.2))
    cv_splits = int(cfg.get("cv_splits", 5))
    n_iter = int(cfg.get("randomized_n_iter", 12))

    clf_random, clf_grid, clf_comp, modelo_clf, metricas_clf = _ejecutar_busquedas(
        tipo="clasificacion",
        data=clf_data,
        target=TARGET_CLF,
        ranking=ranking_clf,
        espacios=_espacios_clasificacion(random_state),
        metrica_ranking="f1",
        scoring="f1",
        cv=StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state),
        random_state=random_state,
        test_size=test_size,
        n_iter=n_iter,
        menor_es_mejor=False,
    )

    reg_random, reg_grid, reg_comp, modelo_reg, metricas_reg = _ejecutar_busquedas(
        tipo="regresion",
        data=reg_data,
        target=TARGET_REG,
        ranking=ranking_reg,
        espacios=_espacios_regresion(random_state),
        metrica_ranking="mae",
        scoring="neg_mean_absolute_error",
        cv=KFold(n_splits=cv_splits, shuffle=True, random_state=random_state),
        random_state=random_state,
        test_size=test_size,
        n_iter=n_iter,
        menor_es_mejor=True,
    )

    resultados_randomized = pd.DataFrame(clf_random + reg_random)
    resultados_grid = pd.DataFrame(clf_grid + reg_grid)
    comparacion = pd.DataFrame(clf_comp + reg_comp)
    metricas_finales = pd.DataFrame([metricas_clf, metricas_reg])

    logger.info(
        "[hyperparameter_tuning] Final clasificacion: %s | f1=%s",
        metricas_clf["modelo_final"],
        metricas_clf.get("f1"),
    )
    logger.info(
        "[hyperparameter_tuning] Final regresion: %s | mae=%s",
        metricas_reg["modelo_final"],
        metricas_reg.get("mae"),
    )

    return (
        resultados_randomized,
        resultados_grid,
        comparacion,
        metricas_finales,
        modelo_clf,
        modelo_reg,
    )

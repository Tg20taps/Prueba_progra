"""
Nodos de aprendizaje no supervisado (Fase 6 - EP2 SCY1101).

Objetivo de negocio:
    Identificar segmentos operativos de envios con patrones similares para
    detectar grupos de mayor riesgo y orientar decisiones logisticas.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)

RANDOM_STATE = 42
TARGET_CLF = "tiene_incidencia"

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
    """Retorna solo columnas disponibles."""
    return [c for c in cols if c in df.columns]


def _preprocesar_para_clustering(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Preprocesa variables numericas y categoricas para clustering."""
    features_num = _features_presentes(df, FEATURES_NUMERICAS)
    features_cat = _features_presentes(df, FEATURES_CATEGORICAS)
    features = features_num + features_cat

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
    preprocesador = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, features_num),
            ("cat", categorical_pipe, features_cat),
        ],
        remainder="drop",
    )
    matriz = preprocesador.fit_transform(df[features])
    return matriz, features


def _metricas_cluster(matriz: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Calcula silhouette y Davies-Bouldin cuando existen al menos 2 clusters."""
    labels_validos = set(labels)
    labels_sin_ruido = labels_validos - {-1}
    if len(labels_sin_ruido) < 2:
        return np.nan, np.nan
    mascara = labels != -1
    return (
        float(silhouette_score(matriz[mascara], labels[mascara])),
        float(davies_bouldin_score(matriz[mascara], labels[mascara])),
    )


def _perfilar_clusters(df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Construye perfil de negocio para cada cluster KMeans."""
    perfil_base = df.copy()
    perfil_base["cluster"] = labels

    agregaciones = {
        TARGET_CLF: ["count", "mean"],
        "peso_kg": "mean",
        "distancia_km": "mean",
        "tiempo_estimado_hrs": "mean",
        "peaje_total": "mean",
        "eficiencia_peso": "mean",
    }
    agregaciones = {
        col: funcs for col, funcs in agregaciones.items() if col in perfil_base.columns
    }

    perfil = perfil_base.groupby("cluster").agg(agregaciones)
    perfil.columns = [
        "_".join(col).strip("_") if isinstance(col, tuple) else col
        for col in perfil.columns.to_flat_index()
    ]
    perfil = perfil.reset_index()
    perfil = perfil.rename(
        columns={
            f"{TARGET_CLF}_count": "n_envios",
            f"{TARGET_CLF}_mean": "tasa_incidencia",
            "peso_kg_mean": "peso_kg_promedio",
            "distancia_km_mean": "distancia_km_promedio",
            "tiempo_estimado_hrs_mean": "tiempo_estimado_hrs_promedio",
            "peaje_total_mean": "peaje_total_promedio",
            "eficiencia_peso_mean": "eficiencia_peso_promedio",
        }
    )

    if "tasa_incidencia" in perfil.columns:
        promedio_global = float(perfil_base[TARGET_CLF].mean())
        condiciones = [
            perfil["tasa_incidencia"] >= promedio_global * 1.25,
            perfil["tasa_incidencia"] <= promedio_global * 0.75,
        ]
        etiquetas = ["riesgo_operativo_alto", "riesgo_operativo_bajo"]
        perfil["interpretacion_negocio"] = np.select(
            condiciones,
            etiquetas,
            default="riesgo_operativo_medio",
        )

    for col in perfil.select_dtypes(include="number").columns:
        if col != "cluster":
            perfil[col] = perfil[col].round(4)

    return perfil


def ejecutar_clustering(
    clf_data: pd.DataFrame,
    parameters: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Ejecuta KMeans, DBSCAN y PCA sobre el dataset limpio de clasificacion.

    Retorna:
        perfil_clusters, metricas_clustering, clusters_pca.
    """
    cfg = parameters.get("unsupervised_learning", {})
    random_state = int(cfg.get("random_state", RANDOM_STATE))
    k_values = cfg.get("kmeans_k_values", [3, 4, 5])
    dbscan_eps = float(cfg.get("dbscan_eps", 2.5))
    dbscan_min_samples = int(cfg.get("dbscan_min_samples", 10))

    matriz, features = _preprocesar_para_clustering(clf_data)
    logger.info(
        "[unsupervised_learning] Matriz clustering: %s | features base: %d",
        matriz.shape,
        len(features),
    )

    filas_metricas: list[dict] = []
    mejores = {"score": -np.inf, "modelo": None, "labels": None, "k": None}

    for k in k_values:
        kmeans = KMeans(n_clusters=int(k), random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(matriz)
        silhouette, db_score = _metricas_cluster(matriz, labels)
        filas_metricas.append(
            {
                "metodo": "KMeans",
                "parametros": f"k={k}",
                "n_clusters": int(k),
                "n_ruido": 0,
                "silhouette_score": round(silhouette, 4),
                "davies_bouldin_score": round(db_score, 4),
                "inertia": round(float(kmeans.inertia_), 4),
            }
        )
        if not np.isnan(silhouette) and silhouette > mejores["score"]:
            mejores = {"score": silhouette, "modelo": kmeans, "labels": labels, "k": k}

    dbscan = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
    labels_dbscan = dbscan.fit_predict(matriz)
    clusters_dbscan = len(set(labels_dbscan) - {-1})
    ruido = int(np.sum(labels_dbscan == -1))
    silhouette_db, db_score_db = _metricas_cluster(matriz, labels_dbscan)
    filas_metricas.append(
        {
            "metodo": "DBSCAN",
            "parametros": f"eps={dbscan_eps}, min_samples={dbscan_min_samples}",
            "n_clusters": clusters_dbscan,
            "n_ruido": ruido,
            "silhouette_score": round(silhouette_db, 4) if not np.isnan(silhouette_db) else np.nan,
            "davies_bouldin_score": round(db_score_db, 4) if not np.isnan(db_score_db) else np.nan,
            "inertia": np.nan,
        }
    )

    labels_kmeans = mejores["labels"]
    perfil = _perfilar_clusters(clf_data, labels_kmeans)

    pca = PCA(n_components=2, random_state=random_state)
    coords = pca.fit_transform(matriz)
    clusters_pca = pd.DataFrame(
        {
            "id_envio": clf_data["id_envio"].values if "id_envio" in clf_data.columns else np.arange(len(clf_data)),
            "pca_1": coords[:, 0].round(4),
            "pca_2": coords[:, 1].round(4),
            "cluster_kmeans": labels_kmeans,
            "cluster_dbscan": labels_dbscan,
            TARGET_CLF: clf_data[TARGET_CLF].values if TARGET_CLF in clf_data.columns else np.nan,
        }
    )
    filas_metricas.append(
        {
            "metodo": "PCA",
            "parametros": "n_components=2",
            "n_clusters": int(mejores["k"]),
            "n_ruido": np.nan,
            "silhouette_score": np.nan,
            "davies_bouldin_score": np.nan,
            "inertia": np.nan,
            "varianza_explicada": round(float(pca.explained_variance_ratio_.sum()), 4),
        }
    )

    metricas = pd.DataFrame(filas_metricas)
    logger.info(
        "[unsupervised_learning] Mejor KMeans: k=%s | silhouette=%.4f",
        mejores["k"],
        mejores["score"],
    )

    return perfil, metricas, clusters_pca

"""Definicion del pipeline de Unsupervised Learning (Fase 6)."""
from __future__ import annotations

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import ejecutar_clustering


def create_pipeline(**kwargs) -> Pipeline:
    """Crea el pipeline de clustering, PCA e interpretacion de segmentos."""
    return pipeline(
        [
            node(
                func=ejecutar_clustering,
                inputs=["model_input_clasificacion", "parameters"],
                outputs=[
                    "perfil_clusters",
                    "metricas_clustering",
                    "clusters_pca",
                ],
                name="ejecutar_clustering_node",
                tags=["unsupervised_learning", "clustering", "pca"],
            )
        ]
    )

"""
Definición del pipeline de Data Transformation (AD 1.3).

DAG resultante:
    envios_clean ──┐
    rutas_clean  ──┤
    vehiculos_clean┤──► unir_datasets ──► crear_features ──► codificar ──► normalizar ──► master_envios
    incidencias_clean┘
"""
from __future__ import annotations

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    codificar_categoricas,
    crear_features,
    normalizar_numericas,
    unir_datasets,
)


def create_pipeline(**kwargs) -> Pipeline:
    """Crea el pipeline de Data Transformation con 4 nodos en cadena."""
    return pipeline(
        [
            node(
                func=unir_datasets,
                inputs=[
                    "envios_clean",
                    "rutas_clean",
                    "vehiculos_clean",
                    "incidencias_clean",
                ],
                outputs="dataset_unificado",
                name="unir_datasets_node",
                tags=["transformation", "join"],
            ),
            node(
                func=crear_features,
                inputs="dataset_unificado",
                outputs="dataset_con_features",
                name="crear_features_node",
                tags=["transformation", "features"],
            ),
            node(
                func=codificar_categoricas,
                inputs=["dataset_con_features", "parameters"],
                outputs="dataset_encoded",
                name="codificar_categoricas_node",
                tags=["transformation", "encoding"],
            ),
            node(
                func=normalizar_numericas,
                inputs=["dataset_encoded", "parameters"],
                outputs="master_envios",
                name="normalizar_numericas_node",
                tags=["transformation", "normalizacion"],
            ),
        ]
    )

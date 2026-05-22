"""
Definición del pipeline de Model Input (Fase 1 — EP2 SCY1101).

DAG resultante:
    master_envios
        └─► preparar_features_base
                ├─► preparar_clasificacion  ──► model_input_clasificacion
                ├─► preparar_regresion      ──► model_input_regresion
                └─► (ambos) generar_reporte ──► model_input_report
"""
from __future__ import annotations

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    generar_reporte_model_input,
    preparar_clasificacion,
    preparar_features_base,
    preparar_regresion,
)


def create_pipeline(**kwargs) -> Pipeline:
    """Crea el pipeline de preparación de datasets ML para EP2 SCY1101."""
    return pipeline(
        [
            node(
                func=preparar_features_base,
                inputs=["master_envios", "parameters"],
                outputs="features_ml_base",
                name="preparar_features_base_node",
                tags=["model_input", "feature_engineering"],
            ),
            node(
                func=preparar_clasificacion,
                inputs=["features_ml_base", "parameters"],
                outputs="model_input_clasificacion",
                name="preparar_clasificacion_node",
                tags=["model_input", "clasificacion"],
            ),
            node(
                func=preparar_regresion,
                inputs=["features_ml_base", "parameters"],
                outputs="model_input_regresion",
                name="preparar_regresion_node",
                tags=["model_input", "regresion"],
            ),
            node(
                func=generar_reporte_model_input,
                inputs=[
                    "master_envios",
                    "model_input_clasificacion",
                    "model_input_regresion",
                ],
                outputs="model_input_report",
                name="generar_reporte_model_input_node",
                tags=["model_input", "reporting"],
            ),
        ]
    )

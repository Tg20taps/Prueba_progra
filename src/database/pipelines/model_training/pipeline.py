"""
Definicion del pipeline de Model Training (Fase 3 — EP2 SCY1101).

DAG resultante:
    model_input_clasificacion ──► entrenar_candidatos_clasificacion ──► ranking_modelos_clasificacion
    model_input_regresion     ──► entrenar_candidatos_regresion     ──► ranking_modelos_regresion
"""
from __future__ import annotations

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    entrenar_candidatos_clasificacion,
    entrenar_candidatos_regresion,
)


def create_pipeline(**kwargs) -> Pipeline:
    """Crea el pipeline de entrenamiento de 15 modelos candidatos para EP2."""
    return pipeline(
        [
            node(
                func=entrenar_candidatos_clasificacion,
                inputs=["model_input_clasificacion", "parameters"],
                outputs="ranking_modelos_clasificacion",
                name="entrenar_clasificacion_node",
                tags=["model_training", "clasificacion"],
            ),
            node(
                func=entrenar_candidatos_regresion,
                inputs=["model_input_regresion", "parameters"],
                outputs="ranking_modelos_regresion",
                name="entrenar_regresion_node",
                tags=["model_training", "regresion"],
            ),
        ]
    )

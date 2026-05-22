"""Definicion del pipeline de Hyperparameter Tuning (Fase 5)."""
from __future__ import annotations

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import optimizar_modelos_finalistas


def create_pipeline(**kwargs) -> Pipeline:
    """Crea el pipeline de seleccion de finalistas y optimizacion."""
    return pipeline(
        [
            node(
                func=optimizar_modelos_finalistas,
                inputs=[
                    "model_input_clasificacion",
                    "model_input_regresion",
                    "ranking_modelos_clasificacion",
                    "ranking_modelos_regresion",
                    "parameters",
                ],
                outputs=[
                    "resultados_randomizedsearch",
                    "resultados_gridsearch",
                    "comparacion_optimizacion",
                    "metricas_modelo_final",
                    "modelo_final_clasificacion",
                    "modelo_final_regresion",
                ],
                name="optimizar_modelos_finalistas_node",
                tags=["hyperparameter_tuning", "model_selection"],
            )
        ]
    )

"""
Definición del pipeline de Data Cleaning (AD 1.2).

DAG resultante:
    envios_raw  + params ──►  limpiar_envios  ──►  envios_clean
    rutas_raw   + params ──►  limpiar_rutas   ──►  rutas_clean
    vehiculos_raw+ params ──►  limpiar_vehiculos►  vehiculos_clean
    incidencias_raw+params ──►  limpiar_incid. ──►  incidencias_clean
"""
from __future__ import annotations

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    limpiar_envios,
    limpiar_incidencias,
    limpiar_rutas,
    limpiar_vehiculos,
)


def create_pipeline(**kwargs) -> Pipeline:
    """Crea el pipeline de Data Cleaning con 4 nodos paralelos."""
    return pipeline(
        [
            node(
                func=limpiar_envios,
                inputs=["envios_raw", "parameters"],
                outputs="envios_clean",
                name="limpiar_envios_node",
                tags=["cleaning", "envios"],
            ),
            node(
                func=limpiar_rutas,
                inputs=["rutas_raw", "parameters"],
                outputs="rutas_clean",
                name="limpiar_rutas_node",
                tags=["cleaning", "rutas"],
            ),
            node(
                func=limpiar_vehiculos,
                inputs=["vehiculos_raw", "parameters"],
                outputs="vehiculos_clean",
                name="limpiar_vehiculos_node",
                tags=["cleaning", "vehiculos"],
            ),
            node(
                func=limpiar_incidencias,
                inputs=["incidencias_raw", "parameters"],
                outputs="incidencias_clean",
                name="limpiar_incidencias_node",
                tags=["cleaning", "incidencias"],
            ),
        ]
    )

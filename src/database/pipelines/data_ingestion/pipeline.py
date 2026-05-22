"""
Definición del pipeline de Data Ingestion (AD 1.1).

DAG resultante:
    envios_raw  ──►  perfilar_envios  ──►  perfil_envios  ──┐
    rutas_raw   ──►  perfilar_rutas   ──►  perfil_rutas   ──┤
    vehiculos_raw►  perfilar_vehiculos►  perfil_vehiculos ──┤──► consolidar ──► ingestion_report
    incidencias_raw► perfilar_incid.  ►  perfil_incidencias┘
"""
from __future__ import annotations

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    consolidar_reporte_ingestion,
    perfilar_envios,
    perfilar_incidencias,
    perfilar_rutas,
    perfilar_vehiculos,
)


def create_pipeline(**kwargs) -> Pipeline:
    """Crea el pipeline de Data Ingestion con 5 nodos."""
    return pipeline(
        [
            node(
                func=perfilar_envios,
                inputs="envios_raw",
                outputs="perfil_envios",
                name="perfilar_envios_node",
                tags=["ingestion", "diagnostico"],
            ),
            node(
                func=perfilar_rutas,
                inputs="rutas_raw",
                outputs="perfil_rutas",
                name="perfilar_rutas_node",
                tags=["ingestion", "diagnostico"],
            ),
            node(
                func=perfilar_vehiculos,
                inputs="vehiculos_raw",
                outputs="perfil_vehiculos",
                name="perfilar_vehiculos_node",
                tags=["ingestion", "diagnostico"],
            ),
            node(
                func=perfilar_incidencias,
                inputs="incidencias_raw",
                outputs="perfil_incidencias",
                name="perfilar_incidencias_node",
                tags=["ingestion", "diagnostico"],
            ),
            node(
                func=consolidar_reporte_ingestion,
                inputs=[
                    "perfil_envios",
                    "perfil_rutas",
                    "perfil_vehiculos",
                    "perfil_incidencias",
                ],
                outputs="ingestion_report",
                name="consolidar_reporte_ingestion_node",
                tags=["ingestion", "reporte"],
            ),
        ]
    )

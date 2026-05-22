"""
Definición del pipeline de Data Validation (AD 1.4).

DAG resultante:
    master_envios ──┬──► validar_con_pandera   ──► res_pandera    ──┐
                    ├──► validar_esquema        ──► res_esquema    ──┤
                    ├──► validar_integridad     ──► res_integridad ──┤──► consolidar ──► validation_report
                    ├──► validar_rangos         ──► res_rangos     ──┤
                    └──► validar_nulos_criticos ──► res_nulos      ──┘
"""
from __future__ import annotations

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    consolidar_reporte_validacion,
    validar_con_pandera,
    validar_esquema,
    validar_integridad_referencial,
    validar_nulos_criticos,
    validar_rangos,
)


def create_pipeline(**kwargs) -> Pipeline:
    """Crea el pipeline de Data Validation con 6 nodos (5 paralelos + 1 consolidador).

    Novedad vs Fase 1: se agrega el nodo validar_con_pandera (Nodo 0), que aplica
    validación declarativa de esquema usando Pandera DataFrameSchema. Este nodo
    complementa los checks manuales con una capa formal y reproducible que verifica
    tipos, constraints de rango y nulos obligatorios en una sola llamada.
    """
    return pipeline(
        [
            node(
                func=validar_con_pandera,
                inputs="master_envios",
                outputs="resultado_pandera",
                name="validar_pandera_node",
                tags=["validation", "pandera"],
            ),
            node(
                func=validar_esquema,
                inputs="master_envios",
                outputs="resultado_esquema",
                name="validar_esquema_node",
                tags=["validation", "schema"],
            ),
            node(
                func=validar_integridad_referencial,
                inputs=["master_envios", "rutas_clean", "vehiculos_clean"],
                outputs="resultado_integridad",
                name="validar_integridad_node",
                tags=["validation", "integridad"],
            ),
            node(
                func=validar_rangos,
                inputs=["master_envios", "parameters"],
                outputs="resultado_rangos",
                name="validar_rangos_node",
                tags=["validation", "rangos"],
            ),
            node(
                func=validar_nulos_criticos,
                inputs=["master_envios", "parameters"],
                outputs="resultado_nulos",
                name="validar_nulos_node",
                tags=["validation", "nulos"],
            ),
            node(
                func=consolidar_reporte_validacion,
                inputs=[
                    "resultado_esquema",
                    "resultado_integridad",
                    "resultado_rangos",
                    "resultado_nulos",
                    "resultado_pandera",
                ],
                outputs="validation_report",
                name="consolidar_validacion_node",
                tags=["validation", "reporte"],
            ),
        ]
    )

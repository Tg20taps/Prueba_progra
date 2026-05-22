"""Project pipelines — registro de los 4 pipelines del proyecto SCY1101."""
from __future__ import annotations

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline


def register_pipelines() -> dict[str, Pipeline]:
    """
    Registra los 4 pipelines del proyecto de logística de transportes.

    Los pipelines se descubren automáticamente con find_pipelines() y se
    encadenan en el pipeline __default__ para ejecución completa con kedro run.

    Returns:
        Diccionario {nombre: Pipeline} con los 4 pipelines nombrados y el default.
    """
    pipelines = find_pipelines(raise_errors=True)
    # El pipeline por defecto ejecuta todos en orden de dependencia
    pipelines["__default__"] = sum(pipelines.values())
    return pipelines

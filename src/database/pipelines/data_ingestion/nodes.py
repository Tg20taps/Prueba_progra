"""
Nodos del pipeline de Data Ingestion (AD 1.1).

Responsabilidad:
    - Recibir los DataFrames crudos desde el catálogo (01_raw).
    - Ejecutar un diagnóstico de calidad inicial sobre cada dataset.
    - Producir un reporte consolidado que queda en 08_reporting.

Técnicas aplicadas:
    - Filtros avanzados con .loc[] para aislar registros problemáticos.
    - Agrupaciones (.groupby + .agg) para estadísticas por columna.
    - Broadcasting booleano para detectar nulos/duplicados de forma vectorizada.
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers privados
# ---------------------------------------------------------------------------

def _diagnostico_columna(serie: pd.Series) -> dict:
    total = len(serie)
    nulos = serie.isna().sum()
    pct_nulos = round(nulos / total * 100, 2) if total > 0 else 0.0
    n_unicos = serie.nunique(dropna=True)

    fila = {
        "tipo_dato": str(serie.dtype),
        "total_registros": total,
        "nulos": int(nulos),
        "pct_nulos": pct_nulos,
        "valores_unicos": int(n_unicos),
    }

    if pd.api.types.is_numeric_dtype(serie):
        fila["min"] = round(float(serie.min()), 4) if nulos < total else np.nan
        fila["max"] = round(float(serie.max()), 4) if nulos < total else np.nan
        fila["media"] = round(float(serie.mean()), 4) if nulos < total else np.nan
        fila["std"] = round(float(serie.std()), 4) if nulos < total else np.nan
    else:
        fila["min"] = np.nan
        fila["max"] = np.nan
        fila["media"] = np.nan
        fila["std"] = np.nan

    return fila


def _perfil_dataframe(df: pd.DataFrame, nombre: str) -> pd.DataFrame:
    filas = []
    for col in df.columns:
        fila = _diagnostico_columna(df[col])
        fila["dataset"] = nombre
        fila["columna"] = col
        filas.append(fila)

    perfil = pd.DataFrame(filas)

    cols_orden = [
        "dataset", "columna", "tipo_dato", "total_registros",
        "nulos", "pct_nulos", "valores_unicos",
        "min", "max", "media", "std"
    ]
    return perfil[cols_orden]


def _detectar_duplicados(df: pd.DataFrame, nombre: str) -> int:
    n_dups = df.duplicated().sum()
    logger.info("[%s] Duplicados detectados: %d", nombre, n_dups)
    return int(n_dups)


# ---------------------------------------------------------------------------
# NODOS
# ---------------------------------------------------------------------------

def perfilar_envios(envios: pd.DataFrame) -> pd.DataFrame:
    nombre = "envios"
    logger.info("[%s] Shape original: %s", nombre, envios.shape)
    logger.info("[%s] Duplicados: %d", nombre, _detectar_duplicados(envios, nombre))

    sin_ruta = envios.loc[envios["id_ruta"].isna()]
    logger.info(
        "[%s] Registros sin id_ruta: %d (%.1f%%)",
        nombre, len(sin_ruta), len(sin_ruta) / len(envios) * 100
    )

    peso_kg_numeric = pd.to_numeric(envios["peso_kg"], errors="coerce")
    q1 = peso_kg_numeric.quantile(0.25)
    q3 = peso_kg_numeric.quantile(0.75)
    iqr = q3 - q1
    outliers_mask = (peso_kg_numeric < q1 - 3 * iqr) | (peso_kg_numeric > q3 + 3 * iqr)

    logger.info("[%s] Posibles outliers en peso_kg: %d", nombre, outliers_mask.sum())

    return _perfil_dataframe(envios, nombre)


def perfilar_rutas(rutas: pd.DataFrame) -> pd.DataFrame:
    nombre = "rutas"
    logger.info("[%s] Shape original: %s", nombre, rutas.shape)
    logger.info("[%s] Duplicados: %d", nombre, _detectar_duplicados(rutas, nombre))

    sin_origen = rutas.loc[rutas["origen"].isna()]
    logger.info("[%s] Registros sin origen: %d", nombre, len(sin_origen))

    if "tipo_via" in rutas.columns:
        con_espacios = rutas.loc[
            rutas["tipo_via"].str.strip() != rutas["tipo_via"].fillna("")
        ]
        logger.info(
            "[%s] tipo_via con espacios extra: %d registros",
            nombre, len(con_espacios)
        )

    return _perfil_dataframe(rutas, nombre)


def perfilar_vehiculos(vehiculos: pd.DataFrame) -> pd.DataFrame:
    nombre = "vehiculos"
    logger.info("[%s] Shape original: %s", nombre, vehiculos.shape)
    logger.info("[%s] Duplicados: %d", nombre, _detectar_duplicados(vehiculos, nombre))

    sin_cap = vehiculos.loc[vehiculos["capacidad_m3"].isna()]
    logger.info("[%s] Veh?culos sin capacidad_m3: %d", nombre, len(sin_cap))

    if "tipo" in vehiculos.columns:
        resumen_tipo = vehiculos.groupby("tipo", observed=True)["id_vehiculo"].count()
        logger.info("[%s] Distribuci?n por tipo:\n%s", nombre, resumen_tipo.to_string())

    return _perfil_dataframe(vehiculos, nombre)


def perfilar_incidencias(incidencias: pd.DataFrame) -> pd.DataFrame:
    nombre = "incidencias"
    logger.info("[%s] Shape original: %s", nombre, incidencias.shape)
    logger.info("[%s] Duplicados: %d", nombre, _detectar_duplicados(incidencias, nombre))

    if "tipo_incidencia" in incidencias.columns:
        resumen = (
            incidencias
            .groupby("tipo_incidencia", observed=True)
            .agg(
                cantidad=("id_incidencia", "count"),
                costo_no_nulos=("costo_impacto", lambda x: x.notna().sum()),
            )
            .reset_index()
        )
        logger.info("[%s] Resumen por tipo_incidencia:\n%s", nombre, resumen.to_string())

    return _perfil_dataframe(incidencias, nombre)


def consolidar_reporte_ingestion(
    perfil_envios: pd.DataFrame,
    perfil_rutas: pd.DataFrame,
    perfil_vehiculos: pd.DataFrame,
    perfil_incidencias: pd.DataFrame,
) -> pd.DataFrame:

    reporte = pd.concat(
        [perfil_envios, perfil_rutas, perfil_vehiculos, perfil_incidencias],
        ignore_index=True,
    )

    # 🔴 FIX PRINCIPAL: SIN EMOJIS (compatibilidad Windows/CSV)
    reporte["alerta_nulos"] = np.where(
        reporte["pct_nulos"] > 0,
        "Tiene nulos",
        "Sin nulos"
    )

    condiciones = [
        reporte["pct_nulos"] > 20,
        reporte["pct_nulos"].between(5, 20),
        reporte["pct_nulos"] > 0,
    ]
    opciones = ["CRITICO", "MODERADO", "BAJO"]
    reporte["nivel_criticidad"] = np.select(condiciones, opciones, default="OK")

    reporte = reporte.sort_values(
        ["nivel_criticidad", "pct_nulos"],
        ascending=[True, False],
    ).reset_index(drop=True)

    logger.info(
        "Reporte de ingesta consolidado: %d columnas analizadas en %d datasets.",
        len(reporte),
        reporte["dataset"].nunique(),
    )

    return reporte
"""
Nodos del pipeline de Data Transformation (AD 1.3).
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

logger = logging.getLogger(__name__)


def unir_datasets(
    envios: pd.DataFrame,
    rutas: pd.DataFrame,
    vehiculos: pd.DataFrame,
    incidencias: pd.DataFrame,
) -> pd.DataFrame:

    logger.info(
        "[join] Shapes — envios: %s, rutas: %s, vehiculos: %s, incidencias: %s",
        envios.shape, rutas.shape, vehiculos.shape, incidencias.shape,
    )

    envios = envios.copy()
    rutas = rutas.copy()
    vehiculos = vehiculos.copy()

    # 🔥 FIX CRÍTICO: asegurar unicidad en claves de join
    rutas = rutas.dropna(subset=["id_ruta"]).drop_duplicates(subset=["id_ruta"])
    vehiculos = vehiculos.dropna(subset=["id_vehiculo"]).drop_duplicates(subset=["id_vehiculo"])

    incidencias_agg = (
        incidencias
        .groupby("id_envio", observed=True)
        .agg(
            total_incidencias=("id_incidencia", "count"),
            costo_impacto_total=("costo_impacto", "sum"),
            tipos_incidencia=("tipo_incidencia", lambda x: "|".join(x.dropna().unique())),
        )
        .reset_index()
    )

    logger.info("[join] Incidencias agregadas: %d", len(incidencias_agg))

    df = envios.merge(
        rutas,
        on="id_ruta",
        how="left",
        validate="m:1"
    )

    logger.info("[join] Env?os sin ruta: %d", df["id_ruta"].isna().sum())

    df = df.merge(
        vehiculos,
        on="id_vehiculo",
        how="left",
        validate="m:1"
    )

    df = df.merge(
        incidencias_agg,
        on="id_envio",
        how="left"
    )

    df["total_incidencias"] = df["total_incidencias"].fillna(0).astype("Int64")
    df["costo_impacto_total"] = df["costo_impacto_total"].fillna(0.0)
    df["tipos_incidencia"] = df["tipos_incidencia"].fillna("Sin Incidencia")

    logger.info(
        "[join] FINAL shape: %s | incidencias: %.1f%%",
        df.shape,
        (df["total_incidencias"] > 0).mean() * 100,
    )

    return df


def crear_features(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    if {"fecha_envio", "fecha_entrega"} <= set(df.columns):
        f_envio = pd.to_datetime(df["fecha_envio"], errors="coerce")
        f_entrega = pd.to_datetime(df["fecha_entrega"], errors="coerce")
        df["dias_en_transito"] = (f_entrega - f_envio).dt.days.clip(0, 60)

    if {"peso_kg", "capacidad_kg"} <= set(df.columns):
        df["eficiencia_peso"] = df["peso_kg"] / df["capacidad_kg"].replace(0, np.nan)

    if {"volumen_m3", "capacidad_m3"} <= set(df.columns):
        df["eficiencia_volumen"] = df["volumen_m3"] / df["capacidad_m3"].replace(0, np.nan)

    if {"peaje_total", "distancia_km"} <= set(df.columns):
        df["costo_por_km"] = df["peaje_total"] / df["distancia_km"].replace(0, np.nan)

    if "total_incidencias" in df.columns:
        df["tiene_incidencia"] = (df["total_incidencias"] > 0).astype("int8")

    return df


def codificar_categoricas(df: pd.DataFrame, parameters: dict) -> pd.DataFrame:

    df = df.copy()

    cols = parameters.get("encoding", {}).get("onehot_cols", ["tipo_carga_norm"])

    for col in cols:
        if col in df.columns:
            df = pd.concat([df, pd.get_dummies(df[col].astype(str), prefix=col)], axis=1)

    if "estado" in df.columns:
        df["estado_encoded"] = LabelEncoder().fit_transform(df["estado"].astype(str))

    if "tipo_via" in df.columns:
        df["tipo_via_encoded"] = LabelEncoder().fit_transform(df["tipo_via"].astype(str))

    return df


def normalizar_numericas(df: pd.DataFrame, parameters: dict) -> pd.DataFrame:

    df = df.copy()

    cols = parameters.get("normalizacion", {}).get("columnas_minmax", [])
    cols = [c for c in cols if c in df.columns]

    if cols:
        scaler = MinMaxScaler()

        df_norm = pd.DataFrame(
            scaler.fit_transform(df[cols]),
            columns=[f"{c}_norm" for c in cols],
            index=df.index,
        )

        df = pd.concat([df, df_norm], axis=1)

    return df
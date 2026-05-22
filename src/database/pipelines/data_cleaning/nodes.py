"""
Nodos del pipeline de Data Cleaning (AD 1.2).

Responsabilidad:
    - Estandarizar formatos de fecha (mixtos dd/mm/yyyy y yyyy-mm-dd).
    - Eliminar duplicados exactos.
    - Tratar valores nulos con imputación estratégica.
    - Detectar y tratar outliers con el método IQR de Tukey.
    - Normalizar texto (strip, title-case) de columnas categóricas.

Técnicas de optimización Pandas aplicadas:
    - Vectorización: operaciones .str.strip(), .str.title() sobre Series completas.
    - Broadcasting booleano: máscaras IQR para filtrar outliers sin loops.
    - .loc[] con condiciones múltiples para filtros avanzados.
    - pd.to_datetime con infer_datetime_format=False para robustez.
    - .fillna() vectorizado para imputación eficiente.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers privados reutilizables
# ---------------------------------------------------------------------------

def _normalizar_fechas(s: pd.Series, nombre: str) -> pd.Series:
    import pandas as pd
    import numpy as np

    s = s.astype(str).str.strip()

    s = s.replace({
        "nan": np.nan,
        "None": np.nan,
        "": np.nan,
        "NaT": np.nan
    })

    resultado = pd.to_datetime(s, errors="coerce", dayfirst=True)

    mask = resultado.isna()
    if mask.any():
        resultado.loc[mask] = pd.to_datetime(
            s[mask],
            errors="coerce",
            format=None
        )

    return resultado


def _eliminar_outliers_iqr(
    df: pd.DataFrame,
    columnas: list[str],
    multiplicador: float = 2.0,
) -> tuple[pd.DataFrame, int]:
    """
    Elimina filas con outliers extremos usando el método IQR de Tukey.

    Broadcasting booleano:
        Se construye una máscara combinada para TODAS las columnas a la vez
        usando operaciones vectorizadas (|, &) en lugar de bucles Python.

    Parámetros
    ----------
    df          : DataFrame a filtrar.
    columnas    : Lista de columnas numéricas a revisar.
    multiplicador : Factor IQR (2.0 = conservador para logística).

    Retorna
    -------
    (DataFrame limpio, cantidad de filas eliminadas)
    """
    mask_total = pd.Series([False] * len(df), index=df.index)  # máscara acumulada

    for col in columnas:
        if col not in df.columns:
            continue
        serie = df[col].dropna()
        if serie.empty:
            continue
        q1 = serie.quantile(0.25)
        q3 = serie.quantile(0.75)
        iqr = q3 - q1
        limite_inf = q1 - multiplicador * iqr
        limite_sup = q3 + multiplicador * iqr

        # Broadcasting: operación sobre toda la columna de una vez
        col_mask = (df[col] < limite_inf) | (df[col] > limite_sup)
        outliers_en_col = col_mask.sum()
        if outliers_en_col > 0:
            logger.info(
                "Outliers en '%s': %d filas (límites: [%.2f, %.2f])",
                col, outliers_en_col, limite_inf, limite_sup,
            )
        mask_total = mask_total | col_mask  # acumular con broadcasting OR

    n_outliers = mask_total.sum()
    df_limpio = df.loc[~mask_total].copy()  # filtro avanzado con .loc[]
    return df_limpio, int(n_outliers)


def _normalizar_texto_col(serie: pd.Series) -> pd.Series:
    """
    Estandariza una columna de texto: strip + title-case.

    Vectorizado: .str.strip() y .str.title() operan sobre toda la Serie.
    """
    return serie.str.strip().str.title()


# ---------------------------------------------------------------------------
# Nodos públicos
# ---------------------------------------------------------------------------

def limpiar_envios(envios: pd.DataFrame, parameters: dict) -> pd.DataFrame:
    df = envios.copy()
    n_inicial = len(df)

    logger.info("[envios] Registros iniciales: %d", n_inicial)

    # =========================
    # 1. DUPLICADOS
    # =========================
    df = df.drop_duplicates()

    # =========================
    # 2. IDS (FIX CRÍTICO PARA JOINS)
    # =========================
    for col in ["id_envio", "id_ruta", "id_vehiculo"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].where(df[col] > 0)
            df[col] = df[col].astype("Int64")

    # =========================
    # 3. FECHAS (CRÍTICO)
    # =========================
    df["fecha_envio"] = _normalizar_fechas(df["fecha_envio"], "envios.fecha_envio")
    df["fecha_entrega"] = _normalizar_fechas(df["fecha_entrega"], "envios.fecha_entrega")

    df["fecha_envio"] = pd.to_datetime(df["fecha_envio"], errors="coerce")
    df["fecha_entrega"] = pd.to_datetime(df["fecha_entrega"], errors="coerce")

    nat_ratio = df["fecha_envio"].isna().mean()
    logger.warning("[envios] NaT ratio fecha_envio: %.2f%%", nat_ratio * 100)

    # 🔥 GUARDRAIL
    if nat_ratio > 0.10:
        logger.error("[ERROR] M?S DE 10% NaT EN FECHA_ENVIO")

    # =========================
    # 4. TEXTO
    # =========================
    df["tipo_carga_norm"] = _normalizar_texto_col(df["tipo_carga"])
    df["estado"] = _normalizar_texto_col(df["estado"])

    # =========================
    # 5. NUMÉRICOS
    # =========================
    for col in ["peso_kg", "volumen_m3"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # =========================
    # 6. RUTA (IMPUTACIÓN)
    # =========================
    moda_ruta = df["id_ruta"].mode()
    if not moda_ruta.empty:
        df["id_ruta"] = df["id_ruta"].fillna(moda_ruta.iloc[0])

    # =========================
    # 7. OUTLIERS
    # =========================
    columnas_outlier = parameters.get("outliers", {}).get("columnas_peso", ["peso_kg", "volumen_m3"])
    mult = parameters.get("outliers", {}).get("iqr_multiplier", 2.0)

    df, _ = _eliminar_outliers_iqr(df, columnas_outlier, mult)

    # =========================
    # 8. TIPOS FINALES
    # =========================
    df["tipo_carga_norm"] = df["tipo_carga_norm"].astype("category")
    df["estado"] = df["estado"].astype("category")

    # =========================
    # 9. CHECK FINAL (CLAVE PARA TU PROBLEMA)
    # =========================
    logger.warning(
        "[envios] ids nulos -> id_envio:%d id_ruta:%d id_vehiculo:%d",
        df["id_envio"].isna().sum(),
        df["id_ruta"].isna().sum(),
        df["id_vehiculo"].isna().sum(),
    )

    logger.info("[envios] Limpieza OK: %d -> %d", n_inicial, len(df))

    return df

def limpiar_rutas(rutas: pd.DataFrame, parameters: dict) -> pd.DataFrame:
    """
    Nodo 2: Limpieza del dataset de rutas.

    Transformaciones:
        1. Eliminar duplicados.
        2. Normalizar tipo_via: strip + title-case.
        3. Imputar origen nulo con valor de parámetro.
        4. Limpiar id_ruta a entero.

    Parámetros
    ----------
    rutas      : DataFrame crudo de rutas.
    parameters : Parámetros del catalog.

    Retorna
    -------
    pd.DataFrame limpio guardado en 02_intermediate/rutas_clean.csv.
    """
    df = rutas.copy()
    n_inicial = len(df)
    logger.info("[rutas] Registros iniciales: %d", n_inicial)

    # 1. Eliminar duplicados
    df = df.drop_duplicates()

    # 2. Normalizar tipo_via (vectorizado: strip + title-case)
    if "tipo_via" in df.columns:
        df["tipo_via"] = _normalizar_texto_col(df["tipo_via"])

    # 3. Imputar origen nulo con parámetro
    origen_default = parameters.get("imputacion", {}).get("origen_desconocido", "Desconocido")
    n_sin_origen = df["origen"].isna().sum()
    df["origen"] = df["origen"].fillna(origen_default)
    if n_sin_origen > 0:
        logger.info("[rutas] %d registros con origen imputado como '%s'.", n_sin_origen, origen_default)

    # 4. Limpiar id_ruta a entero
    df["id_ruta"] = pd.to_numeric(df["id_ruta"], errors="coerce").astype("Int64")

    # 5. Optimizar tipos de datos (gestión de memoria)
    if "tipo_via" in df.columns:
        df["tipo_via"] = df["tipo_via"].astype("category")

    logger.info("[rutas] Limpieza completada. %d -> %d registros.", n_inicial, len(df))
    return df


def limpiar_vehiculos(vehiculos: pd.DataFrame, parameters: dict) -> pd.DataFrame:
    """
    Nodo 3: Limpieza del dataset de vehículos.

    Transformaciones:
        1. Eliminar duplicados.
        2. Imputar capacidad_m3 nula con la mediana por tipo de vehículo
           (groupby + transform: operación vectorizada avanzada).
        3. Normalizar estado_vehiculo.
        4. Limpiar id_vehiculo a entero.

    Parámetros
    ----------
    vehiculos  : DataFrame crudo de vehículos.
    parameters : Parámetros del catalog.

    Retorna
    -------
    pd.DataFrame limpio guardado en 02_intermediate/vehiculos_clean.csv.
    """
    df = vehiculos.copy()
    n_inicial = len(df)
    logger.info("[vehiculos] Registros iniciales: %d", n_inicial)

    # 1. Eliminar duplicados
    df = df.drop_duplicates()

    # 2. Imputar capacidad_m3 con mediana por tipo (groupby + transform vectorizado)
    # .transform() devuelve una Serie del mismo índice, permitiendo fillna directo
    n_sin_cap = df["capacidad_m3"].isna().sum()
    if n_sin_cap > 0:
        mediana_por_tipo = df.groupby("tipo", observed=True)["capacidad_m3"].transform("median")
        df["capacidad_m3"] = df["capacidad_m3"].fillna(mediana_por_tipo)
        # Si aún quedan nulos (todos NaN por tipo), usar mediana global
        df["capacidad_m3"] = df["capacidad_m3"].fillna(df["capacidad_m3"].median())
        logger.info("[vehiculos] %d valores de capacidad_m3 imputados con mediana por tipo.", n_sin_cap)

    # 3. Normalizar estado_vehiculo y tipo (strip + title-case vectorizado)
    for col in ["estado_vehiculo", "tipo"]:
        if col in df.columns:
            df[col] = _normalizar_texto_col(df[col])

    # 4. Limpiar ids
    df["id_vehiculo"] = pd.to_numeric(df["id_vehiculo"], errors="coerce").astype("Int64")

    # 5. Optimizar memoria
    df["tipo"] = df["tipo"].astype("category")
    df["estado_vehiculo"] = df["estado_vehiculo"].astype("category")

    logger.info("[vehiculos] Limpieza completada. %d -> %d registros.", n_inicial, len(df))
    return df


def limpiar_incidencias(incidencias: pd.DataFrame, parameters: dict) -> pd.DataFrame:
    """
    Nodo 4: Limpieza del dataset de incidencias.

    Transformaciones:
        1. Eliminar duplicados.
        2. Normalizar fechas.
        3. Normalizar tipo_incidencia y descripcion.
        4. Limpiar costo_impacto de caracteres especiales.
        5. Tratar costo_impacto nulo con mediana.

    Parámetros
    ----------
    incidencias : DataFrame crudo de incidencias.
    parameters  : Parámetros del catalog.

    Retorna
    -------
    pd.DataFrame limpio guardado en 02_intermediate/incidencias_clean.csv.
    """
    df = incidencias.copy()
    n_inicial = len(df)
    logger.info("[incidencias] Registros iniciales: %d", n_inicial)

    # 1. Eliminar duplicados
    df = df.drop_duplicates()

    # 2. Normalizar fechas (vectorizado)
    df["fecha"] = _normalizar_fechas(df["fecha"], "incidencias.fecha")

    # 3. Normalizar columnas de texto (vectorizado)
    for col in ["tipo_incidencia", "descripcion"]:
        if col in df.columns:
            df[col] = _normalizar_texto_col(df[col])

    # 4. Limpiar costo_impacto: remover $, ~, y "aprox"
    if "costo_impacto" in df.columns:
        # Convertir a string primero, luego limpiar
        df["costo_impacto"] = df["costo_impacto"].astype(str)
        # Remover $, ~, y " aprox" 
        df["costo_impacto"] = df["costo_impacto"].str.replace(r'[\$~]', '', regex=True)
        df["costo_impacto"] = df["costo_impacto"].str.replace(r'\s*aprox.*$', '', regex=True)
        # Convertir a numérico (errores → NaN)
        df["costo_impacto"] = pd.to_numeric(df["costo_impacto"], errors="coerce")
        logger.info("[incidencias] Columna costo_impacto limpiada de caracteres especiales.")

    # 5. Imputar costo_impacto nulo con mediana
    n_sin_costo = df["costo_impacto"].isna().sum()
    if n_sin_costo > 0 and df["costo_impacto"].notna().sum() > 0:
        mediana_costo = df["costo_impacto"].median()
        df["costo_impacto"] = df["costo_impacto"].fillna(mediana_costo)
        logger.info("[incidencias] %d costos imputados con mediana: %.2f.", n_sin_costo, mediana_costo)

    # 6. Limpiar ids
    df["id_incidencia"] = pd.to_numeric(df["id_incidencia"], errors="coerce").astype("Int64")
    df["id_envio"] = pd.to_numeric(df["id_envio"], errors="coerce").astype("Int64")

    # 7. Optimizar memoria
    df["tipo_incidencia"] = df["tipo_incidencia"].astype("category")

    logger.info("[incidencias] Limpieza completada. %d -> %d registros.", n_inicial, len(df))
    return df

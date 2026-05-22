"""
Nodos del pipeline de Model Input (Fase 1 — EP2 SCY1101).

Responsabilidad:
    - Construir datasets ML limpios desde master_envios.
    - Eliminar columnas con data leakage confirmado.
    - Reconstruir categorías limpias de tipo_carga.
    - Imputar nulos con criterio técnico documentado.
    - Separar dataset clasificación (tiene_incidencia) y regresión (dias_en_transito).
    - Generar reporte de preparación ML.

Data Leakage eliminado (README maestro sección 9):
    - total_incidencias     → resume incidencias OCURRIDAS (respuesta al target)
    - costo_impacto_total   → costo de incidencias OCURRIDAS (derivada del evento)
    - tipos_incidencia      → describe qué tipo de incidencia ocurrió
    - costo_impacto_total_norm → versión normalizada del anterior

Variables post-evento excluidas de clasificación:
    - fecha_entrega    → disponible solo DESPUÉS del envío
    - dias_en_transito → calculado de fecha_envio → fecha_entrega (post-evento)
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constantes — fuente de verdad para exclusiones y leakage
# ---------------------------------------------------------------------------

COLUMNAS_LEAKAGE = [
    "total_incidencias",
    "costo_impacto_total",
    "tipos_incidencia",
    "costo_impacto_total_norm",
]

COLUMNAS_POST_EVENTO = [
    "fecha_entrega",
    "dias_en_transito",
]

COLUMNAS_REDUNDANTES = [
    "tipo_carga",          # raw sin normalizar, redundante con tipo_carga_norm
    "placa",               # identificador único de vehículo (alta cardinalidad)
    "estado_encoded",      # encoding anterior, redundante con estado string
    "tipo_via_encoded",    # encoding anterior, redundante con tipo_via string
    "peso_kg_norm",        # normalización previa (sklearn lo hará en Fase 3)
    "volumen_m3_norm",
    "distancia_km_norm",
    "tiempo_estimado_hrs_norm",
    "peaje_total_norm",
    "km_recorridos_norm",
]

FEATURES_NUMERICAS = [
    "peso_kg", "volumen_m3", "distancia_km", "tiempo_estimado_hrs",
    "peaje_total", "capacidad_kg", "capacidad_m3", "km_recorridos",
    "eficiencia_peso", "eficiencia_volumen", "costo_por_km",
]

FEATURES_CATEGORICAS = [
    "tipo_carga_norm", "tipo_via", "tipo", "estado_vehiculo", "estado",
    "origen", "destino",
]

TARGET_CLASIFICACION = "tiene_incidencia"
TARGET_REGRESION = "dias_en_transito"


# ---------------------------------------------------------------------------
# Helpers privados
# ---------------------------------------------------------------------------

def _encontrar_col_anio(df: pd.DataFrame) -> str | None:
    """Localiza la columna año_fabricacion independiente del encoding del nombre."""
    for col in df.columns:
        if "fabricaci" in col.lower():
            return col
    return None


def _eliminar_dummies_sucias(df: pd.DataFrame) -> pd.DataFrame:
    """
    Elimina las columnas dummies sucias de tipo_carga generadas por el
    pipeline de transformación antes de normalizar texto.

    FIX CRITICO: tipo_carga_norm NO debe eliminarse aunque empiece con
    'tipo_carga_'. Se excluye explícitamente del filtro.
    """
    cols_sucias = [
        c for c in df.columns
        if c.startswith("tipo_carga_") and c != "tipo_carga_norm"
    ]
    n_eliminadas = len(cols_sucias)
    df = df.drop(columns=cols_sucias, errors="ignore")
    logger.info(
        "[model_input] %d columnas dummies sucias eliminadas. tipo_carga_norm conservada.",
        n_eliminadas,
    )
    return df


def _imputar_numericas(df: pd.DataFrame, cols: list[str]) -> dict:
    """
    Imputa nulos en columnas numéricas con la mediana.
    Retorna registro de imputaciones para el reporte.
    """
    registro = {}
    for col in cols:
        if col not in df.columns:
            continue
        n_nulos = int(df[col].isna().sum())
        if n_nulos > 0:
            mediana = df[col].median()
            df[col] = df[col].fillna(mediana)
            registro[col] = {"estrategia": "mediana", "n_imputados": n_nulos, "valor": round(float(mediana), 4)}
            logger.info(
                "[model_input] '%s': %d nulos imputados con mediana=%.3f.",
                col, n_nulos, mediana,
            )
    return registro


def _imputar_categoricas(df: pd.DataFrame, cols: list[str]) -> dict:
    """
    Imputa nulos en columnas categóricas con la moda.
    Retorna registro de imputaciones para el reporte.
    """
    registro = {}
    for col in cols:
        if col not in df.columns:
            continue
        n_nulos = int(df[col].isna().sum())
        if n_nulos > 0:
            moda = df[col].mode()
            if not moda.empty:
                valor_moda = moda.iloc[0]
                df[col] = df[col].fillna(valor_moda)
                registro[col] = {"estrategia": "moda", "n_imputados": n_nulos, "valor": str(valor_moda)}
                logger.info(
                    "[model_input] '%s': %d nulos imputados con moda='%s'.",
                    col, n_nulos, valor_moda,
                )
    return registro


def _extraer_features_temporales(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extrae features temporales desde fecha_envio.

    Justificación de negocio: el mes y día de la semana del envío pueden
    correlacionar con incidencias (temporadas altas, fines de semana, etc.).
    Estas features son pre-evento: disponibles antes de que ocurra el envío.
    """
    if "fecha_envio" not in df.columns:
        return df

    fechas = pd.to_datetime(df["fecha_envio"], errors="coerce")
    df["mes_envio"] = fechas.dt.month.astype("float64")
    df["dia_semana_envio"] = fechas.dt.dayofweek.astype("float64")
    df["trimestre_envio"] = fechas.dt.quarter.astype("float64")

    logger.info(
        "[model_input] Features temporales extraídas: mes_envio, dia_semana_envio, trimestre_envio."
    )
    return df


def _filtrar_rangos_modelado(df: pd.DataFrame, parameters: dict) -> pd.DataFrame:
    """
    Excluye registros fuera de rangos tecnicos antes de entrenar modelos.

    Justificacion: el README maestro indica que no se debe entrenar sobre
    registros con rangos no revisados. Para Fase 1 se eliminan del dataset de
    modelado los casos que exceden limites operativos definidos en parameters.
    """
    validacion = parameters.get("validacion", {})
    reglas = {
        "peso_kg": (
            validacion.get("peso_kg_min", 1.0),
            validacion.get("peso_kg_max", 25000.0),
        ),
        "distancia_km": (
            validacion.get("distancia_km_min", 1.0),
            validacion.get("distancia_km_max", 5000.0),
        ),
        "eficiencia_peso": (
            validacion.get("eficiencia_peso_min", 0.0),
            validacion.get("eficiencia_peso_max", 10.0),
        ),
    }

    mascara_fuera_rango = pd.Series(False, index=df.index)
    detalle = {}
    for col, (minimo, maximo) in reglas.items():
        if col not in df.columns:
            continue
        fuera = df[col].isna() | (df[col] < minimo) | (df[col] > maximo)
        detalle[col] = int(fuera.sum())
        mascara_fuera_rango = mascara_fuera_rango | fuera

    n_fuera = int(mascara_fuera_rango.sum())
    if n_fuera > 0:
        logger.warning(
            "[model_input] %d filas excluidas por rangos no aptos para ML: %s",
            n_fuera,
            detalle,
        )
        df = df[~mascara_fuera_rango].copy()
    else:
        logger.info("[model_input] Rangos revisados: sin exclusiones para ML.")

    return df


# ---------------------------------------------------------------------------
# Nodos públicos
# ---------------------------------------------------------------------------

def preparar_features_base(
    master: pd.DataFrame,
    parameters: dict,
) -> pd.DataFrame:
    """
    Nodo 1: Construye el dataset base para ML desde master_envios.

    Transformaciones aplicadas:
        1. Elimina filas sin id_envio (registros no identificables).
        2. Elimina columnas con data leakage confirmado.
        3. Elimina columnas post-evento (fecha_entrega, dias_en_transito).
           NOTA: dias_en_transito se mantiene temporalmente para el nodo de regresión.
        4. Elimina columnas redundantes y dummies sucias.
        5. Extrae features temporales desde fecha_envio.
        6. Localiza columna año_fabricacion (encoding-safe).
        7. Imputa nulos en features numéricas con mediana.
        8. Imputa nulos en features categóricas con moda.
        9. Corrige tipos de datos críticos.

    Parámetros
    ----------
    master     : DataFrame maestro (03_primary/master_envios.csv).
    parameters : Parámetros del proyecto (parameters.yml).

    Retorna
    -------
    pd.DataFrame listo para separar en datasets de clasificación y regresión.
    """
    df = master.copy()
    n_inicial = len(df)
    logger.info("[model_input] Iniciando preparaci?n de features base. Filas: %d, Cols: %d", n_inicial, df.shape[1])

    # 1. Eliminar filas sin id_envio (no identificables)
    n_antes = len(df)
    df = df[df["id_envio"].notna()].copy()
    n_eliminados_id = n_antes - len(df)
    if n_eliminados_id > 0:
        logger.warning(
            "[model_input] %d filas eliminadas por id_envio nulo (registros no identificables).",
            n_eliminados_id,
        )

    # 2. Eliminar columnas con data leakage (README maestro sección 9)
    cols_leakage_presentes = [c for c in COLUMNAS_LEAKAGE if c in df.columns]
    df = df.drop(columns=cols_leakage_presentes, errors="ignore")
    logger.info(
        "[model_input] Leakage eliminado: %s", cols_leakage_presentes
    )

    # 3. Eliminar columnas redundantes y dummies sucias
    df = df.drop(columns=COLUMNAS_REDUNDANTES, errors="ignore")
    df = _eliminar_dummies_sucias(df)

    # 4. Extraer features temporales ANTES de eliminar fecha_envio
    df = _extraer_features_temporales(df)
    df = df.drop(columns=["fecha_envio"], errors="ignore")

    # 5. Localizar columna año_fabricacion (nombre puede variar por encoding)
    col_anio = _encontrar_col_anio(df)
    if col_anio and col_anio != "año_fabricacion":
        df = df.rename(columns={col_anio: "año_fabricacion"})
        logger.info("[model_input] Columna '%s' renombrada a 'a?o_fabricacion'.", col_anio)

    # 6. Imputar features numéricas (mediana)
    cols_num = [c for c in FEATURES_NUMERICAS if c in df.columns]
    if "año_fabricacion" in df.columns:
        cols_num.append("año_fabricacion")
    _imputar_numericas(df, cols_num)

    # 7. Imputar features categóricas (moda)
    _imputar_categoricas(df, [c for c in FEATURES_CATEGORICAS if c in df.columns])

    # 8. Imputar id_vehiculo nulo con moda
    if "id_vehiculo" in df.columns and df["id_vehiculo"].isna().any():
        moda_vehiculo = df["id_vehiculo"].mode()
        if not moda_vehiculo.empty:
            n_nulos_v = int(df["id_vehiculo"].isna().sum())
            df["id_vehiculo"] = df["id_vehiculo"].fillna(moda_vehiculo.iloc[0])
            logger.info("[model_input] 'id_vehiculo': %d nulos imputados con moda.", n_nulos_v)

    # 9. Imputar features temporales nulas (fecha_envio era nula en esas filas)
    for col_temp in ["mes_envio", "dia_semana_envio", "trimestre_envio"]:
        if col_temp in df.columns and df[col_temp].isna().any():
            mediana_t = df[col_temp].median()
            n_t = int(df[col_temp].isna().sum())
            df[col_temp] = df[col_temp].fillna(mediana_t)
            logger.info("[model_input] '%s': %d nulos imputados con mediana=%.1f.", col_temp, n_t, mediana_t)

    # 10. Corregir tipos de datos
    for col in ["id_envio", "id_vehiculo", "id_ruta"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if TARGET_CLASIFICACION in df.columns:
        df[TARGET_CLASIFICACION] = df[TARGET_CLASIFICACION].astype(int)

    # 11. Filtrar rangos no aptos para machine learning
    df = _filtrar_rangos_modelado(df, parameters)

    # Verificacion final de nulos
    total_nulls = df.isnull().sum().sum()
    if total_nulls > 0:
        cols_con_nulos = df.columns[df.isnull().any()].tolist()
        logger.warning(
            "[model_input] Quedan %d nulos en columnas: %s",
            total_nulls, cols_con_nulos,
        )

    logger.info(
        "[model_input] Dataset base listo: %d filas, %d columnas, %d nulos totales.",
        len(df), df.shape[1], total_nulls,
    )
    return df


def preparar_clasificacion(
    df_base: pd.DataFrame,
    parameters: dict,
) -> pd.DataFrame:
    """
    Nodo 2: Dataset de clasificación binaria.

    Pregunta de negocio: ¿Podemos anticipar si un envío tendrá incidencia?
    Target: tiene_incidencia (0 = sin incidencia, 1 = con incidencia)

    Exclusiones adicionales (post-evento para predicción pre-despacho):
        - fecha_entrega: solo disponible después del envío
        - dias_en_transito: calculado post-evento (fecha_entrega - fecha_envio)

    Las columnas excluidas quedan documentadas en el reporte model_input_report.

    Parámetros
    ----------
    df_base    : DataFrame base (salida de preparar_features_base).
    parameters : Parámetros del proyecto.

    Retorna
    -------
    pd.DataFrame guardado en 05_model_input/model_input_clasificacion.csv.
    """
    df = df_base.copy()

    # Excluir columnas post-evento — no disponibles antes del envío
    cols_post = [c for c in COLUMNAS_POST_EVENTO if c in df.columns]
    df = df.drop(columns=cols_post, errors="ignore")

    # Verificar target
    if TARGET_CLASIFICACION not in df.columns:
        raise ValueError(
            f"[model_input] Target '{TARGET_CLASIFICACION}' no encontrado en df_base."
        )

    n_total = len(df)
    n_con_incidencia = int(df[TARGET_CLASIFICACION].sum())
    n_sin_incidencia = n_total - n_con_incidencia
    balance = round(n_con_incidencia / n_total * 100, 2)

    logger.info(
        "[model_input] Clasificación — Filas: %d | Con incidencia: %d (%.1f%%) | Sin: %d (%.1f%%)",
        n_total, n_con_incidencia, balance, n_sin_incidencia, 100 - balance,
    )

    if balance < 10 or balance > 90:
        logger.warning(
            "[model_input] ⚠️ Clases desbalanceadas (%.1f%% positivos). "
            "Considerar class_weight='balanced' o SMOTE en Fase 3.",
            balance,
        )

    logger.info(
        "[model_input] Dataset clasificación listo: %d filas, %d columnas.",
        len(df), df.shape[1],
    )
    return df


def preparar_regresion(
    df_base: pd.DataFrame,
    parameters: dict,
) -> pd.DataFrame:
    """
    Nodo 3: Dataset de regresión.

    Pregunta de negocio: ¿Podemos estimar los días de tránsito de un envío?
    Target: dias_en_transito

    Exclusiones:
        - Filas con dias_en_transito nulo (91 filas — no hay target válido).
        - fecha_entrega: es la variable desde la cual se calcula el target
          (usar fecha_entrega directamente sería leakage del target de regresión).
        - tiene_incidencia: target de clasificación, no debe estar en regresión.

    Parámetros
    ----------
    df_base    : DataFrame base (salida de preparar_features_base).
    parameters : Parámetros del proyecto.

    Retorna
    -------
    pd.DataFrame guardado en 05_model_input/model_input_regresion.csv.
    """
    df = df_base.copy()

    # Eliminar fecha_entrega (derivación directa del target)
    df = df.drop(columns=["fecha_entrega", TARGET_CLASIFICACION], errors="ignore")

    # Verificar target
    if TARGET_REGRESION not in df.columns:
        raise ValueError(
            f"[model_input] Target '{TARGET_REGRESION}' no encontrado en df_base."
        )

    # Eliminar filas sin target (no hay forma de imputar el target)
    n_antes = len(df)
    df = df[df[TARGET_REGRESION].notna()].copy()
    n_eliminados = n_antes - len(df)
    if n_eliminados > 0:
        logger.warning(
            "[model_input] %d filas eliminadas por dias_en_transito nulo "
            "(no se puede imputar el target).",
            n_eliminados,
        )

    # Estadísticas del target
    media = df[TARGET_REGRESION].mean()
    mediana = df[TARGET_REGRESION].median()
    std = df[TARGET_REGRESION].std()
    logger.info(
        "[model_input] Regresión — Filas: %d | dias_en_transito: media=%.2f, mediana=%.2f, std=%.2f",
        len(df), media, mediana, std,
    )

    logger.info(
        "[model_input] Dataset regresión listo: %d filas, %d columnas.",
        len(df), df.shape[1],
    )
    return df


def generar_reporte_model_input(
    master: pd.DataFrame,
    df_clf: pd.DataFrame,
    df_reg: pd.DataFrame,
) -> pd.DataFrame:
    """
    Nodo 4: Genera reporte de trazabilidad de la preparación del dataset ML.

    El reporte documenta cada decisión técnica tomada:
        - Filas iniciales y finales por dataset.
        - Columnas eliminadas y motivo.
        - Balance de clases para clasificación.
        - Estadísticas del target de regresión.

    Este reporte es evidencia académica para la rúbrica EP2.

    Parámetros
    ----------
    master  : Dataset maestro original (para contar diferencias).
    df_clf  : Dataset de clasificación final.
    df_reg  : Dataset de regresión final.

    Retorna
    -------
    pd.DataFrame guardado en 08_reporting/model_input_report.csv.
    """
    filas = []

    # --- Resumen general ---
    filas.append({
        "seccion": "general",
        "item": "filas_master_original",
        "valor": str(len(master)),
        "detalle": "Filas en master_envios antes de preparación ML",
    })
    filas.append({
        "seccion": "general",
        "item": "filas_eliminadas_id_nulo",
        "valor": str(master["id_envio"].isna().sum()),
        "detalle": "Filas con id_envio nulo — registros no identificables, eliminados",
    })
    filas.append({
        "seccion": "general",
        "item": "filas_excluidas_rangos_modelado",
        "valor": str(max(int(master["id_envio"].notna().sum()) - len(df_clf), 0)),
        "detalle": "Filas excluidas del dataset ML por distancia_km, peso_kg o eficiencia_peso fuera de rango",
    })
    filas.append({
        "seccion": "general",
        "item": "columnas_master_original",
        "valor": str(master.shape[1]),
        "detalle": "Columnas en master_envios antes de preparación",
    })

    # --- Data leakage ---
    for col in COLUMNAS_LEAKAGE:
        filas.append({
            "seccion": "leakage_eliminado",
            "item": col,
            "valor": "EXCLUIDA",
            "detalle": f"Data leakage: '{col}' contiene información derivada de las incidencias",
        })

    # --- Post-evento (clasificación) ---
    filas.append({
        "seccion": "post_evento_clasificacion",
        "item": "fecha_entrega",
        "valor": "EXCLUIDA",
        "detalle": "Post-evento: disponible solo DESPUES del envio. No usable para prediccion pre-despacho",
    })
    filas.append({
        "seccion": "post_evento_clasificacion",
        "item": "dias_en_transito",
        "valor": "EXCLUIDA",
        "detalle": "Post-evento: calculado de fecha_entrega - fecha_envio. Target de regresion, no de clasificacion",
    })

    # --- Post-evento (regresión) ---
    filas.append({
        "seccion": "post_evento_regresion",
        "item": "fecha_entrega",
        "valor": "EXCLUIDA",
        "detalle": "Derivacion directa del target dias_en_transito -- excluida para evitar leakage del target",
    })

    # --- Dummies sucias ---
    cols_sucias = [c for c in master.columns if c.startswith("tipo_carga_")]
    filas.append({
        "seccion": "dummies_sucias",
        "item": f"{len(cols_sucias)} columnas tipo_carga_*",
        "valor": "EXCLUIDAS",
        "detalle": (
            "20 dummies generadas con texto sin normalizar (4 variantes por categoria). "
            "Se usa tipo_carga_norm como feature categorica limpia para Fase 3"
        ),
    })

    # --- Columnas redundantes ---
    for col in COLUMNAS_REDUNDANTES:
        filas.append({
            "seccion": "redundantes_excluidas",
            "item": col,
            "valor": "EXCLUIDA",
            "detalle": "Columna redundante o normalizacion previa (sklearn Pipeline lo hara en Fase 3)",
        })

    # --- Dataset clasificación ---
    n_pos = int(df_clf[TARGET_CLASIFICACION].sum()) if TARGET_CLASIFICACION in df_clf.columns else 0
    n_neg = len(df_clf) - n_pos
    filas.extend([
        {
            "seccion": "clasificacion",
            "item": "filas_finales",
            "valor": str(len(df_clf)),
            "detalle": f"Filas en model_input_clasificacion.csv",
        },
        {
            "seccion": "clasificacion",
            "item": "columnas_finales",
            "valor": str(df_clf.shape[1]),
            "detalle": "Columnas en model_input_clasificacion.csv (incluye target)",
        },
        {
            "seccion": "clasificacion",
            "item": "target",
            "valor": TARGET_CLASIFICACION,
            "detalle": "Variable objetivo para clasificación binaria",
        },
        {
            "seccion": "clasificacion",
            "item": "balance_clases",
            "valor": f"{n_pos} positivos / {n_neg} negativos ({round(n_pos/max(len(df_clf),1)*100,1)}%)",
            "detalle": "Balance de clases -- si <10% o >90% considerar class_weight='balanced'",
        },
    ])

    # --- Dataset regresión ---
    if TARGET_REGRESION in df_reg.columns:
        filas.extend([
            {
                "seccion": "regresion",
                "item": "filas_finales",
                "valor": str(len(df_reg)),
                "detalle": "Filas en model_input_regresion.csv (excluye dias_en_transito nulos)",
            },
            {
                "seccion": "regresion",
                "item": "columnas_finales",
                "valor": str(df_reg.shape[1]),
                "detalle": "Columnas en model_input_regresion.csv (incluye target)",
            },
            {
                "seccion": "regresion",
                "item": "target",
                "valor": TARGET_REGRESION,
                "detalle": "Variable objetivo para regresión",
            },
            {
                "seccion": "regresion",
                "item": "target_stats",
                "valor": (
                    f"media={df_reg[TARGET_REGRESION].mean():.2f} | "
                    f"mediana={df_reg[TARGET_REGRESION].median():.2f} | "
                    f"std={df_reg[TARGET_REGRESION].std():.2f}"
                ),
                "detalle": "Estadísticas del target dias_en_transito",
            },
        ])

    reporte = pd.DataFrame(filas)

    # Sanitizar strings: convertir a latin-1 seguro para compatibilidad Windows
    for col in reporte.select_dtypes(include=["object", "string"]).columns:
        reporte[col] = (
            reporte[col]
            .astype(str)
            .str.encode("latin-1", errors="replace")
            .str.decode("latin-1")
        )

    logger.info(
        "[model_input] Reporte generado: %d registros de trazabilidad.", len(reporte)
    )
    return reporte

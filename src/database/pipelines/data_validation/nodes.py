"""
Nodos del pipeline de Data Validation (AD 1.4).

Responsabilidad:
    - Verificar tipos de datos del dataset maestro final (Pandera + manual).
    - Comprobar integridad referencial (ids coherentes entre tablas).
    - Validar rangos numéricos esperados post-limpieza.
    - Asegurar ausencia de nulos en columnas críticas.
    - Generar reporte de validación consolidado en 08_reporting.

Estrategia de doble capa de validación:
    1. Pandera (validar_con_pandera): validación declarativa de esquema,
       tipos y rangos usando DataFrameSchema. Es la capa formal y reproducible.
    2. Validación manual (nodos 1-5): checks de integridad referencial,
       rangos de negocio y nulos críticos que complementan a Pandera.

Técnicas de optimización Pandas aplicadas:
    - Filtros avanzados con .loc[] e .isin() para validación referencial.
    - Broadcasting booleano para verificaciones vectorizadas.
    - .groupby().agg() para estadísticas de validación por dimensión.
    - pd.concat() para consolidar resultados de forma eficiente.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

try:
    import pandera as pa
    from pandera import Column, DataFrameSchema, Check
    _PANDERA_DISPONIBLE = True
except ImportError:
    _PANDERA_DISPONIBLE = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema Pandera (validación declarativa de esquema)
# ---------------------------------------------------------------------------

def _construir_schema_pandera() -> "pa.DataFrameSchema | None":
    """
    Construye el esquema Pandera para el dataset maestro de envíos.

    Valida de forma declarativa:
        - Tipos de datos esperados por columna.
        - Ausencia de nulos en columnas críticas (nullable=False).
        - Rangos de valores válidos (Check.greater_than_or_equal_to, etc.).

    La ventaja de Pandera sobre validación manual es que el esquema es
    autodocumentado y genera errores descriptivos con la fila exacta del error.
    Se usa en conjunto con los checks manuales para cobertura completa.
    """
    if not _PANDERA_DISPONIBLE:
        return None

    schema = DataFrameSchema(
        columns={
            "id_envio": Column(int, nullable=False, description="ID único del envío"),
            "id_ruta": Column(float, nullable=True, description="ID de ruta (puede ser null antes del join)"),
            "id_vehiculo": Column(float, nullable=True, description="ID de vehículo asignado"),
            "peso_kg": Column(
                float,
                checks=Check.greater_than_or_equal_to(0.0, error="peso_kg no puede ser negativo"),
                nullable=True,
                description="Peso del envío en kg",
            ),
            "volumen_m3": Column(
                float,
                checks=Check.greater_than_or_equal_to(0.0, error="volumen_m3 no puede ser negativo"),
                nullable=True,
                description="Volumen del envío en m³",
            ),
            "tiene_incidencia": Column(
                int,
                checks=Check.isin([0, 1], error="tiene_incidencia debe ser 0 o 1"),
                nullable=False,
                description="Indicador binario de incidencia",
            ),
        },
        coerce=False,   # No coercionar; reportar discrepancias tal como están
        strict=False,   # Permitir columnas adicionales al esquema
        name="MasterEnviosSchema",
        description="Esquema Pandera para el dataset maestro de envíos logísticos",
    )
    return schema


def validar_con_pandera(master: pd.DataFrame) -> pd.DataFrame:
    """
    Nodo 0 (Pandera): Validación declarativa del esquema del dataset maestro.

    Usa Pandera DataFrameSchema para verificar:
        - Tipos de datos: id_envio (int), peso_kg (float), tiene_incidencia (int).
        - Constraints: tiene_incidencia ∈ {0,1}, peso_kg ≥ 0, volumen_m3 ≥ 0.
        - Nulos: id_envio y tiene_incidencia no pueden ser nulos.

    La validación Pandera es declarativa: el esquema describe QUÉ se espera
    y Pandera verifica automáticamente TODOS los registros de una vez.
    Esto es superior al chequeo manual columna por columna en bucles.

    Si Pandera detecta una violación, reporta el error sin lanzar excepción
    (se usa lazy=True para recolectar todos los errores antes de reportar).

    Parámetros
    ----------
    master : DataFrame maestro final.

    Retorna
    -------
    pd.DataFrame con una fila por columna validada y su resultado.
    """
    if not _PANDERA_DISPONIBLE:
        logger.warning("[pandera] M?dulo no disponible. Saltando validaci?n Pandera.")
        return pd.DataFrame(columns=["check", "estado", "detalle", "n_afectados", "porcentaje"])

    schema = _construir_schema_pandera()
    resultados: list[dict] = []

    try:
        schema.validate(master, lazy=True)  # lazy=True: recolecta todos los errores
        logger.info("[pandera] OK - Validacion Pandera: dataset maestro cumple el esquema declarativo.")
        resultados.append({
            "check": "pandera:schema_global",
            "estado": "OK",
            "detalle": "DataFrameSchema Pandera validado sin errores sobre todas las columnas definidas.",
            "n_afectados": 0,
            "porcentaje": 0.0,
        })
    except pa.errors.SchemaErrors as exc:
        # Pandera recolecta todos los errores en un DataFrame
        errores_df = exc.failure_cases
        n_errores = len(errores_df)
        columnas_con_error = errores_df["column"].unique().tolist() if "column" in errores_df.columns else []
        detalle = f"{n_errores} violaciones en columnas: {columnas_con_error}"
        logger.warning("[pandera] ADVERTENCIA - Validacion Pandera: %d violaciones detectadas. Columnas: %s",
                       n_errores, columnas_con_error)
        resultados.append({
            "check": "pandera:schema_global",
            "estado": "ADVERTENCIA",
            "detalle": detalle,
            "n_afectados": n_errores,
            "porcentaje": round(n_errores / max(len(master), 1) * 100, 2),
        })
        # Agregar una fila por columna con errores
        for col in columnas_con_error:
            sub = errores_df[errores_df["column"] == col] if "column" in errores_df.columns else errores_df
            resultados.append({
                "check": f"pandera:{col}",
                "estado": "ADVERTENCIA",
                "detalle": f"{len(sub)} registros inválidos en columna '{col}'.",
                "n_afectados": len(sub),
                "porcentaje": round(len(sub) / max(len(master), 1) * 100, 2),
            })
    except Exception as exc:
        logger.error("[pandera] Error inesperado en validaci?n: %s", str(exc))
        resultados.append({
            "check": "pandera:schema_global",
            "estado": "ERROR",
            "detalle": f"Error inesperado: {str(exc)[:120]}",
            "n_afectados": 0,
            "porcentaje": 0.0,
        })

    return pd.DataFrame(resultados)


# Tipos de datos esperados en el dataset maestro post-transformación
SCHEMA_ESPERADO = {
    "id_envio": "Int64",
    "id_ruta": "Int64",
    "id_vehiculo": "Int64",
    "peso_kg": "float64",
    "volumen_m3": "float64",
    "tiene_incidencia": "int8",
    "total_incidencias": "Int64",
}


# ---------------------------------------------------------------------------
# Helpers privados
# ---------------------------------------------------------------------------

def _crear_resultado(
    check: str,
    estado: str,
    detalle: str,
    n_afectados: int = 0,
) -> dict:
    """Crea un diccionario estándar para una fila del reporte de validación."""
    return {
        "check": check,
        "estado": estado,
        "detalle": detalle,
        "n_afectados": n_afectados,
        "porcentaje": round(n_afectados / max(n_afectados, 1) * 100, 2),
    }


# ---------------------------------------------------------------------------
# Nodos públicos
# ---------------------------------------------------------------------------

def validar_esquema(master: pd.DataFrame) -> pd.DataFrame:
    """
    Nodo 1: Verifica que las columnas clave tienen el tipo de dato correcto.

    Usa broadcasting booleano: compara el dtype de cada columna contra
    el esquema esperado de forma vectorizada.

    Parámetros
    ----------
    master : DataFrame maestro final (03_primary/master_envios.csv).

    Retorna
    -------
    pd.DataFrame con resultados del check de esquema.
    """
    resultados = []
    for col, dtype_esperado in SCHEMA_ESPERADO.items():
        if col not in master.columns:
            resultados.append(_crear_resultado(
                f"esquema:{col}", "ADVERTENCIA",
                f"Columna '{col}' no encontrada en el dataset.", 0,
            ))
            logger.warning("[validaci?n] Columna '%s' no encontrada.", col)
        else:
            dtype_actual = str(master[col].dtype)
            # Validación flexible: int8 es subconjunto de int, etc.
            ok = (dtype_esperado.lower() in dtype_actual.lower() or
                  dtype_actual.lower() in dtype_esperado.lower())
            estado = "OK" if ok else "ADVERTENCIA"
            resultados.append(_crear_resultado(
                f"esquema:{col}",
                estado,
                f"dtype actual: {dtype_actual} | esperado: {dtype_esperado}",
                0,
            ))
            if not ok:
                logger.warning(
                    "[validación] Columna '%s': dtype %s ≠ esperado %s.",
                    col, dtype_actual, dtype_esperado,
                )

    logger.info("[validaci?n] Check de esquema: %d columnas verificadas.", len(resultados))
    return pd.DataFrame(resultados)


def validar_integridad_referencial(
    master: pd.DataFrame,
    rutas: pd.DataFrame,
    vehiculos: pd.DataFrame,
) -> pd.DataFrame:
    """
    Nodo 2: Verifica que todos los ids del master existen en sus tablas maestras.

    Técnica: pd.Series.isin() — filtro vectorizado que evita bucles Python.
    El complemento (~) actúa como broadcasting booleano de negación.

    Parámetros
    ----------
    master    : Dataset maestro final.
    rutas     : Dataset limpio de rutas (referencia).
    vehiculos : Dataset limpio de vehículos (referencia).

    Retorna
    -------
    pd.DataFrame con resultados del check de integridad referencial.
    """
    resultados = []
    total = len(master)

    # Check 1: id_ruta en master → debe existir en rutas (filtro .isin() vectorizado)
    if "id_ruta" in master.columns and "id_ruta" in rutas.columns:
        ids_rutas_validas = set(rutas["id_ruta"].dropna())
        mask_invalidos = ~master["id_ruta"].isin(ids_rutas_validas)  # broadcasting ~
        n_invalidos = int(mask_invalidos.sum())

        # Filtro avanzado con .loc[] para mostrar los registros problemáticos
        if n_invalidos > 0:
            ejemplos = master.loc[mask_invalidos, "id_ruta"].head(5).tolist()
            logger.warning(
                "[validación] %d envíos con id_ruta inválido. Ejemplos: %s",
                n_invalidos, ejemplos,
            )
        resultados.append(_crear_resultado(
            "integridad:id_ruta",
            "OK" if n_invalidos == 0 else "ERROR",
            f"{n_invalidos}/{total} envíos con id_ruta no encontrado en tabla rutas.",
            n_invalidos,
        ))

    # Check 2: id_vehiculo en master → debe existir en vehiculos (.isin() vectorizado)
    if "id_vehiculo" in master.columns and "id_vehiculo" in vehiculos.columns:
        ids_vehiculos_validos = set(vehiculos["id_vehiculo"].dropna())
        mask_invalidos_v = ~master["id_vehiculo"].isin(ids_vehiculos_validos)
        n_invalidos_v = int(mask_invalidos_v.sum())

        if n_invalidos_v > 0:
            ejemplos_v = master.loc[mask_invalidos_v, "id_vehiculo"].head(5).tolist()
            logger.warning(
                "[validación] %d envíos con id_vehiculo inválido. Ejemplos: %s",
                n_invalidos_v, ejemplos_v,
            )
        resultados.append(_crear_resultado(
            "integridad:id_vehiculo",
            "OK" if n_invalidos_v == 0 else "ERROR",
            f"{n_invalidos_v}/{total} envíos con id_vehiculo no encontrado en tabla vehículos.",
            n_invalidos_v,
        ))

    logger.info("[validaci?n] Check de integridad referencial completado.")
    return pd.DataFrame(resultados)


def validar_rangos(master: pd.DataFrame, parameters: dict) -> pd.DataFrame:
    """
    Nodo 3: Verifica que los valores numéricos están dentro de rangos esperados.

    Técnicas:
        - Broadcasting booleano para generar máscaras de fuera-de-rango.
        - .between() vectorizado para comparaciones dobles eficientes.
        - .loc[] para extraer registros problemáticos.

    Parámetros
    ----------
    master     : Dataset maestro final.
    parameters : Parámetros de validación (validacion.peso_kg_min, etc.).

    Retorna
    -------
    pd.DataFrame con resultados del check de rangos.
    """
    v = parameters.get("validacion", {})
    resultados = []
    total = len(master)

    # Definir checks de rango como lista de tuplas (col, min, max, nombre)
    checks = []
    if "peso_kg" in master.columns:
        checks.append((
            "peso_kg",
            v.get("peso_kg_min", 1.0),
            v.get("peso_kg_max", 25000.0),
            "rango:peso_kg",
        ))
    if "distancia_km" in master.columns:
        checks.append((
            "distancia_km",
            v.get("distancia_km_min", 1.0),
            v.get("distancia_km_max", 5000.0),
            "rango:distancia_km",
        ))
    if "eficiencia_peso" in master.columns:
        checks.append((
            "eficiencia_peso",
            v.get("eficiencia_peso_min", 0.0),
            v.get("eficiencia_peso_max", 2.0),
            "rango:eficiencia_peso",
        ))
    if "dias_en_transito" in master.columns:
        checks.append((
            "dias_en_transito",
            0,
            v.get("dias_transito_max", 30),
            "rango:dias_en_transito",
        ))

    for col, minv, maxv, nombre in checks:
        # Broadcasting: .between() evalúa toda la columna de una vez
        fuera_rango = ~master[col].between(minv, maxv)
        n_fuera = int(fuera_rango.sum())
        pct_fuera = round(n_fuera / total * 100, 2) if total > 0 else 0.0

        estado = "OK" if n_fuera == 0 else ("ADVERTENCIA" if pct_fuera < 5 else "ERROR")
        resultados.append({
            "check": nombre,
            "estado": estado,
            "detalle": f"Rango esperado [{minv}, {maxv}]. Fuera de rango: {n_fuera} ({pct_fuera}%)",
            "n_afectados": n_fuera,
            "porcentaje": pct_fuera,
        })

        if n_fuera > 0:
            logger.warning(
                "[validación] %s: %d registros fuera del rango [%.2f, %.2f].",
                nombre, n_fuera, minv, maxv,
            )

    logger.info("[validaci?n] Check de rangos: %d columnas verificadas.", len(checks))
    return pd.DataFrame(resultados) if resultados else pd.DataFrame(
        columns=["check", "estado", "detalle", "n_afectados", "porcentaje"]
    )


def validar_nulos_criticos(master: pd.DataFrame, parameters: dict) -> pd.DataFrame:
    """
    Nodo 4: Verifica ausencia de nulos en columnas críticas del dataset maestro.

    Técnicas:
        - .isna().sum() vectorizado para contar nulos por columna.
        - .loc[] con máscara booleana para mostrar registros problemáticos.

    Parámetros
    ----------
    master     : Dataset maestro final.
    parameters : Lista de columnas sin nulos requeridas.

    Retorna
    -------
    pd.DataFrame con resultados del check de nulos.
    """
    cols_criticas = parameters.get("validacion", {}).get("columnas_sin_nulos", [])
    resultados = []
    total = len(master)

    for col in cols_criticas:
        if col not in master.columns:
            resultados.append({
                "check": f"nulos:{col}",
                "estado": "ADVERTENCIA",
                "detalle": f"Columna '{col}' no encontrada.",
                "n_afectados": 0,
                "porcentaje": 0.0,
            })
            continue

        # Broadcasting vectorizado: .isna() sobre toda la columna
        n_nulos = int(master[col].isna().sum())
        pct_nulos = round(n_nulos / total * 100, 2) if total > 0 else 0.0
        estado = "OK" if n_nulos == 0 else "ERROR"

        resultados.append({
            "check": f"nulos:{col}",
            "estado": estado,
            "detalle": f"{n_nulos}/{total} nulos ({pct_nulos}%) en columna crítica.",
            "n_afectados": n_nulos,
            "porcentaje": pct_nulos,
        })

        if n_nulos > 0:
            logger.error(
                "[validación] Columna crítica '%s' tiene %d nulos (%.2f%%).",
                col, n_nulos, pct_nulos,
            )

    logger.info(
        "[validación] Check de nulos críticos: %d columnas verificadas.",
        len(cols_criticas),
    )
    return pd.DataFrame(resultados) if resultados else pd.DataFrame(
        columns=["check", "estado", "detalle", "n_afectados", "porcentaje"]
    )


def consolidar_reporte_validacion(
    res_esquema: pd.DataFrame,
    res_integridad: pd.DataFrame,
    res_rangos: pd.DataFrame,
    res_nulos: pd.DataFrame,
    res_pandera: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Nodo 5: Consolida todos los resultados de validación en un reporte final.

    Incluye los resultados de Pandera (res_pandera) como capa adicional de
    validación declarativa, junto con los checks manuales de esquema, integridad,
    rangos y nulos. Esto garantiza cobertura completa y trazabilidad.

    Técnicas:
        - pd.concat() eficiente para unir DataFrames sin copias intermedias.
        - np.select() con broadcasting para clasificación de criticidad.

    Parámetros
    ----------
    res_esquema   : Resultados del check de tipos de datos.
    res_integridad: Resultados del check de integridad referencial.
    res_rangos    : Resultados del check de rangos numéricos.
    res_nulos     : Resultados del check de nulos críticos.
    res_pandera   : Resultados del check Pandera (DataFrameSchema).

    Retorna
    -------
    pd.DataFrame reporte final guardado en 08_reporting/validation_report.csv.
    """
    frames = [res_esquema, res_integridad, res_rangos, res_nulos]
    if res_pandera is not None and not res_pandera.empty:
        frames.insert(0, res_pandera)  # Pandera va primero (es la capa declarativa)

    # Concatenación eficiente con ignore_index
    reporte = pd.concat(
        frames,
        ignore_index=True,
    )

    # Añadir columna de categoría del check
    reporte["categoria"] = reporte["check"].str.split(":").str[0]

    # Resumen estadístico por estado (agrupación vectorizada)
    resumen = (
        reporte
        .groupby("estado", observed=True)["check"]
        .count()
        .reset_index()
        .rename(columns={"check": "cantidad"})
    )

    n_ok = int(resumen.loc[resumen["estado"] == "OK", "cantidad"].sum())
    n_warn = int(resumen.loc[resumen["estado"] == "ADVERTENCIA", "cantidad"].sum())
    n_err = int(resumen.loc[resumen["estado"] == "ERROR", "cantidad"].sum())
    total_checks = len(reporte)

    logger.info(
        "[validacion] Reporte final --- OK: %d | ADVERTENCIA: %d | ERROR: %d (total: %d checks)",
        n_ok, n_warn, n_err, total_checks,
    )

    if n_err > 0:
        logger.error(
            "[validación] ¡ATENCIÓN! Se encontraron %d errores de validación. "
            "Revisar el reporte en data/08_reporting/validation_report.csv",
            n_err,
        )
    else:
        logger.info("[validacion] OK - Todos los checks criticos pasaron sin errores.")

    # Ordenar: primero los errores, luego advertencias, luego OK
    orden_estado = {"ERROR": 0, "ADVERTENCIA": 1, "OK": 2}
    reporte["orden"] = reporte["estado"].map(orden_estado)
    reporte = reporte.sort_values("orden").drop(columns="orden").reset_index(drop=True)

    return reporte

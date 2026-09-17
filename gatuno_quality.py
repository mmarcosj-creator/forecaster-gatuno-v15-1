"""Monitor V10.3 - Aprendizaje real sin tocar umbrales a mano.

Este modulo responde a una pregunta distinta de auditoria_v10.py:

  auditoria_v10.py  -> "¿acerto o fallo cada pronostico ya jugado?"
  gatuno_quality.py -> "¿el modelo que genera esos pronosticos sigue siendo
                        tan bueno como antes, o se degrado con los datos
                        nuevos?" y "¿la probabilidad que promete coincide con
                        lo que realmente pasa?"

Por que NO se reentrena directamente sobre "acierto/fallo":

  Un pronostico como "MAS DE 4.5 tarjetas" fallado no dice cuanto fallo (?4 o
  10 tarjetas?). El numero real (goles, tarjetas, corners) ya esta disponible
  y gatuno_forecaster.train_models() lo usa completo cada corrida via
  descargar_historico_total(). Reentrenar ademas sobre el bit binario
  acierto/fallo tira esa informacion y arriesga retroalimentacion (el modelo
  aprenderia a imitar sus propios errores de calibracion, no a corregirlos).

Lo que SI hace este modulo, con los datos que ya existen:

  1. Registra, corrida a corrida, las metricas OOS que train_models() ya
     calcula (Brier/MAE, ECE, SuperaBase) - eso es la foto de "que tan bien
     aprendio el modelo de los resultados reales mas recientes".
  2. Compara la corrida de hoy contra la mediana de las ultimas N corridas
     del mismo submodelo. Si empeora mas de la tolerancia, o si un modelo que
     antes superaba la base deja de hacerlo, se marca como degradado.
  3. Cruza la calibracion que el semaforo promete (p. ej. "verde exige
     probabilidad >=68%") contra la tasa de acierto real que ya mide
     auditoria_v10.calcular_metricas(), y muestra la brecha. Esto es
     puramente diagnostico: nunca reescribe umbrales automaticamente.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Umbral nominal minimo de probabilidad que cada color promete para mercados
# binarios, tal como en CRITERIOS_V10.md seccion 6. Se usa solo para comparar
# contra la tasa real observada, nunca para decidir el semaforo.
PROMESA_SEMAFORO = {"VERDE": 0.68, "AMARILLO": 0.59}

METRICS_LOG_COLUMNS = [
    "RunEn", "Modelo", "Target", "Tipo", "N_Validacion",
    "Brier", "MAE", "ECE", "MejoraBrier", "MejoraMAE", "SuperaBase", "CalidadModelo",
]


def registrar_metricas(bundle_metrics: pd.DataFrame, log_path: str | Path, run_en: str | None = None) -> pd.DataFrame:
    """Agrega una fila por submodelo al log historico de calidad. Append-only:
    nunca se borran corridas anteriores, para poder ver la evolucion real.
    """
    run_en = run_en or pd.Timestamp.now(tz="UTC").isoformat()
    log_path = Path(log_path)
    log = pd.read_csv(log_path) if log_path.exists() else pd.DataFrame(columns=METRICS_LOG_COLUMNS)

    filas = []
    for _, row in bundle_metrics.iterrows():
        filas.append({
            "RunEn": run_en,
            "Modelo": row.get("Modelo"),
            "Target": row.get("Target"),
            "Tipo": row.get("Tipo"),
            "N_Validacion": row.get("N_Validacion"),
            "Brier": row.get("Brier", np.nan),
            "MAE": row.get("MAE", np.nan),
            "ECE": row.get("ECE", np.nan),
            "MejoraBrier": row.get("MejoraBrier", np.nan),
            "MejoraMAE": row.get("MejoraMAE", np.nan),
            "SuperaBase": bool(row.get("SuperaBase", False)),
            "CalidadModelo": row.get("CalidadModelo", np.nan),
        })
    nuevo = pd.concat([log, pd.DataFrame(filas)], ignore_index=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    nuevo.to_csv(log_path, index=False, encoding="utf-8-sig")
    return nuevo


def detectar_degradacion(log: pd.DataFrame, ventana: int = 10, tolerancia: float = 0.15) -> list[dict[str, Any]]:
    """Compara la corrida mas reciente de cada modelo contra la mediana de
    las `ventana` corridas previas. No decide nada por si solo: solo avisa,
    para que la revision la haga una persona.
    """
    avisos: list[dict[str, Any]] = []
    if log.empty:
        return avisos

    for modelo, grupo in log.groupby("Modelo"):
        grupo = grupo.sort_values("RunEn")
        if len(grupo) < 2:
            continue
        actual = grupo.iloc[-1]
        previas = grupo.iloc[:-1].tail(ventana)
        if previas.empty:
            continue

        metrica_col = "Brier" if pd.notna(actual.get("Brier")) else "MAE"
        actual_val = actual.get(metrica_col)
        previas_val = previas[metrica_col].dropna()
        if pd.isna(actual_val) or previas_val.empty:
            continue
        mediana_previa = float(previas_val.median())
        if mediana_previa <= 0:
            continue

        empeoro = (actual_val - mediana_previa) / mediana_previa
        if empeoro > tolerancia:
            avisos.append({
                "Modelo": modelo,
                "Motivo": f"{metrica_col} empeoro {empeoro:.0%} vs mediana de las ultimas {len(previas_val)} corridas",
                "Severidad": "ALTA" if empeoro > 2 * tolerancia else "MEDIA",
            })

        if bool(previas.iloc[-1].get("SuperaBase", False)) and not bool(actual.get("SuperaBase", False)):
            avisos.append({
                "Modelo": modelo,
                "Motivo": "Dejo de superar la base OOS en la corrida mas reciente (antes si la superaba)",
                "Severidad": "ALTA",
            })

    return avisos


def comparar_calibracion_real(metricas_auditoria: pd.DataFrame, minimo_fiable: int = 30) -> pd.DataFrame:
    """Contrasta, por mercado y color, la probabilidad minima que el semaforo
    promete contra la tasa de acierto real observada en la auditoria. Solo
    informativo: no ajusta nada. Filas con N_Evaluado < minimo_fiable se
    marcan como no concluyentes en vez de usarse para sacar conclusiones.
    """
    if metricas_auditoria.empty:
        return pd.DataFrame(columns=["Mercado", "Semaforo", "N_Evaluado", "PromesaMinima", "TasaReal", "Brecha", "Lectura"])

    filas = []
    for _, row in metricas_auditoria.iterrows():
        semaforo = row.get("Semaforo")
        promesa = PROMESA_SEMAFORO.get(semaforo)
        if promesa is None:
            continue  # ROJO no promete un piso de probabilidad; no hay nada que contrastar
        n_eval = int(row.get("N_Evaluado", 0))
        tasa = row.get("TasaAcierto")
        if n_eval == 0 or pd.isna(tasa):
            lectura = "SIN DATOS"
            brecha = np.nan
        elif n_eval < minimo_fiable:
            brecha = tasa - promesa
            lectura = "MUESTRA CHICA - NO CONCLUYENTE"
        else:
            brecha = tasa - promesa
            lectura = "CONSISTENTE" if brecha >= -0.05 else "POR DEBAJO DE LO PROMETIDO"
        filas.append({
            "Mercado": row.get("Mercado"),
            "Semaforo": semaforo,
            "N_Evaluado": n_eval,
            "PromesaMinima": promesa,
            "TasaReal": tasa,
            "Brecha": brecha,
            "Lectura": lectura,
        })
    return pd.DataFrame(filas)

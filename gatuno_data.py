# ============================================================
# BET BUILDER V8 SELECTIVE - 7 LIGAS + 5 TORNEOS
# COLAB + PYDROID | SIN TZDATA
# ============================================================
#
# NICHO BASE
#   1) Underdog anota 1+ gol
#   2) Favorito consigue 4+ corners
#   3) Under 4.5 goles
#
# VARIANTES QUE EL MODELO PUEDE COMPARAR
#   BASE
#   BASE + 4+ tarjetas totales (proxy histórico con amarillas)
#   BASE + Over 0.5 goles en 1er tiempo
#
# REGLA ECONOMICA
#   Cuota combinada REAL nunca < 4.20
#   CuotaMinExigida = max(4.20, 1.29 / P_conjunta)
#
# IMPORTANTE
# - La cuota exacta del Bet Builder debe comprobarse en la casa.
# - Clima extremo reduce confianza; no se usa para "fabricar" EV.
# - Altitud puede ajustar ligeramente la pata "underdog marca".
# - Árbitro puede ajustar ligeramente el mercado de tarjetas.
# - Córners de 1T NO activan APOSTAR porque no hay una base histórica
#   homogénea en las fuentes gratuitas usadas por este programa.
#
# FUENTES
# - Football-Data: histórico europeo con corners/tarjetas/HT.
# - ESPN: calendario de las 12 competiciones y respaldo histórico
#   reciente para Brasil, Argentina y Perú.
# - Sofascore: respaldo de calendario diario cuando ESPN y Football-Data
#   no entregan una agenda utilizable. No se usa para fabricar cuotas.
# - Open-Meteo: geocodificación + clima futuro + elevación.
#
# ============================================================

import os
import io
import re
import ssl
import json
import math
import time
import zipfile
import traceback
import unicodedata
import urllib.request
import urllib.error
import urllib.parse

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURACION
# ============================================================

VERSION = "V15.1.1-STABLE-CORE"
TOP_N = 30

# Perú: UTC-5 todo el año. Evita ZoneInfo/tzdata en Pydroid.
TZ_PERU = timezone(timedelta(hours=-5), name="PET")

DIAS_VENTANA = 7
# Conserva cualquier partido del día actual que todavía no haya comenzado.
# Puede aumentarse manualmente si se desea un margen operativo adicional.
MARGEN_PREPARTIDO_MINUTOS = 0

CUOTA_COMBINADA_MIN = 4.20
ROI_OBJETIVO = 0.29

CUOTA_FAVORITO_MIN = 1.20
CUOTA_FAVORITO_MAX = 1.85

# Probabilidad mínima matemática para ROI +29% a 4.20.
P_ROI29_420 = (1.0 + ROI_OBJETIVO) / CUOTA_COMBINADA_MIN

# Filtros de patas base.
P_MIN_DOG_GOL = 0.52
P_MIN_FAV4 = 0.59
P_MIN_UNDER45 = 0.74

# Filtros marginales de patas extra.
P_MIN_CARDS4 = 0.61
P_MIN_GOL1T = 0.62

# Conjuntas.
P_MIN_BASE = max(0.31, P_ROI29_420)
P_MIN_BASE_CARDS = 0.19
P_MIN_BASE_GOL1T = 0.19

P_LCB_MIN_BASE = 1.0 / CUOTA_COMBINADA_MIN

# KNN empírico.
K_VECINOS = 150
K_MINIMO = 45
MAX_TRAIN = 4500
PRIOR_STRENGTH = 24.0
Z_LCB = 0.84

# Historia.
MIN_PARTIDOS_EQUIPO = 6
VENTANAS = (5, 10, 20)

# ESPN histórico para Sudamérica.
ESPN_HIST_DAYS = 560
# Para Elo, posiciones, resultado y goles se conserva una campana amplia. Los
# resúmenes detallados (tarjetas/corners/1T) son más costosos y se limitan a
# una ventana reciente independiente.
MAX_ESPN_RESULT_EVENTS_PER_COMP = 260
MAX_ESPN_HIST_EVENTS_PER_COMP = 90
# Incluye liga argentina/brasilena/peruana y los dos torneos CONMEBOL. El
# limite anterior agotaba el presupuesto antes de llegar a Libertadores y
# Sudamericana, por lo que el modelo no podia aprender cruces interliga.
MAX_ESPN_SUMMARY_CALLS_HIST = 450
PAUSA_ESPN = 0.04

# Futuro.
MAX_FUTURE_SUMMARY_CALLS = 70
MAX_WEATHER_CALLS = 70

# Backtest.
BACKTEST_MAX = 850

# Football-Data temporadas Europeas.
TEMPORADAS_FD = ["2425", "2526", "2627"]


# ============================================================
# 7 LIGAS + 5 TORNEOS
# ============================================================
#
# potencial: prior cualitativo para ranking (NO altera directamente P)
# cards: casa grande suele ofrecer tarjetas en este nivel competitivo
# gol1t: mercado de primera mitad normalmente disponible
#
# ============================================================

COMPETICIONES = {
    # 7 LIGAS
    "BRA": {
        "nombre": "Brasileirão Serie A",
        "espn": "bra.1",
        "tipo": "LIGA",
        "grupo": "SAM",
        "fd_div": None,
        "hist_espn": True,
        "potencial": 100,
        "cards": True,
        "gol1t": True,
    },
    "ESP": {
        "nombre": "LaLiga",
        "espn": "esp.1",
        "tipo": "LIGA",
        "grupo": "EUR",
        "fd_div": "SP1",
        "hist_espn": False,
        "potencial": 97,
        "cards": True,
        "gol1t": True,
    },
    "POR": {
        "nombre": "Liga Portugal",
        "espn": "por.1",
        "tipo": "LIGA",
        "grupo": "EUR",
        "fd_div": "P1",
        "hist_espn": False,
        "potencial": 92,
        "cards": True,
        "gol1t": True,
    },
    "TUR": {
        "nombre": "Süper Lig",
        "espn": "tur.1",
        "tipo": "LIGA",
        "grupo": "EUR",
        "fd_div": "T1",
        "hist_espn": False,
        "potencial": 91,
        "cards": True,
        "gol1t": True,
    },
    "ENG": {
        "nombre": "Premier League",
        "espn": "eng.1",
        "tipo": "LIGA",
        "grupo": "EUR",
        "fd_div": "E0",
        "hist_espn": False,
        "potencial": 90,
        "cards": True,
        "gol1t": True,
    },
    "ARG": {
        "nombre": "Liga Profesional Argentina",
        "espn": "arg.1",
        "tipo": "LIGA",
        "grupo": "SAM",
        "fd_div": None,
        "hist_espn": True,
        "potencial": 88,
        "cards": True,
        "gol1t": True,
    },
    "PER": {
        "nombre": "Liga 1 Perú",
        "espn": "per.1",
        "tipo": "LIGA",
        "grupo": "SAM",
        "fd_div": None,
        "hist_espn": True,
        "potencial": 84,
        "cards": True,
        "gol1t": True,
    },

    # 5 TORNEOS
    "CDB": {
        "nombre": "Copa do Brasil",
        "espn": "bra.copa_do_brazil",
        "tipo": "COPA",
        "grupo": "SAM",
        "fd_div": None,
        "hist_espn": False,
        "potencial": 96,
        "cards": True,
        "gol1t": True,
    },
    "LIB": {
        "nombre": "Copa Libertadores",
        "espn": "conmebol.libertadores",
        "tipo": "COPA",
        "grupo": "SAM",
        "fd_div": None,
        "hist_espn": True,
        "potencial": 94,
        "cards": True,
        "gol1t": True,
    },
    "SUD": {
        "nombre": "Copa Sudamericana",
        "espn": "conmebol.sudamericana",
        "tipo": "COPA",
        "grupo": "SAM",
        "fd_div": None,
        "hist_espn": True,
        "potencial": 93,
        "cards": True,
        "gol1t": True,
    },
    "UEL": {
        "nombre": "UEFA Europa League",
        "espn": "uefa.europa",
        "tipo": "COPA",
        "grupo": "EUR",
        "fd_div": None,
        "hist_espn": False,
        "potencial": 89,
        "cards": True,
        "gol1t": True,
    },
    "UCL": {
        "nombre": "UEFA Champions League",
        "espn": "uefa.champions",
        "tipo": "COPA",
        "grupo": "EUR",
        "fd_div": None,
        "hist_espn": False,
        "potencial": 86,
        "cards": True,
        "gol1t": True,
    },
}

FD_DIV_TO_COMP = {
    info["fd_div"]: key
    for key, info in COMPETICIONES.items()
    if info["fd_div"]
}

BASE_HIST_FD = "https://www.football-data.co.uk/mmz4281/"
FD_FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"


# ============================================================
# OUTPUT / CACHE
# ============================================================

def detectar_output_dir():
    if os.path.isdir("/content"):
        return "/content"

    android = "/storage/emulated/0/Download"
    if os.path.isdir(android):
        return android

    return os.getcwd()


OUTPUT_DIR = detectar_output_dir()
CACHE_DIR = os.path.join(OUTPUT_DIR, "gatuno_cache_v15_1")

try:
    os.makedirs(CACHE_DIR, exist_ok=True)
except Exception:
    CACHE_DIR = os.getcwd()

ARCHIVO_XLSX = os.path.join(OUTPUT_DIR, "BET_BUILDER_V8_TOP30.xlsx")
ARCHIVO_CSV = os.path.join(OUTPUT_DIR, "BET_BUILDER_V8_TOP30.csv")
ARCHIVO_BACKTEST = os.path.join(OUTPUT_DIR, "BET_BUILDER_V8_BACKTEST.csv")
ARCHIVO_VARIANTES = os.path.join(OUTPUT_DIR, "BET_BUILDER_V8_VARIANTES.csv")
ARCHIVO_LOG = os.path.join(OUTPUT_DIR, "BET_BUILDER_V8_LOG.txt")
FIXTURE_CACHE_PATH = os.path.join(CACHE_DIR, "fixtures_latest.csv")


# ============================================================
# LOG
# ============================================================

def iniciar_log():
    with open(ARCHIVO_LOG, "w", encoding="utf-8") as f:
        f.write(f"BET BUILDER {VERSION}\n")
        f.write(f"Ejecución: {datetime.now(TZ_PERU)}\n")
        f.write(f"Salida: {OUTPUT_DIR}\n\n")


def log(msg=""):
    print(msg)
    try:
        with open(ARCHIVO_LOG, "a", encoding="utf-8") as f:
            f.write(str(msg) + "\n")
    except Exception:
        pass


# ============================================================
# HTTP / CACHE
# ============================================================

def parece_html(raw):
    if not raw:
        return True
    h = raw[:600].lstrip().lower()
    return any(
        h.startswith(x)
        for x in (b"<!doctype html", b"<html", b"<head", b"<body")
    )

import os

def descargar_bytes(url, timeout=None, headers=None, reintentos=None):
    fast_mode = os.environ.get("GATUNO_FAST_MODE", "0") == "1"
    if timeout is None:
        timeout = 2.5 if fast_mode else 35
    if reintentos is None:
        reintentos = 0 if fast_mode else 2
        
    # El resto de tu función original (requests.get, manejo de errores, etc.) continúa aquí abajo sin cambios.
def descargar_bytes(url, timeout=None, headers=None, reintentos=None):
    """Descarga con limites aptos para ejecucion interactiva.

    En Streamlit una peticion bloqueada durante 35 segundos y repetida cientos
    de veces puede provocar que el proceso sea reiniciado antes de guardar la
    cache.  El modo rapido usa un unico intento de 12 segundos.  El proceso
    programado conserva los limites amplios originales.
    """
    fast_mode = os.environ.get("GATUNO_FAST_MODE", "0") == "1"
    if timeout is None:
        timeout = 6 if fast_mode else 35
    if reintentos is None:
        reintentos = 1 if fast_mode else 2
    ultimo = None

    hdr = headers or {
        "User-Agent": "Mozilla/5.0 Gatuno-Forecaster/15.1",
        "Accept": "*/*",
        "Cache-Control": "no-cache",
    }

    for intento in range(reintentos):
        try:
            req = urllib.request.Request(url, headers=hdr)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()

        except urllib.error.HTTPError as e:
            ultimo = e
            if e.code == 429:
                time.sleep(2 + intento * 2)
                continue
            break

        except urllib.error.URLError as e:
            ultimo = e
            txt = str(e).lower()

            if "ssl" in txt or "certificate" in txt:
                try:
                    ctx = ssl.create_default_context()
                    ctx.check_hostname = False
                    ctx.verify_mode = ssl.CERT_NONE
                    req = urllib.request.Request(url, headers=hdr)

                    with urllib.request.urlopen(
                        req,
                        context=ctx,
                        timeout=timeout,
                    ) as r:
                        return r.read()
                except Exception as e2:
                    ultimo = e2

            time.sleep(1 + intento)

        except Exception as e:
            ultimo = e
            time.sleep(1 + intento)

    raise RuntimeError(f"No se pudo descargar {url}: {ultimo}")


def descargar_json(url, timeout=None):
    raw = descargar_bytes(
        url,
        timeout=timeout,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Linux; Android 13) AppleWebKit/537.36 "
                "Chrome/124.0 Safari/537.36 Gatuno/10.4.2"
            ),
            "Accept": "application/json,text/plain,*/*",
            "Cache-Control": "no-cache",
        },
    )

    if parece_html(raw):
        raise ValueError("Se recibió HTML en lugar de JSON")

    return json.loads(raw.decode("utf-8", errors="replace"))


def csv_desde_bytes(raw, etiqueta="CSV"):
    if parece_html(raw):
        raise ValueError(f"{etiqueta}: HTML recibido en vez de CSV")

    ultimo = None

    for enc in ("utf-8-sig", "latin-1"):
        try:
            df = pd.read_csv(
                io.BytesIO(raw),
                encoding=enc,
                on_bad_lines="skip",
            )

            df.columns = [
                str(c).replace("\ufeff", "").strip()
                for c in df.columns
            ]

            if any("<HTML" in str(c).upper() for c in df.columns):
                raise ValueError("HTML interpretado como CSV")

            return df

        except Exception as e:
            ultimo = e

    raise ValueError(f"{etiqueta}: {ultimo}")


def cache_json_path(nombre):
    seguro = re.sub(r"[^A-Za-z0-9_.-]+", "_", nombre)
    return os.path.join(CACHE_DIR, seguro + ".json")


def guardar_json_cache(path, data):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
    except Exception:
        pass


def leer_json_cache(path, max_age_hours=None):
    try:
        if not os.path.exists(path):
            return None

        if max_age_hours is not None:
            edad_h = (
                time.time() - os.path.getmtime(path)
            ) / 3600.0

            if edad_h > max_age_hours:
                return None

        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    except Exception:
        return None


# ============================================================
# NORMALIZACION
# ============================================================

def numero(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else np.nan
    except Exception:
        return np.nan


def normalizar_df(df):
    df = df.copy()

    df.columns = [
        str(c).replace("\ufeff", "").strip()
        for c in df.columns
    ]

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(
            df["Date"],
            dayfirst=True,
            errors="coerce",
        )

    for c in [
        "FTHG", "FTAG", "HTHG", "HTAG",
        "HC", "AC", "HY", "AY", "HR", "AR",
        "B365H", "B365D", "B365A",
        "AvgH", "AvgD", "AvgA",
        "PSH", "PSD", "PSA",
        "WHH", "WHD", "WHA",
        "VCH", "VCD", "VCA",
        "MaxH", "MaxD", "MaxA",
        "EspnH", "EspnD", "EspnA",
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def norm_texto(s):
    s = "" if s is None else str(s)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower()

    for a, b in {
        "&": " and ",
        "-": " ",
        ".": " ",
        "'": "",
        "/": " ",
    }.items():
        s = s.replace(a, b)

    tokens = re.findall(r"[a-z0-9]+", s)

    basura = {
        "fc", "afc", "cf", "club", "de", "the",
        "football", "calcio", "ssc", "ac", "sc",
        "cd", "ca", "ec",
    }

    return " ".join(t for t in tokens if t not in basura)


ALIASES = {
    "manchester united": "man united",
    "manchester city": "man city",
    "tottenham hotspur": "tottenham",
    "wolverhampton wanderers": "wolves",
    "west ham united": "west ham",
    "newcastle united": "newcastle",
    "nottingham forest": "nottm forest",
    "brighton hove albion": "brighton",
    "paris saint germain": "paris sg",
    "internazionale": "inter",
    "inter milan": "inter",
    "ac milan": "milan",
    "athletic club": "ath bilbao",
    "celta vigo": "celta",
    "real betis": "betis",
    "real sociedad": "sociedad",
    "deportivo alaves": "alaves",
    "rayo vallecano": "vallecano",
    "borussia dortmund": "dortmund",
    "eintracht frankfurt": "ein frankfurt",
    "sporting cp": "sporting",
    "fc porto": "porto",
}


def resolver_nombre(nombre, candidatos):
    if not candidatos:
        return nombre, 0.0

    objetivo = ALIASES.get(norm_texto(nombre), norm_texto(nombre))

    mapa = {
        norm_texto(c): c
        for c in candidatos
    }

    if objetivo in mapa:
        return mapa[objetivo], 1.0

    to = set(objetivo.split())

    mejor = nombre
    score_mejor = 0.0

    for n, original in mapa.items():
        seq = SequenceMatcher(None, objetivo, n).ratio()

        tn = set(n.split())
        un = len(to | tn)
        jac = len(to & tn) / un if un else 0.0

        cont = 1.0 if objetivo in n or n in objetivo else 0.0

        score = 0.62 * seq + 0.28 * jac + 0.10 * cont

        if score > score_mejor:
            score_mejor = score
            mejor = original

    if score_mejor < 0.57:
        return nombre, score_mejor

    return mejor, score_mejor


# ============================================================
# CUOTAS / FAVORITO
# ============================================================

def primera_cuota(row, columnas):
    for c in columnas:
        if c in row.index:
            v = numero(row.get(c, np.nan))
            if pd.notna(v) and v > 1.0:
                return float(v), c

    return np.nan, ""


def american_a_decimal(ml):
    ml = numero(ml)

    if pd.isna(ml) or ml == 0:
        return np.nan

    if ml > 0:
        return 1.0 + ml / 100.0

    return 1.0 + 100.0 / abs(ml)


def detectar_favorito_mercado(row):
    h, fh = primera_cuota(
        row,
        ["AvgH", "B365H", "PSH", "WHH", "VCH", "MaxH", "EspnH"],
    )
    d, fd = primera_cuota(
        row,
        ["AvgD", "B365D", "PSD", "WHD", "VCD", "MaxD", "EspnD"],
    )
    a, fa = primera_cuota(
        row,
        ["AvgA", "B365A", "PSA", "WHA", "VCA", "MaxA", "EspnA"],
    )

    if pd.isna(h) or pd.isna(a):
        return None

    if h < a:
        tipo = "HOME"
        cuota = h
    elif a < h:
        tipo = "AWAY"
        cuota = a
    else:
        return None

    p_market = np.nan

    if all(pd.notna(x) and x > 1.0 for x in (h, d, a)):
        inv = np.array([1/h, 1/d, 1/a], dtype=float)
        inv = inv / inv.sum()
        p_market = float(inv[0] if tipo == "HOME" else inv[2])

    fuente = (
        "ESPN"
        if fh.startswith("Espn") or fa.startswith("Espn")
        else "Football-Data"
    )

    return {
        "tipo": tipo,
        "cuota": float(cuota),
        "p_market": p_market,
        "fuente": fuente,
    }


# ============================================================
# FOOTBALL-DATA EUROPA
# ============================================================

def descargar_historico_fd():
    partes = []

    divs = {
        info["fd_div"]: key
        for key, info in COMPETICIONES.items()
        if info["fd_div"]
    }

    for temp in TEMPORADAS_FD:
        log(f"Football-Data {temp}...")

        zip_url = f"{BASE_HIST_FD}{temp}/data.zip"

        try:
            raw = descargar_bytes(zip_url)

            if parece_html(raw) or not raw.startswith(b"PK"):
                raise ValueError("ZIP no válido")

            z = zipfile.ZipFile(io.BytesIO(raw))
            nombres = list(z.namelist())

            for div, comp_key in divs.items():
                objetivo = f"{div}.csv".lower()

                arch = next(
                    (
                        n
                        for n in nombres
                        if n.strip("./").lower() == objetivo
                    ),
                    None,
                )

                if not arch:
                    continue

                df = csv_desde_bytes(
                    z.read(arch),
                    f"{div}-{temp}",
                )

                if not {
                    "Date", "HomeTeam", "AwayTeam",
                    "FTHG", "FTAG",
                }.issubset(df.columns):
                    continue

                df = normalizar_df(df)
                df["CompKey"] = comp_key
                df["Grupo"] = COMPETICIONES[comp_key]["grupo"]
                df["Competicion"] = COMPETICIONES[comp_key]["nombre"]
                df["FuenteHist"] = "Football-Data"
                partes.append(df)

        except Exception as e:
            log(f"  ZIP falló: {e}")

            # Si la red completa no respondio, probar cada liga por separado
            # multiplica el bloqueo (varios timeouts por temporada). En la
            # interfaz rapida se continua con las otras fuentes/temporadas.
            if os.environ.get("GATUNO_FAST_MODE", "0") == "1":
                continue

            for div, comp_key in divs.items():
                url = f"{BASE_HIST_FD}{temp}/{div}.csv"

                try:
                    raw = descargar_bytes(url)
                    df = normalizar_df(
                        csv_desde_bytes(raw, f"{div}-{temp}")
                    )

                    if not {
                        "Date", "HomeTeam", "AwayTeam",
                        "FTHG", "FTAG",
                    }.issubset(df.columns):
                        continue

                    df["CompKey"] = comp_key
                    df["Grupo"] = COMPETICIONES[comp_key]["grupo"]
                    df["Competicion"] = COMPETICIONES[comp_key]["nombre"]
                    df["FuenteHist"] = "Football-Data"
                    partes.append(df)

                except Exception:
                    pass

    if not partes:
        return pd.DataFrame()

    h = pd.concat(partes, ignore_index=True, sort=False)

    h = h.dropna(
        subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]
    )

    return h


# ============================================================
# ESPN - PARSERS
# ============================================================

def parse_inline_odds(comp):
    odds = comp.get("odds") or []

    if isinstance(odds, dict):
        odds = [odds]

    for o in odds:
        if not isinstance(o, dict):
            continue

        ho = o.get("homeTeamOdds") or {}
        ao = o.get("awayTeamOdds") or {}
        do = o.get("drawOdds") or {}

        h = american_a_decimal(ho.get("moneyLine"))
        a = american_a_decimal(ao.get("moneyLine"))
        d = american_a_decimal(do.get("moneyLine"))

        if pd.notna(h) and pd.notna(a):
            return h, d, a

    return np.nan, np.nan, np.nan


def buscar_stat(stats, nombres):
    candidatos = {
        str(s.get("name", "")).lower(): s.get("displayValue", s.get("value"))
        for s in (stats or [])
        if isinstance(s, dict)
    }

    for n in nombres:
        if n.lower() in candidatos:
            return numero(candidatos[n.lower()])

    return np.nan


def parse_summary_stats(summary, home_id, away_id):
    out = {
        "HC": np.nan,
        "AC": np.nan,
        "HY": np.nan,
        "AY": np.nan,
        "HTHG": np.nan,
        "HTAG": np.nan,
        "Referee": "",
    }

    # Boxscore stats
    teams = (
        (summary.get("boxscore") or {}).get("teams")
        or []
    )

    for t in teams:
        tid = str(
            ((t.get("team") or {}).get("id"))
            or ""
        )

        stats = t.get("statistics") or []

        corners = buscar_stat(
            stats,
            ["cornerKicks", "corners", "corner kicks"],
        )

        yellow = buscar_stat(
            stats,
            ["yellowCards", "yellow cards"],
        )

        if tid == str(home_id):
            out["HC"] = corners
            out["HY"] = yellow
        elif tid == str(away_id):
            out["AC"] = corners
            out["AY"] = yellow

    # Half-time score from header linescores.
    try:
        comps = (
            (summary.get("header") or {})
            .get("competitions", [])
        )

        if comps:
            competitors = comps[0].get("competitors") or []

            for c in competitors:
                tid = str((c.get("team") or {}).get("id", ""))
                ls = c.get("linescores") or []

                if ls:
                    first = numero(ls[0].get("value"))

                    if tid == str(home_id):
                        out["HTHG"] = first
                    elif tid == str(away_id):
                        out["HTAG"] = first
    except Exception:
        pass

    # Referee
    try:
        game_info = summary.get("gameInfo") or {}
        officials = game_info.get("officials") or []

        if officials:
            o = officials[0]
            out["Referee"] = (
                o.get("displayName")
                or o.get("fullName")
                or ""
            )
    except Exception:
        pass

    return out


def extraer_venue(comp):
    venue = comp.get("venue") or {}
    addr = venue.get("address") or {}

    return {
        "Stadium": venue.get("fullName") or "",
        "VenueCity": addr.get("city") or "",
        "VenueState": addr.get("state") or "",
        "VenueCountry": addr.get("country") or "",
    }


# ============================================================
# ESPN HISTORICO SUDAMERICA
# ============================================================

def resumen_evento_espn(league, event_id, cache_hours=None):
    path = cache_json_path(f"summary_{league}_{event_id}")

    cached = leer_json_cache(
        path,
        max_age_hours=cache_hours,
    )

    if cached is not None:
        return cached

    url = (
        "https://site.api.espn.com/apis/site/v2/sports/"
        f"soccer/{league}/summary?event={event_id}"
    )

    data = descargar_json(url)
    guardar_json_cache(path, data)

    return data


def _scoreboard_cached_request(league, start_date, end_date):
    """Consulta ESPN y conserva la última respuesta válida por intervalo.

    Devuelve ``None`` únicamente cuando no existe respuesta de red ni caché.
    Una lista vacía sí es una respuesta válida: puede no haber partidos.
    """
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    token = start.strftime("%Y%m%d")
    if end.date() != start.date():
        token += "-" + end.strftime("%Y%m%d")
    path = cache_json_path(f"scoreboard_{league}_{token}")
    fresh = leer_json_cache(path, max_age_hours=2)
    if isinstance(fresh, dict) and isinstance(fresh.get("events"), list):
        return fresh["events"]

    url = (
        "https://site.api.espn.com/apis/site/v2/sports/"
        f"soccer/{league}/scoreboard?dates={token}&limit=1000"
    )
    try:
        payload = descargar_json(url, timeout=5)
        if not isinstance(payload, dict) or not isinstance(payload.get("events"), list):
            raise ValueError("ESPN devolvió un calendario inválido")
        guardar_json_cache(path, payload)
        return payload["events"]
    except Exception:
        # Una agenda guardada por hasta 72 horas es preferible a dejar toda la
        # aplicación vacía por una caída transitoria. El filtro de kickoff
        # elimina después cualquier partido ya iniciado.
        stale = leer_json_cache(path, max_age_hours=72)
        if isinstance(stale, dict) and isinstance(stale.get("events"), list):
            return stale["events"]
        return None


def eventos_scoreboard_espn(league, start_date, end_date):
    """Obtiene eventos sin depender de una única consulta extensa.

    El error de V15.0 era pedir 16 días y permitir recuperación sólo cuando
    el intervalo tenía 10 días o menos. V15.1 divide las ventanas futuras en
    bloques de siete días y, si un bloque falla, reintenta sus días. Para
    históricos largos se mantiene una sola petición para evitar cientos de
    llamadas.
    """
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    if end < start:
        return []
    span_days = int((end - start).days) + 1

    if span_days > 31:
        result = _scoreboard_cached_request(league, start, end)
        return result if result is not None else []

    eventos = []
    chunk_start = start
    while chunk_start <= end:
        chunk_end = min(end, chunk_start + pd.Timedelta(days=6))
        result = _scoreboard_cached_request(league, chunk_start, chunk_end)
        if result is None or (not result and chunk_end > chunk_start):
            # ESPN puede responder una lista vacía a una consulta por rango
            # aunque sí entregue eventos al consultar las fechas una a una.
            # También se usa esta ruta cuando la consulta extensa falla.
            # Primero se prueba un solo día; si ni siquiera responde, se evita
            # multiplicar el mismo timeout por cada fecha del intervalo.
            first_daily = _scoreboard_cached_request(league, chunk_start, chunk_start)
            if first_daily is not None:
                eventos.extend(first_daily)
                day = chunk_start + pd.Timedelta(days=1)
                while day <= chunk_end:
                    daily = _scoreboard_cached_request(league, day, day)
                    if daily:
                        eventos.extend(daily)
                    day += pd.Timedelta(days=1)
        else:
            eventos.extend(result)
        chunk_start = chunk_end + pd.Timedelta(days=1)

    unicos = {}
    for event in eventos:
        event_id = str(event.get("id") or "")
        if event_id:
            unicos[event_id] = event
    return list(unicos.values())


def descargar_historico_espn_sam():
    comps = [
        k
        for k, v in COMPETICIONES.items()
        if v["hist_espn"]
    ]

    fin = pd.Timestamp(datetime.now(TZ_PERU).date())
    inicio = fin - pd.Timedelta(days=ESPN_HIST_DAYS)

    rows = []
    summary_calls = 0

    for comp_key in comps:
        info = COMPETICIONES[comp_key]
        league = info["espn"]

        log(f"ESPN histórico {info['nombre']}...")

        try:
            eventos = eventos_scoreboard_espn(
                league,
                inicio,
                fin,
            )
        except Exception as e:
            log(f"  No se pudo obtener scoreboard: {e}")
            continue

        # Solo completados.
        completos = []

        for ev in eventos:
            st = (
                (ev.get("status") or {})
                .get("type", {})
            )

            if st.get("completed") or st.get("state") == "post":
                completos.append(ev)

        completos = sorted(
            completos,
            key=lambda e: e.get("date", ""),
        )[-MAX_ESPN_RESULT_EVENTS_PER_COMP:]

        summary_event_ids = {
            str(e.get("id"))
            for e in completos[-MAX_ESPN_HIST_EVENTS_PER_COMP:]
        }

        log(f"  Eventos a procesar: {len(completos)}")

        for ev in completos:
            comps_ev = ev.get("competitions") or []
            if not comps_ev:
                continue

            c = comps_ev[0]
            sides = {
                x.get("homeAway"): x
                for x in (c.get("competitors") or [])
            }

            if "home" not in sides or "away" not in sides:
                continue

            home_c = sides["home"]
            away_c = sides["away"]

            home_t = home_c.get("team") or {}
            away_t = away_c.get("team") or {}

            home = (
                home_t.get("displayName")
                or home_t.get("shortDisplayName")
                or ""
            )
            away = (
                away_t.get("displayName")
                or away_t.get("shortDisplayName")
                or ""
            )

            if not home or not away:
                continue

            hg = numero(home_c.get("score"))
            ag = numero(away_c.get("score"))

            if pd.isna(hg) or pd.isna(ag):
                continue

            dt = pd.to_datetime(
                ev.get("date"),
                utc=True,
                errors="coerce",
            )

            if pd.isna(dt):
                continue

            hodd, dodd, aodd = parse_inline_odds(c)

            stats = {
                "HC": np.nan,
                "AC": np.nan,
                "HY": np.nan,
                "AY": np.nan,
                "HTHG": np.nan,
                "HTAG": np.nan,
                "Referee": "",
            }

            if (
                str(ev.get("id")) in summary_event_ids
                and summary_calls < MAX_ESPN_SUMMARY_CALLS_HIST
            ):
                try:
                    summary = resumen_evento_espn(
                        league,
                        ev.get("id"),
                    )
                    summary_calls += 1
                    stats = parse_summary_stats(
                        summary,
                        home_t.get("id"),
                        away_t.get("id"),
                    )
                except Exception:
                    pass

            rows.append({
                "Date": dt.tz_convert(TZ_PERU).tz_localize(None),
                "HomeTeam": home,
                "AwayTeam": away,
                "FTHG": hg,
                "FTAG": ag,
                "HC": stats["HC"],
                "AC": stats["AC"],
                "HY": stats["HY"],
                "AY": stats["AY"],
                "HTHG": stats["HTHG"],
                "HTAG": stats["HTAG"],
                "Referee": stats["Referee"],
                "EspnH": hodd,
                "EspnD": dodd,
                "EspnA": aodd,
                "CompKey": comp_key,
                "Grupo": info["grupo"],
                "Competicion": info["nombre"],
                "FuenteHist": "ESPN",
            })

            time.sleep(PAUSA_ESPN)

    if not rows:
        return pd.DataFrame()

    return normalizar_df(pd.DataFrame(rows))


# ============================================================
# HISTORICO TOTAL
# ============================================================

def cargar_historico_empaquetado():
    """Respaldo autocontenido; no necesita un .csv.gz adicional."""
    errors = []
    # El gzip está incorporado en un módulo Python para que GitHub móvil no
    # pueda omitir un archivo binario o colocarlo en otra carpeta.
    try:
        from embedded_history import gzip_bytes

        fallback = pd.read_csv(io.BytesIO(gzip_bytes()), compression="gzip")
        fallback["Date"] = pd.to_datetime(fallback["Date"], errors="coerce")
        fallback = fallback.dropna(
            subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]
        )
        if not fallback.empty:
            log(f"Historico Gatuno autocontenido cargado ({len(fallback)} filas)")
            return normalizar_df(fallback)
    except Exception as exc:
        errors.append(f"embedded_history.py: {exc}")
    detail = "; ".join(errors) if errors else "archivo no encontrado"
    log(f"Historico empaquetado no disponible: {detail}")
    return pd.DataFrame()


def descargar_historico_total():
    fd = descargar_historico_fd()
    sam = descargar_historico_espn_sam()
    bundled = cargar_historico_empaquetado()

    frames = [
        x
        for x in (fd, sam, bundled)
        if x is not None and not x.empty
    ]

    if not frames:
        raise RuntimeError("No se obtuvo histórico utilizable")

    h = pd.concat(
        frames,
        ignore_index=True,
        sort=False,
    )

    h = normalizar_df(h)

    h = h.dropna(
        subset=[
            "Date", "HomeTeam", "AwayTeam",
            "FTHG", "FTAG",
        ]
    )

    # Las fuentes remotas van primero y conservan prioridad. El respaldo solo
    # completa temporadas/equipos ausentes; no reemplaza una observacion nueva.
    h = h.drop_duplicates(
        subset=["Date", "CompKey", "HomeTeam", "AwayTeam"],
        keep="first",
    )

    return h.sort_values(
        ["Date", "CompKey", "HomeTeam"]
    ).reset_index(drop=True)


# ============================================================
# ESTADO DE EQUIPO
# ============================================================

def registro_equipo(row, home=True):
    if home:
        gf = row["FTHG"]
        gc = row["FTAG"]
        cf = row.get("HC", np.nan)
        ca = row.get("AC", np.nan)
        cards_f = row.get("HY", np.nan)
        cards_a = row.get("AY", np.nan)
        ht_gf = row.get("HTHG", np.nan)
        ht_ga = row.get("HTAG", np.nan)
    else:
        gf = row["FTAG"]
        gc = row["FTHG"]
        cf = row.get("AC", np.nan)
        ca = row.get("HC", np.nan)
        cards_f = row.get("AY", np.nan)
        cards_a = row.get("HY", np.nan)
        ht_gf = row.get("HTAG", np.nan)
        ht_ga = row.get("HTHG", np.nan)

    cf = numero(cf)
    ca = numero(ca)
    cards_f = numero(cards_f)
    cards_a = numero(cards_a)
    ht_gf = numero(ht_gf)
    ht_ga = numero(ht_ga)

    return {
        "gf": float(gf),
        "gc": float(gc),
        "cf": cf,
        "ca": ca,
        "marca": int(gf >= 1),
        "concede": int(gc >= 1),
        "under45": int(gf + gc <= 4),
        "c4": int(pd.notna(cf) and cf >= 4),
        "allow_c4": int(pd.notna(ca) and ca >= 4),
        "cards_f": cards_f,
        "cards_a": cards_a,
        "ht_goal": (
            int((ht_gf + ht_ga) >= 1)
            if pd.notna(ht_gf) and pd.notna(ht_ga)
            else np.nan
        ),
        "total": float(gf + gc),
    }


def media(vals, default=0.0):
    vals = [
        float(v)
        for v in vals
        if pd.notna(v)
    ]
    return float(np.mean(vals)) if vals else default


def tasa(vals, prior=0.5, fuerza=4.0):
    vals = [
        float(v)
        for v in vals
        if pd.notna(v)
    ]

    n = len(vals)

    if n == 0:
        return prior

    return float(
        (sum(vals) + prior * fuerza)
        / (n + fuerza)
    )


def snapshot(regs):
    if len(regs) < MIN_PARTIDOS_EQUIPO:
        return None

    out = {"N": len(regs)}

    for w in VENTANAS:
        r = regs[-w:]

        out[f"Marca{w}"] = tasa([x["marca"] for x in r])
        out[f"Concede{w}"] = tasa([x["concede"] for x in r])
        out[f"Under45_{w}"] = tasa([x["under45"] for x in r])
        out[f"C4_{w}"] = tasa([x["c4"] for x in r])
        out[f"AllowC4_{w}"] = tasa([x["allow_c4"] for x in r])
        out[f"HTGoal_{w}"] = tasa([x["ht_goal"] for x in r])

        out[f"GF{w}"] = media([x["gf"] for x in r])
        out[f"GC{w}"] = media([x["gc"] for x in r])
        out[f"CF{w}"] = media([x["cf"] for x in r])
        out[f"CA{w}"] = media([x["ca"] for x in r])
        out[f"CardsF{w}"] = media([x["cards_f"] for x in r])
        out[f"CardsA{w}"] = media([x["cards_a"] for x in r])
        out[f"Total{w}"] = media(
            [x["total"] for x in r],
            default=2.5,
        )

    return out


def agregar_estado(estados, row):
    estados[row["HomeTeam"]].append(
        registro_equipo(row, home=True)
    )
    estados[row["AwayTeam"]].append(
        registro_equipo(row, home=False)
    )


def fuerza_equipo(st):
    if st is None:
        return np.nan

    return (
        0.65 * st["GF10"]
        - 0.55 * st["GC10"]
        + 0.45 * st["Marca10"]
        + 0.12 * st["CF10"]
    )


def favorito_modelo(hs, aws):
    if hs is None or aws is None:
        return None

    fh = fuerza_equipo(hs) + 0.10
    fa = fuerza_equipo(aws)

    if not np.isfinite(fh) or not np.isfinite(fa):
        return None

    dif = fh - fa

    return {
        "tipo": "HOME" if dif >= 0 else "AWAY",
        "cuota": np.nan,
        "p_market": np.nan,
        "fuente": "MODELO_SIN_CUOTA",
    }


# ============================================================
# FEATURES
# ============================================================

FEATURE_WEIGHTS = {
    "DogMarca10": 1.35,
    "DogMarca20": 1.00,
    "FavConcede10": 1.15,
    "FavConcede20": 0.95,
    "FavC4_10": 1.40,
    "FavC4_20": 1.15,
    "DogAllowC4_10": 1.20,
    "DogAllowC4_20": 1.00,
    "HomeUnder45_10": 1.10,
    "AwayUnder45_10": 1.10,
    "AvgTotal10": 1.05,
    "FavCF10": 0.85,
    "DogCA10": 0.85,
    "CardsTotalForm10": 0.70,
    "HTGoalForm10": 0.70,
    "MarketFavProb": 1.10,
    "FavIsHome": 0.45,
}

FEATURES = list(FEATURE_WEIGHTS.keys())


def crear_features(row, hs, aws, fav=None):
    if hs is None or aws is None:
        return None

    if fav is None:
        fav = detectar_favorito_mercado(row)

    if fav is None:
        fav = favorito_modelo(hs, aws)

    if fav is None:
        return None

    if fav["tipo"] == "HOME":
        fs = hs
        ds = aws
        fav_home = 1.0
    else:
        fs = aws
        ds = hs
        fav_home = 0.0

    p_market = fav.get("p_market", np.nan)

    if pd.isna(p_market):
        p_market = 0.60

    return {
        "DogMarca10": ds["Marca10"],
        "DogMarca20": ds["Marca20"],
        "FavConcede10": fs["Concede10"],
        "FavConcede20": fs["Concede20"],
        "FavC4_10": fs["C4_10"],
        "FavC4_20": fs["C4_20"],
        "DogAllowC4_10": ds["AllowC4_10"],
        "DogAllowC4_20": ds["AllowC4_20"],
        "HomeUnder45_10": hs["Under45_10"],
        "AwayUnder45_10": aws["Under45_10"],
        "AvgTotal10": (hs["Total10"] + aws["Total10"]) / 2.0,
        "FavCF10": fs["CF10"],
        "DogCA10": ds["CA10"],
        "CardsTotalForm10": (
            hs["CardsF10"] + aws["CardsF10"]
        ),
        "HTGoalForm10": (
            hs["HTGoal_10"] + aws["HTGoal_10"]
        ) / 2.0,
        "MarketFavProb": p_market,
        "FavIsHome": fav_home,
        "FavoritoTipo": fav["tipo"],
        "CuotaFavorito": fav.get("cuota", np.nan),
        "FuenteFavorito": fav.get("fuente", ""),
    }


# ============================================================
# DATASET SIN FUTURE LEAKAGE
# ============================================================

def construir_dataset(hist):
    estados = defaultdict(list)
    rows = []

    orden = hist.sort_values(
        ["Date", "CompKey", "HomeTeam"]
    ).copy()

    for fecha, bloque in orden.groupby("Date", sort=True):
        # 1) features con datos estrictamente anteriores a la fecha
        for _, row in bloque.iterrows():
            hs = snapshot(estados[row["HomeTeam"]])
            aws = snapshot(estados[row["AwayTeam"]])

            fav = detectar_favorito_mercado(row)

            if fav is None:
                fav = favorito_modelo(hs, aws)

            feat = crear_features(
                row,
                hs,
                aws,
                fav=fav,
            )

            if feat is None:
                continue

            if fav["tipo"] == "HOME":
                dog_gol = int(row["FTAG"] >= 1)
                fav_corner = row.get("HC", np.nan)
            else:
                dog_gol = int(row["FTHG"] >= 1)
                fav_corner = row.get("AC", np.nan)

            fav4 = (
                int(float(fav_corner) >= 4)
                if pd.notna(fav_corner)
                else np.nan
            )

            under45 = int(
                row["FTHG"] + row["FTAG"] <= 4
            )

            base = (
                int(
                    dog_gol
                    and fav4 == 1
                    and under45 == 1
                )
                if pd.notna(fav4)
                else np.nan
            )

            hy = numero(row.get("HY", np.nan))
            ay = numero(row.get("AY", np.nan))

            cards4 = (
                int(hy + ay >= 4)
                if pd.notna(hy) and pd.notna(ay)
                else np.nan
            )

            hthg = numero(row.get("HTHG", np.nan))
            htag = numero(row.get("HTAG", np.nan))

            gol1t = (
                int(hthg + htag >= 1)
                if pd.notna(hthg) and pd.notna(htag)
                else np.nan
            )

            base_cards = (
                int(base == 1 and cards4 == 1)
                if pd.notna(base) and pd.notna(cards4)
                else np.nan
            )

            base_gol1t = (
                int(base == 1 and gol1t == 1)
                if pd.notna(base) and pd.notna(gol1t)
                else np.nan
            )

            rows.append({
                "Date": row["Date"],
                "CompKey": row["CompKey"],
                "Grupo": row["Grupo"],
                "Competicion": row["Competicion"],
                "HomeTeam": row["HomeTeam"],
                "AwayTeam": row["AwayTeam"],
                "Referee": row.get("Referee", ""),
                **feat,
                "Y_DogGol": dog_gol,
                "Y_Fav4": fav4,
                "Y_Under45": under45,
                "Y_BASE": base,
                "Y_Cards4": cards4,
                "Y_Gol1T": gol1t,
                "Y_BASE_CARDS": base_cards,
                "Y_BASE_GOL1T": base_gol1t,
            })

        # 2) recién ahora agregamos resultados de la fecha
        for _, row in bloque.iterrows():
            agregar_estado(estados, row)

    ds = pd.DataFrame(rows)

    if ds.empty:
        raise RuntimeError("Dataset de modelado vacío")

    return (
        ds.sort_values("Date").reset_index(drop=True),
        estados,
    )


# ============================================================
# PERFIL DE ÁRBITROS
# ============================================================

def construir_perfil_arbitros(hist):
    if "Referee" not in hist.columns:
        return {}

    h = hist.copy()

    h["CardsTotal"] = (
        pd.to_numeric(h.get("HY"), errors="coerce")
        + pd.to_numeric(h.get("AY"), errors="coerce")
    )

    h["RefNorm"] = h["Referee"].map(norm_texto)

    h = h[
        (h["RefNorm"] != "")
        & h["CardsTotal"].notna()
    ]

    perfiles = {}

    for ref, g in h.groupby("RefNorm"):
        if len(g) >= 5:
            perfiles[ref] = {
                "n": int(len(g)),
                "cards_avg": float(g["CardsTotal"].mean()),
                "cards4_rate": float((g["CardsTotal"] >= 4).mean()),
            }

    return perfiles


def buscar_perfil_arbitro(nombre, perfiles):
    if not nombre or not perfiles:
        return None

    n = norm_texto(nombre)

    if n in perfiles:
        return perfiles[n]

    mejor = None
    score = 0.0

    for ref, perfil in perfiles.items():
        s = SequenceMatcher(None, n, ref).ratio()

        if s > score:
            score = s
            mejor = perfil

    return mejor if score >= 0.83 else None


# ============================================================
# KNN
# ============================================================

def seleccionar_train(train, comp_key, grupo):
    t = train.copy()

    exacto = t[t["CompKey"] == comp_key]

    if len(exacto) >= K_MINIMO * 2:
        return exacto

    mismo_grupo = t[t["Grupo"] == grupo]

    if len(mismo_grupo) >= K_MINIMO * 3:
        return mismo_grupo

    return t


def knn_probabilidades(
    train,
    x,
    fecha_objetivo,
    comp_key,
    grupo,
):
    t = seleccionar_train(
        train,
        comp_key,
        grupo,
    )

    t = t.dropna(subset=FEATURES)

    if len(t) < K_MINIMO:
        return None

    if len(t) > MAX_TRAIN:
        t = t.tail(MAX_TRAIN)

    X = t[FEATURES].astype(float).to_numpy()
    xv = np.array(
        [float(x[c]) for c in FEATURES],
        dtype=float,
    )

    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0)

    sd[
        (~np.isfinite(sd))
        | (sd < 1e-6)
    ] = 1.0

    Z = (X - mu) / sd
    zv = (xv - mu) / sd

    fw = np.array(
        [FEATURE_WEIGHTS[c] for c in FEATURES],
        dtype=float,
    )

    dist = np.sqrt(
        np.nanmean(
            ((Z - zv) ** 2) * fw,
            axis=1,
        )
    )

    k = min(K_VECINOS, len(t))
    idx = np.argsort(dist)[:k]

    vecinos = t.iloc[idx].copy()
    d = dist[idx]

    w_sim = np.clip(
        1.0 / (0.30 + d),
        0.12,
        3.0,
    )

    edad = (
        pd.Timestamp(fecha_objetivo)
        - pd.to_datetime(vecinos["Date"])
    ).dt.days.to_numpy(dtype=float)

    edad = np.maximum(edad, 0.0)

    w_rec = np.exp(-edad / 540.0)
    weights = w_sim * w_rec

    def posterior(target):
        y = pd.to_numeric(
            vecinos[target],
            errors="coerce",
        ).to_numpy(dtype=float)

        valid = np.isfinite(y)

        if valid.sum() < 22:
            return np.nan, 0.0, 0

        yy = y[valid]
        ww = weights[valid]

        base_series = pd.to_numeric(
            t[target],
            errors="coerce",
        )

        base = (
            float(base_series.mean())
            if base_series.notna().any()
            else 0.5
        )

        sw = float(ww.sum())

        p = (
            float(np.dot(ww, yy))
            + PRIOR_STRENGTH * base
        ) / (
            sw + PRIOR_STRENGTH
        )

        n_eff = (
            sw ** 2
            / max(float(np.dot(ww, ww)), 1e-9)
        )

        return (
            float(np.clip(p, 0.01, 0.99)),
            float(n_eff),
            int(valid.sum()),
        )

    targets = [
        "Y_DogGol",
        "Y_Fav4",
        "Y_Under45",
        "Y_BASE",
        "Y_Cards4",
        "Y_Gol1T",
        "Y_BASE_CARDS",
        "Y_BASE_GOL1T",
    ]

    out = {}

    for target in targets:
        p, n_eff, n_raw = posterior(target)
        out[target] = p
        out[target + "_Neff"] = n_eff
        out[target + "_N"] = n_raw

    if any(
        pd.isna(out[t])
        for t in (
            "Y_DogGol",
            "Y_Fav4",
            "Y_Under45",
            "Y_BASE",
        )
    ):
        return None

    p = out["Y_BASE"]
    n = out["Y_BASE_Neff"]

    se = math.sqrt(
        max(p * (1 - p), 1e-9)
        / max(n + PRIOR_STRENGTH, 1.0)
    )

    out["BASE_LCB"] = max(
        0.0,
        p - Z_LCB * se,
    )

    out["Confianza"] = min(
        1.0,
        n / 80.0,
    )

    return out


# ============================================================
# VENTANA FUTURA ESPN
# ============================================================

def ventana_objetivo(now=None):
    """Ventana móvil: desde hoy, sin excluir todo el día por una hora de corte."""
    now = now or datetime.now(TZ_PERU)
    if now.tzinfo is None:
        now = now.replace(tzinfo=TZ_PERU)
    else:
        now = now.astimezone(TZ_PERU)
    hoy = pd.Timestamp(now.date())
    inicio = hoy
    fin = inicio + pd.Timedelta(days=DIAS_VENTANA - 1)

    return inicio, fin


def extraer_future_event(ev, comp_key):
    info = COMPETICIONES[comp_key]

    comps = ev.get("competitions") or []

    if not comps:
        return None

    c = comps[0]

    competitors = c.get("competitors") or []

    sides = {
        x.get("homeAway"): x
        for x in competitors
    }

    if "home" not in sides or "away" not in sides:
        return None

    hc = sides["home"]
    ac = sides["away"]

    ht = hc.get("team") or {}
    at = ac.get("team") or {}

    home = (
        ht.get("displayName")
        or ht.get("shortDisplayName")
        or ""
    )
    away = (
        at.get("displayName")
        or at.get("shortDisplayName")
        or ""
    )

    if not home or not away:
        return None

    dt = pd.to_datetime(
        ev.get("date"),
        utc=True,
        errors="coerce",
    )

    if pd.isna(dt):
        return None

    venue = extraer_venue(c)

    hodd, dodd, aodd = parse_inline_odds(c)

    # ESPN cambia los nombres/campos de ronda entre competiciones. La capa
    # V9.3 consolida lo disponible sin asumir que "torneo grande" equivale
    # a once titular. Si no hay fase, queda explicitamente NO identificada.
    try:
        from gatuno_context import extract_stage_text
        stage_text = extract_stage_text(ev, c)
    except Exception:
        stage_text = ""

    status_type = (ev.get("status") or {}).get("type") or {}
    status_state = str(status_type.get("state") or "").lower()
    status_name = str(status_type.get("name") or status_type.get("description") or "")
    completed = bool(status_type.get("completed", False)) or status_state == "post"

    return {
        "Date": pd.Timestamp(
            dt.tz_convert(TZ_PERU).date()
        ),
        "KickoffUTC": dt.isoformat(),
        "HoraPeru": dt.tz_convert(TZ_PERU).strftime("%H:%M"),
        "CompKey": comp_key,
        "Grupo": info["grupo"],
        "Competicion": info["nombre"],
        "TipoCompeticion": info["tipo"],
        "StageText": stage_text,
        "HomeOriginal": home,
        "AwayOriginal": away,
        "HomeTeam": home,
        "AwayTeam": away,
        "HomeESPNID": ht.get("id", ""),
        "AwayESPNID": at.get("id", ""),
        "EventID": ev.get("id", ""),
        "StatusState": status_state,
        "StatusName": status_name,
        "Completed": completed,
        "EspnH": hodd,
        "EspnD": dodd,
        "EspnA": aodd,
        "Stadium": venue["Stadium"],
        "VenueCity": venue["VenueCity"],
        "VenueState": venue["VenueState"],
        "VenueCountry": venue["VenueCountry"],
        "FuenteFixture": "ESPN",
    }


def filtrar_fixtures_desde_ahora(
    fixtures,
    now=None,
    margen_minutos=MARGEN_PREPARTIDO_MINUTOS,
):
    """Elimina únicamente partidos iniciados/terminados o demasiado próximos."""
    if fixtures is None or fixtures.empty:
        return fixtures

    now = now or datetime.now(TZ_PERU)
    if now.tzinfo is None:
        now = now.replace(tzinfo=TZ_PERU)
    else:
        now = now.astimezone(TZ_PERU)
    threshold_utc = pd.Timestamp(now).tz_convert("UTC") + pd.Timedelta(
        minutes=int(margen_minutos)
    )

    def keep(row):
        state = str(row.get("StatusState", "")).lower()
        if bool(row.get("Completed", False)) or state in {"in", "post"}:
            return False
        kickoff = pd.to_datetime(row.get("KickoffUTC"), utc=True, errors="coerce")
        if pd.isna(kickoff):
            # Sin hora fiable, no autoriza un partido del mismo día.
            return pd.Timestamp(row.get("Date")) > pd.Timestamp(now.date())
        return kickoff > threshold_utc

    mask = fixtures.apply(keep, axis=1)
    return fixtures.loc[mask].reset_index(drop=True)


def descargar_fixtures_football_data(inicio, fin):
    """Calendario europeo secundario cuando ESPN no responde.

    Football-Data publica una lista semanal de fixtures. La hora del archivo
    se interpreta como hora británica y se convierte a UTC/PET. La fuente no
    cubre todos los torneos sudamericanos, por lo que complementa a ESPN y no
    pretende sustituirla por completo.
    """
    try:
        raw = descargar_bytes(FD_FIXTURES_URL, timeout=10, reintentos=1)
        frame = csv_desde_bytes(raw, "Football-Data fixtures")
    except Exception as exc:
        log(f"  Football-Data fixtures no disponible: {exc}")
        return pd.DataFrame()

    required = {"Div", "Date", "HomeTeam", "AwayTeam"}
    if frame.empty or not required.issubset(frame.columns):
        return pd.DataFrame()

    frame = frame.copy()
    frame["DateParsed"] = pd.to_datetime(frame["Date"], dayfirst=True, errors="coerce")
    frame = frame[
        frame["Div"].astype(str).isin(FD_DIV_TO_COMP)
        & frame["DateParsed"].between(pd.Timestamp(inicio), pd.Timestamp(fin), inclusive="both")
    ]
    if frame.empty:
        return pd.DataFrame()

    rows = []
    london = ZoneInfo("Europe/London")
    for index, source in frame.iterrows():
        comp_key = FD_DIV_TO_COMP.get(str(source.get("Div", "")))
        if not comp_key:
            continue
        match_date = pd.Timestamp(source["DateParsed"])
        time_text = str(source.get("Time", "12:00") or "12:00").strip()
        if not re.fullmatch(r"\d{1,2}:\d{2}", time_text):
            time_text = "12:00"
        naive = pd.to_datetime(
            f"{match_date.date().isoformat()} {time_text}",
            errors="coerce",
        )
        if pd.isna(naive):
            continue
        try:
            kickoff_utc = pd.Timestamp(naive).tz_localize(
                london,
                ambiguous=False,
                nonexistent="shift_forward",
            ).tz_convert("UTC")
        except Exception:
            kickoff_utc = pd.Timestamp(naive).tz_localize("UTC")
        info = COMPETICIONES[comp_key]

        def odd(*names):
            for name in names:
                value = numero(source.get(name, np.nan))
                if pd.notna(value) and value > 1:
                    return float(value)
            return np.nan

        rows.append({
            "Date": pd.Timestamp(match_date.date()),
            "KickoffUTC": kickoff_utc.isoformat(),
            "HoraPeru": kickoff_utc.tz_convert(TZ_PERU).strftime("%H:%M"),
            "CompKey": comp_key,
            "Grupo": info["grupo"],
            "Competicion": info["nombre"],
            "TipoCompeticion": info["tipo"],
            "StageText": "",
            "HomeOriginal": str(source.get("HomeTeam", "")),
            "AwayOriginal": str(source.get("AwayTeam", "")),
            "HomeTeam": str(source.get("HomeTeam", "")),
            "AwayTeam": str(source.get("AwayTeam", "")),
            "HomeESPNID": "",
            "AwayESPNID": "",
            "EventID": f"FD-{comp_key}-{match_date:%Y%m%d}-{index}",
            "StatusState": "pre",
            "StatusName": "SCHEDULED",
            "Completed": False,
            "EspnH": odd("AvgH", "B365H", "PSH"),
            "EspnD": odd("AvgD", "B365D", "PSD"),
            "EspnA": odd("AvgA", "B365A", "PSA"),
            "Stadium": "",
            "VenueCity": "",
            "VenueState": "",
            "VenueCountry": "",
            "FuenteFixture": "Football-Data fixtures",
        })
    return normalizar_df(pd.DataFrame(rows)) if rows else pd.DataFrame()


def _sofa_comp_key(event):
    """Mapea sólo las competiciones objetivo; ante duda devuelve None."""
    tournament = event.get("tournament") or {}
    unique = tournament.get("uniqueTournament") or {}
    category = tournament.get("category") or {}
    name = norm_texto(unique.get("name") or tournament.get("name") or "")
    country = norm_texto(
        category.get("country", {}).get("name")
        if isinstance(category.get("country"), dict)
        else category.get("name") or ""
    )

    if "libertadores" in name:
        return "LIB"
    if "sudamericana" in name:
        return "SUD"
    if "europa league" in name and "conference" not in name:
        return "UEL"
    if "champions league" in name and "women" not in name and "youth" not in name:
        return "UCL"
    if "copa do brasil" in name:
        return "CDB"
    if country == "brazil" and (
        "brasileirao" in name or "brasileiro serie a" in name or name == "serie a"
    ):
        return "BRA"
    if country == "spain" and ("laliga" in name or "la liga" in name):
        return "ESP"
    if country == "portugal" and "liga portugal" in name:
        return "POR"
    if country in {"turkey", "turkiye"} and "super lig" in name:
        return "TUR"
    if country == "england" and name == "premier league":
        return "ENG"
    if country == "argentina" and any(
        token in name for token in ("liga profesional", "primera division", "primera nacional")
    ):
        # Primera Nacional no es la primera división y queda excluida.
        return None if "primera nacional" in name else "ARG"
    if country == "peru" and any(
        token in name for token in ("liga 1", "primera division")
    ):
        return "PER"
    return None


def _eventos_sofascore_fecha(day):
    token = pd.Timestamp(day).strftime("%Y-%m-%d")
    path = cache_json_path(f"sofascore_schedule_{token}")
    fresh = leer_json_cache(path, max_age_hours=2)
    if isinstance(fresh, dict) and isinstance(fresh.get("events"), list):
        return fresh["events"]

    endpoints = (
        f"https://www.sofascore.com/api/v1/sport/football/scheduled-events/{token}",
        f"https://api.sofascore.com/api/v1/sport/football/scheduled-events/{token}",
    )
    last_error = None
    for url in endpoints:
        try:
            payload = descargar_json(url, timeout=8)
            if not isinstance(payload, dict) or not isinstance(payload.get("events"), list):
                raise ValueError("respuesta sin lista de eventos")
            guardar_json_cache(path, payload)
            return payload["events"]
        except Exception as exc:
            last_error = exc

    stale = leer_json_cache(path, max_age_hours=72)
    if isinstance(stale, dict) and isinstance(stale.get("events"), list):
        return stale["events"]
    raise RuntimeError(f"Sofascore {token}: {last_error}")


def descargar_fixtures_sofascore(inicio, fin):
    """Respaldo diario y estricto para las doce competiciones configuradas."""
    days = list(pd.date_range(pd.Timestamp(inicio), pd.Timestamp(fin), freq="D"))
    events = []
    errors = []

    # Siete consultas diarias independientes; tres workers limitan carga y espera.
    with ThreadPoolExecutor(max_workers=3) as executor:
        jobs = {executor.submit(_eventos_sofascore_fecha, day): day for day in days}
        for job in as_completed(jobs):
            try:
                events.extend(job.result())
            except Exception as exc:
                errors.append(str(exc))

    rows = []
    seen = set()
    for event in events:
        comp_key = _sofa_comp_key(event)
        if not comp_key:
            continue
        home_obj = event.get("homeTeam") or {}
        away_obj = event.get("awayTeam") or {}
        home = str(home_obj.get("name") or home_obj.get("shortName") or "").strip()
        away = str(away_obj.get("name") or away_obj.get("shortName") or "").strip()
        timestamp = numero(event.get("startTimestamp"))
        if not home or not away or pd.isna(timestamp):
            continue
        kickoff_utc = pd.to_datetime(int(timestamp), unit="s", utc=True, errors="coerce")
        if pd.isna(kickoff_utc):
            continue
        local_time = kickoff_utc.tz_convert(TZ_PERU)
        match_date = pd.Timestamp(local_time.date())
        if match_date < pd.Timestamp(inicio) or match_date > pd.Timestamp(fin):
            continue

        event_id = str(event.get("id") or "")
        unique_key = event_id or f"{comp_key}-{int(timestamp)}-{norm_texto(home)}-{norm_texto(away)}"
        if unique_key in seen:
            continue
        seen.add(unique_key)

        status = event.get("status") or {}
        status_type = str(status.get("type") or "").lower()
        completed = status_type in {
            "finished", "afterextra", "afterpenalties", "canceled", "postponed"
        }
        status_state = "post" if completed else ("in" if status_type in {"inprogress", "live"} else "pre")
        tournament = event.get("tournament") or {}
        round_info = event.get("roundInfo") or {}
        info = COMPETICIONES[comp_key]
        rows.append({
            "Date": match_date,
            "KickoffUTC": kickoff_utc.isoformat(),
            "HoraPeru": local_time.strftime("%H:%M"),
            "CompKey": comp_key,
            "Grupo": info["grupo"],
            "Competicion": info["nombre"],
            "TipoCompeticion": info["tipo"],
            "StageText": str(round_info.get("name") or tournament.get("name") or ""),
            "HomeOriginal": home,
            "AwayOriginal": away,
            "HomeTeam": home,
            "AwayTeam": away,
            "HomeESPNID": "",
            "AwayESPNID": "",
            "EventID": f"SOFA-{event_id or unique_key}",
            "StatusState": status_state,
            "StatusName": str(status.get("description") or status_type or "SCHEDULED"),
            "Completed": completed,
            "EspnH": np.nan,
            "EspnD": np.nan,
            "EspnA": np.nan,
            "Stadium": "",
            "VenueCity": "",
            "VenueState": "",
            "VenueCountry": "",
            "FuenteFixture": "Sofascore",
        })

    if errors:
        log(f"  Sofascore: {len(errors)} fechas no disponibles")
    if rows:
        log(f"  Sofascore: {len(rows)} partidos objetivo")
        return normalizar_df(pd.DataFrame(rows))
    return pd.DataFrame()


def guardar_cache_fixtures(fixtures):
    """Guarda atómicamente el último calendario utilizable."""
    if fixtures is None or fixtures.empty:
        return
    try:
        os.makedirs(os.path.dirname(FIXTURE_CACHE_PATH), exist_ok=True)
        temp_path = FIXTURE_CACHE_PATH + ".tmp"
        fixtures.to_csv(temp_path, index=False, encoding="utf-8-sig")
        os.replace(temp_path, FIXTURE_CACHE_PATH)
    except Exception as exc:
        log(f"  No se pudo guardar caché de calendario: {exc}")


def cargar_cache_fixtures(inicio, fin, max_age_hours=72):
    """Recupera una agenda reciente si las dos fuentes en vivo fallan."""
    try:
        if not os.path.exists(FIXTURE_CACHE_PATH):
            return pd.DataFrame()
        age_hours = (time.time() - os.path.getmtime(FIXTURE_CACHE_PATH)) / 3600.0
        if age_hours > float(max_age_hours):
            return pd.DataFrame()
        cached = normalizar_df(pd.read_csv(FIXTURE_CACHE_PATH))
        if cached.empty or "Date" not in cached:
            return pd.DataFrame()
        cached["Date"] = pd.to_datetime(cached["Date"], errors="coerce")
        cached = cached[cached["Date"].between(
            pd.Timestamp(inicio), pd.Timestamp(fin), inclusive="both"
        )].copy()
        if not cached.empty:
            if "FuenteFixture" not in cached:
                cached["FuenteFixture"] = "Caché"
            else:
                cached["FuenteFixture"] = cached["FuenteFixture"].astype(str) + " · caché"
        return cached
    except Exception:
        return pd.DataFrame()


def descargar_fixtures_objetivo(inicio, fin):
    rows = []

    def fetch_competition(comp_key, info):
        try:
            eventos = eventos_scoreboard_espn(
                info["espn"],
                inicio,
                fin,
            )
        except Exception as e:
            log(
                f"  {info['nombre']}: error fixtures {e}"
            )
            return comp_key, [], 0

        local_rows = []
        n = 0
        for ev in eventos:
            row = extraer_future_event(
                ev,
                comp_key,
            )

            if row is None:
                continue

            if row["Date"] < inicio or row["Date"] > fin:
                continue

            local_rows.append(row)
            n += 1
        return comp_key, local_rows, n

    # Las competiciones son independientes. Consultarlas en paralelo evita
    # que doce timeouts consecutivos hagan caer la primera ejecución.
    workers = 6 if os.environ.get("GATUNO_FAST_MODE", "0") == "1" else 4
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(fetch_competition, key, info): (key, info)
            for key, info in COMPETICIONES.items()
        }
        for future in as_completed(futures):
            comp_key, info = futures[future]
            try:
                _, local_rows, n = future.result()
            except Exception as exc:
                log(f"  {info['nombre']}: error fixtures {exc}")
                continue
            rows.extend(local_rows)
            log(f"  {info['nombre']}: {n} partidos")

    espn = normalizar_df(pd.DataFrame(rows)) if rows else pd.DataFrame()
    secondary = descargar_fixtures_football_data(inicio, fin)
    # La tercera fuente se consulta sólo si las dos fuentes originales no
    # produjeron ningún partido, para reducir tráfico y superficie de fallo.
    tertiary = pd.DataFrame()
    if espn.empty and secondary.empty:
        try:
            tertiary = descargar_fixtures_sofascore(inicio, fin)
        except Exception as exc:
            log(f"  Sofascore fixtures no disponible: {exc}")
    frames = [
        frame for frame in (espn, secondary, tertiary)
        if frame is not None and not frame.empty
    ]

    if not frames:
        cached = cargar_cache_fixtures(inicio, fin)
        if cached is not None and not cached.empty:
            log(f"  Calendario recuperado desde caché local: {len(cached)} partidos")
            return filtrar_fixtures_desde_ahora(cached)
        raise RuntimeError(
            "No se obtuvo calendario futuro. ESPN, Football-Data y Sofascore no "
            "entregaron partidos y todavía no existe una caché válida."
        )

    fixtures = normalizar_df(pd.concat(frames, ignore_index=True, sort=False))
    # ESPN conserva prioridad; Football-Data precede al respaldo diario.
    priority = {"ESPN": 0, "Football-Data fixtures": 1, "Sofascore": 2}
    fixtures["_source_priority"] = fixtures["FuenteFixture"].map(priority).fillna(9)
    fixtures = fixtures.sort_values("_source_priority").drop_duplicates(
        subset=["CompKey", "Date", "HomeTeam", "AwayTeam"],
        keep="first",
    ).drop(columns="_source_priority")
    guardar_cache_fixtures(fixtures)
    return filtrar_fixtures_desde_ahora(fixtures)


# ============================================================
# OPEN-METEO
# ============================================================

_geocode_mem = {}
_weather_mem = {}


def geocodificar_ciudad(city, country=""):
    if not city:
        return None

    key = norm_texto(city + " " + country)

    if key in _geocode_mem:
        return _geocode_mem[key]

    query = urllib.parse.quote(city)

    url = (
        "https://geocoding-api.open-meteo.com/v1/search?"
        f"name={query}&count=8&language=en&format=json"
    )

    try:
        data = descargar_json(url)
    except Exception:
        _geocode_mem[key] = None
        return None

    resultados = data.get("results") or []

    if not resultados:
        _geocode_mem[key] = None
        return None

    objetivo_pais = norm_texto(country)

    elegido = resultados[0]

    if objetivo_pais:
        for r in resultados:
            rpais = norm_texto(
                r.get("country", "")
            )

            if objetivo_pais in rpais or rpais in objetivo_pais:
                elegido = r
                break

    out = {
        "lat": elegido.get("latitude"),
        "lon": elegido.get("longitude"),
        "elevation": elegido.get("elevation"),
        "name": elegido.get("name", city),
        "country": elegido.get("country", country),
    }

    _geocode_mem[key] = out
    return out


def clima_partido(city, country, kickoff_utc):
    geo = geocodificar_ciudad(
        city,
        country,
    )

    if not geo:
        return {}

    lat = geo.get("lat")
    lon = geo.get("lon")

    if lat is None or lon is None:
        return {}

    key = f"{lat:.3f}_{lon:.3f}"

    if key not in _weather_mem:
        url = (
            "https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat}&longitude={lon}"
            "&hourly=temperature_2m,relative_humidity_2m,"
            "apparent_temperature,precipitation_probability,"
            "wind_speed_10m"
            "&timezone=UTC&forecast_days=10"
        )

        try:
            _weather_mem[key] = descargar_json(url)
        except Exception:
            _weather_mem[key] = None

    data = _weather_mem.get(key)

    if not data:
        return {
            "Elevation": numero(geo.get("elevation"))
        }

    hourly = data.get("hourly") or {}
    tiempos = hourly.get("time") or []

    if not tiempos:
        return {
            "Elevation": numero(
                data.get("elevation", geo.get("elevation"))
            )
        }

    ko = pd.to_datetime(
        kickoff_utc,
        utc=True,
        errors="coerce",
    )

    if pd.isna(ko):
        return {}

    mejor_i = None
    mejor_d = None

    for i, t in enumerate(tiempos):
        dt = pd.to_datetime(
            t,
            utc=True,
            errors="coerce",
        )

        if pd.isna(dt):
            continue

        dif = abs(
            (dt - ko).total_seconds()
        )

        if mejor_d is None or dif < mejor_d:
            mejor_d = dif
            mejor_i = i

    if mejor_i is None:
        return {}

    def val(nombre):
        arr = hourly.get(nombre) or []
        return numero(arr[mejor_i]) if mejor_i < len(arr) else np.nan

    return {
        "TemperatureC": val("temperature_2m"),
        "HumidityPct": val("relative_humidity_2m"),
        "ApparentC": val("apparent_temperature"),
        "RainProbPct": val("precipitation_probability"),
        "WindKmh": val("wind_speed_10m"),
        "Elevation": numero(
            data.get("elevation", geo.get("elevation"))
        ),
    }


# ============================================================
# CONTEXTO: CLIMA / ALTITUD / ARBITRO
# ============================================================

def factor_contexto_clima(
    weather,
    favorito_tipo,
):
    """
    El clima extremo NO aumenta P.
    Solo reduce confianza.
    Altitud sí puede ajustar de forma leve P de dog-gol.
    """
    factor_conf = 1.0
    factor_dog = 1.0
    notas = []

    temp = numero(weather.get("TemperatureC"))
    hum = numero(weather.get("HumidityPct"))
    rain = numero(weather.get("RainProbPct"))
    wind = numero(weather.get("WindKmh"))
    elev = numero(weather.get("Elevation"))

    if pd.notna(temp) and pd.notna(hum):
        if temp >= 30 and hum >= 70:
            factor_conf *= 0.94
            notas.append("calor+humedad altos")
        elif temp >= 28 and hum >= 75:
            factor_conf *= 0.96
            notas.append("humedad exigente")

    if pd.notna(rain) and rain >= 70:
        factor_conf *= 0.96
        notas.append("alta prob. lluvia")

    if pd.notna(wind) and wind >= 35:
        factor_conf *= 0.95
        notas.append("viento fuerte")

    # Altitud: efecto deliberadamente pequeño y acotado.
    if pd.notna(elev) and elev >= 2200:
        if favorito_tipo == "HOME":
            # Si el favorito es local de altura, el underdog visitante
            # tiende a tener una pata de gol más difícil.
            factor_dog *= 0.94
            notas.append("altura favorece local favorito")
        else:
            # Si el underdog es local de altura, ligera compensación.
            factor_dog *= 1.03
            notas.append("underdog local en altura")

    return (
        float(np.clip(factor_conf, 0.88, 1.0)),
        float(np.clip(factor_dog, 0.92, 1.04)),
        "; ".join(notas),
    )


def ajustar_cards_por_arbitro(
    p_cards,
    referee,
    perfiles,
):
    if pd.isna(p_cards):
        return p_cards, ""

    perfil = buscar_perfil_arbitro(
        referee,
        perfiles,
    )

    if not perfil:
        return p_cards, ""

    avg = perfil["cards_avg"]
    factor = 1.0
    nota = ""

    if avg >= 5.5:
        factor = 1.05
        nota = f"árbitro tarjetero ({avg:.1f}/p)"
    elif avg >= 4.8:
        factor = 1.025
        nota = f"árbitro sobre media ({avg:.1f}/p)"
    elif avg <= 3.4:
        factor = 0.95
        nota = f"árbitro poco tarjetero ({avg:.1f}/p)"

    return (
        float(np.clip(p_cards * factor, 0.01, 0.99)),
        nota,
    )


# ============================================================
# CUOTAS OBJETIVO
# ============================================================

def cuota_min_exigida(p):
    if pd.isna(p) or p <= 0:
        return np.nan

    return max(
        CUOTA_COMBINADA_MIN,
        (1.0 + ROI_OBJETIVO) / p,
    )


def cuota_justa(p):
    if pd.isna(p) or p <= 0:
        return np.nan
    return 1.0 / p


# ============================================================
# FUTURO: REFEREE SUMMARY
# ============================================================

def completar_referee_future(
    row,
    calls_counter,
):
    if calls_counter[0] >= MAX_FUTURE_SUMMARY_CALLS:
        return ""

    event_id = row.get("EventID", "")
    comp_key = row.get("CompKey", "")

    if not event_id or comp_key not in COMPETICIONES:
        return ""

    try:
        summary = resumen_evento_espn(
            COMPETICIONES[comp_key]["espn"],
            event_id,
            cache_hours=6,
        )

        calls_counter[0] += 1

        stats = parse_summary_stats(
            summary,
            row.get("HomeESPNID", ""),
            row.get("AwayESPNID", ""),
        )

        return stats.get("Referee", "")

    except Exception:
        return ""


# ============================================================
# PRONOSTICO
# ============================================================

def elegir_variante(pred, comp_info):
    candidatos = []

    # BASE
    p_base = pred["P_BASE_CONTEXT"]

    candidatos.append({
        "Variante": "BASE",
        "P_Variante": p_base,
        "CuotaJusta": cuota_justa(p_base),
        "CuotaMinExigida": cuota_min_exigida(p_base),
        "PataExtra": "Ninguna",
        "ValidaAuto": True,
    })

    # BASE + CARDS
    p_bc = pred.get("P_BASE_CARDS", np.nan)
    p_cards = pred.get("P_Cards4_Ajustada", np.nan)

    if (
        comp_info["cards"]
        and pd.notna(p_bc)
        and pd.notna(p_cards)
        and p_cards >= P_MIN_CARDS4
        and p_bc >= P_MIN_BASE_CARDS
    ):
        candidatos.append({
            "Variante": "BASE + 4+ TARJETAS",
            "P_Variante": p_bc,
            "CuotaJusta": cuota_justa(p_bc),
            "CuotaMinExigida": cuota_min_exigida(p_bc),
            "PataExtra": "4+ tarjetas totales (proxy amarillo; revisar reglas casa)",
            "ValidaAuto": True,
        })

    # BASE + GOL 1T
    p_bg = pred.get("P_BASE_GOL1T", np.nan)
    p_g1 = pred.get("P_Gol1T", np.nan)

    if (
        comp_info["gol1t"]
        and pd.notna(p_bg)
        and pd.notna(p_g1)
        and p_g1 >= P_MIN_GOL1T
        and p_bg >= P_MIN_BASE_GOL1T
    ):
        candidatos.append({
            "Variante": "BASE + GOL 1T",
            "P_Variante": p_bg,
            "CuotaJusta": cuota_justa(p_bg),
            "CuotaMinExigida": cuota_min_exigida(p_bg),
            "PataExtra": "Over 0.5 goles en 1er tiempo",
            "ValidaAuto": True,
        })

    # Selección automática:
    # No elegimos por "cuota más alta". Elegimos la variante con mejor
    # equilibrio P / cuota requerida, penalizando variantes muy frágiles.
    for c in candidatos:
        p = c["P_Variante"]
        q = c["CuotaMinExigida"]

        c["VariantScore"] = (
            100.0 * p
            - 1.4 * max(q - CUOTA_COMBINADA_MIN, 0)
        )

        if c["Variante"] != "BASE":
            c["VariantScore"] -= 1.5

    mejor = max(
        candidatos,
        key=lambda x: x["VariantScore"],
    )

    return mejor, candidatos


def decision_base(pred):
    cuota_fav = pred.get("CuotaFavorito", np.nan)

    mercado_ok = pd.notna(cuota_fav)

    rango_fav = (
        mercado_ok
        and CUOTA_FAVORITO_MIN
        <= cuota_fav
        <= CUOTA_FAVORITO_MAX
    )

    strict = (
        pred["P_DogGol_Context"] >= P_MIN_DOG_GOL
        and pred["P_Fav4"] >= P_MIN_FAV4
        and pred["P_Under45"] >= P_MIN_UNDER45
        and pred["P_BASE_CONTEXT"] >= P_MIN_BASE
        and pred["P_BASE_LCB"] >= P_LCB_MIN_BASE
        and pred["Soporte"] >= 30
    )

    near = (
        pred["P_DogGol_Context"] >= P_MIN_DOG_GOL - 0.05
        and pred["P_Fav4"] >= P_MIN_FAV4 - 0.06
        and pred["P_Under45"] >= P_MIN_UNDER45 - 0.06
        and pred["P_BASE_CONTEXT"] >= 0.26
    )

    if strict and rango_fav:
        return "APOSTAR", "Pasa filtros; validar cuota combinada real"

    if strict and not mercado_ok:
        return "VIGILAR", "Buen perfil; falta cuota 1X2 del favorito"

    if strict and mercado_ok and not rango_fav:
        return "VIGILAR", "Buen perfil; favorito fuera del rango 1.20-1.85"

    if near:
        return "VIGILAR", "Cerca del filtro estricto"

    return "NO APOSTAR", "No supera simultáneamente los filtros"


def pronosticar(
    fixtures,
    ds,
    estados,
    perfiles_arbitro,
):
    rows = []
    variant_rows = []
    ref_calls = [0]
    weather_calls = [0]

    candidatos_estado = list(estados.keys())

    for _, fr in fixtures.iterrows():
        comp_key = fr["CompKey"]
        info = COMPETICIONES[comp_key]

        home_orig = fr["HomeOriginal"]
        away_orig = fr["AwayOriginal"]

        home, sh = resolver_nombre(
            home_orig,
            candidatos_estado,
        )
        away, sa = resolver_nombre(
            away_orig,
            candidatos_estado,
        )

        base = {
            "Fecha": fr["Date"],
            "HoraPeru": fr["HoraPeru"],
            "Competicion": info["nombre"],
            "TipoCompeticion": info["tipo"],
            "CompKey": comp_key,
            "Grupo": info["grupo"],
            "Local": home_orig,
            "Visitante": away_orig,
            "LocalModelo": home,
            "VisitanteModelo": away,
            "MatchLocal": sh,
            "MatchVisitante": sa,
            "Stadium": fr.get("Stadium", ""),
            "VenueCity": fr.get("VenueCity", ""),
            "VenueCountry": fr.get("VenueCountry", ""),
            "EventID": fr.get("EventID", ""),
        }

        # Evita asignaciones fuzzy débiles.
        if sh < 0.57 or sa < 0.57:
            rows.append({
                **base,
                "Accion": "NO APOSTAR",
                "Motivo": "Equipo no emparejado con histórico con suficiente seguridad",
            })
            continue

        hs = snapshot(estados[home])
        aws = snapshot(estados[away])

        if hs is None or aws is None:
            rows.append({
                **base,
                "Accion": "NO APOSTAR",
                "Motivo": "Historial insuficiente de uno de los equipos",
            })
            continue

        # Cuotas ESPN si vienen inline.
        row_model = fr.copy()
        row_model["HomeTeam"] = home
        row_model["AwayTeam"] = away

        fav = detectar_favorito_mercado(row_model)

        if fav is None:
            fav = favorito_modelo(hs, aws)

        feat = crear_features(
            row_model,
            hs,
            aws,
            fav=fav,
        )

        if feat is None:
            continue

        probs = knn_probabilidades(
            ds,
            feat,
            fr["Date"],
            comp_key,
            info["grupo"],
        )

        if probs is None:
            rows.append({
                **base,
                "Accion": "NO APOSTAR",
                "Motivo": "Sin suficientes partidos históricos comparables",
            })
            continue

        # Referee only for candidates with reasonable base probability.
        referee = ""

        if probs["Y_BASE"] >= 0.22:
            referee = completar_referee_future(
                fr,
                ref_calls,
            )

        # Weather.
        weather = {}

        if (
            weather_calls[0] < MAX_WEATHER_CALLS
            and fr.get("VenueCity")
        ):
            weather = clima_partido(
                fr.get("VenueCity", ""),
                fr.get("VenueCountry", ""),
                fr.get("KickoffUTC", ""),
            )
            weather_calls[0] += 1

        factor_conf, factor_dog, climate_note = (
            factor_contexto_clima(
                weather,
                fav["tipo"],
            )
        )

        p_dog_context = float(
            np.clip(
                probs["Y_DogGol"] * factor_dog,
                0.01,
                0.99,
            )
        )

        # Ajuste base solo a través de la pata que sí cambia:
        ratio_dog = (
            p_dog_context
            / max(probs["Y_DogGol"], 0.01)
        )

        p_base_context = float(
            np.clip(
                probs["Y_BASE"] * ratio_dog,
                0.01,
                0.99,
            )
        )

        # Cards referee.
        p_cards_adj, ref_note = ajustar_cards_por_arbitro(
            probs["Y_Cards4"],
            referee,
            perfiles_arbitro,
        )

        # Ajusta joint cards proporcionalmente al marginal de cards.
        p_base_cards = probs["Y_BASE_CARDS"]

        if (
            pd.notna(p_base_cards)
            and pd.notna(probs["Y_Cards4"])
            and probs["Y_Cards4"] > 0
            and pd.notna(p_cards_adj)
        ):
            p_base_cards = float(
                np.clip(
                    p_base_cards
                    * (
                        p_cards_adj
                        / probs["Y_Cards4"]
                    )
                    * ratio_dog,
                    0.01,
                    0.99,
                )
            )

        p_base_gol1t = probs["Y_BASE_GOL1T"]

        if pd.notna(p_base_gol1t):
            p_base_gol1t = float(
                np.clip(
                    p_base_gol1t * ratio_dog,
                    0.01,
                    0.99,
                )
            )

        favorito = (
            home_orig
            if fav["tipo"] == "HOME"
            else away_orig
        )

        dog = (
            away_orig
            if fav["tipo"] == "HOME"
            else home_orig
        )

        pred = {
            **base,
            "Favorito": favorito,
            "Underdog": dog,
            "FavoritoTipo": fav["tipo"],
            "CuotaFavorito": fav.get("cuota", np.nan),
            "FuenteFavorito": fav.get("fuente", ""),
            "Referee": referee,
            "P_DogGol": probs["Y_DogGol"],
            "P_DogGol_Context": p_dog_context,
            "P_Fav4": probs["Y_Fav4"],
            "P_Under45": probs["Y_Under45"],
            "P_BASE": probs["Y_BASE"],
            "P_BASE_CONTEXT": p_base_context,
            "P_BASE_LCB": probs["BASE_LCB"],
            "P_Cards4": probs["Y_Cards4"],
            "P_Cards4_Ajustada": p_cards_adj,
            "P_Gol1T": probs["Y_Gol1T"],
            "P_BASE_CARDS": p_base_cards,
            "P_BASE_GOL1T": p_base_gol1t,
            "Soporte": probs["Y_BASE_Neff"],
            "ConfianzaModelo": probs["Confianza"],
            "FactorConfClima": factor_conf,
            "ClimateNote": climate_note,
            "RefereeNote": ref_note,
            "TemperatureC": weather.get("TemperatureC", np.nan),
            "HumidityPct": weather.get("HumidityPct", np.nan),
            "RainProbPct": weather.get("RainProbPct", np.nan),
            "WindKmh": weather.get("WindKmh", np.nan),
            "ElevationM": weather.get("Elevation", np.nan),
            "PotencialCompeticion": info["potencial"],
        }

        accion, motivo = decision_base(pred)

        pred["Accion"] = accion
        pred["Motivo"] = motivo

        mejor_variante, variantes = elegir_variante(
            pred,
            info,
        )

        pred["BuilderPreferido"] = mejor_variante["Variante"]
        pred["P_BuilderPreferido"] = mejor_variante["P_Variante"]
        pred["CuotaJusta"] = mejor_variante["CuotaJusta"]
        pred["CuotaMinExigida"] = mejor_variante["CuotaMinExigida"]
        pred["PataExtra"] = mejor_variante["PataExtra"]

        pred["ApuestaBase"] = (
            f"{dog} anota 1+ | "
            f"{favorito} 4+ corners | "
            "Under 4.5 goles"
        )

        pred["ApuestaSugerida"] = pred["ApuestaBase"]

        if mejor_variante["Variante"] == "BASE + 4+ TARJETAS":
            pred["ApuestaSugerida"] += " | 4+ tarjetas"
        elif mejor_variante["Variante"] == "BASE + GOL 1T":
            pred["ApuestaSugerida"] += " | Over 0.5 gol 1T"

        # Clima reduce ranking/confianza, NO aumenta probabilidad.
        confianza_contexto = (
            pred["ConfianzaModelo"]
            * factor_conf
        )

        pred["ConfianzaContexto"] = confianza_contexto

        pred["ScoreRanking"] = (
            100.0
            * (
                0.46 * pred["P_BASE_CONTEXT"]
                + 0.18 * pred["P_BASE_LCB"]
                + 0.10 * pred["P_DogGol_Context"]
                + 0.10 * pred["P_Fav4"]
                + 0.08 * pred["P_Under45"]
                + 0.05 * confianza_contexto
            )
            + 0.03 * info["potencial"]
        )

        # Si cuota requerida se dispara, penaliza ranking.
        pred["ScoreRanking"] -= (
            1.2
            * max(
                pred["CuotaMinExigida"]
                - CUOTA_COMBINADA_MIN,
                0,
            )
        )

        # Si no hay cuota del favorito, nunca APOSTAR.
        if (
            pred["Accion"] == "APOSTAR"
            and pd.isna(pred["CuotaFavorito"])
        ):
            pred["Accion"] = "VIGILAR"
            pred["Motivo"] = (
                "Buen perfil; falta confirmar cuota 1X2 del favorito"
            )

        # Regla final visible.
        if pred["Accion"] == "APOSTAR":
            pred["Recomendacion"] = (
                "APOSTAR SOLO SI CUOTA REAL >= "
                f"{pred['CuotaMinExigida']:.2f}"
            )
        elif pred["Accion"] == "VIGILAR":
            pred["Recomendacion"] = (
                "VIGILAR; exigir cuota >= "
                f"{pred['CuotaMinExigida']:.2f}"
            )
        else:
            pred["Recomendacion"] = "NO APOSTAR"

        rows.append(pred)

        for vr in variantes:
            variant_rows.append({
                "Fecha": fr["Date"],
                "Competicion": info["nombre"],
                "Local": home_orig,
                "Visitante": away_orig,
                "AccionBase": pred["Accion"],
                **vr,
            })

    return (
        pd.DataFrame(rows),
        pd.DataFrame(variant_rows),
    )


# ============================================================
# BACKTEST DE VARIANTES
# ============================================================

def backtest(ds):
    valid = ds.dropna(
        subset=FEATURES + ["Y_BASE"]
    ).copy()

    if valid.empty:
        return pd.DataFrame()

    tests = valid.tail(BACKTEST_MAX)
    rows = []

    for _, r in tests.iterrows():
        train = valid[
            valid["Date"] < r["Date"]
        ]

        if len(train) < 250:
            continue

        p = knn_probabilidades(
            train,
            r,
            r["Date"],
            r["CompKey"],
            r["Grupo"],
        )

        if p is None:
            continue

        variantes = [
            ("BASE", p["Y_BASE"], r["Y_BASE"]),
            (
                "BASE + 4+ TARJETAS",
                p["Y_BASE_CARDS"],
                r.get("Y_BASE_CARDS", np.nan),
            ),
            (
                "BASE + GOL 1T",
                p["Y_BASE_GOL1T"],
                r.get("Y_BASE_GOL1T", np.nan),
            ),
        ]

        for nombre, prob, y in variantes:
            if pd.isna(prob) or pd.isna(y):
                continue

            qmin = cuota_min_exigida(prob)

            # Proxy conservador a 4.20 solo cuando 4.20 satisface qmin.
            roi_proxy_420 = np.nan

            if qmin <= 4.20 + 1e-9:
                roi_proxy_420 = (
                    (4.20 - 1.0)
                    if int(y) == 1
                    else -1.0
                )

            rows.append({
                "Fecha": r["Date"],
                "CompKey": r["CompKey"],
                "Competicion": r["Competicion"],
                "Local": r["HomeTeam"],
                "Visitante": r["AwayTeam"],
                "Variante": nombre,
                "ProbModelo": prob,
                "CuotaJusta": cuota_justa(prob),
                "CuotaMinExigida": qmin,
                "Resultado": int(y),
                "ProfitProxy420": roi_proxy_420,
            })

    return pd.DataFrame(rows)


def resumen_backtest(bt):
    if bt.empty:
        return pd.DataFrame()

    rows = []

    for (comp, var), g in bt.groupby(
        ["Competicion", "Variante"]
    ):
        n = len(g)
        hit = float(g["Resultado"].mean())

        gp = g[g["ProfitProxy420"].notna()]

        roi420 = (
            float(gp["ProfitProxy420"].mean())
            if len(gp)
            else np.nan
        )

        rows.append({
            "Competicion": comp,
            "Variante": var,
            "N": n,
            "HitRate": hit,
            "CuotaBreakEvenObservada": (
                1.0 / hit
                if hit > 0
                else np.nan
            ),
            "N_Proxy420": len(gp),
            "ROI_Proxy420": roi420,
        })

    # Total por variante.
    for var, g in bt.groupby("Variante"):
        n = len(g)
        hit = float(g["Resultado"].mean())
        gp = g[g["ProfitProxy420"].notna()]

        rows.append({
            "Competicion": "TOTAL",
            "Variante": var,
            "N": n,
            "HitRate": hit,
            "CuotaBreakEvenObservada": (
                1.0 / hit
                if hit > 0
                else np.nan
            ),
            "N_Proxy420": len(gp),
            "ROI_Proxy420": (
                float(gp["ProfitProxy420"].mean())
                if len(gp)
                else np.nan
            ),
        })

    return pd.DataFrame(rows)


# ============================================================
# TOP 30
# ============================================================

def crear_top30(resultado):
    if resultado.empty:
        return resultado

    r = resultado.copy()

    if "P_BASE_CONTEXT" not in r.columns:
        return pd.DataFrame()

    r = r[
        r["P_BASE_CONTEXT"].notna()
    ].copy()

    orden = {
        "APOSTAR": 0,
        "VIGILAR": 1,
        "NO APOSTAR": 2,
    }

    r["_orden"] = (
        r["Accion"]
        .map(orden)
        .fillna(9)
    )

    r = r.sort_values(
        [
            "_orden",
            "ScoreRanking",
            "P_BASE_CONTEXT",
        ],
        ascending=[
            True,
            False,
            False,
        ],
    ).head(TOP_N)

    r["Ranking"] = np.arange(
        1,
        len(r) + 1,
    )

    # Editable en Excel.
    r["CuotaCombinadaReal"] = ""
    r["CumpleCuota"] = "PENDIENTE"

    columnas = [
        "Ranking",
        "Accion",
        "Fecha",
        "HoraPeru",
        "Competicion",
        "TipoCompeticion",
        "Local",
        "Visitante",
        "Favorito",
        "Underdog",
        "CuotaFavorito",
        "FuenteFavorito",
        "BuilderPreferido",
        "ApuestaSugerida",
        "PataExtra",
        "P_DogGol_Context",
        "P_Fav4",
        "P_Under45",
        "P_BASE_CONTEXT",
        "P_BASE_LCB",
        "P_Cards4_Ajustada",
        "P_Gol1T",
        "P_BuilderPreferido",
        "CuotaJusta",
        "CuotaMinExigida",
        "CuotaCombinadaReal",
        "CumpleCuota",
        "ConfianzaModelo",
        "ConfianzaContexto",
        "Soporte",
        "TemperatureC",
        "HumidityPct",
        "RainProbPct",
        "WindKmh",
        "ElevationM",
        "Referee",
        "RefereeNote",
        "ClimateNote",
        "PotencialCompeticion",
        "ScoreRanking",
        "Motivo",
        "Recomendacion",
    ]

    for c in columnas:
        if c not in r.columns:
            r[c] = np.nan

    return r[columnas]


# ============================================================
# EXPORT EXCEL
# ============================================================

def exportar(
    top,
    resultado,
    variantes,
    bt,
    bt_resumen,
    inicio,
    fin,
):
    top.to_csv(
        ARCHIVO_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    variantes.to_csv(
        ARCHIVO_VARIANTES,
        index=False,
        encoding="utf-8-sig",
    )

    bt.to_csv(
        ARCHIVO_BACKTEST,
        index=False,
        encoding="utf-8-sig",
    )

    apostar = (
        top[top["Accion"] == "APOSTAR"].copy()
        if not top.empty
        else top.copy()
    )

    vigilar = (
        top[top["Accion"] == "VIGILAR"].copy()
        if not top.empty
        else top.copy()
    )

    comps_rows = []

    for key, info in COMPETICIONES.items():
        comps_rows.append({
            "CompKey": key,
            "Competicion": info["nombre"],
            "Tipo": info["tipo"],
            "Grupo": info["grupo"],
            "PotencialPrior": info["potencial"],
            "Tarjetas": info["cards"],
            "Gol1T": info["gol1t"],
        })

    comps_df = pd.DataFrame(comps_rows)

    # Primera hoja: instrucciones claras de apuesta.
    apuestas_claras = crear_apuestas_claras_v81(
        top,
        variants,
    )

    config = pd.DataFrame(
        [
            ["Version", VERSION],
            [
                "Ventana",
                f"{inicio.strftime('%d/%m/%Y')} a "
                f"{fin.strftime('%d/%m/%Y')}",
            ],
            [
                "Nicho base",
                "Underdog anota + Favorito 4+ corners + Under 4.5",
            ],
            ["Cuota mínima absoluta", CUOTA_COMBINADA_MIN],
            ["ROI objetivo", ROI_OBJETIVO],
            ["P necesaria ROI29 a 4.20", P_ROI29_420],
            [
                "Variantes automáticas",
                "BASE | BASE+4 tarjetas | BASE+Gol 1T",
            ],
            [
                "Corners 1T",
                (
                    "Experimental: no activa APOSTAR porque no se "
                    "dispone de histórico homogéneo por mitad"
                ),
            ],
            [
                "Clima",
                (
                    "Temperatura/humedad/lluvia/viento reducen confianza; "
                    "no aumentan probabilidad"
                ),
            ],
            [
                "Altitud",
                (
                    "Ajuste pequeño y limitado sobre underdog marca cuando "
                    "elevación >= 2200 m"
                ),
            ],
            [
                "Regla de cuota real",
                (
                    "Solo apostar si CuotaCombinadaReal >= CuotaMinExigida "
                    "y nunca menor de 4.20"
                ),
            ],
            [
                "Advertencia tarjetas",
                (
                    "El histórico usa principalmente amarillas como proxy. "
                    "Revisar reglas de conteo de la casa para rojas."
                ),
            ],
        ],
        columns=["Parametro", "Valor"],
    )

    try:
        with pd.ExcelWriter(
            ARCHIVO_XLSX,
            engine="openpyxl",
        ) as writer:
            top.to_excel(
                writer,
                sheet_name="TOP_30",
                index=False,
            )
            apostar.to_excel(
                writer,
                sheet_name="APOSTAR",
                index=False,
            )
            vigilar.to_excel(
                writer,
                sheet_name="VIGILAR",
                index=False,
            )
            variantes.to_excel(
                writer,
                sheet_name="VARIANTES",
                index=False,
            )
            bt.to_excel(
                writer,
                sheet_name="BACKTEST_DETALLE",
                index=False,
            )
            bt_resumen.to_excel(
                writer,
                sheet_name="BACKTEST_RESUMEN",
                index=False,
            )
            comps_df.to_excel(
                writer,
                sheet_name="COMPETICIONES",
                index=False,
            )
            config.to_excel(
                writer,
                sheet_name="CONFIG",
                index=False,
            )

            from openpyxl.styles import (
                Font,
                PatternFill,
                Alignment,
            )
            from openpyxl.utils import (
                get_column_letter,
            )

            wb = writer.book

            for ws in wb.worksheets:
                ws.freeze_panes = "A2"
                ws.auto_filter.ref = ws.dimensions

                for cell in ws[1]:
                    cell.font = Font(
                        bold=True,
                        color="FFFFFF",
                    )
                    cell.fill = PatternFill(
                        "solid",
                        fgColor="1F4E78",
                    )
                    cell.alignment = Alignment(
                        horizontal="center",
                    )

                for col in ws.columns:
                    letra = col[0].column_letter
                    max_len = 0

                    for cell in col[:100]:
                        v = (
                            ""
                            if cell.value is None
                            else str(cell.value)
                        )
                        max_len = max(
                            max_len,
                            len(v),
                        )

                    ws.column_dimensions[
                        letra
                    ].width = min(
                        max(10, max_len + 2),
                        42,
                    )

            # Fórmula viva de cuota.
            for name in ("TOP_30", "APOSTAR", "VIGILAR"):
                ws = wb[name]

                headers = {
                    c.value: c.column
                    for c in ws[1]
                }

                if (
                    "CuotaCombinadaReal" in headers
                    and "CuotaMinExigida" in headers
                    and "CumpleCuota" in headers
                ):
                    c_real = get_column_letter(
                        headers["CuotaCombinadaReal"]
                    )
                    c_min = get_column_letter(
                        headers["CuotaMinExigida"]
                    )
                    c_ok = headers["CumpleCuota"]

                    for rr in range(
                        2,
                        ws.max_row + 1,
                    ):
                        ws.cell(
                            rr,
                            c_ok,
                        ).value = (
                            f'=IF({c_real}{rr}="","PENDIENTE",'
                            f'IF({c_real}{rr}>={c_min}{rr},"SI","NO"))'
                        )

                # Porcentajes.
                for h in [
                    "P_DogGol_Context",
                    "P_Fav4",
                    "P_Under45",
                    "P_BASE_CONTEXT",
                    "P_BASE_LCB",
                    "P_Cards4_Ajustada",
                    "P_Gol1T",
                    "P_BuilderPreferido",
                    "ConfianzaModelo",
                    "ConfianzaContexto",
                ]:
                    if h in headers:
                        for rr in range(
                            2,
                            ws.max_row + 1,
                        ):
                            ws.cell(
                                rr,
                                headers[h],
                            ).number_format = "0.0%"

                for h in [
                    "CuotaFavorito",
                    "CuotaJusta",
                    "CuotaMinExigida",
                    "CuotaCombinadaReal",
                    "ScoreRanking",
                ]:
                    if h in headers:
                        for rr in range(
                            2,
                            ws.max_row + 1,
                        ):
                            ws.cell(
                                rr,
                                headers[h],
                            ).number_format = "0.00"

        return True

    except Exception as e:
        log(
            f"Excel no generado; CSV sí disponible: {e}"
        )
        return False


# ============================================================
# COLAB DOWNLOAD
# ============================================================

def descargar_en_colab(xlsx_ok):
    try:
        from google.colab import files

        objetivo = (
            ARCHIVO_XLSX
            if xlsx_ok
            else ARCHIVO_CSV
        )

        files.download(objetivo)

    except Exception:
        pass


# ============================================================
# MAIN
# ============================================================

def main():
    t0 = time.time()
    iniciar_log()

    log("=" * 76)
    log(f"BET BUILDER {VERSION}")
    log("=" * 76)

    inicio, fin = ventana_objetivo()

    log(
        f"Ventana: {inicio.strftime('%d/%m/%Y')} "
        f"a {fin.strftime('%d/%m/%Y')}"
    )

    log("")
    log("1) HISTÓRICO")

    hist = descargar_historico_total()

    log(
        f"Histórico usable: {len(hist)} partidos"
    )

    log("")
    log("2) DATASET PRE-PARTIDO SIN FUTURE LEAKAGE")

    ds, estados = construir_dataset(hist)

    log(
        f"Filas modelables: {len(ds)}"
    )

    perfiles_arbitro = construir_perfil_arbitros(
        hist
    )

    log(
        f"Árbitros con perfil: {len(perfiles_arbitro)}"
    )

    log("")
    log("3) BACKTEST")

    bt = backtest(ds)
    bt_resumen = resumen_backtest(bt)

    if not bt_resumen.empty:
        total = bt_resumen[
            bt_resumen["Competicion"]
            == "TOTAL"
        ]

        for _, r in total.iterrows():
            roi = r["ROI_Proxy420"]

            roi_txt = (
                f"{roi*100:.1f}%"
                if pd.notna(roi)
                else "N/D"
            )

            log(
                f"  {r['Variante']}: "
                f"N={int(r['N'])}, "
                f"hit={r['HitRate']*100:.1f}%, "
                f"ROI proxy 4.20={roi_txt}"
            )

    log("")
    log("4) FIXTURES 7 LIGAS + 5 TORNEOS")

    fixtures = descargar_fixtures_objetivo(
        inicio,
        fin,
    )

    log(
        f"Partidos futuros encontrados: {len(fixtures)}"
    )

    log("")
    log("5) PRONÓSTICO + CLIMA + ALTITUD + ÁRBITRO")

    resultado, variantes = pronosticar(
        fixtures,
        ds,
        estados,
        perfiles_arbitro,
    )

    top = crear_top30(
        resultado
    )

    log(
        f"Top generado: {len(top)}"
    )

    if not top.empty:
        log(
            f"APOSTAR: {(top['Accion']=='APOSTAR').sum()} | "
            f"VIGILAR: {(top['Accion']=='VIGILAR').sum()}"
        )

    log("")
    log("6) EXPORTANDO")

    xlsx_ok = exportar(
        top,
        resultado,
        variantes,
        bt,
        bt_resumen,
        inicio,
        fin,
    )

    log("")
    log("=" * 76)
    log("MEJORES OPCIONES")
    log("=" * 76)

    if top.empty:
        log(
            "No hubo candidatos suficientemente modelables."
        )
    else:
        for _, r in top.head(15).iterrows():
            qfav = r["CuotaFavorito"]

            qfav_txt = (
                f"{qfav:.2f}"
                if pd.notna(qfav)
                else "pendiente"
            )

            log("")
            log(
                f"#{int(r['Ranking'])} {r['Accion']} | "
                f"{r['Competicion']}"
            )

            log(
                f"{r['Local']} vs {r['Visitante']}"
            )

            log(
                f"Builder: {r['BuilderPreferido']} | "
                f"P={r['P_BuilderPreferido']*100:.1f}% | "
                f"Cuota mínima={r['CuotaMinExigida']:.2f}"
            )

            log(
                f"Favorito: {r['Favorito']} | "
                f"cuota 1X2={qfav_txt}"
            )

            if pd.notna(r.get("TemperatureC", np.nan)):
                log(
                    f"Clima: {r['TemperatureC']:.1f}°C | "
                    f"humedad {r['HumidityPct']:.0f}% | "
                    f"altura {r['ElevationM']:.0f} m"
                )

    log("")
    log(f"Excel: {ARCHIVO_XLSX}")
    log(f"CSV Top: {ARCHIVO_CSV}")
    log(f"CSV variantes: {ARCHIVO_VARIANTES}")
    log(f"Backtest: {ARCHIVO_BACKTEST}")
    log(f"Tiempo: {time.time()-t0:.1f}s")

    log("")
    log(
        "REGLA FINAL: una fila APOSTAR NO basta. "
        "La cuota real del Bet Builder debe ser >= CuotaMinExigida "
        "y nunca menor de 4.20."
    )

    descargar_en_colab(
        xlsx_ok
    )




# ============================================================
# V8.1 ROBUST - OVERLAY
# ============================================================
#
# OBJETIVO:
#   - maximizar estabilidad y tasa de éxito, no cantidad de apuestas.
#   - usar el hallazgo mensual como hipótesis, NO como probabilidad fija.
#   - distinguir inicio de temporada / fase estable / tramo tardío.
#   - incorporar descanso y congestión con ajustes pequeños.
#   - reforzar la pata más débil: "underdog marca".
#   - comparar builders y exigir SIEMPRE cuota real >= cuota mínima.
#
# MAX 30 selecciones / 7 días. NUNCA se fuerzan 30.
#
# ============================================================

VERSION = "V15.1.1-STABLE-CORE"

# Umbrales estrictos: deliberadamente más exigentes que V8.
P_MIN_DOG_GOL_STABLE = 0.56
P_MIN_FAV4_STABLE = 0.61
P_MIN_UNDER45_STABLE = 0.76
P_MIN_BASE_STABLE = 0.32
P_MIN_LCB_STABLE = 0.245

# Inicio de temporada: mayor incertidumbre.
P_MIN_DOG_GOL_EARLY = 0.60
P_MIN_FAV4_EARLY = 0.64
P_MIN_UNDER45_EARLY = 0.78
P_MIN_BASE_EARLY = 0.34
P_MIN_LCB_EARLY = 0.255

# Favorito más estrecho para APOSTAR; fuera de esto puede ser VIGILAR.
CUOTA_FAVORITO_STRICT_MIN = 1.25
CUOTA_FAVORITO_STRICT_MAX = 1.78

# Cold start.
MIN_GAMES_SEASON_APOSTAR = 5
MIN_RELIABILITY_APOSTAR = 0.70

# Fatiga / congestión.
REST_CONGESTED_DAYS = 4.0
REST_SEVERE_DAYS = 3.2
MATCHES14_CONGESTED = 4
MATCHES21_HIGH = 6

# Contexto de temporada actual.
CURRENT_SEASON_BLEND_MAX = 0.30

# Contexto ESPN para córners de 1T.
CONTEXT_DAYS = 150
CONTEXT_EVENTS_PER_COMP = 48
CONTEXT_MAX_SUMMARY_CALLS = 560

# Builders.
VARIANT_MIN_PROB = 0.18

# Altitud: veto más fuerte cuando el underdog visitante sube a gran altura.
ALTITUDE_CAUTION = 1800
ALTITUDE_STRONG = 2200
ALTITUDE_VETO = 2800

# Rutas.
ARCHIVO_XLSX_V81 = os.path.join(
    OUTPUT_DIR,
    "BET_BUILDER_V81_ROBUST_TOP30.xlsx",
)

ARCHIVO_CSV_V81 = os.path.join(
    OUTPUT_DIR,
    "BET_BUILDER_V81_ROBUST_TOP30.csv",
)

ARCHIVO_VARIANTES_V81 = os.path.join(
    OUTPUT_DIR,
    "BET_BUILDER_V81_ROBUST_VARIANTES.csv",
)

ARCHIVO_LOG = os.path.join(
    OUTPUT_DIR,
    "BET_BUILDER_V81_ROBUST_LOG.txt",
)

CONTEXT_CACHE = os.path.join(
    CACHE_DIR,
    "v81_context_h1.csv",
)


# ============================================================
# FASE DE TEMPORADA
# ============================================================

def season_start_for(comp_key, date):
    d = pd.Timestamp(date)

    south = {
        "BRA", "ARG", "PER",
        "CDB", "LIB", "SUD",
    }

    if comp_key in south:
        return pd.Timestamp(
            year=d.year,
            month=1,
            day=1,
        )

    # Europa: temporada que cruza dos años.
    year = d.year if d.month >= 7 else d.year - 1

    return pd.Timestamp(
        year=year,
        month=7,
        day=1,
    )


def season_phase_from_games(n):
    if n <= 4:
        return "COLD_START"
    if n <= 8:
        return "EARLY"
    if n <= 24:
        return "STABLE"
    return "LATE"


def phase_confidence_factor(phase):
    return {
        "COLD_START": 0.78,
        "EARLY": 0.88,
        "STABLE": 1.00,
        "LATE": 0.97,
    }.get(phase, 0.90)


# ============================================================
# ÍNDICE DE CARGA DE PARTIDOS
# ============================================================

def construir_schedule_index(hist):
    idx = defaultdict(list)

    h = hist.sort_values("Date")

    for _, r in h.iterrows():
        d = pd.Timestamp(r["Date"])

        idx[str(r["HomeTeam"])].append(
            (d, str(r["CompKey"]))
        )
        idx[str(r["AwayTeam"])].append(
            (d, str(r["CompKey"]))
        )

    return idx


def workload_features(
    schedule_index,
    team,
    date,
    comp_key,
):
    d = pd.Timestamp(date)

    all_games = [
        (gd, ck)
        for gd, ck in schedule_index.get(str(team), [])
        if gd < d
    ]

    if not all_games:
        return {
            "DaysRest": np.nan,
            "Matches7": 0,
            "Matches14": 0,
            "Matches21": 0,
            "GamesSeasonComp": 0,
            "Phase": "COLD_START",
        }

    last_date = max(gd for gd, _ in all_games)

    days_rest = float(
        (d - last_date).days
    )

    def count_days(n):
        start = d - pd.Timedelta(days=n)

        return sum(
            1
            for gd, _ in all_games
            if gd >= start
        )

    s0 = season_start_for(
        comp_key,
        d,
    )

    games_season_comp = sum(
        1
        for gd, ck in all_games
        if gd >= s0 and ck == comp_key
    )

    return {
        "DaysRest": days_rest,
        "Matches7": count_days(7),
        "Matches14": count_days(14),
        "Matches21": count_days(21),
        "GamesSeasonComp": games_season_comp,
        "Phase": season_phase_from_games(
            games_season_comp
        ),
    }


# ============================================================
# CURRENT-SEASON PRIORS (solo datos previos)
# ============================================================

def beta_rate(series, prior=0.5, strength=18.0):
    s = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    n = len(s)

    if n == 0:
        return np.nan, 0

    return (
        float(
            (s.sum() + prior * strength)
            / (n + strength)
        ),
        n,
    )


def current_season_rates(
    ds,
    comp_key,
    grupo,
    date,
):
    d = pd.Timestamp(date)
    s0 = season_start_for(
        comp_key,
        d,
    )

    past = ds[
        (ds["Date"] >= s0)
        & (ds["Date"] < d)
    ]

    exact = past[
        past["CompKey"] == comp_key
    ]

    sample = exact

    if len(sample) < 18:
        group = past[
            past["Grupo"] == grupo
        ]

        if len(group) >= 45:
            sample = group

    targets = [
        "Y_DogGol",
        "Y_Fav4",
        "Y_Under45",
        "Y_BASE",
        "Y_Cards4",
        "Y_Gol1T",
        "Y_BLINDADO_CORE",
    ]

    out = {
        "CurrentSeasonN": len(sample)
    }

    for t in targets:
        if t in sample.columns:
            r, n = beta_rate(
                sample[t],
                prior=0.5,
                strength=18.0,
            )
            out[t] = r
            out[t + "_N"] = n
        else:
            out[t] = np.nan
            out[t + "_N"] = 0

    return out


def calibrate_current_season(
    p_model,
    p_current,
    n_current,
):
    if (
        pd.isna(p_model)
        or pd.isna(p_current)
        or n_current < 12
    ):
        return p_model

    # Peso acotado: la temporada actual importa,
    # pero nunca domina por sí sola.
    w = min(
        CURRENT_SEASON_BLEND_MAX,
        0.36 * n_current / (n_current + 70.0),
    )

    return float(
        np.clip(
            (1.0 - w) * p_model
            + w * p_current,
            0.01,
            0.99,
        )
    )


# ============================================================
# DATASET V8.1: añade FAVORITO 3+ / UNDER 5.5 / BLINDADO CORE
# ============================================================

def construir_dataset_v81(hist):
    estados = defaultdict(list)
    rows = []

    orden = hist.sort_values(
        ["Date", "CompKey", "HomeTeam"]
    ).copy()

    for fecha, bloque in orden.groupby(
        "Date",
        sort=True,
    ):
        for _, row in bloque.iterrows():
            hs = snapshot(
                estados[row["HomeTeam"]]
            )
            aws = snapshot(
                estados[row["AwayTeam"]]
            )

            fav = detectar_favorito_mercado(
                row
            )

            if fav is None:
                fav = favorito_modelo(
                    hs,
                    aws,
                )

            feat = crear_features(
                row,
                hs,
                aws,
                fav=fav,
            )

            if feat is None:
                continue

            if fav["tipo"] == "HOME":
                dog_gol = int(
                    row["FTAG"] >= 1
                )
                fav_corner = row.get(
                    "HC",
                    np.nan,
                )
            else:
                dog_gol = int(
                    row["FTHG"] >= 1
                )
                fav_corner = row.get(
                    "AC",
                    np.nan,
                )

            fav3 = (
                int(float(fav_corner) >= 3)
                if pd.notna(fav_corner)
                else np.nan
            )

            fav4 = (
                int(float(fav_corner) >= 4)
                if pd.notna(fav_corner)
                else np.nan
            )

            total_goals = (
                float(row["FTHG"])
                + float(row["FTAG"])
            )

            under45 = int(
                total_goals <= 4
            )

            under55 = int(
                total_goals <= 5
            )

            y_base = (
                int(
                    dog_gol
                    and fav4 == 1
                    and under45 == 1
                )
                if pd.notna(fav4)
                else np.nan
            )

            y_blindado = (
                int(
                    dog_gol
                    and fav3 == 1
                    and under55 == 1
                )
                if pd.notna(fav3)
                else np.nan
            )

            hy = numero(
                row.get("HY", np.nan)
            )
            ay = numero(
                row.get("AY", np.nan)
            )

            cards4 = (
                int(hy + ay >= 4)
                if pd.notna(hy)
                and pd.notna(ay)
                else np.nan
            )

            hthg = numero(
                row.get("HTHG", np.nan)
            )
            htag = numero(
                row.get("HTAG", np.nan)
            )

            gol1t = (
                int(hthg + htag >= 1)
                if pd.notna(hthg)
                and pd.notna(htag)
                else np.nan
            )

            base_cards = (
                int(
                    y_base == 1
                    and cards4 == 1
                )
                if pd.notna(y_base)
                and pd.notna(cards4)
                else np.nan
            )

            base_gol1t = (
                int(
                    y_base == 1
                    and gol1t == 1
                )
                if pd.notna(y_base)
                and pd.notna(gol1t)
                else np.nan
            )

            rows.append({
                "Date": row["Date"],
                "CompKey": row["CompKey"],
                "Grupo": row["Grupo"],
                "Competicion": row["Competicion"],
                "HomeTeam": row["HomeTeam"],
                "AwayTeam": row["AwayTeam"],
                "Referee": row.get(
                    "Referee",
                    "",
                ),
                **feat,
                "Y_DogGol": dog_gol,
                "Y_Fav3": fav3,
                "Y_Fav4": fav4,
                "Y_Under45": under45,
                "Y_Under55": under55,
                "Y_BASE": y_base,
                "Y_BLINDADO_CORE": y_blindado,
                "Y_Cards4": cards4,
                "Y_Gol1T": gol1t,
                "Y_BASE_CARDS": base_cards,
                "Y_BASE_GOL1T": base_gol1t,
            })

        for _, row in bloque.iterrows():
            agregar_estado(
                estados,
                row,
            )

    ds = pd.DataFrame(rows)

    if ds.empty:
        raise RuntimeError(
            "Dataset V8.1 vacío"
        )

    return (
        ds.sort_values("Date")
        .reset_index(drop=True),
        estados,
    )


# ============================================================
# KNN V8.1
# ============================================================

def knn_probabilidades_v81(
    train,
    x,
    fecha_objetivo,
    comp_key,
    grupo,
):
    t = seleccionar_train(
        train,
        comp_key,
        grupo,
    )

    t = t.dropna(
        subset=FEATURES
    )

    if len(t) < K_MINIMO:
        return None

    if len(t) > MAX_TRAIN:
        t = t.tail(
            MAX_TRAIN
        )

    X = (
        t[FEATURES]
        .astype(float)
        .to_numpy()
    )

    xv = np.array(
        [
            float(x[c])
            for c in FEATURES
        ],
        dtype=float,
    )

    mu = np.nanmean(
        X,
        axis=0,
    )

    sd = np.nanstd(
        X,
        axis=0,
    )

    sd[
        (~np.isfinite(sd))
        | (sd < 1e-6)
    ] = 1.0

    Z = (
        X - mu
    ) / sd

    zv = (
        xv - mu
    ) / sd

    fw = np.array(
        [
            FEATURE_WEIGHTS[c]
            for c in FEATURES
        ],
        dtype=float,
    )

    dist = np.sqrt(
        np.nanmean(
            (
                (Z - zv) ** 2
            ) * fw,
            axis=1,
        )
    )

    k = min(
        K_VECINOS,
        len(t),
    )

    idx = np.argsort(
        dist
    )[:k]

    vecinos = t.iloc[
        idx
    ].copy()

    d = dist[idx]

    w_sim = np.clip(
        1.0 / (0.30 + d),
        0.12,
        3.0,
    )

    edad = (
        pd.Timestamp(
            fecha_objetivo
        )
        - pd.to_datetime(
            vecinos["Date"]
        )
    ).dt.days.to_numpy(
        dtype=float
    )

    edad = np.maximum(
        edad,
        0.0,
    )

    # Más peso a régimen reciente que V8.
    w_rec = np.exp(
        -edad / 360.0
    )

    weights = (
        w_sim * w_rec
    )

    def posterior(target):
        if target not in vecinos.columns:
            return np.nan, 0.0, 0

        y = pd.to_numeric(
            vecinos[target],
            errors="coerce",
        ).to_numpy(
            dtype=float
        )

        valid = np.isfinite(
            y
        )

        if valid.sum() < 22:
            return np.nan, 0.0, int(
                valid.sum()
            )

        yy = y[valid]
        ww = weights[valid]

        base_series = pd.to_numeric(
            t[target],
            errors="coerce",
        )

        base_rate = (
            float(
                base_series.mean()
            )
            if base_series.notna().any()
            else 0.5
        )

        sw = float(
            ww.sum()
        )

        p = (
            float(
                np.dot(
                    ww,
                    yy,
                )
            )
            + PRIOR_STRENGTH
            * base_rate
        ) / (
            sw
            + PRIOR_STRENGTH
        )

        n_eff = (
            sw ** 2
            / max(
                float(
                    np.dot(
                        ww,
                        ww,
                    )
                ),
                1e-9,
            )
        )

        return (
            float(
                np.clip(
                    p,
                    0.01,
                    0.99,
                )
            ),
            float(
                n_eff
            ),
            int(
                valid.sum()
            ),
        )

    targets = [
        "Y_DogGol",
        "Y_Fav3",
        "Y_Fav4",
        "Y_Under45",
        "Y_Under55",
        "Y_BASE",
        "Y_BLINDADO_CORE",
        "Y_Cards4",
        "Y_Gol1T",
        "Y_BASE_CARDS",
        "Y_BASE_GOL1T",
    ]

    out = {}

    for target in targets:
        p, neff, nraw = posterior(
            target
        )

        out[target] = p
        out[target + "_Neff"] = neff
        out[target + "_N"] = nraw

    if any(
        pd.isna(
            out[t]
        )
        for t in (
            "Y_DogGol",
            "Y_Fav4",
            "Y_Under45",
            "Y_BASE",
            "Y_BLINDADO_CORE",
        )
    ):
        return None

    p = out["Y_BASE"]
    neff = out[
        "Y_BASE_Neff"
    ]

    se = math.sqrt(
        max(
            p * (1 - p),
            1e-9,
        )
        / max(
            neff
            + PRIOR_STRENGTH,
            1.0,
        )
    )

    out["BASE_LCB"] = max(
        0.0,
        p
        - Z_LCB * se,
    )

    p_b = out[
        "Y_BLINDADO_CORE"
    ]

    n_b = out[
        "Y_BLINDADO_CORE_Neff"
    ]

    se_b = math.sqrt(
        max(
            p_b * (1-p_b),
            1e-9,
        )
        / max(
            n_b
            + PRIOR_STRENGTH,
            1.0,
        )
    )

    out[
        "BLINDADO_LCB"
    ] = max(
        0.0,
        p_b
        - Z_LCB * se_b,
    )

    out["Confianza"] = min(
        1.0,
        neff / 85.0,
    )

    return out


# ============================================================
# CONTEXTO ESPN: CÓRNERS 1T
# ============================================================

def parse_h1_corners_from_summary(
    summary,
    home_id="",
    away_id="",
):
    """
    Intenta contar eventos 'corner' del primer tiempo.
    Si la estructura ESPN no ofrece play-by-play suficiente,
    devuelve NaN y el booster queda desactivado.
    """
    plays = (
        summary.get("plays")
        or []
    )

    if not plays:
        return {
            "H1CornersTotal": np.nan,
            "H1CornersHome": np.nan,
            "H1CornersAway": np.nan,
        }

    seen = set()
    home = 0
    away = 0
    total = 0

    for i, play in enumerate(plays):
        if not isinstance(
            play,
            dict,
        ):
            continue

        period = play.get(
            "period"
        )

        if isinstance(
            period,
            dict,
        ):
            pnum = numero(
                period.get(
                    "number",
                    period.get("value"),
                )
            )
        else:
            pnum = numero(
                period
            )

        if pd.isna(pnum) or int(
            pnum
        ) != 1:
            continue

        typ = play.get(
            "type"
        ) or {}

        if isinstance(
            typ,
            dict,
        ):
            typ_txt = " ".join(
                [
                    str(
                        typ.get(
                            "text",
                            "",
                        )
                    ),
                    str(
                        typ.get(
                            "name",
                            "",
                        )
                    ),
                ]
            )
        else:
            typ_txt = str(
                typ
            )

        txt = (
            typ_txt
            + " "
            + str(
                play.get(
                    "text",
                    "",
                )
            )
            + " "
            + str(
                play.get(
                    "shortText",
                    "",
                )
            )
        ).lower()

        if "corner" not in txt:
            continue

        event_id = str(
            play.get(
                "id",
                f"{i}-{txt}",
            )
        )

        if event_id in seen:
            continue

        seen.add(
            event_id
        )

        total += 1

        team = (
            play.get("team")
            or {}
        )

        team_id = str(
            team.get(
                "id",
                "",
            )
        )

        if team_id == str(
            home_id
        ):
            home += 1
        elif team_id == str(
            away_id
        ):
            away += 1

    return {
        "H1CornersTotal": (
            float(total)
            if total > 0
            else np.nan
        ),
        "H1CornersHome": (
            float(home)
            if total > 0
            else np.nan
        ),
        "H1CornersAway": (
            float(away)
            if total > 0
            else np.nan
        ),
    }


def descargar_contexto_h1(comp_keys=None):
    requested = {str(x) for x in (comp_keys or []) if str(x)}
    context_cache = CONTEXT_CACHE
    if requested:
        suffix = "_".join(sorted(requested))
        context_cache = os.path.join(CACHE_DIR, f"contexto_h1_{suffix}.csv")
    # Cache corto: el objetivo es régimen actual.
    if os.path.exists(
        context_cache
    ):
        try:
            df = pd.read_csv(
                context_cache
            )

            df["Date"] = pd.to_datetime(
                df["Date"],
                errors="coerce",
            )

            age_h = (
                time.time()
                - os.path.getmtime(
                    context_cache
                )
            ) / 3600.0

            required_v10 = {
                "HomeTeam",
                "AwayTeam",
                "H1CornersTotal",
                "Y_H1C_O45",
            }

            if age_h <= 18 and required_v10.issubset(df.columns):
                log(
                    f"Contexto 1T desde cache: {len(df)}"
                )
                return df

        except Exception:
            pass

    end = pd.Timestamp(
        datetime.now(
            TZ_PERU
        ).date()
    )

    start = (
        end
        - pd.Timedelta(
            days=CONTEXT_DAYS
        )
    )

    rows = []
    calls = 0

    for comp_key, info in COMPETICIONES.items():
        if requested and comp_key not in requested:
            continue
        try:
            events = eventos_scoreboard_espn(
                info["espn"],
                start,
                end,
            )
        except Exception:
            continue

        completed = []

        for ev in events:
            st = (
                (
                    ev.get(
                        "status"
                    )
                    or {}
                ).get(
                    "type",
                    {},
                )
            )

            if (
                st.get(
                    "completed"
                )
                or st.get(
                    "state"
                )
                == "post"
            ):
                completed.append(
                    ev
                )

        completed = sorted(
            completed,
            key=lambda x: x.get(
                "date",
                "",
            ),
        )[-CONTEXT_EVENTS_PER_COMP:]

        for ev in completed:
            if calls >= CONTEXT_MAX_SUMMARY_CALLS:
                break

            comps = ev.get(
                "competitions"
            ) or []

            if not comps:
                continue

            c = comps[0]

            sides = {
                x.get(
                    "homeAway"
                ): x
                for x in (
                    c.get(
                        "competitors"
                    )
                    or []
                )
            }

            if (
                "home"
                not in sides
                or "away"
                not in sides
            ):
                continue

            hc = sides[
                "home"
            ]
            ac = sides[
                "away"
            ]

            ht = hc.get(
                "team"
            ) or {}
            at = ac.get(
                "team"
            ) or {}

            hg = numero(
                hc.get(
                    "score"
                )
            )
            ag = numero(
                ac.get(
                    "score"
                )
            )

            if pd.isna(
                hg
            ) or pd.isna(
                ag
            ):
                continue

            dt = pd.to_datetime(
                ev.get(
                    "date"
                ),
                utc=True,
                errors="coerce",
            )

            if pd.isna(
                dt
            ):
                continue

            hodd, dodd, aodd = parse_inline_odds(
                c
            )

            try:
                summary = resumen_evento_espn(
                    info["espn"],
                    ev.get(
                        "id"
                    ),
                )

                calls += 1

                stats = parse_summary_stats(
                    summary,
                    ht.get(
                        "id"
                    ),
                    at.get(
                        "id"
                    ),
                )

                h1 = parse_h1_corners_from_summary(
                    summary,
                    ht.get(
                        "id"
                    ),
                    at.get(
                        "id"
                    ),
                )

            except Exception:
                continue

            fav_type = None

            if (
                pd.notna(hodd)
                and pd.notna(aodd)
            ):
                if hodd < aodd:
                    fav_type = "HOME"
                elif aodd < hodd:
                    fav_type = "AWAY"

            hc_full = numero(
                stats.get(
                    "HC"
                )
            )
            ac_full = numero(
                stats.get(
                    "AC"
                )
            )

            if fav_type == "HOME":
                dog_goal = int(
                    ag >= 1
                )
                fav_corners = hc_full
            elif fav_type == "AWAY":
                dog_goal = int(
                    hg >= 1
                )
                fav_corners = ac_full
            else:
                dog_goal = np.nan
                fav_corners = np.nan

            under45 = int(
                hg + ag <= 4
            )
            under55 = int(
                hg + ag <= 5
            )

            y_base = (
                int(
                    dog_goal == 1
                    and fav_corners >= 4
                    and under45 == 1
                )
                if pd.notna(
                    dog_goal
                )
                and pd.notna(
                    fav_corners
                )
                else np.nan
            )

            y_blind = (
                int(
                    dog_goal == 1
                    and fav_corners >= 3
                    and under55 == 1
                )
                if pd.notna(
                    dog_goal
                )
                and pd.notna(
                    fav_corners
                )
                else np.nan
            )

            h1_total = h1[
                "H1CornersTotal"
            ]

            y_h1c3 = (
                int(
                    h1_total >= 3
                )
                if pd.notna(
                    h1_total
                )
                else np.nan
            )

            rows.append({
                "Date": dt.tz_convert(
                    TZ_PERU
                ).tz_localize(
                    None
                ),
                "CompKey": comp_key,
                "Grupo": info[
                    "grupo"
                ],
                "HomeTeam": (
                    ht.get("displayName")
                    or ht.get("shortDisplayName")
                    or ""
                ),
                "AwayTeam": (
                    at.get("displayName")
                    or at.get("shortDisplayName")
                    or ""
                ),
                "H1CornersTotal": h1_total,
                "H1CornersHome": h1["H1CornersHome"],
                "H1CornersAway": h1["H1CornersAway"],
                "Y_H1C_O45": (
                    int(h1_total >= 5)
                    if pd.notna(h1_total)
                    else np.nan
                ),
                "Y_BASE": y_base,
                "Y_BLINDADO_CORE": y_blind,
                "Y_H1C3": y_h1c3,
                "Y_BASE_H1C3": (
                    int(
                        y_base == 1
                        and y_h1c3 == 1
                    )
                    if pd.notna(
                        y_base
                    )
                    and pd.notna(
                        y_h1c3
                    )
                    else np.nan
                ),
                "Y_BLINDADO_H1C3": (
                    int(
                        y_blind == 1
                        and y_h1c3 == 1
                    )
                    if pd.notna(
                        y_blind
                    )
                    and pd.notna(
                        y_h1c3
                    )
                    else np.nan
                ),
            })

            time.sleep(
                0.025
            )

    df = pd.DataFrame(
        rows
    )

    if not df.empty:
        try:
            df.to_csv(
                context_cache,
                index=False,
                encoding="utf-8-sig",
            )
        except Exception:
            pass

    return df


def conditional_h1_probs(
    context,
    comp_key,
    grupo,
    date,
):
    out = {
        "P_H1C3_given_BASE": np.nan,
        "P_H1C3_given_BLINDADO": np.nan,
        "H1SampleBase": 0,
        "H1SampleBlindado": 0,
    }

    if (
        context is None
        or context.empty
    ):
        return out

    d = pd.Timestamp(
        date
    )

    c = context[
        context["Date"] < d
    ].copy()

    exact = c[
        c["CompKey"]
        == comp_key
    ]

    sample = exact

    if len(
        sample
    ) < 18:
        grp = c[
            c["Grupo"]
            == grupo
        ]

        if len(
            grp
        ) >= 40:
            sample = grp

    b = sample[
        sample["Y_BASE"]
        == 1
    ]

    bb = sample[
        sample[
            "Y_BLINDADO_CORE"
        ]
        == 1
    ]

    # Beta shrinkage fuerte para evitar sobreajuste.
    if (
        len(b) >= 8
        and b[
            "Y_H1C3"
        ].notna().sum()
        >= 8
    ):
        vals = b[
            "Y_H1C3"
        ].dropna()

        out[
            "P_H1C3_given_BASE"
        ] = float(
            (
                vals.sum()
                + 0.75 * 12
            )
            / (
                len(vals)
                + 12
            )
        )

        out[
            "H1SampleBase"
        ] = len(
            vals
        )

    if (
        len(bb) >= 8
        and bb[
            "Y_H1C3"
        ].notna().sum()
        >= 8
    ):
        vals = bb[
            "Y_H1C3"
        ].dropna()

        out[
            "P_H1C3_given_BLINDADO"
        ] = float(
            (
                vals.sum()
                + 0.75 * 12
            )
            / (
                len(vals)
                + 12
            )
        )

        out[
            "H1SampleBlindado"
        ] = len(
            vals
        )

    return out


# ============================================================
# AJUSTES DE TENDENCIA / FATIGA / ALTITUD
# ============================================================

def trend_factors(
    hs,
    aws,
    fav_type,
):
    if fav_type == "HOME":
        fs = hs
        ds = aws
    else:
        fs = aws
        ds = hs

    f_dog = 1.0
    f_corner = 1.0
    notes = []

    dog_recent = (
        ds["Marca5"]
        - ds["Marca20"]
    )

    corner_recent = (
        fs["C4_5"]
        - fs["C4_20"]
    )

    if dog_recent <= -0.20:
        f_dog *= 0.94
        notes.append(
            "underdog llega con deterioro goleador"
        )
    elif dog_recent >= 0.18:
        f_dog *= 1.02
        notes.append(
            "underdog mejora anotando"
        )

    if corner_recent <= -0.20:
        f_corner *= 0.94
        notes.append(
            "favorito cae en 4+ córners"
        )
    elif corner_recent >= 0.18:
        f_corner *= 1.02
        notes.append(
            "favorito mejora en córners"
        )

    return (
        float(
            np.clip(
                f_dog,
                0.92,
                1.03,
            )
        ),
        float(
            np.clip(
                f_corner,
                0.92,
                1.03,
            )
        ),
        "; ".join(
            notes
        ),
    )


def fatigue_adjustments(
    home_load,
    away_load,
    fav_type,
):
    fav_load = (
        home_load
        if fav_type == "HOME"
        else away_load
    )

    dog_load = (
        away_load
        if fav_type == "HOME"
        else home_load
    )

    f_fav_corner = 1.0
    f_dog_goal = 1.0
    conf = 1.0
    notes = []

    fav_rest = fav_load[
        "DaysRest"
    ]
    dog_rest = dog_load[
        "DaysRest"
    ]

    if (
        pd.notna(
            fav_rest
        )
        and fav_rest
        <= REST_SEVERE_DAYS
    ):
        conf *= 0.94
        f_fav_corner *= 0.97
        notes.append(
            "favorito con descanso <=3 días"
        )

    if (
        pd.notna(
            dog_rest
        )
        and dog_rest
        <= REST_SEVERE_DAYS
    ):
        conf *= 0.95
        f_dog_goal *= 0.97
        notes.append(
            "underdog con descanso <=3 días"
        )

    if (
        fav_load[
            "Matches14"
        ]
        >= MATCHES14_CONGESTED
    ):
        conf *= 0.95
        notes.append(
            "favorito en calendario congestionado"
        )

    if (
        dog_load[
            "Matches14"
        ]
        >= MATCHES14_CONGESTED
    ):
        conf *= 0.96
        notes.append(
            "underdog en calendario congestionado"
        )

    # Fatiga asimétrica: el favorito llega mucho más castigado.
    if (
        pd.notna(
            fav_rest
        )
        and pd.notna(
            dog_rest
        )
        and fav_rest
        <= REST_CONGESTED_DAYS
        and dog_rest >= 6
    ):
        f_fav_corner *= 0.97
        conf *= 0.95
        notes.append(
            "desventaja de recuperación del favorito"
        )

    if (
        fav_load[
            "Matches21"
        ]
        >= MATCHES21_HIGH
        or dog_load[
            "Matches21"
        ]
        >= MATCHES21_HIGH
    ):
        conf *= 0.95
        notes.append(
            "carga alta en 21 días"
        )

    return (
        float(
            np.clip(
                f_fav_corner,
                0.92,
                1.02,
            )
        ),
        float(
            np.clip(
                f_dog_goal,
                0.92,
                1.02,
            )
        ),
        float(
            np.clip(
                conf,
                0.82,
                1.0,
            )
        ),
        "; ".join(
            notes
        ),
    )


def altitude_adjustment_v81(
    elevation,
    fav_type,
):
    elev = numero(
        elevation
    )

    factor_dog = 1.0
    conf = 1.0
    veto = False
    note = ""

    if pd.isna(
        elev
    ):
        return (
            factor_dog,
            conf,
            veto,
            note,
        )

    if (
        fav_type == "HOME"
        and elev >= ALTITUDE_VETO
    ):
        factor_dog = 0.90
        conf = 0.90
        veto = True
        note = (
            "veto de altura: underdog visitante "
            f"a {elev:.0f} m"
        )

    elif (
        fav_type == "HOME"
        and elev >= ALTITUDE_STRONG
    ):
        factor_dog = 0.93
        conf = 0.93
        note = (
            f"altura fuerte {elev:.0f} m contra underdog visitante"
        )

    elif elev >= ALTITUDE_CAUTION:
        factor_dog = (
            1.02
            if fav_type == "AWAY"
            else 0.96
        )

        conf = 0.96

        note = (
            f"altura relevante {elev:.0f} m"
        )

    return (
        factor_dog,
        conf,
        veto,
        note,
    )


# ============================================================
# DECISIÓN V8.1
# ============================================================

def dynamic_thresholds(
    phase_home,
    phase_away,
):
    early = (
        phase_home
        in (
            "COLD_START",
            "EARLY",
        )
        or phase_away
        in (
            "COLD_START",
            "EARLY",
        )
    )

    if early:
        return {
            "Dog": P_MIN_DOG_GOL_EARLY,
            "Fav4": P_MIN_FAV4_EARLY,
            "Under45": P_MIN_UNDER45_EARLY,
            "Base": P_MIN_BASE_EARLY,
            "LCB": P_MIN_LCB_EARLY,
            "Early": True,
        }

    return {
        "Dog": P_MIN_DOG_GOL_STABLE,
        "Fav4": P_MIN_FAV4_STABLE,
        "Under45": P_MIN_UNDER45_STABLE,
        "Base": P_MIN_BASE_STABLE,
        "LCB": P_MIN_LCB_STABLE,
        "Early": False,
    }


def decision_robust(
    pred,
):
    th = dynamic_thresholds(
        pred[
            "PhaseHome"
        ],
        pred[
            "PhaseAway"
        ],
    )

    cuota = pred.get(
        "CuotaFavorito",
        np.nan,
    )

    market_ok = pd.notna(
        cuota
    )

    fav_price_ok = (
        market_ok
        and CUOTA_FAVORITO_STRICT_MIN
        <= cuota
        <= CUOTA_FAVORITO_STRICT_MAX
    )

    cold_veto = (
        pred[
            "GamesSeasonHome"
        ]
        < MIN_GAMES_SEASON_APOSTAR
        or pred[
            "GamesSeasonAway"
        ]
        < MIN_GAMES_SEASON_APOSTAR
    )

    strict = (
        pred[
            "P_DogGol_Ajustada"
        ]
        >= th["Dog"]
        and pred[
            "P_Fav4_Ajustada"
        ]
        >= th["Fav4"]
        and pred[
            "P_Under45_Ajustada"
        ]
        >= th["Under45"]
        and pred[
            "P_BASE_Ajustada"
        ]
        >= th["Base"]
        and pred[
            "P_BASE_LCB"
        ]
        >= th["LCB"]
        and pred[
            "ReliabilityScore"
        ]
        >= MIN_RELIABILITY_APOSTAR
        and pred[
            "Soporte"
        ]
        >= 32
        and not pred[
            "AltitudeVeto"
        ]
        and not cold_veto
    )

    near = (
        pred[
            "P_DogGol_Ajustada"
        ]
        >= th["Dog"] - 0.04
        and pred[
            "P_Fav4_Ajustada"
        ]
        >= th["Fav4"] - 0.05
        and pred[
            "P_Under45_Ajustada"
        ]
        >= th["Under45"] - 0.05
        and pred[
            "P_BASE_Ajustada"
        ]
        >= th["Base"] - 0.04
        and pred[
            "ReliabilityScore"
        ]
        >= 0.58
    )

    if strict and fav_price_ok:
        return (
            "APOSTAR",
            "Pasa filtro robusto; validar cuota real del builder",
        )

    if strict:
        return (
            "VIGILAR",
            "Pasa estadística robusta pero falta/queda fuera cuota 1X2",
        )

    if near:
        return (
            "VIGILAR",
            "Candidato cercano al filtro robusto",
        )

    return (
        "NO APOSTAR",
        "No supera el filtro robusto",
    )


# ============================================================
# VARIANTES V8.1
# ============================================================

def crear_variantes_v81(
    pred,
    h1ctx,
    current_rates,
):
    variants = []

    def add(
        name,
        p,
        desc,
        evidence,
    ):
        if pd.isna(
            p
        ) or p < VARIANT_MIN_PROB:
            return

        variants.append({
            "Variante": name,
            "P_Conjunta": float(
                np.clip(
                    p,
                    0.01,
                    0.99,
                )
            ),
            "CuotaJusta": cuota_justa(
                p
            ),
            "CuotaMinExigida": cuota_min_exigida(
                p
            ),
            "Builder": desc,
            "Evidencia": evidence,
        })

    # BASE
    add(
        "BASE",
        pred[
            "P_BASE_Ajustada"
        ],
        (
            f"{pred['Underdog']} anota 1+ | "
            f"{pred['Favorito']} 4+ córners | "
            "Under 4.5 goles"
        ),
        "KNN directo + temporada actual + contexto",
    )

    # BLINDADO CORE: el hallazgo mensual se usa como candidato,
    # pero la probabilidad viene del histórico KNN, no del mes.
    add(
        "BLINDADO_CORE",
        pred[
            "P_BLINDADO_Ajustada"
        ],
        (
            f"{pred['Underdog']} anota 1+ | "
            f"{pred['Favorito']} 3+ córners | "
            "Under 5.5 goles"
        ),
        (
            "KNN directo. Solo usar si la casa aún ofrece "
            "cuota real >=4.20 y >= cuota mínima."
        ),
    )

    # Booster 3+ córners 1T: condicional, solo si hay muestra.
    p_h1_base = h1ctx.get(
        "P_H1C3_given_BASE",
        np.nan,
    )

    if (
        pd.notna(
            p_h1_base
        )
        and h1ctx.get(
            "H1SampleBase",
            0,
        )
        >= 8
    ):
        p = (
            pred[
                "P_BASE_Ajustada"
            ]
            * p_h1_base
        )

        add(
            "BASE + 3+ CORNERS 1T",
            p,
            (
                f"{pred['Underdog']} anota 1+ | "
                f"{pred['Favorito']} 4+ córners | "
                "Under 4.5 | 3+ córners totales 1T"
            ),
            (
                "Prob. BASE × P(3+ córners 1T | BASE) "
                f"con N={h1ctx['H1SampleBase']}"
            ),
        )

    p_h1_blind = h1ctx.get(
        "P_H1C3_given_BLINDADO",
        np.nan,
    )

    if (
        pd.notna(
            p_h1_blind
        )
        and h1ctx.get(
            "H1SampleBlindado",
            0,
        )
        >= 8
    ):
        p = (
            pred[
                "P_BLINDADO_Ajustada"
            ]
            * p_h1_blind
        )

        add(
            "BLINDADO + 3+ CORNERS 1T",
            p,
            (
                f"{pred['Underdog']} anota 1+ | "
                f"{pred['Favorito']} 3+ córners | "
                "Under 5.5 | 3+ córners totales 1T"
            ),
            (
                "Prob. blindado × P(3+ córners 1T | blindado) "
                f"con N={h1ctx['H1SampleBlindado']}"
            ),
        )

    # Tarjetas: solo si la temporada actual también lo respalda.
    cards_current = current_rates.get(
        "Y_Cards4",
        np.nan,
    )

    cards_n = current_rates.get(
        "Y_Cards4_N",
        0,
    )

    if (
        pd.notna(
            pred[
                "P_BASE_CARDS_Ajustada"
            ]
        )
        and pred[
            "P_Cards4_Ajustada"
        ]
        >= 0.64
        and (
            pd.isna(
                cards_current
            )
            or cards_n < 20
            or cards_current >= 0.61
        )
    ):
        add(
            "BASE + 4+ TARJETAS",
            pred[
                "P_BASE_CARDS_Ajustada"
            ],
            (
                f"{pred['Underdog']} anota 1+ | "
                f"{pred['Favorito']} 4+ córners | "
                "Under 4.5 | 4+ tarjetas"
            ),
            (
                "Joint histórico directo + temporada actual + árbitro"
            ),
        )

    # Gol 1T solo con marginal alto.
    if (
        pd.notna(
            pred[
                "P_BASE_GOL1T_Ajustada"
            ]
        )
        and pred[
            "P_Gol1T_Ajustada"
        ]
        >= 0.66
    ):
        add(
            "BASE + GOL 1T",
            pred[
                "P_BASE_GOL1T_Ajustada"
            ],
            (
                f"{pred['Underdog']} anota 1+ | "
                f"{pred['Favorito']} 4+ córners | "
                "Under 4.5 | Over 0.5 gol 1T"
            ),
            "Joint histórico directo; activado solo con P(Gol1T) alta",
        )

    # Orden conservador:
    # probabilidad alta primero; en empate, cuota mínima menor.
    variants.sort(
        key=lambda x: (
            -x["P_Conjunta"],
            x["CuotaMinExigida"],
        )
    )

    return variants


# ============================================================
# PRONÓSTICO ROBUSTO
# ============================================================

def pronosticar_v81(
    fixtures,
    ds,
    estados,
    perfiles_arbitro,
    schedule_index,
    context_h1,
):
    rows = []
    var_rows = []

    ref_calls = [0]
    weather_calls = [0]

    state_names = list(
        estados.keys()
    )

    for _, fr in fixtures.iterrows():
        comp_key = fr[
            "CompKey"
        ]

        info = COMPETICIONES[
            comp_key
        ]

        home_orig = fr[
            "HomeOriginal"
        ]

        away_orig = fr[
            "AwayOriginal"
        ]

        home, sh = resolver_nombre(
            home_orig,
            state_names,
        )

        away, sa = resolver_nombre(
            away_orig,
            state_names,
        )

        base_meta = {
            "Fecha": fr[
                "Date"
            ],
            "HoraPeru": fr[
                "HoraPeru"
            ],
            "Competicion": info[
                "nombre"
            ],
            "CompKey": comp_key,
            "Local": home_orig,
            "Visitante": away_orig,
            "LocalModelo": home,
            "VisitanteModelo": away,
        }

        if sh < 0.57 or sa < 0.57:
            rows.append({
                **base_meta,
                "Accion": "NO APOSTAR",
                "Motivo": "Emparejamiento de nombre insuficiente",
            })
            continue

        hs = snapshot(
            estados[home]
        )

        aws = snapshot(
            estados[away]
        )

        if hs is None or aws is None:
            rows.append({
                **base_meta,
                "Accion": "NO APOSTAR",
                "Motivo": "Historial insuficiente de equipo",
            })
            continue

        model_row = fr.copy()
        model_row[
            "HomeTeam"
        ] = home
        model_row[
            "AwayTeam"
        ] = away

        fav = detectar_favorito_mercado(
            model_row
        )

        if fav is None:
            fav = favorito_modelo(
                hs,
                aws,
            )

        feat = crear_features(
            model_row,
            hs,
            aws,
            fav=fav,
        )

        if feat is None:
            continue

        probs = knn_probabilidades_v81(
            ds,
            feat,
            fr["Date"],
            comp_key,
            info["grupo"],
        )

        if probs is None:
            rows.append({
                **base_meta,
                "Accion": "NO APOSTAR",
                "Motivo": "Sin vecinos históricos suficientes",
            })
            continue

        # Workload.
        home_load = workload_features(
            schedule_index,
            home,
            fr["Date"],
            comp_key,
        )

        away_load = workload_features(
            schedule_index,
            away,
            fr["Date"],
            comp_key,
        )

        # Season calibration.
        csr = current_season_rates(
            ds,
            comp_key,
            info["grupo"],
            fr["Date"],
        )

        p_dog = calibrate_current_season(
            probs["Y_DogGol"],
            csr.get(
                "Y_DogGol"
            ),
            csr.get(
                "Y_DogGol_N",
                0,
            ),
        )

        p_fav4 = calibrate_current_season(
            probs["Y_Fav4"],
            csr.get(
                "Y_Fav4"
            ),
            csr.get(
                "Y_Fav4_N",
                0,
            ),
        )

        p_under45 = calibrate_current_season(
            probs["Y_Under45"],
            csr.get(
                "Y_Under45"
            ),
            csr.get(
                "Y_Under45_N",
                0,
            ),
        )

        p_base = calibrate_current_season(
            probs["Y_BASE"],
            csr.get(
                "Y_BASE"
            ),
            csr.get(
                "Y_BASE_N",
                0,
            ),
        )

        p_blind = calibrate_current_season(
            probs[
                "Y_BLINDADO_CORE"
            ],
            csr.get(
                "Y_BLINDADO_CORE"
            ),
            csr.get(
                "Y_BLINDADO_CORE_N",
                0,
            ),
        )

        p_cards = calibrate_current_season(
            probs["Y_Cards4"],
            csr.get(
                "Y_Cards4"
            ),
            csr.get(
                "Y_Cards4_N",
                0,
            ),
        )

        p_gol1t = calibrate_current_season(
            probs["Y_Gol1T"],
            csr.get(
                "Y_Gol1T"
            ),
            csr.get(
                "Y_Gol1T_N",
                0,
            ),
        )

        # Trends.
        trend_dog, trend_corner, trend_note = (
            trend_factors(
                hs,
                aws,
                fav["tipo"],
            )
        )

        # Fatigue.
        (
            fatigue_corner,
            fatigue_dog,
            fatigue_conf,
            fatigue_note,
        ) = fatigue_adjustments(
            home_load,
            away_load,
            fav["tipo"],
        )

        # Referee.
        referee = ""

        if p_base >= 0.22:
            referee = completar_referee_future(
                fr,
                ref_calls,
            )

        p_cards_adj, referee_note = (
            ajustar_cards_por_arbitro(
                p_cards,
                referee,
                perfiles_arbitro,
            )
        )

        # Weather.
        weather = {}

        if (
            weather_calls[0]
            < MAX_WEATHER_CALLS
            and fr.get(
                "VenueCity"
            )
        ):
            weather = clima_partido(
                fr.get(
                    "VenueCity",
                    "",
                ),
                fr.get(
                    "VenueCountry",
                    "",
                ),
                fr.get(
                    "KickoffUTC",
                    "",
                ),
            )

            weather_calls[0] += 1

        # Weather does not create probability.
        weather_conf, _, climate_note = (
            factor_contexto_clima(
                weather,
                fav["tipo"],
            )
        )

        alt_dog, alt_conf, alt_veto, alt_note = (
            altitude_adjustment_v81(
                weather.get(
                    "Elevation",
                    np.nan,
                ),
                fav["tipo"],
            )
        )

        # Final marginal adjustments.
        p_dog_adj = float(
            np.clip(
                p_dog
                * trend_dog
                * fatigue_dog
                * alt_dog,
                0.01,
                0.99,
            )
        )

        p_fav4_adj = float(
            np.clip(
                p_fav4
                * trend_corner
                * fatigue_corner,
                0.01,
                0.99,
            )
        )

        p_under45_adj = p_under45

        # Joint BASE adjusts only for relevant legs, with caps.
        ratio_dog = (
            p_dog_adj
            / max(
                p_dog,
                0.01,
            )
        )

        ratio_corner = (
            p_fav4_adj
            / max(
                p_fav4,
                0.01,
            )
        )

        joint_factor = float(
            np.clip(
                ratio_dog
                * ratio_corner,
                0.88,
                1.06,
            )
        )

        p_base_adj = float(
            np.clip(
                p_base
                * joint_factor,
                0.01,
                0.99,
            )
        )

        # Blindado uses same dog context; fav3 is inherently looser.
        p_blind_adj = float(
            np.clip(
                p_blind
                * np.clip(
                    ratio_dog,
                    0.92,
                    1.04,
                ),
                0.01,
                0.99,
            )
        )

        # Cards joint: referee/current regime.
        p_base_cards = probs[
            "Y_BASE_CARDS"
        ]

        if (
            pd.notna(
                p_base_cards
            )
            and pd.notna(
                probs[
                    "Y_Cards4"
                ]
            )
            and probs[
                "Y_Cards4"
            ] > 0
        ):
            p_base_cards_adj = float(
                np.clip(
                    p_base_cards
                    * joint_factor
                    * (
                        p_cards_adj
                        / probs[
                            "Y_Cards4"
                        ]
                    ),
                    0.01,
                    0.99,
                )
            )
        else:
            p_base_cards_adj = np.nan

        p_base_g1 = probs[
            "Y_BASE_GOL1T"
        ]

        if pd.notna(
            p_base_g1
        ):
            p_base_g1_adj = float(
                np.clip(
                    p_base_g1
                    * joint_factor,
                    0.01,
                    0.99,
                )
            )
        else:
            p_base_g1_adj = np.nan

        # Fase de temporada.
        phase_home = home_load[
            "Phase"
        ]
        phase_away = away_load[
            "Phase"
        ]

        phase_conf = min(
            phase_confidence_factor(
                phase_home
            ),
            phase_confidence_factor(
                phase_away
            ),
        )

        reliability = (
            probs[
                "Confianza"
            ]
            * phase_conf
            * fatigue_conf
            * weather_conf
            * alt_conf
        )

        reliability = float(
            np.clip(
                reliability,
                0.0,
                1.0,
            )
        )

        if fav["tipo"] == "HOME":
            favorito = home_orig
            dog = away_orig
        else:
            favorito = away_orig
            dog = home_orig

        pred = {
            **base_meta,
            "Favorito": favorito,
            "Underdog": dog,
            "FavoritoTipo": fav[
                "tipo"
            ],
            "CuotaFavorito": fav.get(
                "cuota",
                np.nan,
            ),
            "FuenteFavorito": fav.get(
                "fuente",
                "",
            ),
            "P_DogGol_Ajustada": p_dog_adj,
            "P_Fav4_Ajustada": p_fav4_adj,
            "P_Under45_Ajustada": p_under45_adj,
            "P_BASE_Ajustada": p_base_adj,
            "P_BASE_LCB": probs[
                "BASE_LCB"
            ],
            "P_BLINDADO_Ajustada": p_blind_adj,
            "P_BLINDADO_LCB": probs[
                "BLINDADO_LCB"
            ],
            "P_Cards4_Ajustada": p_cards_adj,
            "P_Gol1T_Ajustada": p_gol1t,
            "P_BASE_CARDS_Ajustada": p_base_cards_adj,
            "P_BASE_GOL1T_Ajustada": p_base_g1_adj,
            "Soporte": probs[
                "Y_BASE_Neff"
            ],
            "ReliabilityScore": reliability,
            "GamesSeasonHome": home_load[
                "GamesSeasonComp"
            ],
            "GamesSeasonAway": away_load[
                "GamesSeasonComp"
            ],
            "PhaseHome": phase_home,
            "PhaseAway": phase_away,
            "DaysRestHome": home_load[
                "DaysRest"
            ],
            "DaysRestAway": away_load[
                "DaysRest"
            ],
            "Matches14Home": home_load[
                "Matches14"
            ],
            "Matches14Away": away_load[
                "Matches14"
            ],
            "CurrentSeasonN": csr[
                "CurrentSeasonN"
            ],
            "TemperatureC": weather.get(
                "TemperatureC",
                np.nan,
            ),
            "HumidityPct": weather.get(
                "HumidityPct",
                np.nan,
            ),
            "RainProbPct": weather.get(
                "RainProbPct",
                np.nan,
            ),
            "WindKmh": weather.get(
                "WindKmh",
                np.nan,
            ),
            "ElevationM": weather.get(
                "Elevation",
                np.nan,
            ),
            "AltitudeVeto": alt_veto,
            "Referee": referee,
            "TrendNote": trend_note,
            "FatigueNote": fatigue_note,
            "ClimateNote": climate_note,
            "AltitudeNote": alt_note,
            "RefereeNote": referee_note,
        }

        action, reason = decision_robust(
            pred
        )

        pred[
            "Accion"
        ] = action

        pred[
            "Motivo"
        ] = reason

        h1ctx = conditional_h1_probs(
            context_h1,
            comp_key,
            info[
                "grupo"
            ],
            fr[
                "Date"
            ],
        )

        variants = crear_variantes_v81(
            pred,
            h1ctx,
            csr,
        )

        # Variant with highest success probability = "safer".
        if variants:
            safest = variants[0]

            pred[
                "VarianteMasSegura"
            ] = safest[
                "Variante"
            ]

            pred[
                "P_VarianteMasSegura"
            ] = safest[
                "P_Conjunta"
            ]

            pred[
                "CuotaMinMasSegura"
            ] = safest[
                "CuotaMinExigida"
            ]
        else:
            pred[
                "VarianteMasSegura"
            ] = ""

            pred[
                "P_VarianteMasSegura"
            ] = np.nan

            pred[
                "CuotaMinMasSegura"
            ] = np.nan

        # Score: stability > volume.
        pred[
            "ScoreRobusto"
        ] = (
            100.0
            * (
                0.43
                * p_base_adj
                + 0.18
                * probs[
                    "BASE_LCB"
                ]
                + 0.13
                * p_dog_adj
                + 0.09
                * p_fav4_adj
                + 0.07
                * p_under45_adj
                + 0.10
                * reliability
            )
        )

        if (
            phase_home
            == "COLD_START"
            or phase_away
            == "COLD_START"
        ):
            pred[
                "ScoreRobusto"
            ] -= 5.0

        if alt_veto:
            pred[
                "ScoreRobusto"
            ] -= 8.0

        rows.append(
            pred
        )

        for v in variants:
            var_rows.append({
                "Fecha": fr[
                    "Date"
                ],
                "HoraPeru": fr[
                    "HoraPeru"
                ],
                "Competicion": info[
                    "nombre"
                ],
                "Local": home_orig,
                "Visitante": away_orig,
                "AccionPartido": action,
                "ReliabilityScore": reliability,
                **v,
                "CuotaReal": "",
                "CumpleCuota": "PENDIENTE",
                "EVReal": "",
            })

    return (
        pd.DataFrame(
            rows
        ),
        pd.DataFrame(
            var_rows
        ),
    )


# ============================================================
# TOP 30: NO FORZAR
# ============================================================

def crear_top30_v81(
    resultado
):
    if resultado.empty:
        return resultado

    r = resultado[
        resultado[
            "Accion"
        ].isin(
            [
                "APOSTAR",
                "VIGILAR",
            ]
        )
    ].copy()

    if r.empty:
        return r

    order = {
        "APOSTAR": 0,
        "VIGILAR": 1,
    }

    r[
        "_orden"
    ] = r[
        "Accion"
    ].map(
        order
    )

    r = r.sort_values(
        [
            "_orden",
            "ScoreRobusto",
            "P_BASE_Ajustada",
            "ReliabilityScore",
        ],
        ascending=[
            True,
            False,
            False,
            False,
        ],
    ).head(
        TOP_N
    )

    r[
        "Ranking"
    ] = np.arange(
        1,
        len(r) + 1,
    )

    cols = [
        "Ranking",
        "Accion",
        "Fecha",
        "HoraPeru",
        "Competicion",
        "Local",
        "Visitante",
        "Favorito",
        "Underdog",
        "CuotaFavorito",
        "P_DogGol_Ajustada",
        "P_Fav4_Ajustada",
        "P_Under45_Ajustada",
        "P_BASE_Ajustada",
        "P_BASE_LCB",
        "P_BLINDADO_Ajustada",
        "VarianteMasSegura",
        "P_VarianteMasSegura",
        "CuotaMinMasSegura",
        "ReliabilityScore",
        "GamesSeasonHome",
        "GamesSeasonAway",
        "PhaseHome",
        "PhaseAway",
        "DaysRestHome",
        "DaysRestAway",
        "Matches14Home",
        "Matches14Away",
        "CurrentSeasonN",
        "TemperatureC",
        "HumidityPct",
        "RainProbPct",
        "WindKmh",
        "ElevationM",
        "Referee",
        "TrendNote",
        "FatigueNote",
        "ClimateNote",
        "AltitudeNote",
        "RefereeNote",
        "ScoreRobusto",
        "Motivo",
    ]

    for c in cols:
        if c not in r.columns:
            r[
                c
            ] = np.nan

    return r[
        cols
    ]



# ============================================================
# HOJA SIMPLE PARA EL USUARIO
# ============================================================

def crear_apuestas_claras_v81(top, variants):
    """
    Crea una vista de una fila por partido con los mercados separados
    como los ve el usuario en la casa de apuestas.

    IMPORTANTE:
    - No inventa mercados no modelados.
    - "RESULTADO" queda como NO INCLUIDO porque V8.1 actual no calcula
      1X2 / doble oportunidad como pata del builder.
    """
    columnas = [
        "N°",
        "ESTADO",
        "FECHA",
        "HORA",
        "COMPETICIÓN",
        "PARTIDO",
        "RESULTADO",
        "GOL DE EQUIPO",
        "CÓRNERS DE EQUIPO",
        "TOTAL GOLES",
        "TARJETAS",
        "CÓRNERS 1T",
        "GOL 1T",
        "CUOTA MÍNIMA",
        "CUOTA REAL",
        "CUMPLE CUOTA",
        "FIABILIDAD",
        "PROB. BUILDER",
        "APUESTA COMPLETA",
        "OBSERVACIÓN",
    ]

    if top is None or top.empty:
        return pd.DataFrame(columns=columnas)

    rows = []

    for _, r in top.iterrows():
        variante = str(r.get("VarianteMasSegura", "") or "")
        favorito = str(r.get("Favorito", "") or "")
        dog = str(r.get("Underdog", "") or "")
        local = str(r.get("Local", "") or "")
        visita = str(r.get("Visitante", "") or "")

        # Etiqueta local/visitante para que sea aún más fácil construirla.
        if favorito == local:
            fav_label = f"{favorito} (LOCAL)"
        elif favorito == visita:
            fav_label = f"{favorito} (VISITA)"
        else:
            fav_label = favorito

        if dog == local:
            dog_label = f"{dog} (LOCAL)"
        elif dog == visita:
            dog_label = f"{dog} (VISITA)"
        else:
            dog_label = dog

        resultado_market = "NO INCLUIDO"
        gol_equipo = f"{dog_label}: MARCA 1+ GOL"
        corners_equipo = f"{fav_label}: 4+ CÓRNERS"
        total_goles = "MENOS DE 4.5 GOLES"
        tarjetas = "—"
        corners_1t = "—"
        gol_1t = "—"

        if variante.startswith("BLINDADO"):
            corners_equipo = f"{fav_label}: 3+ CÓRNERS"
            total_goles = "MENOS DE 5.5 GOLES"

        if "4+ TARJETAS" in variante:
            tarjetas = "4+ TARJETAS TOTALES"

        if "3+ CORNERS 1T" in variante:
            corners_1t = "3+ CÓRNERS TOTALES 1T"

        if "GOL 1T" in variante:
            gol_1t = "MÁS DE 0.5 GOLES 1T"

        patas = [
            gol_equipo,
            corners_equipo,
            total_goles,
        ]

        if tarjetas != "—":
            patas.append(tarjetas)

        if corners_1t != "—":
            patas.append(corners_1t)

        if gol_1t != "—":
            patas.append(gol_1t)

        apuesta_completa = " + ".join(patas)

        rows.append({
            "N°": int(r.get("Ranking", len(rows) + 1)),
            "ESTADO": str(r.get("Accion", "")),
            "FECHA": r.get("Fecha", ""),
            "HORA": r.get("HoraPeru", ""),
            "COMPETICIÓN": r.get("Competicion", ""),
            "PARTIDO": f"{local} vs {visita}",
            "RESULTADO": resultado_market,
            "GOL DE EQUIPO": gol_equipo,
            "CÓRNERS DE EQUIPO": corners_equipo,
            "TOTAL GOLES": total_goles,
            "TARJETAS": tarjetas,
            "CÓRNERS 1T": corners_1t,
            "GOL 1T": gol_1t,
            "CUOTA MÍNIMA": r.get("CuotaMinMasSegura", np.nan),
            "CUOTA REAL": "",
            "CUMPLE CUOTA": "PENDIENTE",
            "FIABILIDAD": r.get("ReliabilityScore", np.nan),
            "PROB. BUILDER": r.get("P_VarianteMasSegura", np.nan),
            "APUESTA COMPLETA": apuesta_completa,
            "OBSERVACIÓN": (
                "APOSTAR solo si CUOTA REAL >= CUOTA MÍNIMA"
                if str(r.get("Accion", "")) == "APOSTAR"
                else "VIGILAR: todavía no cumple todos los filtros estrictos"
            ),
        })

    return pd.DataFrame(rows, columns=columnas)



# ============================================================
# EXPORT V8.1
# ============================================================

def exportar_v81(
    top,
    variants,
    resultado,
    inicio,
    fin,
):
    top.to_csv(
        ARCHIVO_CSV_V81,
        index=False,
        encoding="utf-8-sig",
    )

    variants.to_csv(
        ARCHIVO_VARIANTES_V81,
        index=False,
        encoding="utf-8-sig",
    )

    apostar = (
        top[
            top["Accion"]
            == "APOSTAR"
        ].copy()
        if not top.empty
        else top.copy()
    )

    vigilar = (
        top[
            top["Accion"]
            == "VIGILAR"
        ].copy()
        if not top.empty
        else top.copy()
    )

    config = pd.DataFrame(
        [
            [
                "Version",
                VERSION,
            ],
            [
                "Ventana",
                (
                    f"{inicio.strftime('%d/%m/%Y')} "
                    f"a {fin.strftime('%d/%m/%Y')}"
                ),
            ],
            [
                "Máximo selecciones",
                TOP_N,
            ],
            [
                "Forzar 30",
                "NO",
            ],
            [
                "Cuota combinada mínima",
                CUOTA_COMBINADA_MIN,
            ],
            [
                "ROI objetivo",
                ROI_OBJETIVO,
            ],
            [
                "Cold start",
                (
                    "0-4 partidos temporada: no APOSTAR; "
                    "5-8 umbrales más duros"
                ),
            ],
            [
                "Congestión",
                (
                    "descanso <=4 días / 4+ partidos en 14d "
                    "reduce confianza y aplica ajustes pequeños"
                ),
            ],
            [
                "Current season",
                (
                    "probabilidades KNN se recalibran con temporada "
                    "actual, peso máximo 30%"
                ),
            ],
            [
                "Altitud",
                (
                    ">=2200m penaliza gol underdog visitante; "
                    ">=2800m puede vetar APOSTAR"
                ),
            ],
            [
                "BLINDADO_CORE",
                "Dog 1+ + Fav 3+ corners + Under 5.5",
            ],
            [
                "BOOSTER 1T",
                (
                    "3+ corners 1T solo si ESPN ofrece play-by-play "
                    "y muestra condicional suficiente"
                ),
            ],
            [
                "Regla de precio",
                (
                    "CuotaReal >= CuotaMinExigida y nunca <4.20"
                ),
            ],
            [
                "Advertencia",
                (
                    "Más estable no significa seguro ni garantiza beneficio."
                ),
            ],
        ],
        columns=[
            "Parametro",
            "Valor",
        ],
    )

    with pd.ExcelWriter(
        ARCHIVO_XLSX_V81,
        engine="openpyxl",
    ) as writer:
        # PRIMERA HOJA: lectura simple para apostar.
        # Dejamos cuatro filas arriba para título/leyenda.
        apuestas_claras.to_excel(
            writer,
            sheet_name="APUESTAS_CLARAS",
            index=False,
            startrow=4,
        )

        top.to_excel(
            writer,
            sheet_name="TOP_30_MAX",
            index=False,
        )

        apostar.to_excel(
            writer,
            sheet_name="APOSTAR",
            index=False,
        )

        vigilar.to_excel(
            writer,
            sheet_name="VIGILAR",
            index=False,
        )

        variants.to_excel(
            writer,
            sheet_name="VARIANTES_PRECIO",
            index=False,
        )

        resultado.to_excel(
            writer,
            sheet_name="TODOS",
            index=False,
        )

        config.to_excel(
            writer,
            sheet_name="CONFIG",
            index=False,
        )

        from openpyxl.styles import (
            Font,
            PatternFill,
            Alignment,
            Border,
            Side,
        )

        from openpyxl.utils import (
            get_column_letter,
        )

        wb = writer.book

        for ws in wb.worksheets:
            ws.freeze_panes = "A2"
            ws.auto_filter.ref = ws.dimensions

            for cell in ws[1]:
                cell.font = Font(
                    bold=True,
                    color="FFFFFF",
                )

                cell.fill = PatternFill(
                    "solid",
                    fgColor="1F4E78",
                )

                cell.alignment = Alignment(
                    horizontal="center",
                )

            for col in ws.columns:
                letter = col[
                    0
                ].column_letter

                max_len = 0

                for cell in col[:100]:
                    val = (
                        ""
                        if cell.value is None
                        else str(
                            cell.value
                        )
                    )

                    max_len = max(
                        max_len,
                        len(
                            val
                        ),
                    )

                ws.column_dimensions[
                    letter
                ].width = min(
                    max(
                        10,
                        max_len + 2,
                    ),
                    42,
                )

        # ====================================================
        # APUESTAS_CLARAS: hoja principal, visual y fácil de leer
        # ====================================================
        ws_simple = wb["APUESTAS_CLARAS"]

        max_col_simple = 20

        # Título
        ws_simple.merge_cells(
            start_row=1,
            start_column=1,
            end_row=1,
            end_column=max_col_simple,
        )
        title = ws_simple.cell(1, 1)
        title.value = "BET BUILDER V8.1 ROBUST — QUÉ APOSTAR"
        title.font = Font(
            bold=True,
            color="FFFFFF",
            size=18,
        )
        title.fill = PatternFill(
            "solid",
            fgColor="102A43",
        )
        title.alignment = Alignment(
            horizontal="center",
            vertical="center",
        )
        ws_simple.row_dimensions[1].height = 30

        # Instrucción breve
        ws_simple.merge_cells(
            start_row=2,
            start_column=1,
            end_row=2,
            end_column=max_col_simple,
        )
        instruction = ws_simple.cell(2, 1)
        instruction.value = (
            "LEE ESTA HOJA PRIMERO: construye exactamente las patas indicadas. "
            "Solo una fila APOSTAR es válida si la CUOTA REAL alcanza la CUOTA MÍNIMA. "
            "VIGILAR = no apostar todavía."
        )
        instruction.font = Font(
            bold=True,
            color="203040",
            size=10,
        )
        instruction.fill = PatternFill(
            "solid",
            fgColor="D9EAF7",
        )
        instruction.alignment = Alignment(
            wrap_text=True,
            vertical="center",
        )
        ws_simple.row_dimensions[2].height = 36

        # Leyenda
        ws_simple["A3"] = "🟢 APOSTAR"
        ws_simple["B3"] = "🟡 VIGILAR"
        ws_simple["C3"] = "CUOTA REAL: escribir la ofrecida por la casa"
        ws_simple["F3"] = "RESULTADO = NO INCLUIDO mientras el modelo no calcule 1X2/doble oportunidad"

        ws_simple["A3"].fill = PatternFill("solid", fgColor="C6EFCE")
        ws_simple["B3"].fill = PatternFill("solid", fgColor="FFF2CC")
        ws_simple["C3"].fill = PatternFill("solid", fgColor="DDEBF7")
        ws_simple["F3"].fill = PatternFill("solid", fgColor="F2F2F2")

        for cell in ("A3", "B3", "C3", "F3"):
            ws_simple[cell].font = Font(bold=True)
            ws_simple[cell].alignment = Alignment(
                wrap_text=True,
                vertical="center",
            )

        # Cabecera real está en fila 5 por startrow=4.
        header_row = 5

        header_map = {
            c.value: c.column
            for c in ws_simple[header_row]
        }

        for cell in ws_simple[header_row]:
            cell.font = Font(
                bold=True,
                color="FFFFFF",
                size=10,
            )
            cell.fill = PatternFill(
                "solid",
                fgColor="1F4E78",
            )
            cell.alignment = Alignment(
                horizontal="center",
                vertical="center",
                wrap_text=True,
            )

        ws_simple.freeze_panes = "A6"
        ws_simple.auto_filter.ref = (
            f"A5:{get_column_letter(max_col_simple)}"
            f"{max(ws_simple.max_row, 5)}"
        )

        # Anchos pensados para celular/Excel.
        simple_widths = {
            "A": 5,
            "B": 12,
            "C": 12,
            "D": 8,
            "E": 21,
            "F": 28,
            "G": 17,
            "H": 31,
            "I": 32,
            "J": 20,
            "K": 22,
            "L": 24,
            "M": 21,
            "N": 14,
            "O": 14,
            "P": 15,
            "Q": 13,
            "R": 14,
            "S": 58,
            "T": 46,
        }

        for letter, width in simple_widths.items():
            ws_simple.column_dimensions[letter].width = width

        thin = Side(
            style="thin",
            color="D9E2F3",
        )

        for rr in range(6, ws_simple.max_row + 1):
            for cc in range(1, max_col_simple + 1):
                cell = ws_simple.cell(rr, cc)
                cell.alignment = Alignment(
                    vertical="top",
                    wrap_text=True,
                )
                cell.border = Border(
                    bottom=thin,
                )

            estado = str(ws_simple.cell(
                rr,
                header_map.get("ESTADO", 2),
            ).value or "")

            if estado == "APOSTAR":
                row_fill = "E2F0D9"
                strong_fill = "70AD47"
                strong_font = "FFFFFF"
            elif estado == "VIGILAR":
                row_fill = "FFF2CC"
                strong_fill = "FFC000"
                strong_font = "5B4500"
            else:
                row_fill = "F2F2F2"
                strong_fill = "A6A6A6"
                strong_font = "FFFFFF"

            # Color suave de toda la fila.
            for cc in range(1, max_col_simple + 1):
                ws_simple.cell(rr, cc).fill = PatternFill(
                    "solid",
                    fgColor=row_fill,
                )

            # ESTADO destacado.
            estado_cell = ws_simple.cell(
                rr,
                header_map["ESTADO"],
            )
            estado_cell.fill = PatternFill(
                "solid",
                fgColor=strong_fill,
            )
            estado_cell.font = Font(
                bold=True,
                color=strong_font,
            )
            estado_cell.alignment = Alignment(
                horizontal="center",
                vertical="center",
            )

            # CUOTA REAL editable.
            real_cell = ws_simple.cell(
                rr,
                header_map["CUOTA REAL"],
            )
            real_cell.fill = PatternFill(
                "solid",
                fgColor="FFF2CC",
            )
            real_cell.number_format = "0.00"

            # CUOTA MÍNIMA.
            min_cell = ws_simple.cell(
                rr,
                header_map["CUOTA MÍNIMA"],
            )
            min_cell.fill = PatternFill(
                "solid",
                fgColor="D9EAF7",
            )
            min_cell.number_format = "0.00"
            min_cell.font = Font(
                bold=True,
                color="1F4E78",
            )

            # CUMPLE CUOTA fórmula.
            c_real = get_column_letter(
                header_map["CUOTA REAL"]
            )
            c_min = get_column_letter(
                header_map["CUOTA MÍNIMA"]
            )
            ok_cell = ws_simple.cell(
                rr,
                header_map["CUMPLE CUOTA"],
            )
            ok_cell.value = (
                f'=IF({c_real}{rr}="","PENDIENTE",'
                f'IF(AND({c_real}{rr}>={c_min}{rr},'
                f'{c_real}{rr}>={CUOTA_COMBINADA_MIN}),"SI","NO"))'
            )
            ok_cell.font = Font(
                bold=True,
            )
            ok_cell.alignment = Alignment(
                horizontal="center",
            )

            # % fiabilidad y probabilidad.
            ws_simple.cell(
                rr,
                header_map["FIABILIDAD"],
            ).number_format = "0.0%"

            ws_simple.cell(
                rr,
                header_map["PROB. BUILDER"],
            ).number_format = "0.0%"

            # APUESTA COMPLETA resaltada.
            full_cell = ws_simple.cell(
                rr,
                header_map["APUESTA COMPLETA"],
            )
            full_cell.fill = PatternFill(
                "solid",
                fgColor="EAF2F8",
            )
            full_cell.font = Font(
                bold=True,
                color="17365D",
            )

            ws_simple.row_dimensions[rr].height = 52

        # Precio real en VARIANTES.
        ws = wb[
            "VARIANTES_PRECIO"
        ]

        headers = {
            c.value: c.column
            for c in ws[1]
        }

        if all(
            h in headers
            for h in [
                "P_Conjunta",
                "CuotaMinExigida",
                "CuotaReal",
                "CumpleCuota",
                "EVReal",
            ]
        ):
            c_p = get_column_letter(
                headers[
                    "P_Conjunta"
                ]
            )

            c_min = get_column_letter(
                headers[
                    "CuotaMinExigida"
                ]
            )

            c_real = get_column_letter(
                headers[
                    "CuotaReal"
                ]
            )

            c_ok = headers[
                "CumpleCuota"
            ]

            c_ev = headers[
                "EVReal"
            ]

            for rr in range(
                2,
                ws.max_row + 1,
            ):
                ws.cell(
                    rr,
                    c_ok,
                ).value = (
                    f'=IF({c_real}{rr}="","PENDIENTE",'
                    f'IF(AND({c_real}{rr}>={c_min}{rr},'
                    f'{c_real}{rr}>={CUOTA_COMBINADA_MIN}),'
                    '"SI","NO"))'
                )

                ws.cell(
                    rr,
                    c_ev,
                ).value = (
                    f'=IF({c_real}{rr}="","",'
                    f'{c_p}{rr}*{c_real}{rr}-1)'
                )

            for rr in range(
                2,
                ws.max_row + 1,
            ):
                ws.cell(
                    rr,
                    headers[
                        "P_Conjunta"
                    ],
                ).number_format = "0.0%"

                ws.cell(
                    rr,
                    c_ev,
                ).number_format = "0.0%"

                ws.cell(
                    rr,
                    headers[
                        "CuotaMinExigida"
                    ],
                ).number_format = "0.00"

    return True


# ============================================================
# MAIN V8.1
# ============================================================

def main_v81():
    t0 = time.time()
    iniciar_log()

    log(
        "=" * 78
    )

    log(
        f"BET BUILDER {VERSION}"
    )

    log(
        "=" * 78
    )

    inicio, fin = ventana_objetivo()

    log(
        f"Ventana: "
        f"{inicio.strftime('%d/%m/%Y')} -> "
        f"{fin.strftime('%d/%m/%Y')}"
    )

    log("")
    log(
        "1) HISTÓRICO"
    )

    hist = descargar_historico_total()

    log(
        f"Histórico: {len(hist)}"
    )

    log("")
    log(
        "2) DATASET V8.1"
    )

    ds, estados = construir_dataset_v81(
        hist
    )

    schedule_index = construir_schedule_index(
        hist
    )

    perfiles_arbitro = construir_perfil_arbitros(
        hist
    )

    log(
        f"Dataset: {len(ds)} | "
        f"árbitros: {len(perfiles_arbitro)}"
    )

    log("")
    log(
        "3) CONTEXTO CÓRNERS 1T"
    )

    context_h1 = descargar_contexto_h1()

    log(
        f"Contexto H1 disponible: "
        f"{0 if context_h1 is None else len(context_h1)}"
    )

    log("")
    log(
        "4) FIXTURES"
    )

    fixtures = descargar_fixtures_objetivo(
        inicio,
        fin,
    )

    log(
        f"Fixtures: {len(fixtures)}"
    )

    log("")
    log(
        "5) MODELO ROBUSTO"
    )

    resultado, variants = pronosticar_v81(
        fixtures,
        ds,
        estados,
        perfiles_arbitro,
        schedule_index,
        context_h1,
    )

    top = crear_top30_v81(
        resultado
    )

    n_ap = (
        int(
            (
                top[
                    "Accion"
                ]
                == "APOSTAR"
            ).sum()
        )
        if not top.empty
        else 0
    )

    n_vi = (
        int(
            (
                top[
                    "Accion"
                ]
                == "VIGILAR"
            ).sum()
        )
        if not top.empty
        else 0
    )

    log(
        f"Selecciones: {len(top)} "
        f"(APOSTAR={n_ap}, VIGILAR={n_vi})"
    )

    if len(
        top
    ) < TOP_N:
        log(
            "No se fuerzan 30: la estabilidad tiene prioridad."
        )

    log("")
    log(
        "6) EXPORT"
    )

    exportar_v81(
        top,
        variants,
        resultado,
        inicio,
        fin,
    )

    log("")
    log(
        "=" * 78
    )
    log(
        "TOP V8.1"
    )
    log(
        "=" * 78
    )

    if top.empty:
        log(
            "No hay apuestas con calidad suficiente."
        )
    else:
        for _, r in top.head(
            15
        ).iterrows():
            log(
                f"#{int(r['Ranking'])} "
                f"{r['Accion']} | "
                f"{r['Competicion']} | "
                f"{r['Local']} vs {r['Visitante']} | "
                f"Pbase={r['P_BASE_Ajustada']*100:.1f}% | "
                f"Reliability={r['ReliabilityScore']*100:.1f}% | "
                f"{r['VarianteMasSegura']}"
            )

    log("")
    log(
        f"Excel: {ARCHIVO_XLSX_V81}"
    )

    log(
        f"Tiempo: {(time.time()-t0)/60:.1f} min"
    )

    log("")
    log(
        "REGLA: APOSTAR solo si la cuota real de la variante "
        "es >= CuotaMinExigida y nunca <4.20."
    )

    try:
        from google.colab import files

        files.download(
            ARCHIVO_XLSX_V81
        )

    except Exception:
        pass


if __name__ == "__main__":
    try:
        main_v81()

    except Exception as e:
        try:
            log("")
            log("ERROR CONTROLADO")
            log(str(e))
            log(
                traceback.format_exc()
            )
        except Exception:
            print(
                traceback.format_exc()
            )

        raise

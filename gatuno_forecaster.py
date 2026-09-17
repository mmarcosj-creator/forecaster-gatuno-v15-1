from __future__ import annotations
"""FORECASTER FUTBOL V10 PRO.

Motor de pronosticos individuales por partido.  No inventa cuotas y no
construye combinadas.  Las probabilidades se calculan con informacion
disponible antes del encuentro y se validan en orden temporal.

Mercados publicados por cada partido:
  * Resultado 1X2 (local / empate / visitante).
  * Goles del primer tiempo, linea 1.5.
  * Corners del primer tiempo, linea 4.5 (solo con cobertura real).
  * Tarjetas amarillas totales, linea 4.5.
  * Gol del local, linea 0.5.
  * Gol del visitante, linea 0.5.

La etiqueta de color describe evidencia estadistica; no es una orden de
apuesta.  Si faltan datos, el partido sigue visible, pero el mercado se marca
como NO MODELABLE / ROJO en vez de fabricar una seleccion.
"""
import os
import pickle

CACHE_FILE = "trained_model_cache.pkl"

def obtener_modelo_optimizado():
    """
    Carga el modelo desde la caché local o ejecuta el entrenamiento si no existe.
    """
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "rb") as f:
                modelo = pickle.load(f)
            return modelo
        except Exception:
            pass

    # Reemplaza 'entrenar_modelo_nuevo()' por la función que ya usa tu archivo para entrenar
    modelo = entrenar_modelo_nuevo() 
    
    try:
        with open(CACHE_FILE, "wb") as f:
            pickle.dump(modelo, f)
    except Exception:
        pass
        
    return modelo


import hashlib
import json
import math
import os
import pickle
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import gatuno_data as base
import gatuno_context as context93

try:
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression, PoissonRegressor
    from sklearn.metrics import accuracy_score, log_loss, mean_absolute_error
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
except Exception as exc:  # pragma: no cover - mensaje explicito en despliegue
    raise RuntimeError(
        "V10 necesita scikit-learn. Instala requirements.txt antes de ejecutar."
    ) from exc


VERSION = "V15.1.1-STABLE-GATUNO"
APP_DATA_DIR = Path("app_data_v15_1")
LINEUP_HISTORY_PATH = APP_DATA_DIR / "lineup_history.json"
MODEL_CACHE_PATH = APP_DATA_DIR / "trained_model_cache.pkl"

# La nacionalidad de un club se aprende de su participacion previa en una
# liga domestica. No se deduce por el nombre y no se rellena a mano: si el
# origen no puede demostrarse con datos anteriores queda como UNKNOWN.
DOMESTIC_ORIGIN = {
    "ARG": "ARG",
    "BRA": "BRA",
    "PER": "PER",
    "ESP": "ESP",
    "ENG": "ENG",
    "POR": "POR",
    "TUR": "TUR",
}
SOUTH_AMERICAN_INTERNATIONAL = {"LIB", "SUD"}
MIN_ARG_CROSS_SAMPLE = 30

MIN_HISTORY_FOR_TRAIN = 4
MAX_TEAM_RECORDS = 36
TEAM_DECAY_DAYS = 210.0
ELO_HOME_ADVANTAGE = 55.0
ELO_K = 22.0

RESULT_LABELS = {0: "LOCAL", 1: "EMPATE", 2: "VISITANTE"}

MARKET_CODES = {
    "Resultado 1X2": "RESULT_1X2",
    "Goles 1.er tiempo": "H1_GOALS_OU15",
    "Corners 1.er tiempo": "H1_CORNERS_OU45",
    "Tarjetas amarillas totales": "YELLOW_CARDS_OU45",
}

NUMERIC_FEATURES = [
    "HomeElo",
    "AwayElo",
    "EloDiff",
    "HomePPG",
    "AwayPPG",
    "PPGDiff",
    "HomeGF",
    "HomeGA",
    "AwayGF",
    "AwayGA",
    "HomeVenueGF",
    "HomeVenueGA",
    "AwayVenueGF",
    "AwayVenueGA",
    "HomeH1GF",
    "HomeH1GA",
    "AwayH1GF",
    "AwayH1GA",
    "HomeCornersFor",
    "HomeCornersAgainst",
    "AwayCornersFor",
    "AwayCornersAgainst",
    "HomeCardsFor",
    "HomeCardsAgainst",
    "AwayCardsFor",
    "AwayCardsAgainst",
    "HomeScoreRate",
    "HomeConcedeRate",
    "AwayScoreRate",
    "AwayConcedeRate",
    "HomeDaysRest",
    "AwayDaysRest",
    "HomeMatches14",
    "AwayMatches14",
    "HomeSeasonPPG",
    "AwaySeasonPPG",
    "SeasonPPGDiff",
    "HomePositionPct",
    "AwayPositionPct",
    "PositionDiff",
    "HomeSeasonGames",
    "AwaySeasonGames",
    "CompHomeGoals",
    "CompAwayGoals",
    "CompH1Goals",
    "CompCards",
    "CompHomeWinRate",
    "CompDrawRate",
    "IsCup",
    "HomeIsArgentine",
    "AwayIsArgentine",
    "ArgentineCrossLeague",
    "ArgentineCrossHome",
    "ArgentineCrossAway",
    "InternationalCrossCountry",
    "OriginKnown",
]

CATEGORICAL_FEATURES = ["CompKey", "Grupo"]
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


@dataclass
class SequentialState:
    histories: dict[str, list[dict[str, Any]]] = field(
        default_factory=lambda: defaultdict(list)
    )
    elos: dict[str, float] = field(default_factory=lambda: defaultdict(lambda: 1500.0))
    standings: dict[str, dict[str, dict[str, float]]] = field(
        default_factory=lambda: defaultdict(dict)
    )
    competition_history: dict[str, list[dict[str, Any]]] = field(
        default_factory=lambda: defaultdict(list)
    )
    origin_counts: dict[str, dict[str, int]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(int))
    )


@dataclass
class ProbabilityModel:
    name: str
    target: str
    model: Any
    kind: str
    calibrator: Any
    metrics: dict[str, Any]


@dataclass
class CountModel:
    name: str
    target: str
    model: Any
    metrics: dict[str, Any]


@dataclass
class ModelBundle:
    probability_models: dict[str, ProbabilityModel]
    count_models: dict[str, CountModel]
    metrics: pd.DataFrame
    trained_through: pd.Timestamp


def _num(value: Any) -> float:
    return base.numero(value)


def _clip_probability(p: float) -> float:
    return float(np.clip(float(p), 0.005, 0.995))


def _season_id(comp_key: str, date: Any) -> str:
    start = base.season_start_for(comp_key, pd.Timestamp(date))
    return f"{comp_key}|{pd.Timestamp(start).date()}"


def _weighted_average(records: list[dict[str, Any]], key: str, now: Any) -> float:
    if not records:
        return np.nan
    target = pd.Timestamp(now)
    values: list[float] = []
    weights: list[float] = []
    recent = records[-MAX_TEAM_RECORDS:]
    for index, row in enumerate(recent):
        value = _num(row.get(key, np.nan))
        if pd.isna(value):
            continue
        age = max(0.0, float((target - pd.Timestamp(row["date"])).days))
        order_age = len(recent) - 1 - index
        weight = math.exp(-age / TEAM_DECAY_DAYS) * math.exp(-order_age / 18.0)
        values.append(float(value))
        weights.append(float(weight))
    if not weights or sum(weights) <= 0:
        return np.nan
    return float(np.average(values, weights=weights))


def _profile(records: list[dict[str, Any]], now: Any, venue: str | None = None) -> dict[str, float]:
    all_records = list(records[-MAX_TEAM_RECORDS:])
    selected = [r for r in all_records if venue is None or r.get("venue") == venue]
    # El split local/visita se encoge hacia la forma total para no exagerar
    # muestras de dos o tres partidos.
    overall = {
        key: _weighted_average(all_records, key, now)
        for key in (
            "gf", "ga", "ppg", "h1gf", "h1ga", "corners_for",
            "corners_against", "cards_for", "cards_against", "scored",
            "conceded",
        )
    }
    if venue is None:
        return {**overall, "n": float(len(all_records))}
    venue_values = {
        key: _weighted_average(selected, key, now)
        for key in overall
    }
    n = len(selected)
    shrink = n / (n + 5.0)
    result: dict[str, float] = {}
    for key in overall:
        ov = overall[key]
        vv = venue_values[key]
        if pd.isna(vv):
            result[key] = ov
        elif pd.isna(ov):
            result[key] = vv
        else:
            result[key] = float(shrink * vv + (1.0 - shrink) * ov)
    result["n"] = float(n)
    return result


def _competition_profile(records: list[dict[str, Any]], now: Any) -> dict[str, float]:
    recent = records[-500:]
    defaults = {
        "home_goals": 1.42,
        "away_goals": 1.16,
        "h1_goals": 1.12,
        "cards": 4.6,
        "home_win": 0.44,
        "draw": 0.27,
    }
    out = {}
    for key, default in defaults.items():
        value = _weighted_average(recent, key, now)
        out[key] = default if pd.isna(value) else float(value)
    out["n"] = float(len(recent))
    return out


def _standing_snapshot(table: dict[str, dict[str, float]], team: str) -> dict[str, float]:
    entry = table.get(team, {"games": 0.0, "points": 0.0, "gd": 0.0, "gf": 0.0})
    games = float(entry.get("games", 0.0))
    ppg = float(entry.get("points", 0.0)) / games if games > 0 else np.nan
    teams = list(table)
    if team not in teams or len(teams) < 2:
        position_pct = 0.5
    else:
        ordered = sorted(
            teams,
            key=lambda t: (
                -float(table[t].get("points", 0.0)),
                -float(table[t].get("gd", 0.0)),
                -float(table[t].get("gf", 0.0)),
                t,
            ),
        )
        rank = ordered.index(team) + 1
        position_pct = (rank - 1.0) / max(1.0, len(ordered) - 1.0)
    return {"games": games, "ppg": ppg, "position_pct": float(position_pct)}


def _rest_features(records: list[dict[str, Any]], now: Any) -> tuple[float, int]:
    if not records:
        return np.nan, 0
    target = pd.Timestamp(now)
    dates = [pd.Timestamp(r["date"]) for r in records if pd.Timestamp(r["date"]) < target]
    if not dates:
        return np.nan, 0
    rest = float((target - max(dates)).days)
    matches14 = sum(1 for d in dates if d >= target - pd.Timedelta(days=14))
    return rest, int(matches14)


def _team_origin(state: SequentialState, team: str) -> str:
    counts = state.origin_counts.get(str(team), {})
    if not counts:
        return "UNKNOWN"
    return str(max(counts, key=lambda key: (counts[key], key)))


def _origin_flags(
    state: SequentialState,
    comp_key: str,
    home: str,
    away: str,
) -> dict[str, Any]:
    home_origin = _team_origin(state, home)
    away_origin = _team_origin(state, away)
    is_international = str(comp_key) in SOUTH_AMERICAN_INTERNATIONAL
    known = home_origin != "UNKNOWN" and away_origin != "UNKNOWN"
    arg_cross = bool(
        is_international
        and known
        and ((home_origin == "ARG") ^ (away_origin == "ARG"))
    )
    return {
        "HomeOrigin": home_origin,
        "AwayOrigin": away_origin,
        "HomeIsArgentine": float(home_origin == "ARG"),
        "AwayIsArgentine": float(away_origin == "ARG"),
        "ArgentineCrossLeague": float(arg_cross),
        "ArgentineCrossHome": float(arg_cross and home_origin == "ARG"),
        "ArgentineCrossAway": float(arg_cross and away_origin == "ARG"),
        "InternationalCrossCountry": float(
            is_international and known and home_origin != away_origin
        ),
        "OriginKnown": float(known),
    }


def _feature_row(
    state: SequentialState,
    date: Any,
    comp_key: str,
    grupo: str,
    home: str,
    away: str,
) -> dict[str, Any]:
    date = pd.Timestamp(date)
    hp = _profile(state.histories.get(home, []), date)
    ap = _profile(state.histories.get(away, []), date)
    hv = _profile(state.histories.get(home, []), date, "H")
    av = _profile(state.histories.get(away, []), date, "A")
    cp = _competition_profile(state.competition_history.get(comp_key, []), date)
    hrest, hm14 = _rest_features(state.histories.get(home, []), date)
    arest, am14 = _rest_features(state.histories.get(away, []), date)
    table = state.standings.get(_season_id(comp_key, date), {})
    hs = _standing_snapshot(table, home)
    aws = _standing_snapshot(table, away)
    helo = float(state.elos.get(home, 1500.0))
    aelo = float(state.elos.get(away, 1500.0))
    comp_info = base.COMPETICIONES.get(comp_key, {})
    is_cup = float(str(comp_info.get("tipo", "")).upper() != "LIGA")
    origin = _origin_flags(state, comp_key, home, away)

    return {
        "Date": date,
        "CompKey": str(comp_key),
        "Grupo": str(grupo),
        "HomeTeam": str(home),
        "AwayTeam": str(away),
        "HomeElo": helo,
        "AwayElo": aelo,
        "EloDiff": helo - aelo,
        "HomePPG": hp["ppg"],
        "AwayPPG": ap["ppg"],
        "PPGDiff": hp["ppg"] - ap["ppg"] if pd.notna(hp["ppg"]) and pd.notna(ap["ppg"]) else np.nan,
        "HomeGF": hp["gf"],
        "HomeGA": hp["ga"],
        "AwayGF": ap["gf"],
        "AwayGA": ap["ga"],
        "HomeVenueGF": hv["gf"],
        "HomeVenueGA": hv["ga"],
        "AwayVenueGF": av["gf"],
        "AwayVenueGA": av["ga"],
        "HomeH1GF": hp["h1gf"],
        "HomeH1GA": hp["h1ga"],
        "AwayH1GF": ap["h1gf"],
        "AwayH1GA": ap["h1ga"],
        "HomeCornersFor": hp["corners_for"],
        "HomeCornersAgainst": hp["corners_against"],
        "AwayCornersFor": ap["corners_for"],
        "AwayCornersAgainst": ap["corners_against"],
        "HomeCardsFor": hp["cards_for"],
        "HomeCardsAgainst": hp["cards_against"],
        "AwayCardsFor": ap["cards_for"],
        "AwayCardsAgainst": ap["cards_against"],
        "HomeScoreRate": hp["scored"],
        "HomeConcedeRate": hp["conceded"],
        "AwayScoreRate": ap["scored"],
        "AwayConcedeRate": ap["conceded"],
        "HomeDaysRest": hrest,
        "AwayDaysRest": arest,
        "HomeMatches14": hm14,
        "AwayMatches14": am14,
        "HomeSeasonPPG": hs["ppg"],
        "AwaySeasonPPG": aws["ppg"],
        "SeasonPPGDiff": hs["ppg"] - aws["ppg"] if pd.notna(hs["ppg"]) and pd.notna(aws["ppg"]) else np.nan,
        "HomePositionPct": hs["position_pct"],
        "AwayPositionPct": aws["position_pct"],
        "PositionDiff": aws["position_pct"] - hs["position_pct"],
        "HomeSeasonGames": hs["games"],
        "AwaySeasonGames": aws["games"],
        "CompHomeGoals": cp["home_goals"],
        "CompAwayGoals": cp["away_goals"],
        "CompH1Goals": cp["h1_goals"],
        "CompCards": cp["cards"],
        "CompHomeWinRate": cp["home_win"],
        "CompDrawRate": cp["draw"],
        "IsCup": is_cup,
        **origin,
        "HomeHistoryN": len(state.histories.get(home, [])),
        "AwayHistoryN": len(state.histories.get(away, [])),
        "CompetitionHistoryN": int(cp["n"]),
    }


def _match_record(row: pd.Series, home: bool) -> dict[str, Any]:
    hg, ag = _num(row.get("FTHG")), _num(row.get("FTAG"))
    hthg, htag = _num(row.get("HTHG")), _num(row.get("HTAG"))
    hc, ac = _num(row.get("HC")), _num(row.get("AC"))
    hy, ay = _num(row.get("HY")), _num(row.get("AY"))
    if home:
        gf, ga, h1gf, h1ga = hg, ag, hthg, htag
        cf, ca, cardsf, cardsa, venue = hc, ac, hy, ay, "H"
    else:
        gf, ga, h1gf, h1ga = ag, hg, htag, hthg
        cf, ca, cardsf, cardsa, venue = ac, hc, ay, hy, "A"
    if gf > ga:
        ppg = 3.0
    elif gf == ga:
        ppg = 1.0
    else:
        ppg = 0.0
    return {
        "date": pd.Timestamp(row["Date"]),
        "venue": venue,
        "gf": gf,
        "ga": ga,
        "ppg": ppg,
        "h1gf": h1gf,
        "h1ga": h1ga,
        "corners_for": cf,
        "corners_against": ca,
        "cards_for": cardsf,
        "cards_against": cardsa,
        "scored": float(gf >= 1),
        "conceded": float(ga >= 1),
    }


def _update_standing(table: dict[str, dict[str, float]], team: str, gf: float, ga: float) -> None:
    entry = table.setdefault(team, {"games": 0.0, "points": 0.0, "gd": 0.0, "gf": 0.0})
    entry["games"] += 1.0
    entry["gf"] += gf
    entry["gd"] += gf - ga
    entry["points"] += 3.0 if gf > ga else (1.0 if gf == ga else 0.0)


def _elo_delta(home_elo: float, away_elo: float, hg: float, ag: float) -> float:
    expected = 1.0 / (1.0 + 10.0 ** (-(home_elo + ELO_HOME_ADVANTAGE - away_elo) / 400.0))
    actual = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)
    multiplier = 1.0 + 0.35 * math.log1p(abs(hg - ag))
    return float(ELO_K * multiplier * (actual - expected))


def build_feature_dataset(hist: pd.DataFrame) -> tuple[pd.DataFrame, SequentialState]:
    """Construye variables estrictamente prepartido y conserva el estado final."""
    required = {"Date", "CompKey", "Grupo", "HomeTeam", "AwayTeam", "FTHG", "FTAG"}
    missing = required - set(hist.columns)
    if missing:
        raise ValueError(f"Historico incompleto; faltan columnas: {sorted(missing)}")

    data = hist.copy()
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    data = data.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"])
    data = data.sort_values(["Date", "CompKey", "HomeTeam"]).reset_index(drop=True)
    state = SequentialState()
    rows: list[dict[str, Any]] = []

    for date, block in data.groupby("Date", sort=True):
        pending: list[tuple[pd.Series, float]] = []
        for _, match in block.iterrows():
            home, away = str(match["HomeTeam"]), str(match["AwayTeam"])
            if min(len(state.histories[home]), len(state.histories[away])) >= MIN_HISTORY_FOR_TRAIN:
                features = _feature_row(
                    state,
                    date,
                    str(match["CompKey"]),
                    str(match["Grupo"]),
                    home,
                    away,
                )
                hg, ag = _num(match.get("FTHG")), _num(match.get("FTAG"))
                hthg, htag = _num(match.get("HTHG")), _num(match.get("HTAG"))
                hy, ay = _num(match.get("HY")), _num(match.get("AY"))
                if hg > ag:
                    result_class = 0
                elif hg == ag:
                    result_class = 1
                else:
                    result_class = 2
                features.update({
                    "Competicion": match.get("Competicion", match.get("CompKey", "")),
                    "ResultClass": result_class,
                    "Y_H1_UNDER15": float(hthg + htag <= 1) if pd.notna(hthg) and pd.notna(htag) else np.nan,
                    "Y_CARDS_OVER45": float(hy + ay >= 5) if pd.notna(hy) and pd.notna(ay) else np.nan,
                    "Y_HOME_SCORE": float(hg >= 1),
                    "Y_AWAY_SCORE": float(ag >= 1),
                    "HomeGoals": hg,
                    "AwayGoals": ag,
                    "H1Goals": hthg + htag if pd.notna(hthg) and pd.notna(htag) else np.nan,
                    "CardsTotal": hy + ay if pd.notna(hy) and pd.notna(ay) else np.nan,
                    "OddsH": _num(match.get("AvgH", match.get("B365H", np.nan))),
                    "OddsD": _num(match.get("AvgD", match.get("B365D", np.nan))),
                    "OddsA": _num(match.get("AvgA", match.get("B365A", np.nan))),
                })
                rows.append(features)
            delta = _elo_delta(float(state.elos[home]), float(state.elos[away]), _num(match["FTHG"]), _num(match["FTAG"]))
            pending.append((match, delta))

        # Todo el bloque de la misma fecha se actualiza despues de producir
        # sus variables; asi no se filtra informacion de otro juego del dia.
        for match, delta in pending:
            home, away = str(match["HomeTeam"]), str(match["AwayTeam"])
            hg, ag = _num(match["FTHG"]), _num(match["FTAG"])
            state.histories[home].append(_match_record(match, True))
            state.histories[away].append(_match_record(match, False))
            state.elos[home] += delta
            state.elos[away] -= delta
            domestic_origin = DOMESTIC_ORIGIN.get(str(match["CompKey"]))
            if domestic_origin:
                state.origin_counts[home][domestic_origin] += 1
                state.origin_counts[away][domestic_origin] += 1
            key = _season_id(str(match["CompKey"]), match["Date"])
            table = state.standings[key]
            _update_standing(table, home, hg, ag)
            _update_standing(table, away, ag, hg)
            hthg, htag = _num(match.get("HTHG")), _num(match.get("HTAG"))
            hy, ay = _num(match.get("HY")), _num(match.get("AY"))
            state.competition_history[str(match["CompKey"])].append({
                "date": pd.Timestamp(match["Date"]),
                "home_goals": hg,
                "away_goals": ag,
                "h1_goals": hthg + htag if pd.notna(hthg) and pd.notna(htag) else np.nan,
                "cards": hy + ay if pd.notna(hy) and pd.notna(ay) else np.nan,
                "home_win": float(hg > ag),
                "draw": float(hg == ag),
            })

    dataset = pd.DataFrame(rows).sort_values("Date").reset_index(drop=True)
    if dataset.empty:
        raise RuntimeError("No hay suficientes partidos para entrenar V10.")
    return dataset, state


def _preprocessor() -> ColumnTransformer:
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=8, sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric, NUMERIC_FEATURES),
        ("cat", categorical, CATEGORICAL_FEATURES),
    ], remainder="drop")


def _new_classifier() -> Pipeline:
    return Pipeline([
        ("prep", _preprocessor()),
        ("model", LogisticRegression(C=0.32, max_iter=1500, solver="lbfgs")),
    ])


def _new_count_model() -> Pipeline:
    return Pipeline([
        ("prep", _preprocessor()),
        ("model", PoissonRegressor(alpha=1.15, max_iter=1000)),
    ])


def _time_weight(dates: pd.Series) -> np.ndarray:
    end = pd.Timestamp(pd.to_datetime(dates).max())
    age = (end - pd.to_datetime(dates)).dt.days.to_numpy(dtype=float)
    return np.clip(np.exp(-np.maximum(age, 0.0) / 720.0), 0.18, 1.0)


def _fit_pipeline(model: Pipeline, frame: pd.DataFrame, target: str) -> Pipeline:
    sample_weight = _time_weight(frame["Date"])
    model.fit(frame[MODEL_FEATURES], frame[target], model__sample_weight=sample_weight)
    return model


def _date_cutoffs(frame: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    dates = np.array(sorted(pd.to_datetime(frame["Date"]).dropna().unique()))
    if len(dates) < 20:
        raise RuntimeError("Se requieren al menos 20 fechas historicas distintas.")
    def at(frac: float) -> pd.Timestamp:
        return pd.Timestamp(dates[min(len(dates) - 1, max(1, int(len(dates) * frac)))])
    return at(0.58), at(0.73), at(0.88)


class _IdentityCalibrator:
    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        return np.asarray(probabilities, dtype=float)


class _SigmoidCalibrator:
    def __init__(self) -> None:
        self.model = LogisticRegression(C=10.0, max_iter=1000, solver="lbfgs")

    def fit(self, probabilities: np.ndarray, y: np.ndarray) -> "_SigmoidCalibrator":
        p = np.clip(np.asarray(probabilities, dtype=float), 1e-5, 1 - 1e-5)
        logits = np.log(p / (1.0 - p)).reshape(-1, 1)
        self.model.fit(logits, np.asarray(y, dtype=int))
        return self

    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        p = np.clip(np.asarray(probabilities, dtype=float), 1e-5, 1 - 1e-5)
        logits = np.log(p / (1.0 - p)).reshape(-1, 1)
        return self.model.predict_proba(logits)[:, 1]


class _TemperatureCalibrator:
    def __init__(self, temperature: float = 1.0) -> None:
        self.temperature = float(temperature)

    @staticmethod
    def _apply(probabilities: np.ndarray, temperature: float) -> np.ndarray:
        logits = np.log(np.clip(probabilities, 1e-9, 1.0)) / temperature
        logits -= logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        return exp / exp.sum(axis=1, keepdims=True)

    def fit(self, probabilities: np.ndarray, y: np.ndarray) -> "_TemperatureCalibrator":
        best_t, best_loss = 1.0, float("inf")
        labels = np.asarray(y, dtype=int)
        for temperature in np.linspace(0.60, 2.40, 73):
            calibrated = self._apply(probabilities, float(temperature))
            loss = log_loss(labels, calibrated, labels=[0, 1, 2])
            if loss < best_loss:
                best_t, best_loss = float(temperature), float(loss)
        self.temperature = best_t
        return self

    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        return self._apply(np.asarray(probabilities, dtype=float), self.temperature)


def _ece_binary(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    total = len(y)
    if total == 0:
        return np.nan
    error = 0.0
    for low, high in zip(np.linspace(0, 1, bins + 1)[:-1], np.linspace(0, 1, bins + 1)[1:]):
        mask = (p >= low) & (p < high if high < 1 else p <= high)
        if mask.any():
            error += mask.mean() * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return float(error)


def _ece_multiclass(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    confidence = p.max(axis=1)
    correct = (p.argmax(axis=1) == np.asarray(y, dtype=int)).astype(float)
    return _ece_binary(correct, confidence, bins=bins)


def _binary_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    prevalence = float(y.mean())
    brier = float(np.mean((p - y) ** 2))
    baseline_brier = float(np.mean((prevalence - y) ** 2))
    return {
        "N_Validacion": int(len(y)),
        "Brier": brier,
        "BrierBase": baseline_brier,
        "MejoraBrier": float((baseline_brier - brier) / max(baseline_brier, 1e-9)),
        "LogLoss": float(log_loss(y, p, labels=[0, 1])),
        "Exactitud": float(accuracy_score(y, p >= 0.5)),
        "ExactitudBase": max(prevalence, 1.0 - prevalence),
        "ECE": _ece_binary(y, p),
    }


def _multiclass_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    onehot = np.eye(3)[y]
    priors = onehot.mean(axis=0)
    brier = float(np.mean(np.sum((p - onehot) ** 2, axis=1)))
    baseline_brier = float(np.mean(np.sum((priors - onehot) ** 2, axis=1)))
    return {
        "N_Validacion": int(len(y)),
        "Brier": brier,
        "BrierBase": baseline_brier,
        "MejoraBrier": float((baseline_brier - brier) / max(baseline_brier, 1e-9)),
        "LogLoss": float(log_loss(y, p, labels=[0, 1, 2])),
        "Exactitud": float(accuracy_score(y, p.argmax(axis=1))),
        "ExactitudBase": float(priors.max()),
        "ECE": _ece_multiclass(y, p),
    }


def _quality_score(metrics: dict[str, Any]) -> float:
    improvement = float(metrics.get("MejoraBrier", 0.0) or 0.0)
    ece = float(metrics.get("ECE", 0.20) or 0.20)
    n = float(metrics.get("N_Validacion", 0) or 0)
    sample = min(1.0, n / 350.0)
    raw = float(np.clip(0.46 + 0.85 * improvement - 1.20 * ece + 0.16 * sample, 0.18, 0.94))
    # Un modelo que no supera la prediccion base en datos posteriores no puede
    # originar un verde ni un amarillo, aunque entregue una P extrema.
    if improvement <= 0.0:
        return min(raw, 0.42)
    if improvement < 0.01:
        return min(raw, 0.55)
    return raw


def _fit_probability_model(frame: pd.DataFrame, name: str, target: str, kind: str) -> ProbabilityModel:
    usable = frame.dropna(subset=[target]).copy()
    usable = usable.sort_values("Date").reset_index(drop=True)
    if len(usable) < 240:
        raise RuntimeError(f"Muestra insuficiente para {name}: {len(usable)}")
    c1, c2, c3 = _date_cutoffs(usable)

    oos_probabilities: list[np.ndarray] = []
    oos_targets: list[np.ndarray] = []
    for train_end, valid_end in ((c1, c2), (c2, c3)):
        train = usable[usable["Date"] < train_end]
        valid = usable[(usable["Date"] >= train_end) & (usable["Date"] < valid_end)]
        if len(train) < 180 or len(valid) < 40:
            continue
        model = _fit_pipeline(_new_classifier(), train, target)
        raw = model.predict_proba(valid[MODEL_FEATURES])
        if kind == "binary":
            raw = raw[:, list(model.named_steps["model"].classes_).index(1)]
        oos_probabilities.append(raw)
        oos_targets.append(valid[target].astype(int).to_numpy())

    if not oos_probabilities:
        calibrator: Any = _IdentityCalibrator()
    else:
        raw_oos = np.concatenate(oos_probabilities, axis=0)
        y_oos = np.concatenate(oos_targets, axis=0)
        if kind == "binary" and len(np.unique(y_oos)) == 2:
            calibrator = _SigmoidCalibrator().fit(raw_oos, y_oos)
        elif kind == "multiclass" and len(np.unique(y_oos)) == 3:
            calibrator = _TemperatureCalibrator().fit(raw_oos, y_oos)
        else:
            calibrator = _IdentityCalibrator()

    train_eval = usable[usable["Date"] < c3]
    test = usable[usable["Date"] >= c3]
    eval_model = _fit_pipeline(_new_classifier(), train_eval, target)
    raw_test = eval_model.predict_proba(test[MODEL_FEATURES])
    if kind == "binary":
        raw_test = raw_test[:, list(eval_model.named_steps["model"].classes_).index(1)]
        calibrated_test = calibrator.transform(raw_test)
        metrics = _binary_metrics(test[target].astype(int).to_numpy(), calibrated_test)
    else:
        calibrated_test = calibrator.transform(raw_test)
        metrics = _multiclass_metrics(test[target].astype(int).to_numpy(), calibrated_test)
    metrics.update({"Modelo": name, "Target": target, "Tipo": kind})
    metrics["SuperaBase"] = bool(
        metrics.get("MejoraBrier", 0.0) > 0.0
        and metrics.get("ECE", 1.0) <= 0.12
    )
    metrics["EstadoValidacion"] = (
        "SUPERA BASE OOS" if metrics["SuperaBase"] else "NO SUPERA BASE OOS"
    )

    # Auditoría específica de cruces argentinos interliga. El indicador está
    # disponible para todos los modelos, pero se usa como cortacircuito en los
    # mercados donde la hipótesis táctica es directa (tarjetas y córners). No
    # se aplica una bonificación o penalización fija por nacionalidad.
    train_arg_flag = (
        pd.to_numeric(train_eval["ArgentineCrossLeague"], errors="coerce").fillna(0)
        if "ArgentineCrossLeague" in train_eval
        else pd.Series(0.0, index=train_eval.index)
    )
    test_arg_flag = (
        pd.to_numeric(test["ArgentineCrossLeague"], errors="coerce").fillna(0)
        if "ArgentineCrossLeague" in test
        else pd.Series(0.0, index=test.index)
    )
    train_arg = train_eval[train_arg_flag >= 0.5]
    test_arg_mask = test_arg_flag >= 0.5
    test_arg = test.loc[test_arg_mask]
    arg_n = int(len(test_arg))
    metrics.update({
        "ArgCrossNEntrenamiento": int(len(train_arg)),
        "ArgCrossNValidacion": arg_n,
        "ArgCrossBrier": np.nan,
        "ArgCrossBrierBase": np.nan,
        "ArgCrossMejoraBrier": np.nan,
        "ArgCrossECE": np.nan,
        "ArgCrossSuperaBase": False,
    })
    if arg_n:
        if kind == "binary":
            arg_positions = np.flatnonzero(test_arg_mask.to_numpy())
            arg_probabilities = np.asarray(calibrated_test)[arg_positions]
            arg_targets = test_arg[target].astype(int).to_numpy()
            reference_rows = train_arg if len(train_arg) >= 20 else train_eval
            reference = float(reference_rows[target].mean())
            arg_brier = float(np.mean((arg_probabilities - arg_targets) ** 2))
            arg_base = float(np.mean((reference - arg_targets) ** 2))
            arg_ece = _ece_binary(arg_targets, arg_probabilities, bins=5)
        else:
            arg_positions = np.flatnonzero(test_arg_mask.to_numpy())
            arg_probabilities = np.asarray(calibrated_test)[arg_positions]
            arg_targets = test_arg[target].astype(int).to_numpy()
            reference_rows = train_arg if len(train_arg) >= 20 else train_eval
            priors = (
                reference_rows[target].value_counts(normalize=True)
                .reindex([0, 1, 2], fill_value=0.0)
                .to_numpy(dtype=float)
            )
            onehot = np.eye(3)[arg_targets]
            arg_brier = float(np.mean(np.sum((arg_probabilities - onehot) ** 2, axis=1)))
            arg_base = float(np.mean(np.sum((priors - onehot) ** 2, axis=1)))
            arg_ece = _ece_multiclass(arg_targets, arg_probabilities, bins=5)
        arg_improvement = float((arg_base - arg_brier) / max(arg_base, 1e-9))
        metrics.update({
            "ArgCrossBrier": arg_brier,
            "ArgCrossBrierBase": arg_base,
            "ArgCrossMejoraBrier": arg_improvement,
            "ArgCrossECE": arg_ece,
            "ArgCrossSuperaBase": bool(
                arg_n >= MIN_ARG_CROSS_SAMPLE
                and arg_improvement > 0.0
                and arg_ece <= 0.16
            ),
        })
    metrics["CalidadModelo"] = _quality_score(metrics)

    final_model = _fit_pipeline(_new_classifier(), usable, target)
    return ProbabilityModel(name, target, final_model, kind, calibrator, metrics)


def _fit_count_model(frame: pd.DataFrame, name: str, target: str) -> CountModel:
    usable = frame.dropna(subset=[target]).sort_values("Date").reset_index(drop=True)
    if len(usable) < 240:
        raise RuntimeError(f"Muestra insuficiente para {name}: {len(usable)}")
    _, _, cutoff = _date_cutoffs(usable)
    train, test = usable[usable["Date"] < cutoff], usable[usable["Date"] >= cutoff]
    eval_model = _fit_pipeline(_new_count_model(), train, target)
    pred = np.clip(eval_model.predict(test[MODEL_FEATURES]), 0.02, 8.0)
    baseline_prediction = float(train[target].mean())
    baseline_mae = float(
        mean_absolute_error(test[target], np.full(len(test), baseline_prediction))
    )
    model_mae = float(mean_absolute_error(test[target], pred))
    metrics = {
        "Modelo": name,
        "Target": target,
        "Tipo": "count",
        "N_Validacion": int(len(test)),
        "MAE": model_mae,
        "MAEBase": baseline_mae,
        "MejoraMAE": float((baseline_mae - model_mae) / max(baseline_mae, 1e-9)),
        "MediaReal": float(test[target].mean()),
        "MediaPredicha": float(np.mean(pred)),
    }
    metrics["SuperaBase"] = bool(metrics["MejoraMAE"] > 0.0)
    metrics["EstadoValidacion"] = (
        "SUPERA BASE OOS" if metrics["SuperaBase"] else "NO SUPERA BASE OOS"
    )
    metrics["CalidadModelo"] = float(np.clip(1.0 - metrics["MAE"] / max(1.0, metrics["MediaReal"] + 0.5), 0.18, 0.90))
    final_model = _fit_pipeline(_new_count_model(), usable, target)
    return CountModel(name, target, final_model, metrics)


def _unavailable_probability_model(
    name: str, target: str, kind: str, error: Exception
) -> ProbabilityModel:
    """Representa un mercado sin muestra sin derribar los otros mercados."""
    metrics = {
        "Modelo": name,
        "Target": target,
        "Tipo": kind,
        "Disponible": False,
        "N_Validacion": 0,
        "Brier": np.nan,
        "BrierBase": np.nan,
        "MejoraBrier": np.nan,
        "LogLoss": np.nan,
        "Exactitud": np.nan,
        "ExactitudBase": np.nan,
        "ECE": np.nan,
        "SuperaBase": False,
        "EstadoValidacion": "SIN COBERTURA",
        "CalidadModelo": 0.0,
        "ArgCrossNEntrenamiento": 0,
        "ArgCrossNValidacion": 0,
        "ArgCrossSuperaBase": False,
        "ErrorModelo": str(error),
    }
    return ProbabilityModel(name, target, None, kind, _IdentityCalibrator(), metrics)


def _unavailable_count_model(name: str, target: str, error: Exception) -> CountModel:
    metrics = {
        "Modelo": name,
        "Target": target,
        "Tipo": "count",
        "Disponible": False,
        "N_Validacion": 0,
        "MAE": np.nan,
        "MAEBase": np.nan,
        "MejoraMAE": np.nan,
        "SuperaBase": False,
        "EstadoValidacion": "SIN COBERTURA",
        "CalidadModelo": 0.0,
        "ErrorModelo": str(error),
    }
    return CountModel(name, target, None, metrics)


def train_models(dataset: pd.DataFrame) -> ModelBundle:
    specifications = [
        ("RESULTADO_1X2", "ResultClass", "multiclass"),
        ("GOLES_1T_U15", "Y_H1_UNDER15", "binary"),
        ("TARJETAS_O45", "Y_CARDS_OVER45", "binary"),
        ("LOCAL_MARCA", "Y_HOME_SCORE", "binary"),
        ("VISITANTE_MARCA", "Y_AWAY_SCORE", "binary"),
    ]
    probability_models: dict[str, ProbabilityModel] = {}
    metric_rows: list[dict[str, Any]] = []
    for name, target, kind in specifications:
        try:
            fitted = _fit_probability_model(dataset, name, target, kind)
            fitted.metrics["Disponible"] = True
        except Exception as exc:
            fitted = _unavailable_probability_model(name, target, kind, exc)
        probability_models[name] = fitted
        metric_rows.append(fitted.metrics)

    count_models: dict[str, CountModel] = {}
    for name, target in (
        ("GOLES_ESPERADOS_LOCAL", "HomeGoals"),
        ("GOLES_ESPERADOS_VISITANTE", "AwayGoals"),
        ("GOLES_ESPERADOS_1T", "H1Goals"),
        ("TARJETAS_ESPERADAS", "CardsTotal"),
    ):
        try:
            fitted_count = _fit_count_model(dataset, name, target)
            fitted_count.metrics["Disponible"] = True
        except Exception as exc:
            fitted_count = _unavailable_count_model(name, target, exc)
        count_models[name] = fitted_count
        metric_rows.append(fitted_count.metrics)

    return ModelBundle(
        probability_models=probability_models,
        count_models=count_models,
        metrics=pd.DataFrame(metric_rows),
        trained_through=pd.Timestamp(dataset["Date"].max()),
    )


def temporal_backtest_sample(
    dataset: pd.DataFrame,
    n_matches: int = 100,
    random_seed: int = 20260916,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Muestra reproducible de partidos posteriores al entrenamiento.

    La fecha de corte se fija antes de sortear los encuentros. Cada modelo se
    entrena y calibra solo con filas anteriores. No calcula beneficio porque
    no existen cuotas reales historicas completas para estos mercados.
    """
    ordered = dataset.sort_values("Date").reset_index(drop=True)
    _, _, cutoff = _date_cutoffs(ordered)
    train = ordered[ordered["Date"] < cutoff].copy()
    test = ordered[ordered["Date"] >= cutoff].copy()
    if test.empty:
        raise RuntimeError("No existe tramo temporal reservado para backtest.")
    sample_n = min(int(n_matches), len(test))
    sample = test.sample(n=sample_n, random_state=random_seed).sort_values("Date")

    specifications = [
        ("RESULTADO_1X2", "ResultClass", "multiclass", "Resultado 1X2"),
        ("GOLES_1T_U15", "Y_H1_UNDER15", "binary", "Goles 1T U/O 1.5"),
        ("TARJETAS_O45", "Y_CARDS_OVER45", "binary", "Tarjetas U/O 4.5"),
        ("LOCAL_MARCA", "Y_HOME_SCORE", "binary", "Gol local 0.5"),
        ("VISITANTE_MARCA", "Y_AWAY_SCORE", "binary", "Gol visitante 0.5"),
    ]
    rows: list[dict[str, Any]] = []
    for model_name, target, kind, market_label in specifications:
        fitted = _fit_probability_model(train, model_name, target, kind)
        evaluable = sample.dropna(subset=[target]).copy()
        if evaluable.empty:
            continue
        probabilities = _predict_probability(fitted, evaluable)
        for position, (_, observed) in enumerate(evaluable.iterrows()):
            if kind == "multiclass":
                vector = probabilities[position]
                predicted = int(np.argmax(vector))
                actual = int(observed[target])
                probability = float(vector[predicted])
                prediction_label = RESULT_LABELS[predicted]
                actual_label = RESULT_LABELS[actual]
            else:
                positive_probability = float(probabilities[position])
                predicted = int(positive_probability >= 0.5)
                actual = int(observed[target])
                probability = (
                    positive_probability if predicted == 1 else 1.0 - positive_probability
                )
                if model_name == "GOLES_1T_U15":
                    labels = {1: "MENOS DE 1.5", 0: "MAS DE 1.5"}
                elif model_name == "TARJETAS_O45":
                    labels = {1: "MAS DE 4.5", 0: "MENOS DE 4.5"}
                else:
                    labels = {1: "MARCA 1+", 0: "NO MARCA"}
                prediction_label = labels[predicted]
                actual_label = labels[actual]
            rows.append({
                "Fecha": observed["Date"],
                "Competicion": observed.get("Competicion", observed.get("CompKey", "")),
                "Local": observed["HomeTeam"],
                "Visitante": observed["AwayTeam"],
                "Mercado": market_label,
                "Pronostico": prediction_label,
                "Real": actual_label,
                "ProbabilidadPronostico": probability,
                "Correcto": bool(predicted == actual),
                "CorteEntrenamiento": cutoff,
                "Semilla": int(random_seed),
            })

    detail = pd.DataFrame(rows)
    summaries: list[dict[str, Any]] = []
    if not detail.empty:
        for market, group in detail.groupby("Mercado", sort=False):
            counts = group["Real"].value_counts(normalize=True)
            summaries.append({
                "Mercado": market,
                "N": int(len(group)),
                "Aciertos": int(group["Correcto"].sum()),
                "TasaAcierto": float(group["Correcto"].mean()),
                "ExactitudBaseMayoritaria": float(counts.max()),
                "ProbabilidadMediaSeleccion": float(group["ProbabilidadPronostico"].mean()),
                "CorteEntrenamiento": cutoff,
                "Semilla": int(random_seed),
            })
    return detail, pd.DataFrame(summaries)


def _predict_probability(model: ProbabilityModel, row: pd.DataFrame) -> np.ndarray:
    raw = model.model.predict_proba(row[MODEL_FEATURES])
    if model.kind == "binary":
        classes = list(model.model.named_steps["model"].classes_)
        positive = raw[:, classes.index(1)]
        return np.asarray(model.calibrator.transform(positive), dtype=float)
    return np.asarray(model.calibrator.transform(raw), dtype=float)


def _wilson_lower(p: float, n: float, z: float = 1.28) -> float:
    p = float(np.clip(p, 0.0, 1.0))
    n = max(float(n), 1.0)
    z2 = z * z
    centre = p + z2 / (2.0 * n)
    radius = z * math.sqrt(max(p * (1.0 - p) / n + z2 / (4.0 * n * n), 0.0))
    return float(max(0.0, (centre - radius) / (1.0 + z2 / n)))


def _support_and_reliability(
    feature: dict[str, Any], model_quality: float, context_factor: float
) -> tuple[int, float]:
    team_n = min(int(feature.get("HomeHistoryN", 0)), int(feature.get("AwayHistoryN", 0)))
    comp_n = int(feature.get("CompetitionHistoryN", 0))
    support = max(1, min(180, int(0.60 * comp_n + 2.0 * team_n)))
    team_factor = min(1.0, team_n / 18.0)
    comp_factor = min(1.0, comp_n / 160.0)
    phase_factor = min(1.0, min(float(feature.get("HomeSeasonGames", 0)), float(feature.get("AwaySeasonGames", 0))) / 8.0 + 0.28)
    reliability = float(np.clip(
        model_quality * (0.52 + 0.20 * team_factor + 0.18 * comp_factor + 0.10 * phase_factor) * context_factor,
        0.0,
        0.98,
    ))
    return support, reliability


def _semaphore_binary(p_selected: float, lcb: float, reliability: float, support: int) -> tuple[str, str]:
    if p_selected >= 0.68 and lcb >= 0.59 and reliability >= 0.66 and support >= 55:
        return "VERDE", "ALTA/BUENA"
    if p_selected >= 0.58 and lcb >= 0.50 and reliability >= 0.50 and support >= 35:
        return "AMARILLO", "MEDIA/CAUTELA"
    return "ROJO", "MUY RIESGOSA"


def _semaphore_result(
    p_selected: float, margin: float, lcb: float, reliability: float, support: int
) -> tuple[str, str]:
    if p_selected >= 0.49 and margin >= 0.11 and lcb >= 0.40 and reliability >= 0.67 and support >= 70:
        return "VERDE", "ALTA/BUENA"
    if p_selected >= 0.40 and margin >= 0.06 and lcb >= 0.32 and reliability >= 0.50 and support >= 40:
        return "AMARILLO", "MEDIA/CAUTELA"
    return "ROJO", "MUY RIESGOSA"


def _validated_signal(
    semaphore: str,
    level: str,
    passes_oos: bool,
) -> tuple[str, str]:
    """Impide que un modelo que no supera su referencia publique una señal.

    La interfaz nunca debe reconstruir el color usando solo la probabilidad.
    Esta compuerta es deliberadamente unidireccional: puede degradar una señal,
    pero jamás promocionarla.
    """
    if not bool(passes_oos):
        return "ROJO", "NO SUPERA BASE OOS"
    return str(semaphore), str(level)


def _market_code(market: str, fixture: pd.Series) -> str:
    if market in MARKET_CODES:
        return MARKET_CODES[market]
    home = str(fixture.get("HomeOriginal", fixture.get("HomeTeam", "")))
    away = str(fixture.get("AwayOriginal", fixture.get("AwayTeam", "")))
    if market == f"Goles {home}":
        return "HOME_SCORE_OU05"
    if market == f"Goles {away}":
        return "AWAY_SCORE_OU05"
    return "UNKNOWN"


def _prediction_code(market_code: str, prediction: str) -> str:
    value = str(prediction).strip().upper()
    if value in {"", "SIN PRONOSTICO", "NAN"}:
        return "ABSTAIN"
    if market_code == "RESULT_1X2":
        return {"LOCAL": "HOME", "EMPATE": "DRAW", "VISITANTE": "AWAY"}.get(value, "ABSTAIN")
    if market_code in {"H1_GOALS_OU15", "H1_CORNERS_OU45", "YELLOW_CARDS_OU45"}:
        return "OVER" if value.startswith("MAS") else "UNDER" if value.startswith("MENOS") else "ABSTAIN"
    if market_code in {"HOME_SCORE_OU05", "AWAY_SCORE_OU05"}:
        return "YES" if value.startswith("MARCA") else "NO" if value.startswith("NO MARCA") else "ABSTAIN"
    return "ABSTAIN"


def _mark_best_option(rows: list[dict[str, Any]]) -> None:
    """Marca como máximo una opción verde por partido, sin crear combinadas."""
    candidates: list[tuple[float, int]] = []
    for index, row in enumerate(rows):
        probability = float(row.get("Probabilidad", np.nan))
        conservative = float(row.get("PConservadora", np.nan))
        reliability = float(row.get("Fiabilidad", np.nan))
        support = float(row.get("Soporte", 0) or 0)
        if (
            str(row.get("Semaforo", "")) == "VERDE"
            and str(row.get("PronosticoCodigo", "")) != "ABSTAIN"
            and np.isfinite(probability)
            and np.isfinite(conservative)
            and np.isfinite(reliability)
        ):
            score = 0.45 * conservative + 0.35 * reliability + 0.20 * min(1.0, support / 100.0)
            row["PuntajeSeleccion"] = float(score)
            candidates.append((score, index))
        else:
            row["PuntajeSeleccion"] = np.nan
    if candidates:
        rows[max(candidates)[1]]["MejorOpcion"] = True


def _market_row(
    fixture: pd.Series,
    market: str,
    line: str,
    prediction: str,
    p_selected: float,
    p_alternative: float | None,
    lcb: float,
    reliability: float,
    support: int,
    semaphore: str,
    level: str,
    reason: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    market_code = _market_code(market, fixture)
    row = {
        "Fecha": fixture.get("Date"),
        "KickoffUTC": fixture.get("KickoffUTC", ""),
        "EventID": str(fixture.get("EventID", "")),
        "HomeESPNID": str(fixture.get("HomeESPNID", "")),
        "AwayESPNID": str(fixture.get("AwayESPNID", "")),
        "HoraPeru": fixture.get("HoraPeru", ""),
        "CompKey": fixture.get("CompKey", ""),
        "Competicion": fixture.get("Competicion", fixture.get("CompKey", "")),
        "Local": fixture.get("HomeOriginal", fixture.get("HomeTeam", "")),
        "Visitante": fixture.get("AwayOriginal", fixture.get("AwayTeam", "")),
        "Mercado": market,
        "MercadoCodigo": market_code,
        "Linea": line,
        "Pronostico": prediction,
        "PronosticoCodigo": _prediction_code(market_code, prediction),
        "Probabilidad": p_selected,
        "ProbAlternativa": p_alternative,
        "PConservadora": lcb,
        "Fiabilidad": reliability,
        "Soporte": support,
        "Semaforo": semaphore,
        "SemaforoModelo": semaphore,
        "SemaforoFinal": semaphore,
        "Nivel": level,
        "Motivo": reason,
        "MejorOpcion": False,
        "Version": VERSION,
    }
    if extra:
        row.update(extra)
    return row


def _resolve_team(name: str, candidates: list[str]) -> tuple[str, float]:
    resolved, score = base.resolver_nombre(name, candidates)
    # Los equipos nuevos siguen apareciendo, pero sin fingir historial.
    return (resolved if score >= 0.57 else name), float(score)


_LINEUP_CACHE: dict[str, dict[str, Any]] = {}
_LINEUP_CALLS = 0
MAX_LINEUP_CALLS = 36


def _lineup_context(fixture: pd.Series) -> dict[str, Any]:
    global _LINEUP_CALLS
    event_id = str(fixture.get("EventID", ""))
    if not event_id or fixture.get("CompKey") not in base.COMPETICIONES:
        return context93.parse_confirmed_lineups({})
    kickoff = pd.to_datetime(fixture.get("KickoffUTC"), utc=True, errors="coerce")
    now = pd.Timestamp.now(tz="UTC")
    cached = _LINEUP_CACHE.get(event_id)
    if cached and (cached["result"]["status"] == "CONFIRMADA" or now - cached["checked_at"] < pd.Timedelta(minutes=15)):
        return cached["result"]
    if pd.isna(kickoff) or kickoff - now > pd.Timedelta(hours=6) or _LINEUP_CALLS >= MAX_LINEUP_CALLS:
        return context93.parse_confirmed_lineups({})
    try:
        league = base.COMPETICIONES[str(fixture["CompKey"])]["espn"]
        summary = base.resumen_evento_espn(league, event_id, cache_hours=0.5)
        _LINEUP_CALLS += 1
        result = context93.parse_confirmed_lineups(
            summary,
            fixture.get("HomeESPNID", ""),
            fixture.get("AwayESPNID", ""),
        )
    except Exception:
        result = context93.parse_confirmed_lineups({})
    _LINEUP_CACHE[event_id] = {"checked_at": now, "result": result}
    return result


def _fixture_context(
    fixture: pd.Series,
    home: str,
    away: str,
    schedule_index: dict[str, Any],
    calendar_index: dict[str, Any],
) -> dict[str, Any]:
    home_load = base.workload_features(schedule_index, home, fixture["Date"], fixture["CompKey"])
    away_load = base.workload_features(schedule_index, away, fixture["Date"], fixture["CompKey"])
    lineup = _lineup_context(fixture)
    history = context93.load_lineup_history(LINEUP_HISTORY_PATH)
    home_profile = context93.lineup_strength_profile(
        history,
        fixture.get("HomeOriginal", home),
        lineup.get("home_starters", []),
        fixture["Date"],
        fixture.get("EventID", ""),
    )
    away_profile = context93.lineup_strength_profile(
        history,
        fixture.get("AwayOriginal", away),
        lineup.get("away_starters", []),
        fixture["Date"],
        fixture.get("EventID", ""),
    )
    if lineup.get("status") == "CONFIRMADA":
        context93.archive_confirmed_lineups(LINEUP_HISTORY_PATH, fixture, lineup)
    evaluated = context93.evaluate_fixture_context(
        fixture,
        home,
        away,
        home_load,
        away_load,
        "HOME",
        calendar_index,
        lineup_status=lineup.get("status", "NO_DISPONIBLE"),
        lineup_home=home_profile,
        lineup_away=away_profile,
    )
    evaluated["home_load"] = home_load
    evaluated["away_load"] = away_load
    return evaluated


def _h1_corner_oos_validation(sample: pd.DataFrame) -> dict[str, Any]:
    """Comprueba en el tramo mas reciente si el ajuste por equipos aporta."""
    ordered = sample.sort_values("Date").copy()
    dates = np.array(sorted(ordered["Date"].dropna().unique()))
    if len(ordered) < 60 or len(dates) < 18:
        return {
            "n": 0,
            "brier": np.nan,
            "brier_base": np.nan,
            "improvement": np.nan,
            "ece": np.nan,
            "passes": False,
            "note": "validacion temporal insuficiente",
        }
    cutoff = pd.Timestamp(dates[max(1, int(len(dates) * 0.78))])
    train = ordered[ordered["Date"] < cutoff].copy()
    valid = ordered[ordered["Date"] >= cutoff].copy()
    if len(train) < 40 or len(valid) < 12:
        return {
            "n": int(len(valid)),
            "brier": np.nan,
            "brier_base": np.nan,
            "improvement": np.nan,
            "ece": np.nan,
            "passes": False,
            "note": "validacion temporal insuficiente",
        }

    train_y = (train["H1CornersTotal"] >= 5).astype(float)
    base_rate = float((train_y.sum() + 0.50 * 24.0) / (len(train_y) + 24.0))
    team_stats: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
    if {"HomeTeam", "AwayTeam"}.issubset(train.columns):
        for _, observed in train.iterrows():
            outcome = float(observed["H1CornersTotal"] >= 5)
            for column in ("HomeTeam", "AwayTeam"):
                name = base.norm_texto(observed.get(column, ""))
                if name:
                    team_stats[name][0] += outcome
                    team_stats[name][1] += 1.0

    probabilities = []
    for _, observed in valid.iterrows():
        rates = []
        for column in ("HomeTeam", "AwayTeam"):
            name = base.norm_texto(observed.get(column, ""))
            successes, count = team_stats.get(name, [0.0, 0.0])
            if count > 0:
                rates.append(float((successes + base_rate * 12.0) / (count + 12.0)))
        probability = (
            float(0.62 * base_rate + 0.38 * np.mean(rates))
            if rates
            else base_rate
        )
        probabilities.append(_clip_probability(probability))

    y = (valid["H1CornersTotal"] >= 5).astype(int).to_numpy()
    p = np.asarray(probabilities, dtype=float)
    brier = float(np.mean((p - y) ** 2))
    brier_base = float(np.mean((base_rate - y) ** 2))
    improvement = float((brier_base - brier) / max(brier_base, 1e-9))
    ece = _ece_binary(y, p, bins=6)
    passes = bool(improvement > 0.0 and ece <= 0.15)
    return {
        "n": int(len(valid)),
        "brier": brier,
        "brier_base": brier_base,
        "improvement": improvement,
        "ece": ece,
        "passes": passes,
        "note": "SUPERA BASE OOS" if passes else "NO SUPERA BASE OOS",
    }


def _argentine_corner_oos_validation(
    all_international: pd.DataFrame,
    argentine_crosses: pd.DataFrame,
) -> dict[str, Any]:
    """Valida temporalmente si el subgrupo argentino aporta frente a la base.

    Compara una tasa del subgrupo encogida con la tasa general disponible antes
    del corte. Así, una diferencia descriptiva no se convierte automáticamente
    en regla de apuesta.
    """
    group = argentine_crosses.sort_values("Date").copy()
    if len(group) < MIN_ARG_CROSS_SAMPLE:
        return {
            "n": int(len(group)), "n_valid": 0, "improvement": np.nan,
            "brier": np.nan, "brier_base": np.nan, "ece": np.nan,
            "passes": False, "note": "muestra argentina insuficiente",
        }
    dates = np.array(sorted(group["Date"].dropna().unique()))
    if len(dates) < 10:
        return {
            "n": int(len(group)), "n_valid": 0, "improvement": np.nan,
            "brier": np.nan, "brier_base": np.nan, "ece": np.nan,
            "passes": False, "note": "pocas fechas independientes",
        }
    cutoff = pd.Timestamp(dates[max(1, int(len(dates) * 0.72))])
    train_all = all_international[all_international["Date"] < cutoff]
    train_group = group[group["Date"] < cutoff]
    valid = group[group["Date"] >= cutoff]
    if len(train_all) < 40 or len(train_group) < 20 or len(valid) < 8:
        return {
            "n": int(len(group)), "n_valid": int(len(valid)), "improvement": np.nan,
            "brier": np.nan, "brier_base": np.nan, "ece": np.nan,
            "passes": False, "note": "corte temporal argentino insuficiente",
        }
    base_rate = float((train_all["H1CornersTotal"].ge(5).sum() + 0.5 * 24.0) / (len(train_all) + 24.0))
    group_rate = float((train_group["H1CornersTotal"].ge(5).sum() + base_rate * 24.0) / (len(train_group) + 24.0))
    y = valid["H1CornersTotal"].ge(5).astype(float).to_numpy()
    brier = float(np.mean((group_rate - y) ** 2))
    brier_base = float(np.mean((base_rate - y) ** 2))
    improvement = float((brier_base - brier) / max(brier_base, 1e-9))
    ece = float(abs(group_rate - y.mean()))
    passes = bool(improvement > 0.0 and ece <= 0.18)
    return {
        "n": int(len(group)),
        "n_valid": int(len(valid)),
        "improvement": improvement,
        "brier": brier,
        "brier_base": brier_base,
        "ece": ece,
        "passes": passes,
        "note": "SUPERA BASE OOS" if passes else "NO SUPERA BASE OOS",
    }


def predict_h1_corners(
    context: pd.DataFrame,
    fixture: pd.Series,
    state: SequentialState,
    resolved_home: str,
    resolved_away: str,
) -> dict[str, Any]:
    """Pronostica O/U 4.5 solo desde conteos reales del primer tiempo."""
    if context is None or context.empty or "H1CornersTotal" not in context.columns:
        return {"available": False, "reason": "Sin historico homogeneo de corners 1T"}
    data = context.copy()
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    data["H1CornersTotal"] = pd.to_numeric(data["H1CornersTotal"], errors="coerce")
    data = data[(data["Date"] < pd.Timestamp(fixture["Date"])) & data["H1CornersTotal"].notna()]
    if data.empty:
        return {"available": False, "reason": "Sin observaciones previas de corners 1T"}
    exact = data[data["CompKey"].astype(str) == str(fixture["CompKey"])]
    if len(exact) < 28:
        if "Grupo" in data.columns:
            group = data[data["Grupo"].astype(str) == str(fixture.get("Grupo", ""))]
        else:
            group = pd.DataFrame(columns=data.columns)
        sample = group if len(group) >= 55 else exact
    else:
        sample = exact
    if len(sample) < 28:
        return {"available": False, "reason": f"Cobertura insuficiente de corners 1T (n={len(sample)})"}

    validation = _h1_corner_oos_validation(sample)
    event = (sample["H1CornersTotal"] >= 5).astype(float)
    global_rate = float((event.sum() + 0.50 * 24.0) / (len(event) + 24.0))
    home_name = base.norm_texto(fixture.get("HomeOriginal", fixture.get("HomeTeam", "")))
    away_name = base.norm_texto(fixture.get("AwayOriginal", fixture.get("AwayTeam", "")))
    team_rates = []
    team_ns = []
    if {"HomeTeam", "AwayTeam"}.issubset(sample.columns):
        for name in (home_name, away_name):
            mask = sample["HomeTeam"].map(base.norm_texto).eq(name) | sample["AwayTeam"].map(base.norm_texto).eq(name)
            sub = sample.loc[mask, "H1CornersTotal"]
            n = len(sub)
            if n:
                rate = float(((sub >= 5).sum() + global_rate * 12.0) / (n + 12.0))
                team_rates.append(rate)
                team_ns.append(n)
    if team_rates:
        p_over = float(0.62 * global_rate + 0.38 * np.mean(team_rates))
    else:
        p_over = global_rate

    # Nicho argentino interliga: se trata como una interaccion empirica, no
    # como una penalizacion fija. El ajuste solo se activa cuando existen 30
    # cruces previos identificables y siempre se encoge hacia la base.
    current_flags = _origin_flags(
        state,
        str(fixture.get("CompKey", "")),
        resolved_home,
        resolved_away,
    )
    arg_cross_n = 0
    arg_cross_rate = np.nan
    arg_corner_delta = np.nan
    arg_validation = {
        "n": 0, "n_valid": 0, "improvement": np.nan, "brier": np.nan,
        "brier_base": np.nan, "ece": np.nan, "passes": False,
        "note": "no aplica",
    }
    adjustment_note = ""
    if current_flags["ArgentineCrossLeague"] == 1.0:
        origins_by_normalized = {
            base.norm_texto(team): _team_origin(state, team)
            for team in state.origin_counts
            if _team_origin(state, team) != "UNKNOWN"
        }

        def origin_for_display(name: Any) -> str:
            normalized = base.norm_texto(name)
            if normalized in origins_by_normalized:
                return origins_by_normalized[normalized]
            candidates = list(origins_by_normalized)
            if not candidates:
                return "UNKNOWN"
            resolved, score = base.resolver_nombre(normalized, candidates)
            return origins_by_normalized.get(resolved, "UNKNOWN") if score >= 0.72 else "UNKNOWN"

        cross_mask = []
        for _, observed in data.iterrows():
            if str(observed.get("CompKey", "")) not in SOUTH_AMERICAN_INTERNATIONAL:
                cross_mask.append(False)
                continue
            home_origin = origin_for_display(observed.get("HomeTeam", ""))
            away_origin = origin_for_display(observed.get("AwayTeam", ""))
            cross_mask.append(
                home_origin != "UNKNOWN"
                and away_origin != "UNKNOWN"
                and ((home_origin == "ARG") ^ (away_origin == "ARG"))
            )
        arg_sample = data.loc[np.asarray(cross_mask, dtype=bool)]
        arg_cross_n = int(len(arg_sample))
        international = data[data["CompKey"].astype(str).isin(SOUTH_AMERICAN_INTERNATIONAL)]
        arg_validation = _argentine_corner_oos_validation(international, arg_sample)
        if arg_cross_n >= MIN_ARG_CROSS_SAMPLE:
            raw_rate = float((arg_sample["H1CornersTotal"] >= 5).mean())
            arg_cross_rate = float(
                ((arg_sample["H1CornersTotal"] >= 5).sum() + global_rate * 24.0)
                / (arg_cross_n + 24.0)
            )
            arg_mean = float(arg_sample["H1CornersTotal"].mean())
            base_mean = float(sample["H1CornersTotal"].mean())
            arg_corner_delta = arg_mean - base_mean
            if arg_validation["passes"]:
                weight = min(0.45, arg_cross_n / (arg_cross_n + 55.0))
                p_over = float((1.0 - weight) * p_over + weight * arg_cross_rate)
                adjustment_note = (
                    f"; cruce argentino interliga validado n={arg_cross_n}, "
                    f"delta corners 1T={arg_corner_delta:+.2f}, tasa O4.5={raw_rate:.1%}"
                )
            else:
                adjustment_note = (
                    f"; diferencia argentina observada pero bloqueada: "
                    f"{arg_validation['note']} (validacion n={arg_validation['n_valid']})"
                )
        else:
            adjustment_note = (
                f"; cruce argentino detectado pero sin ajuste: "
                f"n={arg_cross_n}<{MIN_ARG_CROSS_SAMPLE}"
            )
    support = int(min(160, len(sample) + 2 * sum(team_ns)))
    reliability = float(np.clip(0.34 + 0.45 * min(1.0, len(sample) / 120.0) + 0.21 * min(1.0, min(team_ns or [0]) / 10.0), 0.25, 0.88))
    if not validation["passes"]:
        reliability = min(reliability, 0.42)
    if current_flags["ArgentineCrossLeague"] == 1.0 and not arg_validation["passes"]:
        reliability *= 0.78
    effective_oos = bool(
        validation["passes"]
        and (
            current_flags["ArgentineCrossLeague"] != 1.0
            or arg_validation["passes"]
        )
    )
    return {
        "available": True,
        "p_over": _clip_probability(p_over),
        "support": support,
        "reliability": reliability,
        "reason": (
            f"Conteos reales 1T; muestra competencia/grupo n={len(sample)}; "
            f"{validation['note']} (n={validation['n']}){adjustment_note}"
        ),
        "oos_brier": validation["brier"],
        "oos_brier_base": validation["brier_base"],
        "oos_improvement": validation["improvement"],
        "oos_ece": validation["ece"],
        "oos_passes": effective_oos,
        "arg_cross_n": arg_cross_n,
        "arg_cross_rate": arg_cross_rate,
        "arg_corner_delta": arg_corner_delta,
        "arg_oos_n": arg_validation["n_valid"],
        "arg_oos_improvement": arg_validation["improvement"],
        "arg_oos_ece": arg_validation["ece"],
        "arg_oos_passes": arg_validation["passes"],
    }


def forecast_fixture(
    fixture: pd.Series,
    state: SequentialState,
    bundle: ModelBundle,
    h1_corner_context: pd.DataFrame,
    schedule_index: dict[str, Any],
    calendar_index: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    candidates = list(state.histories)
    home, home_match = _resolve_team(str(fixture.get("HomeOriginal", fixture.get("HomeTeam", ""))), candidates)
    away, away_match = _resolve_team(str(fixture.get("AwayOriginal", fixture.get("AwayTeam", ""))), candidates)
    feature = _feature_row(state, fixture["Date"], str(fixture["CompKey"]), str(fixture["Grupo"]), home, away)
    feature_frame = pd.DataFrame([feature])
    context = _fixture_context(fixture, home, away, schedule_index, calendar_index)
    context_factor = float(context.get("context_factor", 0.80))
    rows: list[dict[str, Any]] = []

    result_model = bundle.probability_models["RESULTADO_1X2"]
    if result_model.model is None:
        rows.append(_market_row(
            fixture, "Resultado 1X2", "Local / Empate / Visitante", "SIN PRONOSTICO",
            np.nan, np.nan, np.nan, 0.0, 0, "ROJO", "NO MODELABLE",
            str(result_model.metrics.get("ErrorModelo", "Muestra insuficiente")),
            {"ModeloSuperaBase": False, "ModeloEstadoValidacion": "SIN COBERTURA"},
        ))
    else:
        result_probs = _predict_probability(result_model, feature_frame)[0]
        order = np.argsort(result_probs)[::-1]
        choice, runner_up = int(order[0]), int(order[1])
        p_choice = float(result_probs[choice])
        margin = p_choice - float(result_probs[runner_up])
        support, reliability = _support_and_reliability(feature, result_model.metrics["CalidadModelo"], context_factor)
        lcb = _wilson_lower(p_choice, support)
        semaphore, level = _semaphore_result(p_choice, margin, lcb, reliability, support)
        semaphore, level = _validated_signal(
            semaphore, level, result_model.metrics.get("SuperaBase", False)
        )
        rows.append(_market_row(
            fixture, "Resultado 1X2", "Local / Empate / Visitante", RESULT_LABELS[choice],
            p_choice, float(result_probs[runner_up]), lcb, reliability, support,
            semaphore, level,
            (
                f"Ventaja sobre segunda opcion {margin:.1%}; modelo temporal calibrado; "
                f"{result_model.metrics['EstadoValidacion']}"
            ),
            {
                "P_Local": float(result_probs[0]),
                "P_Empate": float(result_probs[1]),
                "P_Visitante": float(result_probs[2]),
                "ModeloSuperaBase": bool(result_model.metrics.get("SuperaBase", False)),
                "ModeloEstadoValidacion": result_model.metrics.get("EstadoValidacion", ""),
            },
        ))

    def add_binary(model_key: str, market: str, line: str, positive: str, negative: str) -> None:
        model = bundle.probability_models[model_key]
        if model.model is None:
            rows.append(_market_row(
                fixture, market, line, "SIN PRONOSTICO", np.nan, np.nan,
                np.nan, 0.0, 0, "ROJO", "NO MODELABLE",
                str(model.metrics.get("ErrorModelo", "Muestra insuficiente")),
                {"ModeloSuperaBase": False, "ModeloEstadoValidacion": "SIN COBERTURA"},
            ))
            return
        p_positive = float(_predict_probability(model, feature_frame)[0])
        if p_positive >= 0.5:
            prediction, selected, alternative = positive, p_positive, 1.0 - p_positive
        else:
            prediction, selected, alternative = negative, 1.0 - p_positive, p_positive
        support_b, rel_b = _support_and_reliability(feature, model.metrics["CalidadModelo"], context_factor)
        lcb_b = _wilson_lower(selected, support_b)
        sem, lev = _semaphore_binary(selected, lcb_b, rel_b, support_b)
        passes_oos = bool(model.metrics.get("SuperaBase", False))
        arg_subgroup_note = ""
        if feature.get("ArgentineCrossLeague", 0.0) == 1.0 and model_key == "TARJETAS_O45":
            arg_passes = bool(model.metrics.get("ArgCrossSuperaBase", False))
            passes_oos = passes_oos and arg_passes
            arg_subgroup_note = (
                f"; validacion cruce argentino n={int(model.metrics.get('ArgCrossNValidacion', 0) or 0)}: "
                + ("SUPERA BASE" if arg_passes else "NO SUPERA BASE/SIN MUESTRA")
            )
        sem, lev = _validated_signal(sem, lev, passes_oos)
        rows.append(_market_row(
            fixture, market, line, prediction, selected, alternative, lcb_b,
            rel_b, support_b, sem, lev,
            (
                f"Brier OOS={model.metrics['Brier']:.3f}; "
                f"base={model.metrics['BrierBase']:.3f}; "
                f"ECE={model.metrics['ECE']:.3f}; {model.metrics['EstadoValidacion']}"
                + (
                    "; cruce argentino interliga incluido como interaccion"
                    if feature.get("ArgentineCrossLeague", 0.0) == 1.0
                    else ""
                )
                + arg_subgroup_note
            ),
            {
                "P_OpcionPositiva": p_positive,
                "ModeloSuperaBase": passes_oos,
                "ModeloEstadoValidacion": (
                    "SUPERA BASE OOS" if passes_oos else "NO SUPERA BASE OOS"
                ),
                "ArgCrossNValidacion": model.metrics.get("ArgCrossNValidacion", 0),
                "ArgCrossSuperaBase": model.metrics.get("ArgCrossSuperaBase", False),
            },
        ))

    add_binary("GOLES_1T_U15", "Goles 1.er tiempo", "1.5", "MENOS DE 1.5", "MAS DE 1.5")

    corner = predict_h1_corners(h1_corner_context, fixture, state, home, away)
    if corner.get("available"):
        p_over = float(corner["p_over"])
        if p_over >= 0.5:
            prediction, selected, alternative = "MAS DE 4.5", p_over, 1.0 - p_over
        else:
            prediction, selected, alternative = "MENOS DE 4.5", 1.0 - p_over, p_over
        corner_rel = float(corner["reliability"]) * context_factor
        corner_support = int(corner["support"])
        corner_lcb = _wilson_lower(selected, corner_support)
        sem, lev = _semaphore_binary(selected, corner_lcb, corner_rel, corner_support)
        sem, lev = _validated_signal(sem, lev, corner.get("oos_passes", False))
        rows.append(_market_row(
            fixture, "Corners 1.er tiempo", "4.5", prediction, selected, alternative,
            corner_lcb, corner_rel, corner_support, sem, lev, str(corner["reason"]),
            {
                "P_Mas45": p_over,
                "ArgCrossN": corner.get("arg_cross_n", 0),
                "ArgCrossRateO45": corner.get("arg_cross_rate", np.nan),
                "ArgCornerDelta1T": corner.get("arg_corner_delta", np.nan),
                "ArgCornerNValidacion": corner.get("arg_oos_n", 0),
                "ArgCornerMejoraOOS": corner.get("arg_oos_improvement", np.nan),
                "ArgCornerECE": corner.get("arg_oos_ece", np.nan),
                "ArgCornerSuperaBase": corner.get("arg_oos_passes", False),
                "CornerBrierOOS": corner.get("oos_brier", np.nan),
                "CornerBrierBase": corner.get("oos_brier_base", np.nan),
                "CornerMejoraBrier": corner.get("oos_improvement", np.nan),
                "CornerECE": corner.get("oos_ece", np.nan),
                "CornerSuperaBase": corner.get("oos_passes", False),
                "ModeloSuperaBase": bool(corner.get("oos_passes", False)),
                "ModeloEstadoValidacion": (
                    "SUPERA BASE OOS" if corner.get("oos_passes", False) else "NO SUPERA BASE OOS"
                ),
            },
        ))
    else:
        rows.append(_market_row(
            fixture, "Corners 1.er tiempo", "4.5", "SIN PRONOSTICO", np.nan, np.nan,
            np.nan, 0.0, 0, "ROJO", "NO MODELABLE", str(corner.get("reason", "Sin datos")),
            {"ModeloSuperaBase": False, "ModeloEstadoValidacion": "SIN COBERTURA"},
        ))

    add_binary("TARJETAS_O45", "Tarjetas amarillas totales", "4.5", "MAS DE 4.5", "MENOS DE 4.5")
    add_binary("LOCAL_MARCA", f"Goles {fixture.get('HomeOriginal', 'local')}", "0.5", "MARCA 1+", "NO MARCA")
    add_binary("VISITANTE_MARCA", f"Goles {fixture.get('AwayOriginal', 'visitante')}", "0.5", "MARCA 1+", "NO MARCA")

    for row in rows:
        row["EstadoAlineacion"] = context.get("lineup_status", "NO_DISPONIBLE")
        row["RiesgoRotacion"] = context.get("rotation_risk", "")
        row["FactorContexto"] = context_factor

    _mark_best_option(rows)

    def expected_count(model_key: str, low: float, high: float) -> float:
        count_model = bundle.count_models[model_key]
        if count_model.model is None:
            return np.nan
        try:
            prediction = count_model.model.predict(feature_frame[MODEL_FEATURES])[0]
            return float(np.clip(prediction, low, high))
        except Exception:
            return np.nan

    expected_home = expected_count("GOLES_ESPERADOS_LOCAL", 0.05, 5.0)
    expected_away = expected_count("GOLES_ESPERADOS_VISITANTE", 0.05, 5.0)
    expected_h1 = expected_count("GOLES_ESPERADOS_1T", 0.02, 4.0)
    expected_cards = expected_count("TARJETAS_ESPERADAS", 0.10, 10.0)

    match_summary = {
        "Fecha": fixture.get("Date"),
        "KickoffUTC": fixture.get("KickoffUTC", ""),
        "EventID": str(fixture.get("EventID", "")),
        "HomeESPNID": str(fixture.get("HomeESPNID", "")),
        "AwayESPNID": str(fixture.get("AwayESPNID", "")),
        "HoraPeru": fixture.get("HoraPeru", ""),
        "CompKey": fixture.get("CompKey", ""),
        "Competicion": fixture.get("Competicion", fixture.get("CompKey", "")),
        "Local": fixture.get("HomeOriginal", fixture.get("HomeTeam", "")),
        "Visitante": fixture.get("AwayOriginal", fixture.get("AwayTeam", "")),
        "GolesEsperadosLocal": expected_home,
        "GolesEsperadosVisitante": expected_away,
        "GolesEsperados1T": expected_h1,
        "TarjetasEsperadas": expected_cards,
        "RiesgoRotacion": context.get("rotation_risk", ""),
        "EstadoAlineacion": context.get("lineup_status", ""),
        "FaseCompetitiva": context.get("stage", ""),
        "ImportanciaPartido": context.get("importance", np.nan),
        "NotaContexto": context.get("note", ""),
        "CoincidenciaNombreLocal": home_match,
        "CoincidenciaNombreVisitante": away_match,
        "HistorialLocalN": feature["HomeHistoryN"],
        "HistorialVisitanteN": feature["AwayHistoryN"],
        "HistorialCompeticionN": feature["CompetitionHistoryN"],
        "OrigenLocal": feature.get("HomeOrigin", "UNKNOWN"),
        "OrigenVisitante": feature.get("AwayOrigin", "UNKNOWN"),
        "CruceArgentinoInterliga": bool(feature.get("ArgentineCrossLeague", 0.0)),
        "EntrenadoHasta": bundle.trained_through,
        "Version": VERSION,
    }
    return match_summary, rows


def _notify(progress: Any, message: str) -> None:
    if callable(progress):
        progress(message)


@contextmanager
def _runtime_limits(fast_mode: bool):
    """Reduce llamadas remotas en la sesion web y restaura la configuracion."""
    global MAX_LINEUP_CALLS, _LINEUP_CALLS
    old_env = os.environ.get("GATUNO_FAST_MODE")
    old_lineup_limit = MAX_LINEUP_CALLS
    old_lineup_calls = _LINEUP_CALLS
    names = {
        "TEMPORADAS_FD": ["2526", "2627"],
        "ESPN_HIST_DAYS": 220,
        "MAX_ESPN_RESULT_EVENTS_PER_COMP": 100,
        "MAX_ESPN_HIST_EVENTS_PER_COMP": 10,
        "MAX_ESPN_SUMMARY_CALLS_HIST": 50,
        "CONTEXT_DAYS": 90,
        "CONTEXT_EVENTS_PER_COMP": 4,
        "CONTEXT_MAX_SUMMARY_CALLS": 24,
        "MAX_FUTURE_SUMMARY_CALLS": 16,
        "MAX_WEATHER_CALLS": 16,
    }
    previous = {name: getattr(base, name) for name in names}
    try:
        if fast_mode:
            os.environ["GATUNO_FAST_MODE"] = "1"
            for name, value in names.items():
                setattr(base, name, value)
            MAX_LINEUP_CALLS = 12
            _LINEUP_CALLS = 0
        else:
            os.environ["GATUNO_FAST_MODE"] = "0"
        yield
    finally:
        for name, value in previous.items():
            setattr(base, name, value)
        MAX_LINEUP_CALLS = old_lineup_limit
        _LINEUP_CALLS = old_lineup_calls
        if old_env is None:
            os.environ.pop("GATUNO_FAST_MODE", None)
        else:
            os.environ["GATUNO_FAST_MODE"] = old_env


def _history_signature(history: pd.DataFrame) -> str:
    """Firma breve para no reutilizar un modelo con datos distintos."""
    ordered = history.sort_values("Date")
    cols = [c for c in ("Date", "CompKey", "HomeTeam", "AwayTeam", "FTHG", "FTAG") if c in ordered]
    tail = ordered[cols].tail(250).astype(str).to_csv(index=False)
    payload = f"{VERSION}|{len(ordered)}|{tail}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _state_to_cache(state: SequentialState) -> dict[str, Any]:
    """Convierte los defaultdict con factorías lambda a estructuras serializables."""
    return {
        "histories": {key: list(value) for key, value in state.histories.items()},
        "elos": dict(state.elos),
        "standings": {key: dict(value) for key, value in state.standings.items()},
        "competition_history": {
            key: list(value) for key, value in state.competition_history.items()
        },
        "origin_counts": {
            key: dict(value) for key, value in state.origin_counts.items()
        },
    }


def _state_from_cache(payload: dict[str, Any]) -> SequentialState:
    state = SequentialState()
    state.histories.update(payload.get("histories", {}))
    state.elos.update(payload.get("elos", {}))
    state.standings.update(payload.get("standings", {}))
    state.competition_history.update(payload.get("competition_history", {}))
    for key, value in payload.get("origin_counts", {}).items():
        state.origin_counts[key].update(value)
    return state


def _load_or_train_models(history: pd.DataFrame, progress: Any = None):
    """Carga una calibración vigente o la construye una sola vez.

    La caché contiene únicamente objetos derivados del histórico y se invalida
    automáticamente si cambia la versión o llegan partidos nuevos.
    """
    signature = _history_signature(history)
    try:
        with MODEL_CACHE_PATH.open("rb") as handle:
            cached = pickle.load(handle)
        if cached.get("signature") == signature and cached.get("version") == VERSION:
            _notify(progress, "Modelo calibrado recuperado desde caché segura…")
            return _state_from_cache(cached["state"]), cached["bundle"]
    except Exception:
        pass

    _notify(progress, f"Construyendo variables prepartido ({len(history):,} partidos)…")
    dataset, state = build_feature_dataset(history)
    _notify(progress, "Calibrando mercados; esta etapa ocurre sólo cuando cambian los datos…")
    bundle = train_models(dataset)
    try:
        APP_DATA_DIR.mkdir(parents=True, exist_ok=True)
        temp = MODEL_CACHE_PATH.with_name(MODEL_CACHE_PATH.name + ".tmp")
        with temp.open("wb") as handle:
            pickle.dump(
                {
                    "version": VERSION,
                    "signature": signature,
                    "state": _state_to_cache(state),
                    "bundle": bundle,
                },
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        os.replace(temp, MODEL_CACHE_PATH)
    except Exception:
        try:
            temp.unlink(missing_ok=True)
        except Exception:
            pass
    return state, bundle


def _run_v10_impl(progress: Any = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    start, end = base.ventana_objetivo()
    # Primero se comprueba el calendario. V15.0 hacía todo el entrenamiento y
    # sólo después descubría que las fuentes de fixtures estaban caídas.
    _notify(progress, "Consultando el calendario antes de iniciar el entrenamiento…")
    fixtures_context = base.descargar_fixtures_objetivo(start, end)
    fixtures = fixtures_context[
        (fixtures_context["Date"] >= start) & (fixtures_context["Date"] <= end)
    ].copy()
    fixtures = base.filtrar_fixtures_desde_ahora(fixtures)
    if fixtures.empty:
        raise RuntimeError(
            "Las fuentes respondieron, pero no devolvieron partidos futuros dentro de los próximos 7 días."
        )
    _notify(progress, f"Calendario listo: {len(fixtures)} partidos futuros encontrados.")

    _notify(progress, "Actualizando histórico compacto y verificable…")
    history = base.descargar_historico_total()
    state, bundle = _load_or_train_models(history, progress=progress)
    schedule_index = base.construir_schedule_index(history)
    calendar_index = context93.build_calendar_index(history, fixtures_context)

    _notify(progress, "Recuperando contexto 1T solo para las competiciones encontradas…")
    try:
        fixture_comp_keys = fixtures["CompKey"].dropna().astype(str).unique().tolist()
        h1_context = base.descargar_contexto_h1(comp_keys=fixture_comp_keys)
    except Exception:
        h1_context = pd.DataFrame()

    match_rows: list[dict[str, Any]] = []
    market_rows: list[dict[str, Any]] = []
    _notify(progress, f"Generando seis mercados para {len(fixtures)} partidos…")
    for _, fixture in fixtures.iterrows():
        try:
            match, markets = forecast_fixture(
                fixture, state, bundle, h1_context, schedule_index, calendar_index
            )
            match_rows.append(match)
            market_rows.extend(markets)
        except Exception as exc:
            # El partido no desaparece: queda auditable como error de datos.
            match_rows.append({
                "Fecha": fixture.get("Date"),
                "KickoffUTC": fixture.get("KickoffUTC", ""),
                "HoraPeru": fixture.get("HoraPeru", ""),
                "CompKey": fixture.get("CompKey", ""),
                "Competicion": fixture.get("Competicion", fixture.get("CompKey", "")),
                "Local": fixture.get("HomeOriginal", fixture.get("HomeTeam", "")),
                "Visitante": fixture.get("AwayOriginal", fixture.get("AwayTeam", "")),
                "ErrorDatos": str(exc),
                "Version": VERSION,
            })
            for market, line in (
                ("Resultado 1X2", "Local / Empate / Visitante"),
                ("Goles 1.er tiempo", "1.5"),
                ("Corners 1.er tiempo", "4.5"),
                ("Tarjetas amarillas totales", "4.5"),
                (f"Goles {fixture.get('HomeOriginal', 'local')}", "0.5"),
                (f"Goles {fixture.get('AwayOriginal', 'visitante')}", "0.5"),
            ):
                market_rows.append(_market_row(
                    fixture, market, line, "SIN PRONOSTICO", np.nan, np.nan,
                    np.nan, 0.0, 0, "ROJO", "NO MODELABLE", str(exc),
                ))

    matches = pd.DataFrame(match_rows)
    markets = pd.DataFrame(market_rows)
    if not matches.empty:
        matches = matches.sort_values(["Fecha", "HoraPeru", "Competicion", "Local"]).reset_index(drop=True)
        matches.insert(0, "Ranking", np.arange(1, len(matches) + 1))
    if not markets.empty:
        markets = markets.sort_values(["Fecha", "HoraPeru", "Competicion", "Local", "Mercado"]).reset_index(drop=True)
    return matches, markets, bundle.metrics, start, end


def run_v10(
    fast_mode: bool = False,
    progress: Any = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    """Ejecuta el motor; la interfaz usa limites de red conservadores."""
    with _runtime_limits(bool(fast_mode)):
        return _run_v10_impl(progress=progress)


def remove_margin(decimal_odds: list[float]) -> dict[str, Any]:
    """Convierte cuotas reales de un mercado completo a probabilidades sin margen."""
    odds = np.asarray(decimal_odds, dtype=float)
    if odds.ndim != 1 or len(odds) < 2 or np.any(~np.isfinite(odds)) or np.any(odds <= 1.0):
        raise ValueError("Incluye todas las cuotas decimales validas del mismo mercado.")
    raw = 1.0 / odds
    booksum = float(raw.sum())
    return {
        "probabilidades_brutas": raw.tolist(),
        "probabilidades_sin_margen": (raw / booksum).tolist(),
        "overround": booksum - 1.0,
        "booksum": booksum,
    }


def save_debug_snapshot(path: str | Path, matches: pd.DataFrame, markets: pd.DataFrame, metrics: pd.DataFrame) -> None:
    """CSV/JSON local para diagnostico; el Excel se genera en la interfaz."""
    root = Path(path)
    root.mkdir(parents=True, exist_ok=True)
    matches.to_csv(root / "partidos.csv", index=False, encoding="utf-8-sig")
    markets.to_csv(root / "mercados.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(root / "validacion.csv", index=False, encoding="utf-8-sig")
    (root / "meta.json").write_text(
        json.dumps({"version": VERSION, "generated_at": datetime.now(base.TZ_PERU).isoformat()}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

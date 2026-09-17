"""Real-result audit for Gatuno V15 Clean.

Principles:
- Freeze every prediction before kickoff.
- Resolve only with observable post-match data.
- Never invent a result when the source is incomplete.
- Keep a criterion snapshot so later adjustments do not rewrite history.
"""
from __future__ import annotations
import json, math
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
import numpy as np

DATA_DIR = Path("app_data_v15_1")
HISTORY_FILE = DATA_DIR / "prediction_history.csv"
POLICY_LOG = DATA_DIR / "policy_decisions.csv"

def _ensure():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

def _load():
    _ensure()
    if not HISTORY_FILE.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(HISTORY_FILE)
    except Exception:
        return pd.DataFrame()

def load_history():
    return _load()

def _actual_from_summary(row, summary):
    code = str(row.get("MercadoCodigo",""))
    pred = str(row.get("PronosticoCodigo",""))
    comp = (summary.get("header") or {}).get("competitions") or []
    competitors = (comp[0].get("competitors") if comp else []) or []
    home_id, away_id = str(row.get("HomeESPNID","")), str(row.get("AwayESPNID",""))
    hg = ag = None
    h1g = a1g = None
    for c in competitors:
        tid = str((c.get("team") or {}).get("id",""))
        score = c.get("score")
        try:
            score = float(score)
        except Exception:
            score = None
        if tid == home_id: hg = score
        if tid == away_id: ag = score
        ls = c.get("linescores") or []
        if ls:
            try:
                first = float(ls[0].get("value"))
            except Exception:
                first = None
            if tid == home_id: h1g = first
            if tid == away_id: a1g = first

    if code == "RESULT_1X2":
        if hg is None or ag is None: return None, "DATOS_INCOMPLETOS"
        actual = "HOME" if hg > ag else "DRAW" if hg == ag else "AWAY"
        return actual == pred, actual

    if code == "H1_GOALS_OU15":
        if h1g is None or a1g is None: return None, "1T_NO_DISPONIBLE"
        total = h1g + a1g
        actual = "OVER" if total > 1.5 else "UNDER"
        return actual == pred, actual

    if code in {"HOME_SCORE_OU05","AWAY_SCORE_OU05"}:
        goals = hg if code == "HOME_SCORE_OU05" else ag
        if goals is None: return None, "DATOS_INCOMPLETOS"
        actual = "YES" if goals >= 1 else "NO"
        return actual == pred, actual

    if code == "YELLOW_CARDS_OU45":
        teams = (summary.get("boxscore") or {}).get("teams") or []
        total = 0.0
        found = False
        for t in teams:
            stats = t.get("statistics") or []
            for s in stats:
                name = str(s.get("name","")).lower()
                if name in {"yellowcards","yellow cards"}:
                    try:
                        total += float(s.get("displayValue", s.get("value")))
                        found = True
                    except Exception:
                        pass
        if not found: return None, "TARJETAS_NO_DISPONIBLES"
        actual = "OVER" if total > 4.5 else "UNDER"
        return actual == pred, actual

    # 1T corners require trustworthy event-level 1T coverage.
    if code == "H1_CORNERS_OU45":
        return None, "CORNERS_1T_NO_DISPONIBLE"

    return None, "MERCADO_NO_RECONOCIDO"

def _append(rows):
    _ensure()
    df = pd.DataFrame(rows)
    if HISTORY_FILE.exists():
        old = _load()
        df = pd.concat([old, df], ignore_index=True)
    if not df.empty:
        keys = [c for c in ["EventID","MercadoCodigo","VersionPronostico"] if c in df]
        if keys:
            df = df.drop_duplicates(keys, keep="last")
        df.to_csv(HISTORY_FILE, index=False, encoding="utf-8-sig")
    return df

def record_predictions(markets):
    if markets is None or markets.empty:
        return load_history(), {"insertados": 0}
    old = load_history()
    existing = set()
    if not old.empty:
        existing = set(zip(old.get("EventID",""), old.get("MercadoCodigo",""), old.get("VersionPronostico","")))
    rows = []
    now = datetime.now(timezone.utc).isoformat()
    for _, r in markets.iterrows():
        kickoff = str(r.get("KickoffUTC",""))
        if not kickoff: continue
        key = (str(r.get("EventID","")), str(r.get("MercadoCodigo","")), str(r.get("Version","")))
        if key in existing: continue
        if str(r.get("PronosticoCodigo","ABSTAIN")) == "ABSTAIN": continue
        rows.append({
            "EventID": str(r.get("EventID","")),
            "KickoffUTC": kickoff,
            "Fecha": r.get("Fecha",""),
            "Competicion": r.get("Competicion",""),
            "CompKey": r.get("CompKey",""),
            "Local": r.get("Local",""),
            "Visitante": r.get("Visitante",""),
            "HomeESPNID": str(r.get("HomeESPNID","")),
            "AwayESPNID": str(r.get("AwayESPNID","")),
            "Mercado": r.get("Mercado",""),
            "MercadoCodigo": r.get("MercadoCodigo",""),
            "Pronostico": r.get("Pronostico",""),
            "PronosticoCodigo": r.get("PronosticoCodigo",""),
            "Probabilidad": r.get("Probabilidad",np.nan),
            "PConservadora": r.get("PConservadora",np.nan),
            "Fiabilidad": r.get("Fiabilidad",np.nan),
            "Soporte": r.get("Soporte",0),
            "SemaforoInicial": r.get("SemaforoFinal",r.get("Semaforo","")),
            "AjusteAplicado": r.get("AjusteAdaptativo","SIN AJUSTE"),
            "VersionPronostico": r.get("Version",""),
            "Estado": "PENDIENTE",
            "Correcto": "",
            "Real": "",
            "ErrorResolucion": "",
            "RegistradoUTC": now,
            "ResueltoUTC": "",
        })
    out = _append(rows)
    return out, {"insertados": len(rows)}

def resolve_pending(max_events=24):
    """Cierra pendientes con una sola llamada por partido.

    V15.0 repetía la descarga del resumen por cada mercado del mismo encuentro;
    seis pronósticos podían provocar seis llamadas idénticas. V15.1 agrupa por
    EventID, limita el trabajo de cada actualización y reutiliza una caché corta.
    """
    import gatuno_data as base
    hist = load_history()
    if hist.empty:
        return hist, {"cerrados":0, "partidos_consultados":0, "pendientes":0, "errores":[]}
    for column in ("Estado", "Real", "ErrorResolucion", "ResueltoUTC"):
        if column not in hist:
            hist[column] = ""
        hist[column] = hist[column].fillna("").astype("object")

    pending = hist[hist["Estado"].astype(str).eq("PENDIENTE")].copy()
    pending["_kickoff"] = pd.to_datetime(pending.get("KickoffUTC"), utc=True, errors="coerce")
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(minutes=20)
    pending = pending[pending["_kickoff"].notna() & (pending["_kickoff"] <= cutoff)]
    pending = pending.sort_values("_kickoff")

    closed, consulted, errors = 0, 0, []
    event_keys = pending[["CompKey", "EventID"]].drop_duplicates().head(int(max_events))
    for _, event in event_keys.iterrows():
        comp_key = str(event.get("CompKey", ""))
        event_id = str(event.get("EventID", ""))
        mask = pending["CompKey"].astype(str).eq(comp_key) & pending["EventID"].astype(str).eq(event_id)
        group = pending[mask]
        if not event_id or event_id.lower() == "nan":
            continue
        league = base.COMPETICIONES.get(comp_key, {}).get("espn")
        if not league:
            continue
        try:
            summary = base.resumen_evento_espn(league, event_id, cache_hours=0.25)
            consulted += 1
            for idx, row in group.iterrows():
                ok, actual = _actual_from_summary(row, summary)
                if ok is None:
                    hist.at[idx, "ErrorResolucion"] = actual
                    continue
                hist.at[idx, "Correcto"] = int(bool(ok))
                hist.at[idx, "Real"] = actual
                hist.at[idx, "Estado"] = "EVALUADO"
                hist.at[idx, "ResueltoUTC"] = datetime.now(timezone.utc).isoformat()
                hist.at[idx, "ErrorResolucion"] = ""
                closed += 1
        except Exception as exc:
            errors.append(f"{event_id}: {exc}")
            for idx in group.index:
                hist.at[idx, "ErrorResolucion"] = str(exc)[:250]

    _ensure()
    hist.to_csv(HISTORY_FILE,index=False,encoding="utf-8-sig")
    return hist, {
        "cerrados":closed,
        "partidos_consultados":consulted,
        "pendientes":int((hist["Estado"]=="PENDIENTE").sum()),
        "errores":errors,
    }

def _lcb(k,n,z=1.96):
    if n <= 0: return np.nan
    p=k/n
    den=1+z*z/n
    centre=(p+z*z/(2*n))/den
    adj=z*math.sqrt((p*(1-p)+z*z/(4*n))/n)/den
    return centre-adj

def audit_summary(history):
    if history is None or history.empty:
        return {"state":"SIN_DATOS","total":0,"pending":0,"evaluated":0,"green_evaluated":0,"green_accuracy":np.nan,"green_lcb95":np.nan,"message":"Aún no hay historial."}, pd.DataFrame()
    h=history.copy()
    evaluated=h[h["Estado"].astype(str).eq("EVALUADO")].copy()
    green=evaluated[evaluated["SemaforoInicial"].astype(str).eq("VERDE")]
    n=len(green); k=int(pd.to_numeric(green["Correcto"],errors="coerce").fillna(0).sum())
    acc=k/n if n else np.nan
    overview={"state":"OPERATIVO" if len(evaluated) else "RECOLECTANDO","total":len(h),"pending":int((h["Estado"]=="PENDIENTE").sum()),"evaluated":len(evaluated),"green_evaluated":n,"green_accuracy":acc,"green_lcb95":_lcb(k,n),"message":"La tasa se calcula solo sobre pronósticos realmente resueltos."}
    detail=evaluated.groupby(["MercadoCodigo","Mercado"],dropna=False).agg(N=("Correcto","size"),Aciertos=("Correcto","sum")).reset_index()
    if not detail.empty: detail["TasaAcierto"]=detail["Aciertos"]/detail["N"]
    return overview,detail

def adaptive_safety_gate(markets, history):
    # Preserve the model's original semaphore. Adaptive policy is applied separately.
    if markets is None or markets.empty: return markets
    out=markets.copy()
    if "SemaforoPreAdaptativo" not in out:
        out["SemaforoPreAdaptativo"]=out.get("SemaforoFinal",out.get("Semaforo","ROJO"))
    if "AjusteAdaptativo" not in out: out["AjusteAdaptativo"]="SIN AJUSTE"
    return out

def sync_future_signals(markets):
    return markets

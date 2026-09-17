"""Adaptive monitor for Gatuno V15.1 Stable.

A 7-day window is treated as an alert sample, not as proof. The system can
propose a bounded temporary adjustment, but the user must confirm it.
"""
from __future__ import annotations
from pathlib import Path
from datetime import datetime, timedelta, timezone
import json
import numpy as np
import pandas as pd

DATA_DIR=Path("app_data_v15_1")
POLICY_FILE=DATA_DIR/"adaptive_policy.json"
PROPOSALS_FILE=DATA_DIR/"adaptive_proposals.csv"
MAX_DAYS=7

DEFAULT_POLICY={"rules":{}}

def _ensure(): DATA_DIR.mkdir(parents=True,exist_ok=True)
def load_policy():
    _ensure()
    if not POLICY_FILE.exists(): return DEFAULT_POLICY.copy()
    try: return json.loads(POLICY_FILE.read_text(encoding="utf-8"))
    except Exception: return DEFAULT_POLICY.copy()
def _save(p):
    _ensure(); POLICY_FILE.write_text(json.dumps(p,ensure_ascii=False,indent=2),encoding="utf-8")

def _market_key(code): return str(code)
def analyze_markets(history):
    if history is None or history.empty: return pd.DataFrame()
    h=history.copy()
    h=h[h["Estado"].astype(str).eq("EVALUADO")].copy()
    h["ResueltoUTC"]=pd.to_datetime(h["ResueltoUTC"],utc=True,errors="coerce")
    cutoff=pd.Timestamp.now(tz="UTC")-pd.Timedelta(days=7)
    recent=h[h["ResueltoUTC"]>=cutoff].copy()
    rows=[]
    for code,g in recent.groupby("MercadoCodigo",dropna=False):
        n=len(g); k=int(pd.to_numeric(g["Correcto"],errors="coerce").fillna(0).sum())
        rate=k/n if n else np.nan
        probs=pd.to_numeric(g["Probabilidad"],errors="coerce")
        expected=float(probs.mean()) if probs.notna().any() else np.nan
        gap=(rate-expected) if np.isfinite(expected) else np.nan
        days=g["ResueltoUTC"].dt.date.nunique()
        # Repeated deterioration requires both sample and temporal repetition.
        if n < 8 or days < 3:
            state="SIN_MUESTRA"
        elif rate < 0.45 and np.isfinite(gap) and gap < -0.10:
            state="CRITICA"
        elif rate < 0.55 and np.isfinite(gap) and gap < -0.06:
            state="ALERTA"
        elif rate < 0.62:
            state="OBSERVAR"
        else:
            state="ESTABLE"
        rows.append({"MercadoCodigo":code,"Mercado":g["Mercado"].iloc[0],"N":n,"Aciertos":k,
                     "TasaAcierto":rate,"ProbMedia":expected,"Brecha":gap,"DiasActivos":days,
                     "Estado":state,"VentanaDias":7})
    return pd.DataFrame(rows).sort_values(["Estado","N"],ascending=[True,False]) if rows else pd.DataFrame()

def load_proposals():
    _ensure()
    if not PROPOSALS_FILE.exists(): return pd.DataFrame()
    try: return pd.read_csv(PROPOSALS_FILE)
    except Exception: return pd.DataFrame()

def _save_proposals(df): _ensure(); df.to_csv(PROPOSALS_FILE,index=False,encoding="utf-8-sig")

def update_proposals(history):
    diag=analyze_markets(history)
    old=load_proposals()
    rows=[]
    now=datetime.now(timezone.utc)
    for _,r in diag.iterrows():
        if r["Estado"] not in {"ALERTA","CRITICA"}: continue
        code=str(r["MercadoCodigo"])
        # Existing pending proposal remains pending; don't spam duplicates.
        exists=False
        if not old.empty:
            exists=bool(((old["MercadoCodigo"].astype(str)==code)&(old["Estado"].astype(str)=="PENDIENTE")).any())
        if exists: continue
        action="REDUCIR_UMBRAL_CONFIANZA_TEMPORAL" if r["Estado"]=="ALERTA" else "PAUSAR_PROMOCION_VERDE_TEMPORAL"
        rows.append({"ProposalID":f"{code}-{now.strftime('%Y%m%d%H%M%S')}",
                     "MercadoCodigo":code,"Mercado":r["Mercado"],"Severidad":r["Estado"],
                     "N":int(r["N"]),"Aciertos":int(r["Aciertos"]),"TasaAcierto":float(r["TasaAcierto"]),
                     "ProbMedia":float(r["ProbMedia"]) if np.isfinite(r["ProbMedia"]) else np.nan,
                     "AccionPropuesta":action,
                     "Motivo":f"7 días: {r['Aciertos']}/{r['N']} aciertos, brecha {r['Brecha']:.1%}; requiere confirmación.",
                     "Estado":"PENDIENTE","Creado":now.isoformat()})
    if rows: old=pd.concat([old,pd.DataFrame(rows)],ignore_index=True)
    _save_proposals(old)
    return diag,old

def decide_proposal(proposal_id, decision):
    p=load_proposals()
    if p.empty:return
    mask=p["ProposalID"].astype(str)==str(proposal_id)
    p.loc[mask,"Estado"]="APLICADO" if decision=="APLICAR" else "OBSERVAR_7_DIAS"
    if decision=="APLICAR":
        code=str(p.loc[mask,"MercadoCodigo"].iloc[0])
        policy=load_policy()
        policy.setdefault("rules",{})[code]={
            "mode":str(p.loc[mask,"AccionPropuesta"].iloc[0]),
            "created":datetime.now(timezone.utc).isoformat(),
            "expires":(datetime.now(timezone.utc)+timedelta(days=7)).isoformat()
        }
        _save(policy)
    _save_proposals(p)

def active_policy_rows():
    policy=load_policy(); rows=[]; now=datetime.now(timezone.utc)
    changed=False
    for code,rule in list(policy.get("rules",{}).items()):
        try: exp=pd.Timestamp(rule["expires"])
        except Exception: exp=pd.Timestamp.now(tz="UTC")
        if exp.tzinfo is None: exp=exp.tz_localize("UTC")
        if exp<=now:
            del policy["rules"][code]; changed=True; continue
        rows.append({"MercadoCodigo":code,"Mercado":code,"Modo":rule.get("mode",""),"Vence":exp.isoformat()})
    if changed:_save(policy)
    return pd.DataFrame(rows)

def revert_market(code):
    p=load_policy(); p.setdefault("rules",{}).pop(str(code),None); _save(p)

def apply_policy(markets, policy):
    if markets is None or markets.empty:return markets
    out=markets.copy()
    if "SemaforoPreAdaptativo" not in out: out["SemaforoPreAdaptativo"]=out.get("SemaforoFinal",out.get("Semaforo","ROJO"))
    out["SemaforoFinal"]=out["SemaforoPreAdaptativo"]; out["Semaforo"]=out["SemaforoPreAdaptativo"]; out["AjusteAdaptativo"]="SIN AJUSTE"
    now=pd.Timestamp.now(tz="UTC")
    for code,rule in policy.get("rules",{}).items():
        mask=out["MercadoCodigo"].astype(str).eq(str(code))
        try: active=pd.Timestamp(rule["expires"])>now
        except Exception: active=False
        if not active: continue
        mode=str(rule.get("mode",""))
        if mode=="PAUSAR_PROMOCION_VERDE_TEMPORAL":
            out.loc[mask & out["SemaforoFinal"].eq("VERDE"),"SemaforoFinal"]="AMARILLO"
            out.loc[mask,"AjusteAdaptativo"]="PAUSA VERDE TEMPORAL"
        elif mode=="REDUCIR_UMBRAL_CONFIANZA_TEMPORAL":
            out.loc[mask & out["SemaforoFinal"].eq("VERDE"),"AjusteAdaptativo"]="UMBRAL REVISADO TEMPORAL"
    return out

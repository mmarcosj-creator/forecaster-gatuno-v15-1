from __future__ import annotations
from datetime import datetime
from io import BytesIO
from pathlib import Path
import html, json, os, time, traceback
import numpy as np
import pandas as pd
import streamlit as st

os.environ.setdefault("GATUNO_FAST_MODE","1")

import gatuno_data as base
import gatuno_forecaster as v10
import gatuno_audit as audit
import gatuno_adaptive as adaptive
import gatuno_quality as quality_monitor

APP_VERSION="V15.1.1-FUENTES-GATUNO"
DATA_DIR=Path("app_data_v15_1"); DATA_DIR.mkdir(exist_ok=True)
MATCH_FILE=DATA_DIR/"latest_matches.csv"; MARKET_FILE=DATA_DIR/"latest_markets.csv"
METRIC_FILE=DATA_DIR/"latest_validation.csv"; META_FILE=DATA_DIR/"meta.json"
RUN_FILE=DATA_DIR/"run_state.json"; METRIC_LOG=DATA_DIR/"model_metrics_log.csv"

st.set_page_config(page_title="Gatuno V15 Clean",page_icon="🐾",layout="wide",initial_sidebar_state="collapsed")

def esc(x): return html.escape(str(x if x is not None else ""))
def atomic_csv(df,p):
    tmp=p.with_suffix(p.suffix+".tmp"); df.to_csv(tmp,index=False,encoding="utf-8-sig"); os.replace(tmp,p)
def atomic_json(x,p):
    tmp=p.with_suffix(p.suffix+".tmp"); tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding="utf-8"); os.replace(tmp,p)
def load_cached():
    return pd.read_csv(MATCH_FILE),pd.read_csv(MARKET_FILE),pd.read_csv(METRIC_FILE)
def filter_future(df):
    if df.empty or "KickoffUTC" not in df:return df
    k=pd.to_datetime(df["KickoffUTC"],utc=True,errors="coerce")
    return df[k.notna()&(k>pd.Timestamp.now(tz="UTC"))].reset_index(drop=True)

def refresh():
    started=time.monotonic()
    hist,closure=audit.resolve_pending()
    matches,markets,metrics,start,end=v10.run_v10(fast_mode=True,progress=lambda x: st.write("🐾",x))
    matches=filter_future(matches); markets=filter_future(markets)
    markets=audit.adaptive_safety_gate(markets,hist)
    markets=adaptive.apply_policy(markets,adaptive.load_policy())
    hist,rec=audit.record_predictions(markets)
    adaptive.update_proposals(hist)
    atomic_csv(matches,MATCH_FILE); atomic_csv(markets,MARKET_FILE); atomic_csv(metrics,METRIC_FILE)
    try: quality_monitor.registrar_metricas(metrics,METRIC_LOG)
    except Exception: pass
    atomic_json({"version":APP_VERSION,"generated_at":datetime.now(base.TZ_PERU).isoformat(),
                 "matches":len(matches),"markets":len(markets),"closed":closure.get("cerrados",0),
                 "inserted":rec.get("insertados",0),
                 "duration_seconds":round(time.monotonic()-started,1)},META_FILE)
    return matches,markets,metrics,hist

def signal(x):
    return {"VERDE":"🟢","AMARILLO":"🟡","ROJO":"🔴"}.get(str(x).upper(),"⚪")

st.markdown(f"# 🐾 Gatuno V15.1.1 Estable\n**Motor limpio, recuperable y auditable · {APP_VERSION}**")
st.caption("Arquitectura nueva: pronóstico → congelado → cierre real → diagnóstico 7 días → propuesta → confirmación → ajuste temporal reversible.")

meta=json.loads(META_FILE.read_text(encoding="utf-8")) if META_FILE.exists() else {}
ready=MATCH_FILE.exists() and MARKET_FILE.exists() and METRIC_FILE.exists() and meta.get("version")==APP_VERSION

if not ready:
    st.info(
        "🐾 Primera ejecución limpia. Primero comprobará el calendario y sólo después calibrará "
        "el motor. La primera calibración puede tardar varios minutos; las siguientes reutilizan "
        "la caché mientras el histórico no cambie."
    )
    if st.button("🐾 GENERAR PRONÓSTICOS V15.1.1",type="primary",use_container_width=True):
        try:
            with st.status("Construyendo motor limpio…",expanded=True):
                refresh()
            st.success("Motor V15.1.1 listo.")
            st.rerun()
        except Exception as e:
            message=str(e)
            if "calendario" in message.lower() or "partidos futuros" in message.lower():
                st.error("Las tres fuentes del calendario no entregaron partidos. No se perdió ningún modelo ni historial.")
                st.warning("V15.1.1 consultó ESPN, Football-Data, Sofascore y la caché antes de detenerse.")
            else:
                st.error("El motor no terminó.")
            st.code(message)
            with st.expander("Detalle técnico"):
                st.code(traceback.format_exc())
    st.stop()

try:
    matches,markets,metrics=load_cached()
    hist=audit.load_history()
except Exception:
    st.warning("La caché de V15 no pudo leerse. Pulsa actualizar.")
    if st.button("🔄 Reconstruir V15",type="primary"): refresh(); st.rerun()
    st.stop()

duration=meta.get("duration_seconds")
if duration is not None:
    st.caption(
        f"⏱️ Última actualización: {float(duration):.1f} s. "
        "La próxima reutilizará el modelo mientras no cambie el histórico."
    )

overview,detail=audit.audit_summary(hist)
acc=overview.get("green_accuracy")
st.markdown(f"### 📊 Auditoría real: **{overview.get('state')}**")
c1,c2,c3,c4=st.columns(4)
c1.metric("Pronósticos congelados",overview.get("total",0))
c2.metric("Evaluados",overview.get("evaluated",0))
c3.metric("Verdes evaluados",overview.get("green_evaluated",0))
c4.metric("Acierto verde","—" if not np.isfinite(acc) else f"{acc:.1%}")

diag,props=adaptive.update_proposals(hist)
st.markdown("### 🐱🔎 Vigilante Gatuno · ventana de 7 días")
st.caption("La semana es una señal de deterioro, no una sentencia estadística. El sistema no cambia criterios en silencio.")

if diag.empty:
    st.info("🐾 Recolectando resultados suficientes por mercado.")
else:
    st.dataframe(diag[["Mercado","N","Aciertos","TasaAcierto","ProbMedia","Brecha","DiasActivos","Estado"]],use_container_width=True,hide_index=True)

pending=props[props["Estado"].astype(str).eq("PENDIENTE")] if not props.empty else pd.DataFrame()
if not pending.empty:
    st.warning("🙀 Hay propuestas de ajuste pendientes de confirmación.")
    for _,p in pending.iterrows():
        st.markdown(f"**🐾 {p['Mercado']} · {p['Severidad']}** — {int(p['Aciertos'])}/{int(p['N'])} · propuesta: `{p['AccionPropuesta']}`")
        st.caption(str(p["Motivo"]))
        a,b=st.columns(2)
        if a.button("🐾 Aplicar ajuste temporal",key="apply_"+str(p["ProposalID"])):
            adaptive.decide_proposal(p["ProposalID"],"APLICAR"); st.rerun()
        if b.button("🔎 Observar 7 días más",key="watch_"+str(p["ProposalID"])):
            adaptive.decide_proposal(p["ProposalID"],"OBSERVAR_7_DIAS"); st.rerun()

active=adaptive.active_policy_rows()
if not active.empty:
    st.markdown("### 😺 Ajustes temporales activos")
    for _,r in active.iterrows():
        a,b=st.columns([4,1]); a.write(f"**{r['MercadoCodigo']}** · {r['Modo']} · vence {r['Vence']}")
        if b.button("↩️ Revertir",key="rev_"+str(r["MercadoCodigo"])):
            adaptive.revert_market(r["MercadoCodigo"]); st.rerun()

st.markdown("### ⚽ Pronósticos próximos")
for event,group in markets.groupby("EventID",sort=False):
    first=group.iloc[0]
    st.markdown(f"**{esc(first.get('Local',''))} vs {esc(first.get('Visitante',''))}** · {esc(first.get('Competicion',''))} · {esc(first.get('HoraPeru',''))} PET")
    for _,r in group.iterrows():
        s=str(r.get("SemaforoFinal",r.get("Semaforo","ROJO")))
        note=str(r.get("AjusteAdaptativo",""))
        st.write(f"{signal(s)} **{r.get('Mercado','')}** → **{r.get('Pronostico','')}** · P={float(r.get('Probabilidad',np.nan)):.1%} · {note}")
    st.divider()

if st.button("🔄 Actualizar, cerrar resultados y auditar",use_container_width=True):
    try:
        refresh(); st.success("🐾 Actualización completada."); st.rerun()
    except Exception as e:
        st.error(str(e)); st.code(traceback.format_exc())

st.caption("Sistema experimental: las tasas se calculan con resultados observados y no garantizan aciertos ni rentabilidad. No se persiguen pérdidas.")

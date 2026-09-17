"""Context layer for Gatuno V15 Clean.

Design goal:
- No hard dependency on the old V9.3 context package.
- Lineups are informational only; they never block a signal.
- No synthetic lineup/player data is generated.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import pandas as pd

def parse_confirmed_lineups(summary: dict | None, home_id="", away_id="") -> dict[str, Any]:
    # We deliberately do not fabricate starters when the source does not expose them.
    return {
        "status": "NO_DISPONIBLE",
        "home_starters": [],
        "away_starters": [],
        "note": "Alineación no disponible; no bloquea la señal.",
    }

def load_lineup_history(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []

def archive_confirmed_lineups(path, fixture, lineup) -> None:
    # Only persist genuinely confirmed lineups. Current adapter normally receives none.
    if str(lineup.get("status")) != "CONFIRMADA":
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    rows = load_lineup_history(p)
    rows.append({
        "event_id": str(fixture.get("EventID", "")),
        "date": str(fixture.get("Date", "")),
        "home": str(fixture.get("HomeOriginal", fixture.get("HomeTeam", ""))),
        "away": str(fixture.get("AwayOriginal", fixture.get("AwayTeam", ""))),
        "home_starters": list(lineup.get("home_starters", [])),
        "away_starters": list(lineup.get("away_starters", [])),
    })
    p.write_text(json.dumps(rows[-1000:], ensure_ascii=False, indent=2), encoding="utf-8")

def lineup_strength_profile(history, team, starters, date, event_id="") -> dict[str, Any]:
    # No player ratings are invented. The profile is neutral until real historical
    # lineup observations exist.
    return {"available": False, "strength": 0.0, "n": 0}

def build_calendar_index(history: pd.DataFrame, fixtures: pd.DataFrame) -> dict[str, Any]:
    # Calendar pressure is already handled by the base workload features.
    return {}

def evaluate_fixture_context(
    fixture, home, away, home_load, away_load, venue, calendar_index,
    lineup_status="NO_DISPONIBLE", lineup_home=None, lineup_away=None
) -> dict[str, Any]:
    # Neutral context: workload remains a feature in the forecaster itself.
    return {
        "context_factor": 1.0,
        "importance": 0.5,
        "stage": "",
        "rotation_risk": "",
        "lineup_status": str(lineup_status or "NO_DISPONIBLE"),
        "note": "Contexto limpio: alineación informativa, sin bloqueo.",
    }

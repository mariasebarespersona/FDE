# app.py  — HappyRobots FDE Metrics API
# Run:  uvicorn app:app --host 0.0.0.0 --port 8081 --reload
# Env:  API_KEY=devkey123  METRICS_DB=metrics.db

from fastapi import FastAPI, Request, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone
import sqlite3, os, json


from fastapi.middleware.cors import CORSMiddleware




API_KEY = os.environ.get("API_KEY", "devkey123")
DB_PATH = os.environ.get("METRICS_DB", "metrics.db")
TABLE   = "events"

app = FastAPI(title="FDE Metrics API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # or lock down to your Streamlit URL later
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# ---------- SQLite helpers ----------

SCHEMA_COLS = [
    ("id", "INTEGER PRIMARY KEY AUTOINCREMENT"),
    ("ts", "TEXT NOT NULL"),
    ("run_id", "TEXT"),
    ("call_id", "TEXT"),
    ("stage", "TEXT"),
    ("outcome", "TEXT"),
    ("carrier_mc", "TEXT"),
    ("fmcsa_status", "TEXT"),
    ("load_id", "TEXT"),
    ("origin", "TEXT"),
    ("destination", "TEXT"),
    ("equipment_type", "TEXT"),
    ("loadboard_rate", "REAL"),
    ("model_offer", "REAL"),
    ("carrier_counter", "INTEGER"),
    ("agreed_rate", "REAL"),
    ("discount_pct", "REAL"),
    ("negotiation_rounds", "INTEGER"),
    ("sentiment", "TEXT"),
    ("sentiment_score", "REAL"),
    ("duration_sec", "REAL"),
    ("transferred_to_rep", "INTEGER"),
    ("error", "TEXT"),
]

COL_NAMES = [c for c, _ in SCHEMA_COLS]

def connect():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def ensure_table():
    with connect() as con:
        cols_sql = ", ".join([f"{c} {t}" for c, t in SCHEMA_COLS])
        con.execute(f"CREATE TABLE IF NOT EXISTS {TABLE} ({cols_sql});")
ensure_table()

def rows_to_dicts(rows, cols):
    return [dict(zip(cols, r)) for r in rows]

def coerce_num(v, typ="float"):
    if v is None or v == "":
        return None
    try:
        return float(v) if typ == "float" else int(v)
    except Exception:
        return None

def normalize_event(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Map arbitrary webhook JSON into our table columns, preserving types where possible.
    """
    # base row with None defaults
    row = {c: None for c in COL_NAMES}
    # ISO timestamp (UTC) for ts
    row["ts"] = datetime.now(timezone.utc).isoformat()

    # direct string-ish fields (coerce to str, trim empties)
    for key in ["run_id","call_id","stage","outcome","carrier_mc","fmcsa_status",
                "load_id","origin","destination","equipment_type","sentiment","error"]:
        v = payload.get(key)
        if v is not None:
            s = str(v).strip()
            row[key] = s if s and s.lower() not in ("none","null","nan") else None

    # numeric fields (float)
    row["loadboard_rate"]     = coerce_num(payload.get("loadboard_rate"), "float")
    row["model_offer"]        = coerce_num(payload.get("model_offer"), "float")
    row["agreed_rate"]        = coerce_num(payload.get("agreed_rate"), "float")
    row["discount_pct"]       = coerce_num(payload.get("discount_pct"), "float")
    row["sentiment_score"]    = coerce_num(payload.get("sentiment_score"), "float")
    row["duration_sec"]       = coerce_num(payload.get("duration_sec"), "float")

    # numeric fields (int)
    row["carrier_counter"]      = coerce_num(payload.get("carrier_counter"), "int")
    row["negotiation_rounds"]   = coerce_num(payload.get("negotiation_rounds"), "int")

    # booleans -> ints
    def as_bool(x):
        if isinstance(x, bool): return int(x)
        if isinstance(x, (int, float)) and x in (0,1): return int(x)
        if isinstance(x, str):
            return 1 if x.strip().lower() in ("true","yes","y","1") else 0
        return None

    row["transferred_to_rep"] = as_bool(payload.get("transferred_to_rep"))

    # light canonicalization for downstream logic
    if row["stage"]:
        row["stage"] = row["stage"].strip()
    if row["outcome"]:
        row["outcome"] = row["outcome"].strip().lower()

    # if equipment is present in different keys, prefer them
    if not row.get("equipment_type"):
        for alt in ("equipment", "equipmentType", "equipment_type_norm", "data_equipment_type"):
            v = payload.get(alt)
            if v:
                row["equipment_type"] = str(v).strip()
                break

    return row

# ---------- Routes ----------

@app.get("/healthz")
def healthz():
    return {"ok": True, "ts": datetime.now(timezone.utc).isoformat()}

@app.post("/events")
async def ingest(request: Request, x_api_key: Optional[str] = Header(default=None)):
    # simple header auth
    if (x_api_key or "").strip() != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    # allow envelope {"event": {...}} or raw object
    if isinstance(payload, dict) and "event" in payload and isinstance(payload["event"], dict):
        event = payload["event"]
    elif isinstance(payload, dict):
        event = payload
    else:
        raise HTTPException(status_code=400, detail="JSON object expected")

    row = normalize_event(event)

    cols = [c for c in COL_NAMES if c != "id"]
    ph   = ",".join(["?"] * len(cols))
    sql  = f"INSERT INTO {TABLE} ({','.join(cols)}) VALUES ({ph})"

    with connect() as con:
        con.execute(sql, [row[c] for c in cols])
        con.commit()

    return {"ok": True}

@app.get("/events")
def list_events(
    limit: int = Query(2000, ge=1, le=10000),
    since: Optional[str] = Query(None, description="ISO timestamp, UTC"),
    stage: Optional[str] = Query(None),
    outcome: Optional[str] = Query(None),
    equipment: Optional[str] = Query(None, description="equipment_type exact match"),
    run_id: Optional[str] = Query(None),
    order: str = Query("desc", pattern="^(asc|desc)$"),
):
    where: List[str] = []
    params: List[Any] = []

    if since:
        where.append("ts >= ?")
        params.append(since)
    if stage:
        where.append("stage = ?")
        params.append(stage)
    if outcome:
        where.append("outcome = ?")
        params.append(outcome.lower())
    if equipment:
        where.append("equipment_type = ?")
        params.append(equipment)
    if run_id:
        where.append("run_id = ?")
        params.append(run_id)

    sql = f"SELECT * FROM {TABLE}"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += f" ORDER BY ts {'ASC' if order=='asc' else 'DESC'} LIMIT ?"
    params.append(limit)

    with connect() as con:
        cur = con.execute(sql, params)
        cols = [c[0] for c in cur.description]
        rows = cur.fetchall()

    return JSONResponse(content=rows_to_dicts(rows, cols))

# convenience to wipe data during dev (protect with API key)
@app.post("/reset")
def reset(x_api_key: Optional[str] = Header(default=None)):
    if (x_api_key or "").strip() != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")
    with connect() as con:
        con.execute(f"DELETE FROM {TABLE}")
        con.commit()
    return {"ok": True, "cleared": True}

# dashboard.py
# Streamlit dashboard for HappyRobots FDE Metrics
# - Public-friendly: works on Streamlit Cloud / Railway
# - Robust normalization: merges canonical columns with your ls_*/nr_* fallbacks
# - Clear KPIs, accepted-deal table, outcomes by equipment, raw events, spotlight

import os
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests
import streamlit as st

# --------------------------- Config ---------------------------

# Prefer Streamlit secrets; then env; fallback to your Railway URL
API_BASE = (
    st.secrets.get("API_BASE")
    or os.getenv("API_BASE")
    or "https://fde-production.up.railway.app"
).rstrip("/")

API_KEY = st.secrets.get("API_KEY") or os.getenv("API_KEY") or ""
API_EVENTS = f"{API_BASE}/events"
API_HEALTH = f"{API_BASE}/healthz"
HEADERS = {"X-API-KEY": API_KEY} if API_KEY else {}

st.set_page_config(
    page_title="FDE Use-Case Metrics",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --------------------------- Helpers --------------------------

def _to_ts(x):
    try:
        return pd.to_datetime(x, utc=True)
    except Exception:
        return pd.NaT

@st.cache_data(show_spinner=False, ttl=5)
def ping_health() -> bool:
    try:
        r = requests.get(API_HEALTH, headers=HEADERS, timeout=4)
        r.raise_for_status()
        data = r.json()
        return bool(data.get("ok", False))
    except Exception:
        return False

@st.cache_data(show_spinner=False, ttl=5)
def fetch_events() -> pd.DataFrame:
    """
    Expected canonical schema (if present):
      id, ts, run_id, call_id, stage, outcome, carrier_mc, fmcsa_status,
      load_id, origin, destination, equipment_type, loadboard_rate, model_offer,
      carrier_counter, agreed_rate, discount_pct, negotiation_rounds,
      sentiment, sentiment_score, duration_sec, transferred_to_rep, error

    We also merge common alternates produced by your nodes:
      ls_origin, ls_destination, ls_equipment_type, ls_loadboard_rate
      nr_outcome, nr_agreed_rate, nr_model_offer, nr_discount_pct, nr_negotiation_rounds
    """
    try:
        r = requests.get(API_EVENTS, headers=HEADERS, timeout=8)
        r.raise_for_status()
        data = r.json()
    except Exception:
        data = []

    df = pd.DataFrame(data)
    if df.empty:
        return pd.DataFrame(columns=[
            "id","ts","run_id","call_id","stage","outcome","carrier_mc","fmcsa_status",
            "load_id","origin","destination","equipment_type","loadboard_rate","model_offer",
            "carrier_counter","agreed_rate","discount_pct","negotiation_rounds",
            "sentiment","sentiment_score","duration_sec","transferred_to_rep","error"
        ])

    # --- Normalize timestamps & numerics
    if "ts" not in df.columns:
        df["ts"] = pd.NaT
    df["ts"] = df["ts"].apply(_to_ts)

    for col in [
        "loadboard_rate","model_offer","agreed_rate","discount_pct",
        "negotiation_rounds","sentiment_score","duration_sec",
        "carrier_counter",
        # num versions of ls_/nr_ fields (in case API typed them as strings)
        "ls_loadboard_rate","nr_agreed_rate","nr_model_offer","nr_discount_pct","nr_negotiation_rounds"
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # --- String hygiene
    for col in [
        "stage","outcome","equipment_type","origin","destination",
        "sentiment","fmcsa_status","carrier_mc","ls_origin","ls_destination",
        "ls_equipment_type","nr_outcome"
    ]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            df.loc[df[col].str.lower().isin(["", "none", "null", "nan"]), col] = np.nan

    # --- Merge fallbacks into canonical columns
    # origin/destination/equipment
    if "origin" not in df.columns: df["origin"] = np.nan
    if "destination" not in df.columns: df["destination"] = np.nan
    if "equipment_type" not in df.columns: df["equipment_type"] = np.nan
    if "loadboard_rate" not in df.columns: df["loadboard_rate"] = np.nan
    if "agreed_rate" not in df.columns: df["agreed_rate"] = np.nan
    if "model_offer" not in df.columns: df["model_offer"] = np.nan
    if "discount_pct" not in df.columns: df["discount_pct"] = np.nan
    if "negotiation_rounds" not in df.columns: df["negotiation_rounds"] = np.nan
    if "outcome" not in df.columns: df["outcome"] = np.nan

    # Prefer canonical values; if NaN, take ls_/nr_ where appropriate
    df["origin"]           = df["origin"].fillna(df.get("ls_origin"))
    df["destination"]      = df["destination"].fillna(df.get("ls_destination"))
    df["equipment_type"]   = df["equipment_type"].fillna(df.get("ls_equipment_type"))
    df["loadboard_rate"]   = df["loadboard_rate"].fillna(df.get("ls_loadboard_rate"))

    df["agreed_rate"]          = df["agreed_rate"].fillna(df.get("nr_agreed_rate"))
    df["model_offer"]          = df["model_offer"].fillna(df.get("nr_model_offer"))
    df["discount_pct"]         = df["discount_pct"].fillna(df.get("nr_discount_pct"))
    df["negotiation_rounds"]   = df["negotiation_rounds"].fillna(df.get("nr_negotiation_rounds"))

    # Outcome normalization: if canonical missing, adopt nr_outcome
    df["outcome"] = df["outcome"].fillna(df.get("nr_outcome"))

    # Equipment display (title-case for grouping)
    df["equipment_display"] = (
        df["equipment_type"].fillna("").astype(str).str.strip().str.title()
    )
    df.loc[df["equipment_display"] == "", "equipment_display"] = np.nan

    # Derived flags
    df["stage"] = df["stage"].fillna("")
    low_outcome = df["outcome"].fillna("").str.lower()

    df["is_fmcsa_active"]  = (df["stage"].eq("fmcsa_verify")  & (low_outcome.eq("active")))
    df["is_load_selected"] = (df["stage"].eq("load_selected"))
    df["is_negotiate"]     = (df["stage"].eq("negotiate_result"))

    # "Accepted" if outcome says so OR we have an agreed_rate > 0
    df["is_accepted"] = df["is_negotiate"] & (
        low_outcome.isin(["accepted","success","won","deal"]) | (pd.to_numeric(df["agreed_rate"], errors="coerce") > 0)
    )

    return df.sort_values("ts", ascending=True).reset_index(drop=True)

def now_utc():
    return datetime.now(timezone.utc)

# --------------------------- UI -------------------------------

st.title("FDE Use-Case Metrics")

ok = ping_health()
st.caption(("🟢 API connected" if ok else "🔴 API unreachable") + f" · {API_BASE}")

# Time window
tw_options = {"Last 24h": timedelta(days=1), "Last 7d": timedelta(days=7), "Last 30d": timedelta(days=30), "All time": None}
c1, c2, c3 = st.columns([1.2, 1.2, 1.2])

with c1:
    tw_key = st.selectbox("Time window", list(tw_options.keys()), index=0)

df_all = fetch_events()
df = df_all.copy()
if tw_options[tw_key]:
    start = now_utc() - tw_options[tw_key]
    df = df[df["ts"] >= start]

# Filters
equip_opts = ["All"] + sorted([e for e in df_all["equipment_display"].dropna().unique().tolist()])
neg_opts_raw = (
    df_all.loc[df_all["is_negotiate"], "outcome"]
    .dropna().str.title().unique().tolist()
)
neg_opts = ["All"] + sorted(neg_opts_raw)

with c2:
    equip_sel = st.multiselect("Equipment type", equip_opts, default=["All"])
with c3:
    neg_sel = st.multiselect("Negotiation outcome", neg_opts, default=["All"])

if "All" not in equip_sel:
    df = df[df["equipment_display"].isin(equip_sel)]
if "All" not in neg_sel:
    df = df[df["is_negotiate"] & df["outcome"].str.title().isin(neg_sel)]

# --------------------------- KPIs -----------------------------

total_calls    = df["run_id"].nunique()
verified       = df.loc[df["is_fmcsa_active"], "run_id"].nunique()
loads_pitched  = int(df.loc[df["is_load_selected"]].shape[0])
agreements     = int(df.loc[df["is_accepted"]].shape[0])

neg_rows = df.loc[df["is_negotiate"]]
win_rate = (agreements / max(1, len(neg_rows))) * 100.0

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total Calls", total_calls)
k2.metric("Verified (FMCSA active)", verified)
k3.metric("Loads Pitched", loads_pitched)
k4.metric("Agreements", agreements)
k5.metric("Win Rate", f"{win_rate:.0f}%")

st.markdown("### Pricing")
acc = df.loc[df["is_accepted"]]
avg_discount = float(acc["discount_pct"].dropna().mean()) if not acc.empty else 0.0
st.caption(f"Avg discount vs loadboard: **{avg_discount:.1f}%**")

# -------------------- Sentiment & Rounds ----------------------

st.markdown("### Sentiment & Rounds")

have_sent_cols = any(c in df.columns for c in ["sentiment", "sentiment_score"])
if have_sent_cols and not df.empty:
    last_by_run = (
        df.sort_values("ts")
          .groupby("run_id")
          .tail(1)[["run_id","outcome","sentiment","sentiment_score","duration_sec"]]
          .reset_index(drop=True)
    )
    last_by_run["sentiment"] = last_by_run["sentiment"].fillna("UNKNOWN")
    last_by_run["sentiment_score"] = pd.to_numeric(last_by_run["sentiment_score"], errors="coerce").fillna(0.5)
    last_by_run["duration_sec"] = pd.to_numeric(last_by_run["duration_sec"], errors="coerce").fillna(0).astype(int)
    st.dataframe(last_by_run, use_container_width=True, hide_index=True)
else:
    st.info("No sentiment rows yet.")

# -------------------- Agreed vs Loadboard ---------------------

st.markdown("### Agreed vs Loadboard (accepted only)")
cols = ["run_id","equipment_display","origin","destination","loadboard_rate","agreed_rate","discount_pct","negotiation_rounds","ts"]
acc = df.loc[df["is_accepted"], cols].copy()
if acc.empty:
    st.info("No accepted deals yet.")
else:
    acc = acc.sort_values("ts", ascending=False)
    acc["equipment_display"]   = acc["equipment_display"].fillna("Unknown")
    acc["negotiation_rounds"]  = pd.to_numeric(acc["negotiation_rounds"], errors="coerce").fillna(0).astype(int)
    acc["discount_pct"]        = pd.to_numeric(acc["discount_pct"], errors="coerce").fillna(0).round(1)
    st.dataframe(acc.drop(columns=["ts"]), use_container_width=True, hide_index=True)

# -------------------- Outcomes by Equipment -------------------

st.markdown("### Outcomes by Equipment")
roll = (
    df.loc[df["is_negotiate"]]
      .assign(out=df["outcome"].fillna("Unknown").str.title())
      .groupby(["equipment_display","out"], dropna=False)
      .size().rename("count").reset_index()
)
if roll.empty:
    st.caption("No negotiation outcomes yet for the selected window/filters.")
else:
    pivot = roll.pivot_table(index="equipment_display", columns="out", values="count", fill_value=0)
    st.dataframe(pivot, use_container_width=True)

# -------------------- Raw Events ------------------------------

st.markdown("### Raw Events")
show_cols = [
    "id","ts","run_id","call_id","stage","outcome","carrier_mc","fmcsa_status",
    "load_id","origin","destination","equipment_type","loadboard_rate","model_offer",
    "carrier_counter","agreed_rate","discount_pct","negotiation_rounds",
    # show your fallbacks so you can debug logging quickly
    "ls_origin","ls_destination","ls_equipment_type","ls_loadboard_rate",
    "nr_outcome","nr_agreed_rate","nr_model_offer","nr_discount_pct","nr_negotiation_rounds",
]
present = [c for c in show_cols if c in df.columns]
st.dataframe(df[present].sort_values("ts", ascending=False), use_container_width=True)

# -------------------- Latest Call Spotlight -------------------

st.markdown("---")
st.markdown("#### Latest Call (spotlight)")
if df.empty:
    st.caption("No events yet.")
else:
    last_run = df.sort_values("ts").iloc[-1]["run_id"]
    spot = df[df["run_id"] == last_run].copy()

    equip = spot["equipment_display"].dropna().tail(1).tolist()
    equip_str = equip[-1] if equip else "Unknown"
    o = spot["origin"].dropna().tail(1).tolist()
    d = spot["destination"].dropna().tail(1).tolist()
    route = f"{(o[-1] if o else 'None')} → {(d[-1] if d else 'None')}"

    has_pitch = bool(spot["is_load_selected"].any())
    has_neg   = bool(spot["is_negotiate"].any())
    accepted  = bool(spot["is_accepted"].any())

    st.write(
        f"**Run:** `{last_run}` | **Equipment:** {equip_str} | **From→To:** {route} | "
        f"**Pitched:** {'✅' if has_pitch else '—'} | "
        f"**Negotiated:** {'✅' if has_neg else '—'} | "
        f"**Accepted:** {'✅' if accepted else '—'}"
    )

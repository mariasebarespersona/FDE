# dashboard.py
# Streamlit dashboard for HappyRobots FDE Metrics
# - Pulls raw events from your metrics API (FastAPI) at METRICS_URL
# - Computes KPIs + per-run sentiment/rounds
# - Robust filters (time window, equipment, negotiation outcome)
# - Shows your latest accepted call clearly

import os
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests
import streamlit as st

# --------------------------- Config ---------------------------

API_BASE = os.environ.get("METRICS_URL", "http://localhost:8081").rstrip("/")
API_EVENTS = f"{API_BASE}/events"          # GET returns JSON list of events rows
API_HEALTH = f"{API_BASE}/healthz"

st.set_page_config(
    page_title="FDE Use-Case Metrics",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --------------------------- Helpers --------------------------

@st.cache_data(show_spinner=False, ttl=5)
def ping_health():
    try:
        r = requests.get(API_HEALTH, timeout=3)
        r.raise_for_status()
        data = r.json()
        return bool(data.get("ok", False))
    except Exception:
        return False

@st.cache_data(show_spinner=False, ttl=5)
def fetch_events() -> pd.DataFrame:
    """
    Expected schema from API:
      id, ts, run_id, call_id, stage, outcome, carrier_mc, fmcsa_status,
      load_id, origin, destination, equipment_type, loadboard_rate, model_offer,
      carrier_counter, agreed_rate, discount_pct, negotiation_rounds,
      sentiment, sentiment_score, duration_sec, transferred_to_rep, error
    Unknown/missing fields are OK; we normalize below.
    """
    r = requests.get(API_EVENTS, timeout=8)
    r.raise_for_status()
    data = r.json()
    if not isinstance(data, list):
        data = []

    df = pd.DataFrame(data)
    if df.empty:
        # Ensure downstream code has columns
        df = pd.DataFrame(columns=[
            "id","ts","run_id","call_id","stage","outcome","carrier_mc",
            "fmcsa_status","load_id","origin","destination","equipment_type",
            "loadboard_rate","model_offer","carrier_counter","agreed_rate",
            "discount_pct","negotiation_rounds","sentiment","sentiment_score",
            "duration_sec","transferred_to_rep","error"
        ])

    # Normalize dtypes
    def to_ts(x):
        try:
            return pd.to_datetime(x, utc=True)
        except Exception:
            return pd.NaT

    df["ts"] = df["ts"].apply(to_ts)
    for col in ["loadboard_rate","model_offer","agreed_rate","discount_pct",
                "negotiation_rounds","sentiment_score","duration_sec"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Strings; trim/standardize empties
    for col in ["stage","outcome","equipment_type","origin","destination",
                "sentiment","fmcsa_status","carrier_mc"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            df.loc[df[col].isin(["", "none", "null", "nan", "NaN"]), col] = np.nan

    # Canonicalize equipment display
    if "equipment_type" not in df.columns:
        df["equipment_type"] = np.nan
    df["equipment_display"] = df["equipment_type"].fillna("").astype(str).str.strip().str.title()
    df.loc[df["equipment_display"] == "", "equipment_display"] = np.nan

    # Useful derived flags
    df["is_fmcsa_active"] = (df["stage"].eq("fmcsa_verify") &
                             df["outcome"].fillna("").str.lower().eq("active"))
    df["is_load_selected"] = df["stage"].eq("load_selected")
    df["is_negotiate"] = df["stage"].eq("negotiate_result")
    df["is_accepted"] = (df["is_negotiate"] &
                         df["outcome"].fillna("").str.lower().eq("accepted"))

    return df.sort_values("ts", ascending=True).reset_index(drop=True)

def now_utc():
    return datetime.now(timezone.utc)

# --------------------------- UI -------------------------------

st.title("FDE Use-Case Metrics")

# Top: connectivity pill
ok = ping_health()
st.caption(("🟢 API connected" if ok else "🔴 API unreachable") + f" · {API_BASE}")

# Time window selector
tw_label = "Time window"
tw_options = {
    "Last 24h": timedelta(days=1),
    "Last 7d": timedelta(days=7),
    "Last 30d": timedelta(days=30),
    "All time": None,
}
c1, c2, c3 = st.columns([1.2, 1.2, 1.2])
with c1:
    tw_key = st.selectbox(tw_label, list(tw_options.keys()), index=0)

df_all = fetch_events()

# Apply time window
if tw_options[tw_key] is not None:
    window_start = now_utc() - tw_options[tw_key]
    df = df_all[df_all["ts"] >= window_start].copy()
else:
    df = df_all.copy()

# Build equipment options from the *wider* set (to avoid empty dropdowns)
equip_wide = (df_all["equipment_display"].dropna().unique().tolist())
equip_opts = ["All"] + sorted(equip_wide) if equip_wide else ["All"]

# Negotiation outcome options
neg_opts_raw = df_all.loc[df_all["is_negotiate"], "outcome"].dropna().str.title().unique().tolist()
neg_opts = ["All"] + sorted(neg_opts_raw) if neg_opts_raw else ["All"]

with c2:
    equip_sel = st.multiselect("Equipment type", equip_opts, default=["All"])
with c3:
    neg_sel = st.multiselect("Negotiation outcome", neg_opts, default=["All"])

# Apply filters
if "All" not in equip_sel:
    df = df[df["equipment_display"].isin(equip_sel)]
if "All" not in neg_sel:
    df = df[df["stage"].eq("negotiate_result") & df["outcome"].str.title().isin(neg_sel)]

# --------------------------- KPIs -----------------------------

# Total calls -> distinct run ids that have any stage
total_calls = df["run_id"].nunique()

# Verified (FMCSA active) -> distinct run ids with fmcsa_verify outcome active
verified_active = df.loc[df["is_fmcsa_active"], "run_id"].nunique()

# Loads pitched -> count of load_selected rows
loads_pitched = int(df.loc[df["is_load_selected"]].shape[0])

# Agreements -> count of negotiate_result with outcome accepted
agreements = int(df.loc[df["is_accepted"]].shape[0])

# Win rate -> accepted / (accepted + declined)
neg_df = df.loc[df["is_negotiate"]].copy()
neg_total = int(neg_df.shape[0])
win_rate = (agreements / max(1, neg_total)) * 100.0

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total Calls", total_calls)
k2.metric("Verified (FMCSA active)", verified_active)
k3.metric("Loads Pitched", loads_pitched)
k4.metric("Agreements", agreements)
k5.metric("Win Rate", f"{win_rate:.0f}%")

st.markdown("### Pricing")
# Average discount among accepted
if "discount_pct" in df.columns and not df.loc[df["is_accepted"], "discount_pct"].dropna().empty:
    avg_discount = float(df.loc[df["is_accepted"], "discount_pct"].mean())
else:
    avg_discount = 0.0
st.caption(f"Avg discount vs loadboard: **{avg_discount:.1f}%**")

# -------------------- Sentiment & Rounds ----------------------

st.markdown("### Sentiment & Rounds")

# Most recent sentiment per run (if present)
sent_cols = ["run_id", "outcome", "sentiment", "sentiment_score", "duration_sec", "ts"]
have_sent = [c for c in ["sentiment_score","sentiment","duration_sec"] if c in df.columns]
if have_sent:
    # take the last sentiment row per run_id
    sent = (df.dropna(subset=["run_id"])
              .sort_values("ts")
              .groupby("run_id")
              .tail(1)[["run_id","outcome","sentiment","sentiment_score","duration_sec"]]
              .reset_index(drop=True))
    sent["sentiment"] = sent["sentiment"].fillna("UNKNOWN")
    sent["sentiment_score"] = sent["sentiment_score"].fillna(0.5)
    sent["duration_sec"] = sent["duration_sec"].fillna(0).astype(int)
    st.dataframe(sent, use_container_width=True, hide_index=True)
else:
    st.info("No sentiment rows yet.")

# -------------------- Agreed vs Loadboard ---------------------

st.markdown("### Agreed vs Loadboard (accepted only)")
acc = df.loc[df["is_accepted"], ["run_id","equipment_display","origin","destination",
                                 "loadboard_rate","agreed_rate","discount_pct","negotiation_rounds","ts"]].copy()
if acc.empty:
    st.info("No accepted deals yet.")
else:
    acc["equipment_display"] = acc["equipment_display"].fillna("Unknown")
    acc["negotiation_rounds"] = acc["negotiation_rounds"].fillna(0).astype(int)
    acc["discount_pct"] = acc["discount_pct"].fillna(0.0).round(1)
    acc = acc.sort_values("ts", ascending=False)
    st.dataframe(acc.drop(columns=["ts"]), use_container_width=True, hide_index=True)

# -------------------- Outcomes by Equipment -------------------

st.markdown("### Outcomes by Equipment")
equip_roll = (df.loc[df["is_negotiate"]]
                .assign(out=lambda d: d["outcome"].str.title().fillna("Unknown"))
                .groupby(["equipment_display","out"], dropna=False)
                .size().rename("count").reset_index())
if equip_roll.empty:
    st.caption("No negotiation outcomes yet for the selected window/filters.")
else:
    # small pivot for readability
    pivot = equip_roll.pivot_table(index="equipment_display", columns="out", values="count", fill_value=0)
    st.dataframe(pivot, use_container_width=True)

# -------------------- Raw Events ------------------------------

st.markdown("### Raw Events")
show_cols = [
    "id","ts","run_id","call_id","stage","outcome","carrier_mc","fmcsa_status",
    "load_id","origin","destination","equipment_type","loadboard_rate","model_offer",
    "carrier_counter","agreed_rate","discount_pct","negotiation_rounds"
]
present_cols = [c for c in show_cols if c in df.columns]
if present_cols:
    st.dataframe(df[present_cols].sort_values("ts", ascending=False), use_container_width=True)
else:
    st.write(df.tail(50))

# -------------------- “Latest call spotlight” -----------------

st.markdown("---")
st.markdown("#### Latest Call (spotlight)")
if df.empty:
    st.caption("No events yet.")
else:
    last_run = df.sort_values("ts").iloc[-1]["run_id"]
    spot = df[df["run_id"] == last_run].copy()
    # summarize
    has_pitch = bool(spot["is_load_selected"].any())
    has_negotiate = bool(spot["is_negotiate"].any())
    accepted = bool(spot["is_accepted"].any())
    equip = spot["equipment_display"].dropna().tail(1).tolist()
    equip_str = equip[-1] if equip else "Unknown"
    st.write(
        f"**Run:** `{last_run}`  |  **Equipment:** {equip_str}  |  "
        f"**Pitched:** {'✅' if has_pitch else '—'}  |  "
        f"**Negotiated:** {'✅' if has_negotiate else '—'}  |  "
        f"**Accepted:** {'✅' if accepted else '—'}"
    )

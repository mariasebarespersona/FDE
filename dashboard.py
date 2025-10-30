# # dashboard.py
# # Streamlit dashboard for HappyRobots FDE Metrics
# # - Pulls raw events from your metrics API (FastAPI) at METRICS_URL
# # - Computes KPIs + per-run sentiment/rounds
# # - Robust filters (time window, equipment, negotiation outcome)
# # - Shows your latest accepted call clearly

# import os
# import time
# from datetime import datetime, timedelta, timezone

# import numpy as np
# import pandas as pd
# import requests
# import streamlit as st

# # --------------------------- Config ---------------------------

# # API_BASE = os.environ.get("METRICS_URL", "http://localhost:8081").rstrip("/")
# # API_EVENTS = f"{API_BASE}/events"          # GET returns JSON list of events rows
# # API_HEALTH = f"{API_BASE}/healthz"


# import os


# # --- Resolve API base and key (secrets > env > sane default) ---
# API_BASE = (
#     st.secrets.get("API_BASE")
#     or os.getenv("API_BASE")
#     or "https://fde-production.up.railway.app"
# ).rstrip("/")

# API_KEY = st.secrets.get("API_KEY") or os.getenv("API_KEY") or ""

# API_EVENTS = f"{API_BASE}/events"
# API_HEALTH = f"{API_BASE}/healthz"

# HEADERS = {"X-API-KEY": API_KEY} if API_KEY else {}


# st.set_page_config(
#     page_title="FDE Use-Case Metrics",
#     layout="wide",
#     initial_sidebar_state="collapsed",
# )

# # --------------------------- Helpers --------------------------

# @st.cache_data(show_spinner=False, ttl=5)
# def ping_health():
#     try:
#         r = requests.get(API_HEALTH, timeout=3)
#         r.raise_for_status()
#         data = r.json()
#         return bool(data.get("ok", False))
#     except Exception:
#         return False

# @st.cache_data(show_spinner=False, ttl=5)
# def fetch_events() -> pd.DataFrame:
#     """
#     Expected schema from API:
#       id, ts, run_id, call_id, stage, outcome, carrier_mc, fmcsa_status,
#       load_id, origin, destination, equipment_type, loadboard_rate, model_offer,
#       carrier_counter, agreed_rate, discount_pct, negotiation_rounds,
#       sentiment, sentiment_score, duration_sec, transferred_to_rep, error
#     Unknown/missing fields are OK; we normalize below.
#     """
#     r = requests.get(API_EVENTS, timeout=8)
#     r.raise_for_status()
#     data = r.json()
#     if not isinstance(data, list):
#         data = []

#     df = pd.DataFrame(data)
#     if df.empty:
#         # Ensure downstream code has columns
#         df = pd.DataFrame(columns=[
#             "id","ts","run_id","call_id","stage","outcome","carrier_mc",
#             "fmcsa_status","load_id","origin","destination","equipment_type",
#             "loadboard_rate","model_offer","carrier_counter","agreed_rate",
#             "discount_pct","negotiation_rounds","sentiment","sentiment_score",
#             "duration_sec","transferred_to_rep","error"
#         ])

#     # Normalize dtypes
#     def to_ts(x):
#         try:
#             return pd.to_datetime(x, utc=True)
#         except Exception:
#             return pd.NaT

#     df["ts"] = df["ts"].apply(to_ts)
#     for col in ["loadboard_rate","model_offer","agreed_rate","discount_pct",
#                 "negotiation_rounds","sentiment_score","duration_sec"]:
#         if col in df.columns:
#             df[col] = pd.to_numeric(df[col], errors="coerce")

#     # Strings; trim/standardize empties
#     for col in ["stage","outcome","equipment_type","origin","destination",
#                 "sentiment","fmcsa_status","carrier_mc"]:
#         if col in df.columns:
#             df[col] = df[col].astype(str).str.strip()
#             df.loc[df[col].isin(["", "none", "null", "nan", "NaN"]), col] = np.nan

#     # Canonicalize equipment display
#     if "equipment_type" not in df.columns:
#         df["equipment_type"] = np.nan
#     df["equipment_display"] = df["equipment_type"].fillna("").astype(str).str.strip().str.title()
#     df.loc[df["equipment_display"] == "", "equipment_display"] = np.nan

#     # Useful derived flags
#     df["is_fmcsa_active"] = (df["stage"].eq("fmcsa_verify") &
#                              df["outcome"].fillna("").str.lower().eq("active"))
#     df["is_load_selected"] = df["stage"].eq("load_selected")
#     df["is_negotiate"] = df["stage"].eq("negotiate_result")
#     df["is_accepted"] = (df["is_negotiate"] &
#                          df["outcome"].fillna("").str.lower().eq("accepted"))

#     return df.sort_values("ts", ascending=True).reset_index(drop=True)

# def now_utc():
#     return datetime.now(timezone.utc)

# # --------------------------- UI -------------------------------

# st.title("FDE Use-Case Metrics")

# # Top: connectivity pill
# ok = ping_health()
# st.caption(("🟢 API connected" if ok else "🔴 API unreachable") + f" · {API_BASE}")

# # Time window selector
# tw_label = "Time window"
# tw_options = {
#     "Last 24h": timedelta(days=1),
#     "Last 7d": timedelta(days=7),
#     "Last 30d": timedelta(days=30),
#     "All time": None,
# }
# c1, c2, c3 = st.columns([1.2, 1.2, 1.2])
# with c1:
#     tw_key = st.selectbox(tw_label, list(tw_options.keys()), index=0)

# df_all = fetch_events()

# # Apply time window
# if tw_options[tw_key] is not None:
#     window_start = now_utc() - tw_options[tw_key]
#     df = df_all[df_all["ts"] >= window_start].copy()
# else:
#     df = df_all.copy()

# # Build equipment options from the *wider* set (to avoid empty dropdowns)
# equip_wide = (df_all["equipment_display"].dropna().unique().tolist())
# equip_opts = ["All"] + sorted(equip_wide) if equip_wide else ["All"]

# # Negotiation outcome options
# neg_opts_raw = df_all.loc[df_all["is_negotiate"], "outcome"].dropna().str.title().unique().tolist()
# neg_opts = ["All"] + sorted(neg_opts_raw) if neg_opts_raw else ["All"]

# with c2:
#     equip_sel = st.multiselect("Equipment type", equip_opts, default=["All"])
# with c3:
#     neg_sel = st.multiselect("Negotiation outcome", neg_opts, default=["All"])

# # Apply filters
# if "All" not in equip_sel:
#     df = df[df["equipment_display"].isin(equip_sel)]
# if "All" not in neg_sel:
#     df = df[df["stage"].eq("negotiate_result") & df["outcome"].str.title().isin(neg_sel)]

# # --------------------------- KPIs -----------------------------

# # Total calls -> distinct run ids that have any stage
# total_calls = df["run_id"].nunique()

# # Verified (FMCSA active) -> distinct run ids with fmcsa_verify outcome active
# verified_active = df.loc[df["is_fmcsa_active"], "run_id"].nunique()

# # Loads pitched -> count of load_selected rows
# loads_pitched = int(df.loc[df["is_load_selected"]].shape[0])

# # Agreements -> count of negotiate_result with outcome accepted
# agreements = int(df.loc[df["is_accepted"]].shape[0])

# # Win rate -> accepted / (accepted + declined)
# neg_df = df.loc[df["is_negotiate"]].copy()
# neg_total = int(neg_df.shape[0])
# win_rate = (agreements / max(1, neg_total)) * 100.0

# k1, k2, k3, k4, k5 = st.columns(5)
# k1.metric("Total Calls", total_calls)
# k2.metric("Verified (FMCSA active)", verified_active)
# k3.metric("Loads Pitched", loads_pitched)
# k4.metric("Agreements", agreements)
# k5.metric("Win Rate", f"{win_rate:.0f}%")

# st.markdown("### Pricing")
# # Average discount among accepted
# if "discount_pct" in df.columns and not df.loc[df["is_accepted"], "discount_pct"].dropna().empty:
#     avg_discount = float(df.loc[df["is_accepted"], "discount_pct"].mean())
# else:
#     avg_discount = 0.0
# st.caption(f"Avg discount vs loadboard: **{avg_discount:.1f}%**")

# # -------------------- Sentiment & Rounds ----------------------

# st.markdown("### Sentiment & Rounds")

# # Most recent sentiment per run (if present)
# sent_cols = ["run_id", "outcome", "sentiment", "sentiment_score", "duration_sec", "ts"]
# have_sent = [c for c in ["sentiment_score","sentiment","duration_sec"] if c in df.columns]
# if have_sent:
#     # take the last sentiment row per run_id
#     sent = (df.dropna(subset=["run_id"])
#               .sort_values("ts")
#               .groupby("run_id")
#               .tail(1)[["run_id","outcome","sentiment","sentiment_score","duration_sec"]]
#               .reset_index(drop=True))
#     sent["sentiment"] = sent["sentiment"].fillna("UNKNOWN")
#     sent["sentiment_score"] = sent["sentiment_score"].fillna(0.5)
#     sent["duration_sec"] = sent["duration_sec"].fillna(0).astype(int)
#     st.dataframe(sent, use_container_width=True, hide_index=True)
# else:
#     st.info("No sentiment rows yet.")

# # -------------------- Agreed vs Loadboard ---------------------

# st.markdown("### Agreed vs Loadboard (accepted only)")
# acc = df.loc[df["is_accepted"], ["run_id","equipment_display","origin","destination",
#                                  "loadboard_rate","agreed_rate","discount_pct","negotiation_rounds","ts"]].copy()
# if acc.empty:
#     st.info("No accepted deals yet.")
# else:
#     acc["equipment_display"] = acc["equipment_display"].fillna("Unknown")
#     acc["negotiation_rounds"] = acc["negotiation_rounds"].fillna(0).astype(int)
#     acc["discount_pct"] = acc["discount_pct"].fillna(0.0).round(1)
#     acc = acc.sort_values("ts", ascending=False)
#     st.dataframe(acc.drop(columns=["ts"]), use_container_width=True, hide_index=True)

# # -------------------- Outcomes by Equipment -------------------

# st.markdown("### Outcomes by Equipment")
# equip_roll = (df.loc[df["is_negotiate"]]
#                 .assign(out=lambda d: d["outcome"].str.title().fillna("Unknown"))
#                 .groupby(["equipment_display","out"], dropna=False)
#                 .size().rename("count").reset_index())
# if equip_roll.empty:
#     st.caption("No negotiation outcomes yet for the selected window/filters.")
# else:
#     # small pivot for readability
#     pivot = equip_roll.pivot_table(index="equipment_display", columns="out", values="count", fill_value=0)
#     st.dataframe(pivot, use_container_width=True)

# # -------------------- Raw Events ------------------------------

# st.markdown("### Raw Events")
# show_cols = [
#     "id","ts","run_id","call_id","stage","outcome","carrier_mc","fmcsa_status",
#     "load_id","origin","destination","equipment_type","loadboard_rate","model_offer",
#     "carrier_counter","agreed_rate","discount_pct","negotiation_rounds"
# ]
# present_cols = [c for c in show_cols if c in df.columns]
# if present_cols:
#     st.dataframe(df[present_cols].sort_values("ts", ascending=False), use_container_width=True)
# else:
#     st.write(df.tail(50))

# # -------------------- “Latest call spotlight” -----------------

# st.markdown("---")
# st.markdown("#### Latest Call (spotlight)")
# if df.empty:
#     st.caption("No events yet.")
# else:
#     last_run = df.sort_values("ts").iloc[-1]["run_id"]
#     spot = df[df["run_id"] == last_run].copy()
#     # summarize
#     has_pitch = bool(spot["is_load_selected"].any())
#     has_negotiate = bool(spot["is_negotiate"].any())
#     accepted = bool(spot["is_accepted"].any())
#     equip = spot["equipment_display"].dropna().tail(1).tolist()
#     equip_str = equip[-1] if equip else "Unknown"
#     st.write(
#         f"**Run:** `{last_run}`  |  **Equipment:** {equip_str}  |  "
#         f"**Pitched:** {'✅' if has_pitch else '—'}  |  "
#         f"**Negotiated:** {'✅' if has_negotiate else '—'}  |  "
#         f"**Accepted:** {'✅' if accepted else '—'}"
#     )


# dashboard.py
# Streamlit dashboard for HappyRobots FDE Metrics (public-friendly)
# - Pulls raw events from your metrics API (FastAPI) at API_BASE
# - Coalesces fields from alternates (ls_origin, ls_loadboard_rate, etc.)
# - Builds per-run rollups to avoid "None" after later stages
# - KPIs, filters, accepted-deals table, outcomes by equipment, raw events, spotlight

import os
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests
import streamlit as st

# --------------------------- Config ---------------------------

st.set_page_config(
    page_title="FDE Use-Case Metrics",
    layout="wide",
    initial_sidebar_state="collapsed",
)

def _get_secret_or_env(name: str, default: str = "") -> str:
    # Streamlit Secrets -> Env -> default
    try:
        val = st.secrets.get(name)
        if val:
            return str(val)
    except Exception:
        pass
    val = os.getenv(name, "")
    return str(val) if val else default

API_BASE = _get_secret_or_env("API_BASE", "https://fde-production.up.railway.app").rstrip("/")
API_KEY  = _get_secret_or_env("API_KEY", "")
API_EVENTS = f"{API_BASE}/events"
API_HEALTH = f"{API_BASE}/healthz"
HEADERS = {"X-API-KEY": API_KEY} if API_KEY else {}

# --------------------------- Helpers --------------------------

def _safe_to_ts(x):
    try:
        return pd.to_datetime(x, utc=True)
    except Exception:
        return pd.NaT

def _coalesce(series_list):
    """Return the first non-null series among the list, element-wise."""
    out = None
    for s in series_list:
        if s is None:
            continue
        if out is None:
            out = s.copy()
        else:
            out = out.fillna(s)
    return out

@st.cache_data(show_spinner=False, ttl=5)
def ping_health() -> bool:
    try:
        r = requests.get(API_HEALTH, timeout=4, headers=HEADERS)
        r.raise_for_status()
        j = r.json()
        return bool(j.get("ok", False))
    except Exception:
        return False

@st.cache_data(show_spinner=False, ttl=5)
def fetch_events() -> pd.DataFrame:
    """
    Expected schema (any missing are handled):
      id, ts, run_id, call_id, stage, outcome, carrier_mc, fmcsa_status,
      load_id, origin, destination, equipment_type,
      loadboard_rate, model_offer, carrier_counter, agreed_rate,
      discount_pct, negotiation_rounds, sentiment, sentiment_score,
      duration_sec, transferred_to_rep, error

    Also supports alternates logged by earlier nodes:
      ls_origin, ls_destination, ls_equipment_type, ls_loadboard_rate
    """
    r = requests.get(API_EVENTS, timeout=10, headers=HEADERS)
    r.raise_for_status()
    data = r.json()
    if not isinstance(data, list):
        data = []
    df = pd.DataFrame(data)

    # If no data, return an empty frame with expected columns to avoid UI errors
    base_cols = [
        "id","ts","run_id","call_id","stage","outcome","carrier_mc","fmcsa_status",
        "load_id","origin","destination","equipment_type","loadboard_rate","model_offer",
        "carrier_counter","agreed_rate","discount_pct","negotiation_rounds","sentiment",
        "sentiment_score","duration_sec","transferred_to_rep","error",
        # alternates
        "ls_origin","ls_destination","ls_equipment_type","ls_loadboard_rate",
    ]
    for c in base_cols:
        if c not in df.columns:
            df[c] = np.nan

    # Normalize
    df["ts"] = df["ts"].apply(_safe_to_ts)

    # numerics
    num_cols = [
        "loadboard_rate","model_offer","carrier_counter","agreed_rate",
        "discount_pct","negotiation_rounds","sentiment_score","duration_sec",
        "ls_loadboard_rate"
    ]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # tiny string cleanup
    str_cols = [
        "stage","outcome","equipment_type","origin","destination","sentiment",
        "fmcsa_status","carrier_mc","ls_origin","ls_destination","ls_equipment_type"
    ]
    for c in str_cols:
        df[c] = df[c].astype(str).str.strip()
        df.loc[df[c].isin(["", "none", "null", "nan", "NaN"]), c] = np.nan

    # Markers for convenience
    df["is_fmcsa_verify"] = df["stage"].eq("fmcsa_verify")
    df["is_load_selected"] = df["stage"].eq("load_selected")
    df["is_negotiate"] = df["stage"].eq("negotiate_result")
    df["is_accepted"] = df["is_negotiate"] & df["outcome"].str.lower().eq("accepted")
    df["is_fmcsa_active_row"] = df["is_fmcsa_verify"] & df["outcome"].str.lower().eq("active")

    # For display dropdowns
    df["equipment_display_raw"] = _coalesce([
        df["equipment_type"],
        df["ls_equipment_type"]
    ]).fillna("").str.title()
    df.loc[df["equipment_display_raw"] == "", "equipment_display_raw"] = np.nan

    # Keep time order for forward-filling
    df = df.sort_values("ts", ascending=True).reset_index(drop=True)
    return df

def build_run_rollup(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a rollup table with the latest known values per run_id,
    forward-filling from earlier stages so the 'happy path' shows complete info.
    """
    if df.empty:
        return pd.DataFrame(columns=[
            "run_id","first_ts","last_ts","equipment_display","origin","destination",
            "loadboard_rate","agreed_rate","discount_pct","negotiation_rounds",
            "sentiment","sentiment_score","duration_sec","accepted","pitched","negotiated"
        ])

    # Coalesced fields first (per row)
    df = df.copy()
    df["origin_all"] = _coalesce([df["origin"], df["ls_origin"]])
    df["destination_all"] = _coalesce([df["destination"], df["ls_destination"]])
    df["equipment_all"] = _coalesce([df["equipment_type"], df["ls_equipment_type"]])

    # prefer canonical numeric values, fallback to ls_*
    df["loadboard_all"] = _coalesce([df["loadboard_rate"], df["ls_loadboard_rate"]])

    # Per-run forward-fill
    fcols = [
        "origin_all","destination_all","equipment_all",
        "loadboard_all","agreed_rate","discount_pct","negotiation_rounds",
        "sentiment","sentiment_score","duration_sec"
    ]
    roll = []
    for run_id, g in df.groupby("run_id", dropna=True):
        gg = g.sort_values("ts").copy()
        gg[fcols] = gg[fcols].ffill()

        record = {
            "run_id": run_id,
            "first_ts": gg["ts"].min(),
            "last_ts": gg["ts"].max(),
            "equipment_display": (gg["equipment_all"].dropna().tail(1).astype(str).str.title().iloc[0]
                                  if gg["equipment_all"].notna().any() else np.nan),
            "origin": gg["origin_all"].dropna().tail(1).iloc[0] if gg["origin_all"].notna().any() else np.nan,
            "destination": gg["destination_all"].dropna().tail(1).iloc[0] if gg["destination_all"].notna().any() else np.nan,
            "loadboard_rate": float(gg["loadboard_all"].dropna().tail(1).iloc[0]) if gg["loadboard_all"].notna().any() else np.nan,
            "agreed_rate": float(gg["agreed_rate"].dropna().tail(1).iloc[0]) if gg["agreed_rate"].notna().any() else np.nan,
            "discount_pct": float(gg["discount_pct"].dropna().tail(1).iloc[0]) if gg["discount_pct"].notna().any() else np.nan,
            "negotiation_rounds": int(gg["negotiation_rounds"].dropna().tail(1).iloc[0]) if gg["negotiation_rounds"].notna().any() else 0,
            "sentiment": gg["sentiment"].dropna().tail(1).iloc[0] if gg["sentiment"].notna().any() else np.nan,
            "sentiment_score": float(gg["sentiment_score"].dropna().tail(1).iloc[0]) if gg["sentiment_score"].notna().any() else np.nan,
            "duration_sec": int(gg["duration_sec"].dropna().tail(1).iloc[0]) if gg["duration_sec"].notna().any() else 0,
            "accepted": bool((gg["stage"].eq("negotiate_result") & gg["outcome"].str.lower().eq("accepted")).any()),
            "pitched": bool(gg["stage"].eq("load_selected").any()),
            "negotiated": bool(gg["stage"].eq("negotiate_result").any()),
        }
        roll.append(record)

    rollup = pd.DataFrame(roll).sort_values("last_ts", ascending=False).reset_index(drop=True)
    # polish display
    if "equipment_display" in rollup:
        rollup["equipment_display"] = rollup["equipment_display"].fillna("Unknown").str.title()
    return rollup

def now_utc():
    return datetime.now(timezone.utc)

# --------------------------- UI -------------------------------

st.title("FDE Use-Case Metrics")

api_ok = ping_health()
pill = "🟢 API connected" if api_ok else "🔴 API unreachable"
st.caption(f"{pill} · {_get_secret_or_env('API_BASE', API_BASE)}")

# Time window selector
tw_options = {
    "Last 24h": timedelta(days=1),
    "Last 7d": timedelta(days=7),
    "Last 30d": timedelta(days=30),
    "All time": None,
}
c1, c2, c3 = st.columns([1.2, 1.2, 1.2])
with c1:
    tw_key = st.selectbox("Time window", list(tw_options.keys()), index=0)

# Fetch raw
df_all = fetch_events()

# Apply time window before building rollups (so counts reflect filter)
if tw_options[tw_key] is None:
    df = df_all.copy()
else:
    window_start = now_utc() - tw_options[tw_key]
    df = df_all[df_all["ts"] >= window_start].copy()

# Equipment & outcome filters built from whole dataset (friendlier UX)
equip_opts = ["All"] + sorted(df_all["equipment_display_raw"].dropna().unique().tolist()) if not df_all.empty else ["All"]
neg_outcomes = ["All"] + sorted(
    df_all.loc[df_all["is_negotiate"], "outcome"].dropna().str.title().unique().tolist()
) if not df_all.empty else ["All"]

with c2:
    equip_sel = st.multiselect("Equipment type", equip_opts, default=["All"])
with c3:
    neg_sel = st.multiselect("Negotiation outcome", neg_outcomes, default=["All"])

# Build rollups for this filtered time window
rollup = build_run_rollup(df)

# Apply equipment filter
if "All" not in equip_sel:
    rollup = rollup[rollup["equipment_display"].isin(equip_sel)]

# Apply negotiation outcome filter
if "All" not in neg_sel:
    # Map to accepted/declined
    desired = set([s.lower() for s in neg_sel])
    # treat missing outcome as neither accepted nor declined; we only filter on negotiated runs
    mask = np.zeros(len(rollup), dtype=bool)
    if "accepted" in desired:
        mask |= rollup["accepted"].fillna(False)
    if "declined" in desired:
        mask |= (~rollup["accepted"].fillna(False)) & rollup["negotiated"].fillna(False)
    rollup = rollup[mask]

# --------------------------- KPIs -----------------------------

total_calls = rollup["run_id"].nunique()
verified_active = df.loc[df["is_fmcsa_active_row"], "run_id"].nunique()
loads_pitched = int(df.loc[df["is_load_selected"]].shape[0])
agreements = int(df.loc[df["is_accepted"]].shape[0])

neg_rows = df.loc[df["is_negotiate"]].shape[0]
win_rate = (agreements / max(1, neg_rows)) * 100.0

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total Calls", total_calls)
k2.metric("Verified (FMCSA active)", verified_active)
k3.metric("Loads Pitched", loads_pitched)
k4.metric("Agreements", agreements)
k5.metric("Win Rate", f"{win_rate:.0f}%")

# Pricing
st.markdown("### Pricing")
acc_roll = rollup[rollup["accepted"]]
avg_discount = float(acc_roll["discount_pct"].dropna().mean()) if not acc_roll.empty else 0.0
st.caption(f"Avg discount vs loadboard: **{avg_discount:.1f}%**")

# -------------------- Sentiment & Rounds ----------------------

st.markdown("### Sentiment & Rounds")
if rollup.empty:
    st.info("No runs in the selected window/filters.")
else:
    sent_view = rollup[[
        "run_id","accepted","sentiment","sentiment_score","duration_sec"
    ]].copy()
    sent_view["sentiment"] = sent_view["sentiment"].fillna("UNKNOWN")
    sent_view["sentiment_score"] = sent_view["sentiment_score"].fillna(0.5).round(2)
    sent_view["duration_sec"] = sent_view["duration_sec"].fillna(0).astype(int)
    sent_view.rename(columns={"accepted":"outcome"}, inplace=True)
    sent_view["outcome"] = sent_view["outcome"].map({True:"accepted", False:"None"})
    st.dataframe(sent_view, use_container_width=True, hide_index=True)

# -------------------- Agreed vs Loadboard ---------------------

st.markdown("### Agreed vs Loadboard (accepted only)")
if acc_roll.empty:
    st.info("No accepted deals yet.")
else:
    view = acc_roll[[
        "run_id","equipment_display","origin","destination",
        "loadboard_rate","agreed_rate","discount_pct","negotiation_rounds","last_ts"
    ]].copy()
    view["equipment_display"] = view["equipment_display"].fillna("Unknown")
    view["negotiation_rounds"] = view["negotiation_rounds"].fillna(0).astype(int)
    view["discount_pct"] = view["discount_pct"].fillna(0.0).round(1)
    view = view.sort_values("last_ts", ascending=False).drop(columns=["last_ts"])
    st.dataframe(view, use_container_width=True, hide_index=True)

# -------------------- Outcomes by Equipment -------------------

st.markdown("### Outcomes by Equipment")
neg_df = df.loc[df["is_negotiate"]].copy()
if neg_df.empty:
    st.caption("No negotiation outcomes yet for the selected window/filters.")
else:
    neg_df["equipment_display"] = df_all["equipment_display_raw"]
    neg_df["equipment_display"] = neg_df["equipment_display"].fillna("Unknown")
    neg_df["out_norm"] = neg_df["outcome"].str.title().fillna("Unknown")
    roll = (neg_df
            .groupby(["equipment_display","out_norm"], dropna=False)
            .size().rename("count")
            .reset_index())
    pivot = roll.pivot_table(index="equipment_display", columns="out_norm", values="count", fill_value=0)
    st.dataframe(pivot, use_container_width=True)

# -------------------- Raw Events (for debugging) --------------

st.markdown("### Raw Events")
raw_cols = [
    "id","ts","run_id","call_id","stage","outcome",
    "carrier_mc","fmcsa_status","load_id",
    "origin","destination","equipment_type",
    "ls_origin","ls_destination","ls_equipment_type",
    "loadboard_rate","ls_loadboard_rate","model_offer",
    "carrier_counter","agreed_rate","discount_pct","negotiation_rounds"
]
present_cols = [c for c in raw_cols if c in df.columns]
if present_cols:
    st.dataframe(
        df[present_cols].sort_values("ts", ascending=False),
        use_container_width=True
    )
else:
    st.write(df.tail(50))

# -------------------- “Latest call spotlight” -----------------

st.markdown("---")
st.markdown("#### Latest Call (spotlight)")
if rollup.empty:
    st.caption("No runs in the selected window/filters.")
else:
    latest = rollup.sort_values("last_ts").iloc[-1]
    st.write(
        f"**Run:** `{latest['run_id']}`  |  "
        f"**Equipment:** {latest.get('equipment_display','Unknown')}  |  "
        f"**From→To:** {latest.get('origin','—')} → {latest.get('destination','—')}  |  "
        f"**Pitched:** {'✅' if latest.get('pitched') else '—'}  |  "
        f"**Negotiated:** {'✅' if latest.get('negotiated') else '—'}  |  "
        f"**Accepted:** {'✅' if latest.get('accepted') else '—'}"
    )

# app.py
# Device Event Insights — Pro (Outliers + Local DB)
# - Robust outliers (median+MAD) for walking gaps, dwell, and hourly volume
# - Drill-down with H:MM:SS, per-visit durations, CSV exports
# - Local persistence to SQLite (and Parquet fallback)
# - Safe datetime slider, duplicate-column handling

from datetime import timedelta
from typing import Dict, Tuple, List
import hashlib
import json
import os
import re
import sqlite3

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ---------- CONFIG ----------
st.set_page_config(page_title="Device Event Insights", layout="wide", initial_sidebar_state="expanded")

DEFAULT_COLMAP = {
    "datetime": "TransactionDateTime",
    "device":   "Device",
    "user":     "UserName",
    "type":     "TransactionType",
    # optional:
    "desc":     "MedDescription",
    "qty":      "Quantity",
    "medid":    "MedID",
}

DEFAULT_IDLE_MIN = 30       # seconds to consider walking/travel gap
LOCAL_HISTORY_FILE = "event_history.parquet"
SQLITE_DB = "event_history.db"   # local DB
USE_SQLITE = True                 # set False if you want Parquet-only behavior

# ---------- Helpers ----------
def fmt_hms(x) -> str:
    if pd.isna(x):
        return ""
    x = int(round(float(x)))
    h, r = divmod(x, 3600); m, s = divmod(r, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"

def dedupe_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()].copy()
    df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]
    return df

def parse_datetime_series(s: pd.Series) -> pd.Series:
    out = pd.to_datetime(s, errors="coerce", utc=True)
    if pd.api.types.is_datetime64tz_dtype(out):
        out = out.dt.tz_convert("UTC").dt.tz_localize(None)
    return out

def ensure_parquet_safe(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    # normalize datetimes to tz-naive
    for c in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[c]):
            s = out[c]
            if pd.api.types.is_datetime64tz_dtype(s):
                out[c] = s.dt.tz_convert("UTC").dt.tz_localize(None)
    # force all python-objects to strings
    obj_cols = out.select_dtypes(include=["object"]).columns
    for c in obj_cols:
        if out[c].apply(lambda x: isinstance(x, (list, dict))).any():
            out[c] = out[c].apply(lambda x: json.dumps(x) if isinstance(x, (list, dict)) else ("" if pd.isna(x) else str(x)))
        else:
            out[c] = out[c].astype(str)
    try:
        out = out.convert_dtypes()
    except Exception:
        pass
    return out

def base_clean(df_raw: pd.DataFrame, colmap: Dict[str, str]) -> pd.DataFrame:
    out = dedupe_columns(df_raw).copy()
    dtcol = colmap["datetime"]
    s = out[dtcol]
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    out[dtcol] = parse_datetime_series(s)

    if colmap.get("qty") and colmap["qty"] in out.columns:
        out[colmap["qty"]] = pd.to_numeric(out[colmap["qty"]], errors="coerce")

    for key in ["device", "user", "type", "desc", "medid"]:
        c = colmap.get(key)
        if c and c in out.columns:
            out[c] = out[c].astype("string").str.strip()

    out = out.dropna(subset=[dtcol]).copy()
    out = out.sort_values(dtcol).reset_index(drop=True)
    out["__date"] = out[dtcol].dt.date
    out["__hour"] = out[dtcol].dt.hour
    out["__dow"]  = out[dtcol].dt.day_name()
    return out

def build_pk(df: pd.DataFrame, colmap: Dict[str, str]) -> pd.Series:
    cols = []
    for k in ["datetime","device","user","type","desc","qty","medid"]:
        c = colmap.get(k)
        if c in df.columns:
            cols.append(df[c].astype(str))
        else:
            cols.append(pd.Series([""], index=df.index))
    arr = np.vstack([c.values for c in cols]).T
    out = [hashlib.sha1("|".join(row).encode("utf-8")).hexdigest() for row in arr]
    return pd.Series(out, index=df.index, dtype="string")

# ---------- SQLite persistence ----------
def init_db():
    with sqlite3.connect(SQLITE_DB) as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS events (
                pk TEXT PRIMARY KEY,
                ts TEXT,
                device TEXT,
                user TEXT,
                type TEXT,
                desc TEXT,
                qty REAL,
                medid TEXT
            )
        """)
        con.execute("""
            CREATE TABLE IF NOT EXISTS baselines (
                scope TEXT,              -- 'walk_user', 'dwell_device', 'vol_devhour'
                key1 TEXT,               -- user or device or device|hour
                median REAL,
                mad REAL,
                n INTEGER,
                PRIMARY KEY (scope, key1)
            )
        """)
    return True

def to_rows_for_sqlite(df: pd.DataFrame, colmap: Dict[str, str]) -> List[tuple]:
    ts, dev, usr, typ = (colmap["datetime"], colmap["device"], colmap["user"], colmap["type"])
    desc = colmap.get("desc"); qty = colmap.get("qty"); medid = colmap.get("medid")
    def get(col): return df[col] if (col and col in df.columns) else pd.Series([None]*len(df))
    rows = list(zip(
        df["pk"].astype(str),
        pd.to_datetime(df[ts]).dt.strftime("%Y-%m-%d %H:%M:%S"),
        df[dev].astype(str),
        df[usr].astype(str),
        df[typ].astype(str),
        get(desc).astype(str) if desc else pd.Series([None]*len(df)),
        pd.to_numeric(get(qty), errors="coerce") if qty else pd.Series([None]*len(df)),
        get(medid).astype(str) if medid else pd.Series([None]*len(df)),
    ))
    return rows

def upsert_events_sqlite(df: pd.DataFrame, colmap: Dict[str, str]) -> int:
    if df.empty: return 0
    init_db()
    rows = to_rows_for_sqlite(df, colmap)
    with sqlite3.connect(SQLITE_DB) as con:
        cur = con.cursor()
        cur.executemany("""
            INSERT OR IGNORE INTO events (pk, ts, device, user, type, desc, qty, medid)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, rows)
        con.commit()
        return cur.rowcount

def fetch_history_sqlite() -> pd.DataFrame:
    if not os.path.exists(SQLITE_DB):
        return pd.DataFrame()
    with sqlite3.connect(SQLITE_DB) as con:
        df = pd.read_sql_query("SELECT * FROM events", con)
    if df.empty:
        return df
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    df = df.rename(columns={"ts":"TransactionDateTime",
                            "device":"Device","user":"UserName",
                            "type":"TransactionType",
                            "desc":"MedDescription","qty":"Quantity",
                            "medid":"MedID"})
    return df

# ---------- Analytics ----------
def build_delivery_analytics(
    ev: pd.DataFrame, colmap: Dict[str, str], idle_min: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ts  = colmap["datetime"]; dev = colmap["device"]; usr = colmap["user"]; typ = colmap["type"]
    needed = [ts, dev, usr, typ]
    if colmap.get("desc")  and colmap["desc"]  in ev.columns: needed.append(colmap["desc"])
    if colmap.get("medid") and colmap["medid"] in ev.columns: needed.append(colmap["medid"])
    if colmap.get("qty")   and colmap["qty"]   in ev.columns: needed.append(colmap["qty"])

    data = ev[needed].sort_values([usr, ts]).copy()
    data["__next_ts"]  = data.groupby(usr)[ts].shift(-1)
    data["__next_dev"] = data.groupby(usr)[dev].shift(-1)
    data["__gap_s"] = (data["__next_ts"] - data[ts]).dt.total_seconds()
    data["__device_change"] = (data[dev] != data["__next_dev"]) & data["__next_dev"].notna()

    data["__walk_gap_s"] = np.where(
        (data["__device_change"]) & (data["__gap_s"] >= idle_min),
        data["__gap_s"], np.nan
    )
    data["__dwell_s"] = np.where(~data["__device_change"], data["__gap_s"], np.nan)

    # visit id per user whenever device changes
    data["__visit_id"] = (
        data.groupby(usr)[dev].apply(lambda x: (x != x.shift()).cumsum()).reset_index(level=0, drop=True)
    )

    visit = (
        data.groupby([usr, "__visit_id", dev])
            .agg(start=(ts,"min"), end=(ts,"max"))
            .reset_index()
    )
    visit["visit_duration_s"] = (visit["end"] - visit["start"]).dt.total_seconds()
    data = data.merge(visit[[usr,"__visit_id","visit_duration_s"]], on=[usr,"__visit_id"], how="left")

    # device stats
    cnt = data.groupby(dev).size().rename("events")
    dwell = (data.loc[~data["__device_change"], "__gap_s"].groupby(data[dev]).median().rename("median_dwell_s"))
    device_stats = pd.concat([cnt, dwell], axis=1).fillna(0).sort_values("events", ascending=False).reset_index()

    # tech stats
    tcnt = data.groupby(usr).size().rename("events")
    twalk = data["__walk_gap_s"].groupby(data[usr]).median().rename("median_walk_gap_s")
    tech_stats = pd.concat([tcnt, twalk], axis=1).fillna(0).sort_values("events", ascending=False).reset_index()

    # runs
    data["__is_break"] = (data["__walk_gap_s"] >= idle_min).fillna(False)
    data["__run_id"] = data.groupby(usr)["__is_break"].cumsum()
    run_stats = (
        data.groupby([usr,"__run_id"])
        .agg(start=(ts,"min"), end=(ts,"max"),
             n_events=(ts,"count"),
             n_devices=(dev,"nunique"),
             total_walk_s=("__walk_gap_s", lambda s: np.nansum(s.values)))
        .reset_index()
    )
    run_stats["duration_s"] = (run_stats["end"] - run_stats["start"]).dt.total_seconds()

    # hourly
    hourly = data.groupby(data[ts].dt.floor("H")).size().rename("events").reset_index().rename(columns={ts:"hour"})
    return data, device_stats, tech_stats, run_stats, hourly, visit

def weekly_summary(ev: pd.DataFrame, colmap: Dict[str, str]) -> pd.DataFrame:
    ts, dev, usr, typ = (colmap["datetime"], colmap["device"], colmap["user"], colmap["type"])
    df = ev.copy()
    df["week"] = df[ts].dt.to_period("W-SUN").apply(lambda p: p.start_time.date())
    out = (df.groupby("week").agg(events=(typ,"count"), devices=(dev,"nunique"), techs=(usr,"nunique"))
           .reset_index().sort_values("week"))
    return out

# ---------- Baselines + Outliers ----------
def mad(values: np.ndarray) -> float:
    med = np.nanmedian(values)
    return np.nanmedian(np.abs(values - med))

def compute_baselines(history: pd.DataFrame, data: pd.DataFrame, colmap: Dict[str,str]) -> dict:
    """Return dict of DataFrames with robust baselines and also store them in SQLite."""
    baselines = {}
    if history.empty and data.empty:
        return baselines

    ts  = colmap["datetime"]; dev = colmap["device"]; usr = colmap["user"]

    # WALK: per user (from all historical data's computed gaps if present, else fallback to current)
    walk_source = data[["__walk_gap_s", usr]].dropna() if "__walk_gap_s" in data else pd.DataFrame()
    # DWELL: per device
    dwell_source = data.loc[~data["__device_change"], ["__gap_s", dev]].dropna() if "__gap_s" in data else pd.DataFrame()
    # VOLUME: per device x hour
    vol_df = None
    if not history.empty:
        h2 = history.copy()
        h2["__hour_floor"] = pd.to_datetime(h2[ts]).dt.floor("H")
        vol_df = h2.groupby([dev, "__hour_floor"]).size().rename("events").reset_index()
        vol_df["hour"] = vol_df["__hour_floor"].dt.hour

    # Build/refresh SQLite table
    init_db()
    rows = []

    if not walk_source.empty:
        g = walk_source.groupby(usr)["__walk_gap_s"].apply(lambda s: pd.Series({
            "median": float(np.nanmedian(s)), "mad": float(mad(s)), "n": int(s.notna().sum())
        })).reset_index()
        g["scope"] = "walk_user"
        g["key1"]  = g[usr]
        baselines["walk_user"] = g[["scope","key1","median","mad","n"]]
        rows += list(g[["scope","key1","median","mad","n"]].itertuples(index=False, name=None))

    if not dwell_source.empty:
        g = dwell_source.groupby(dev)["__gap_s"].apply(lambda s: pd.Series({
            "median": float(np.nanmedian(s)), "mad": float(mad(s)), "n": int(s.notna().sum())
        })).reset_index()
        g["scope"] = "dwell_device"
        g["key1"]  = g[dev]
        baselines["dwell_device"] = g[["scope","key1","median","mad","n"]]
        rows += list(g[["scope","key1","median","mad","n"]].itertuples(index=False, name=None))

    if vol_df is not None and not vol_df.empty:
        g = vol_df.groupby([dev, "hour"])["events"].apply(lambda s: pd.Series({
            "median": float(np.nanmedian(s)), "mad": float(mad(s)), "n": int(s.notna().sum())
        })).reset_index()
        g["scope"] = "vol_devhour"
        g["key1"]  = g[dev] + "|" + g["hour"].astype(str)
        baselines["vol_devhour"] = g[["scope","key1","median","mad","n"]]
        rows += list(g[["scope","key1","median","mad","n"]].itertuples(index=False, name=None))

    if rows:
        with sqlite3.connect(SQLITE_DB) as con:
            con.executemany("""
                INSERT INTO baselines (scope, key1, median, mad, n)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(scope, key1) DO UPDATE SET
                    median=excluded.median,
                    mad=excluded.mad,
                    n=excluded.n
            """, rows)
            con.commit()

    return baselines

def find_outliers(data: pd.DataFrame, ev: pd.DataFrame, baselines: dict, colmap: Dict[str,str],
                  k_walk: float, k_dwell: float, k_vol: float):
    """
    Return three DataFrames: walk_outliers, dwell_outliers, vol_outliers
    with deviation, z_mad and friendly H:MM:SS fields.
    """
    ts  = colmap["datetime"]; dev = colmap["device"]; usr = colmap["user"]
    walk_out = pd.DataFrame(); dwell_out = pd.DataFrame(); vol_out = pd.DataFrame()

    # WALK outliers (per user baseline)
    if "walk_user" in baselines and "__walk_gap_s" in data:
        b = baselines["walk_user"].rename(columns={"key1": usr})
        d = data[[ts, usr, dev, "__walk_gap_s"]].dropna().merge(b[[usr,"median","mad"]], on=usr, how="left")
        d["mad"].replace(0, np.nan, inplace=True)  # avoid div0
        d["z_mad"] = (d["__walk_gap_s"] - d["median"]) / d["mad"]
        walk_out = d[d["z_mad"] >= k_walk].copy()
        walk_out["gap_hms"] = walk_out["__walk_gap_s"].map(fmt_hms)
        walk_out["baseline_hms"] = walk_out["median"].map(fmt_hms)
        walk_out = walk_out.sort_values("z_mad", ascending=False)

    # DWELL outliers (per device baseline)
    if "dwell_device" in baselines and "__gap_s" in data:
        base = baselines["dwell_device"].rename(columns={"key1": dev})
        # only same-device gaps
        d2 = data.loc[~data["__device_change"], [ts, usr, dev, "__gap_s"]].dropna().merge(
            base[[dev,"median","mad"]], on=dev, how="left"
        )
        d2["mad"].replace(0, np.nan, inplace=True)
        d2["z_mad"] = (d2["__gap_s"] - d2["median"]) / d2["mad"]
        dwell_out = d2[d2["z_mad"] >= k_dwell].copy()
        dwell_out["dwell_hms"] = dwell_out["__gap_s"].map(fmt_hms)
        dwell_out["baseline_hms"] = dwell_out["median"].map(fmt_hms)
        dwell_out = dwell_out.sort_values("z_mad", ascending=False)

    # VOLUME outliers (per device-hour baseline) — use current filtered ev
    if "vol_devhour" in baselines and not ev.empty:
        temp = ev.copy()
        temp["hour_floor"] = temp[ts].dt.floor("H")
        temp["hour"] = temp["hour_floor"].dt.hour
        curr = temp.groupby([dev, "hour"]).size().rename("events_now").reset_index()
        base = baselines["vol_devhour"].copy()
        base[["Device","hour"]] = base["key1"].str.split("|", n=1, expand=True)
        base["hour"] = base["hour"].astype(int)
        m = curr.merge(base[["Device","hour","median","mad","n"]], on=["Device","hour"], how="left")
        m["mad"].replace(0, np.nan, inplace=True)
        m["z_mad"] = (m["events_now"] - m["median"]) / m["mad"]
        vol_out = m[m["z_mad"] >= k_vol].copy()
        vol_out = vol_out.sort_values("z_mad", ascending=False)

    return walk_out, dwell_out, vol_out

# ---------- Q&A ----------
def qa_answer(question: str, ev: pd.DataFrame, data: pd.DataFrame, colmap: Dict[str,str]) -> Tuple[str, pd.DataFrame]:
    q = question.strip().lower()
    ts  = colmap["datetime"]; dev = colmap["device"]; usr = colmap["user"]; typ = colmap["type"]

    if re.search(r"\b(top|most)\b.*\bdevices?\b", q):
        t = ev.groupby(dev).size().rename("events").reset_index().sort_values("events", ascending=False).head(10)
        return f"Top devices by event volume (showing {len(t)}):", t

    if "longest" in q and "dwell" in q:
        if data.empty: return "No dwell data in current filter.", pd.DataFrame()
        t = (data.loc[~data["__device_change"], ["__gap_s"]]
             .groupby(data[dev]).median().rename(columns={"__gap_s":"median_dwell_s"})
             .sort_values("median_dwell_s", ascending=False).head(10).reset_index())
        t["median_dwell_hms"] = t["median_dwell_s"].map(fmt_hms)
        return "Devices with longest median dwell:", t

    m = re.search(r"median .*walk.* for (.+)", q)
    if m:
        name = m.group(1).strip()
        sub = data[data[usr].str.lower()==name.lower()]
        if sub.empty: return f"No rows found for user '{name}'.", pd.DataFrame()
        val = np.nanmedian(sub["__walk_gap_s"].values)
        return f"Median walk gap for {name}: {fmt_hms(val)} ({int(val)}s)", pd.DataFrame()

    if "hour" in q:
        t = ev.groupby(ev[ts].dt.floor("H")).size().rename("events").reset_index().rename(columns={ts:"hour"})
        if t.empty: return "No hourly data in current filter.", pd.DataFrame()
        top = t.sort_values("events", ascending=False).head(1).iloc[0]
        return f"Busiest hour: {top['hour']:%Y-%m-%d %H:%M} with {int(top['events'])} events.", t.sort_values("hour")

    if "which tech" in q and "median walk" in q:
        if data.empty: return "No walk-gap data in current filter.", pd.DataFrame()
        t = data.groupby(usr)["__walk_gap_s"].median().reset_index().rename(columns={"__walk_gap_s":"median_walk_s"})
        t = t.sort_values("median_walk_s", ascending=False)
        top = t.iloc[0]
        return f"Highest median walk gap: {top[usr]} at {fmt_hms(top['median_walk_s'])}.", t

    ans = "Try: 'top devices', 'longest dwell devices', 'median walk gap for Melissa', 'busiest hour'."
    tbl = ev[[ts, usr, dev, typ]].head(50)
    return ans, tbl

def safe_unique(df: pd.DataFrame, col: str) -> List[str]:
    if col not in df.columns: return []
    return sorted([x for x in df[col].dropna().astype(str).unique()])

# ---------- UI ----------
st.title("All Device Event Insights — Pro")

# 1) Upload
st.sidebar.header("1) Upload")
up = st.sidebar.file_uploader("Drag & drop daily XLSX/CSV", type=["xlsx","csv"])
if not up:
    st.info("Upload your daily export to get started.")
    st.stop()

# Load
if up.name.lower().endswith(".xlsx"):
    df_raw = pd.read_excel(up)
else:
    df_raw = pd.read_csv(up)

# 2) Column mapping
st.sidebar.header("2) Map columns")
colmap = {}
for k, default in DEFAULT_COLMAP.items():
    opts = list(df_raw.columns)
    sel = st.sidebar.selectbox(
        f"{k.capitalize()} column",
        options=opts,
        index=opts.index(default) if default in opts else 0,
        key=f"map_{k}",
        help="Pick the matching column from your export",
    )
    colmap[k] = sel

# Clean + Engineer
ev = base_clean(df_raw, colmap)

# PK + optional persist current upload
if "pk" not in ev.columns:
    ev["pk"] = build_pk(ev, colmap)

# Save to SQLite (and Parquet fallback) so app learns over time
saved_rows = 0
if USE_SQLITE:
    saved_rows = upsert_events_sqlite(ev, colmap)
    if saved_rows:
        st.sidebar.success(f"Saved {saved_rows} new rows to local DB ({SQLITE_DB}).")
    history = fetch_history_sqlite()
else:
    # Parquet path
    new_ev = ensure_parquet_safe(ev.copy())
    # load old
    if os.path.exists(LOCAL_HISTORY_FILE):
        try:
            history = pd.read_parquet(LOCAL_HISTORY_FILE)
        except Exception:
            history = pd.DataFrame()
    else:
        history = pd.DataFrame()
    if "pk" not in new_ev.columns:
        new_ev["pk"] = build_pk(new_ev, colmap)
    if not history.empty and "pk" not in history.columns:
        history["pk"] = build_pk(history, colmap)
    combined = pd.concat([history, new_ev], ignore_index=True).drop_duplicates(subset=["pk"]).reset_index(drop=True)
    combined = ensure_parquet_safe(combined)
    try:
        combined.to_parquet(LOCAL_HISTORY_FILE, index=False)
        st.sidebar.success(f"Saved local history: {len(combined):,} rows → {LOCAL_HISTORY_FILE}")
    except Exception as e:
        fallback = LOCAL_HISTORY_FILE.replace(".parquet",".csv")
        try:
            combined.to_csv(fallback, index=False)
            st.sidebar.warning(f"Parquet save failed ({e}). Saved CSV instead → {fallback}")
        except Exception as e2:
            st.sidebar.error(f"Could not save local history: {e2}")
    history = combined

# 3) Filters (native datetime slider)
st.sidebar.header("3) Filters")
ts = colmap["datetime"]
_min = pd.to_datetime(ev[ts].min()); _max = pd.to_datetime(ev[ts].max())
min_ts = _min.to_pydatetime(); max_ts = _max.to_pydatetime()
if min_ts == max_ts:
    max_ts = min_ts + timedelta(minutes=1)

rng = st.sidebar.slider(
    "Time range", min_value=min_ts, max_value=max_ts, value=(min_ts, max_ts), format="YYYY-MM-DD HH:mm"
)
pick_devices = st.sidebar.multiselect("Devices", safe_unique(ev, colmap["device"]))
pick_users   = st.sidebar.multiselect("Users", safe_unique(ev, colmap["user"]))
pick_types   = st.sidebar.multiselect("Transaction types", safe_unique(ev, colmap["type"]))
idle_min = st.sidebar.number_input("Walk gap threshold (seconds)", min_value=5, max_value=900, value=DEFAULT_IDLE_MIN, step=5)

mask = (
    (ev[ts] >= pd.to_datetime(rng[0])) & (ev[ts] <= pd.to_datetime(rng[1]))
)
if pick_devices:
    mask &= ev[colmap["device"]].isin(pick_devices)
if pick_users:
    mask &= ev[colmap["user"]].isin(pick_users)
if pick_types:
    mask &= ev[colmap["type"]].isin(pick_types)

ev = ev.loc[mask].copy()
if ev.empty:
    st.warning("No events in current filter range.")
    st.stop()

# Compute analytics for current filtered view
data, device_stats, tech_stats, run_stats, hourly, visit = build_delivery_analytics(ev, colmap, idle_min=idle_min)

# Pre-format H:MM:SS fields for drill-down
data["gap_hms"]      = data["__gap_s"].map(fmt_hms)
data["walk_gap_hms"] = data["__walk_gap_s"].map(fmt_hms)
data["dwell_hms"]    = data["__dwell_s"].map(fmt_hms)
data["visit_hms"]    = data["visit_duration_s"].map(fmt_hms)

# Baselines from history + current (so the model gets smarter)
baselines = compute_baselines(history, data, colmap)

# Tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs(
    ["📈 Overview", "🚶 Delivery Analytics", "🧑‍🔧 Tech Comparison",
     "📦 Devices", "⏱ Hourly", "🧪 Drill-down", "🚨 Outliers", "🔟 Weekly Top 10", "❓ Ask the data"]
)

with tab1:
    st.subheader("Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Events", f"{len(ev):,}")
    c2.metric("Devices", f"{ev[colmap['device']].nunique():,}")
    c3.metric("Users", f"{ev[colmap['user']].nunique():,}")
    c4.metric("Types", f"{ev[colmap['type']].nunique():,}")

    week_df = weekly_summary(ev, colmap)
    if not week_df.empty:
        fig = px.bar(week_df, x="week", y="events", title="Weekly events")
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Delivery Analytics (per-tech sequences)")
    c1, c2 = st.columns(2)
    hg = data["__walk_gap_s"].dropna()
    if not hg.empty:
        fig = px.histogram(hg, nbins=40, title="Walking/Travel gaps (seconds)")
        c1.plotly_chart(fig, use_container_width=True)
        c1.caption("X-axis = seconds between finishing a device and starting the next (≥ threshold & device changed).")
    dw = data.loc[~data["__device_change"], "__gap_s"].dropna()
    if not dw.empty:
        fig2 = px.histogram(dw, nbins=40, title="Same-device dwell gaps (seconds)")
        c2.plotly_chart(fig2, use_container_width=True)
    st.markdown("**Tip:** Use the *Drill-down* tab to inspect rows behind any long gaps.")

with tab3:
    st.subheader("Tech comparison")
    st.dataframe(tech_stats, use_container_width=True)
    if not tech_stats.empty:
        fig = px.bar(tech_stats, x=colmap["user"], y="median_walk_gap_s", title="Median walk gap by tech (s)")
        st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.subheader("Devices")
    st.dataframe(device_stats, use_container_width=True)
    if not device_stats.empty:
        fig = px.bar(device_stats.head(25), x=colmap["device"], y="events", title="Top devices by event volume")
        st.plotly_chart(fig, use_container_width=True)

with tab5:
    st.subheader("Hourly cadence")
    st.dataframe(hourly, use_container_width=True)
    if not hourly.empty:
        fig = px.line(hourly, x="hour", y="events", markers=True, title="Events by hour")
        st.plotly_chart(fig, use_container_width=True)

with tab6:
    st.subheader("Drill-down (click a row → review source)")
    show_cols = [
        colmap["datetime"], colmap["user"], colmap["device"], colmap["type"],
        "gap_hms", "walk_gap_hms", "dwell_hms", "visit_hms",
        "__gap_s", "__walk_gap_s", "__dwell_s", "visit_duration_s",
        "__device_change",
    ]
    if colmap.get("desc") and colmap["desc"] in data.columns: show_cols.insert(4, colmap["desc"])
    if colmap.get("qty")  and colmap["qty"]  in data.columns: show_cols.insert(5, colmap["qty"])
    if colmap.get("medid")and colmap["medid"]in data.columns: show_cols.insert(6, colmap["medid"])
    show_cols = [c for c in show_cols if c in data.columns]
    table = data[show_cols].copy()
    st.dataframe(table, use_container_width=True, height=520)
    st.download_button("Download current drill-down as CSV",
                       data=table.to_csv(index=False).encode("utf-8"),
                       file_name="drilldown.csv", mime="text/csv")

    st.markdown("### Per-visit summary (time per continuous stop at a device)")
    usr = colmap["user"]; dev = colmap["device"]
    visit_show = visit[[usr, dev, "start", "end", "visit_duration_s"]].copy()
    visit_show["visit_hms"] = visit_show["visit_duration_s"].map(fmt_hms)
    st.dataframe(visit_show[[usr, dev, "start", "end", "visit_hms", "visit_duration_s"]],
                 use_container_width=True, height=360)
    st.download_button("Download visit summary as CSV",
                       data=visit_show.to_csv(index=False).encode("utf-8"),
                       file_name="visit_summary.csv", mime="text/csv")

with tab7:
    st.subheader("🚨 Outliers (robust median + MAD baselines)")
    c1, c2, c3 = st.columns(3)
    k_walk = c1.slider("Walk gap sensitivity (k·MAD)", 1.0, 6.0, 3.0, 0.5)
    k_dwell= c2.slider("Dwell gap sensitivity (k·MAD)", 1.0, 6.0, 3.0, 0.5)
    k_vol  = c3.slider("Volume sensitivity (k·MAD)",   1.0, 6.0, 3.0, 0.5)

    walk_out, dwell_out, vol_out = find_outliers(data, ev, baselines, colmap, k_walk, k_dwell, k_vol)

    st.markdown("#### Walking outliers (time **between** devices)")
    if walk_out.empty:
        st.info("No walking outliers at current sensitivity.")
    else:
        show = walk_out[[colmap["datetime"], colmap["user"], colmap["device"],
                         "gap_hms","baseline_hms","__walk_gap_s","median","z_mad"]].rename(
            columns={"median":"baseline_s","__walk_gap_s":"walk_gap_s"}
        )
        st.dataframe(show, use_container_width=True, height=300)
        st.download_button("Download walking outliers CSV",
                           data=show.to_csv(index=False).encode("utf-8"),
                           file_name="walk_outliers.csv", mime="text/csv")

    st.markdown("#### Dwell outliers (time **on** a device)")
    if dwell_out.empty:
        st.info("No dwell outliers at current sensitivity.")
    else:
        show = dwell_out[[colmap["datetime"], colmap["user"], colmap["device"],
                          "dwell_hms","baseline_hms","__gap_s","median","z_mad"]].rename(
            columns={"median":"baseline_s","__gap_s":"dwell_s"}
        )
        st.dataframe(show, use_container_width=True, height=300)
        st.download_button("Download dwell outliers CSV",
                           data=show.to_csv(index=False).encode("utf-8"),
                           file_name="dwell_outliers.csv", mime="text/csv")

    st.markdown("#### Volume outliers (events per device-hour)")
    if vol_out.empty:
        st.info("No volume outliers at current sensitivity.")
    else:
        show = vol_out.rename(columns={"events_now":"events_now", "median":"baseline_events"})
        st.dataframe(show[["Device","hour","events_now","baseline_events","mad","z_mad","n"]],
                     use_container_width=True, height=240)
        st.download_button("Download volume outliers CSV",
                           data=show.to_csv(index=False).encode("utf-8"),
                           file_name="volume_outliers.csv", mime="text/csv")

with tab8:
    st.subheader("Weekly Top 10 (last 7d vs prior 7d)")
    # Simple digest: we can reuse hourly and device deltas later; placeholder
    st.info("Top 10 weekly anomalies will populate as history grows (kept in local DB).")

with tab9:
    st.subheader("Ask the data")
    q = st.text_input("Type a question (e.g., 'top devices', 'median walk gap for Melissa')")
    if q:
        ans, tbl = qa_answer(q, ev, data, colmap)
        st.write(ans)
        if not tbl.empty:
            st.dataframe(tbl, use_container_width=True, height=420)
            st.download_button("Download answer table as CSV",
                               tbl.to_csv(index=False).encode("utf-8"),
                               "qa_answer.csv", "text/csv")
    else:
        st.info("Examples: 'top devices', 'longest dwell devices', 'median walk gap for <name>', 'busiest hour'.")

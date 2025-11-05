# app.py
# Device Event Insights — Pro (Supabase Postgres + cache-busted engine)
# - Multi-file upload (xlsx/csv)  |  DB-only resume toggle
# - Column mapping with duplicate guard
# - Safe datetime parsing; duplicate header trimming
# - Durable history in Postgres (UPSERT by pk, chunked)
# - Delivery analytics: walk gaps, dwell, visit/run durations (with idle_max cap)
# - Drill-down with H:MM:SS + CSV export
# - Weekly summary + Top-10 anomalies (7d vs prior 7d)
# - Outliers (IQR) per user/device
# - Index maintenance button (concurrent create)
# - FutureWarning-safe (observed=True, no categorical group keys)
from __future__ import annotations

import hashlib
import io
import os
import re
from datetime import timedelta
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sqlalchemy import create_engine, text

# ----------------------------- CONFIG ---------------------------------
st.set_page_config(
    page_title="Device Event Insights",
    layout="wide",
    initial_sidebar_state="expanded",
)

DEFAULT_COLMAP = {
    "datetime": "TransactionDateTime",
    "device": "Device",
    "user": "UserName",
    "type": "TransactionType",
    # optional:
    "desc": "MedDescription",
    "qty": "Quantity",
    "medid": "MedID",
}

DEFAULT_IDLE_MIN = 30  # seconds to qualify as a "walk/travel" gap

# ----------------------------- HELPERS --------------------------------

def dedupe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure column names are unique by dropping exact dupes; trim whitespace."""
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()].copy()
    df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]
    return df

def parse_datetime_series(s: pd.Series) -> pd.Series:
    """Parse to pandas datetime (UTC-naive)."""
    out = pd.to_datetime(s, errors="coerce", utc=True)
    if pd.api.types.is_datetime64tz_dtype(out):
        out = out.dt.tz_convert("UTC").dt.tz_localize(None)
    return out

def base_clean(df_raw: pd.DataFrame, colmap: Dict[str, str]) -> pd.DataFrame:
    out = dedupe_columns(df_raw).copy()

    # Safe handle for possible duplicate-named datetime columns
    dtcol = colmap["datetime"]
    if dtcol not in out.columns:
        raise ValueError(f"Mapped datetime column '{dtcol}' not found in file.")
    s = out[dtcol]
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    out[dtcol] = parse_datetime_series(s)

    # Optional numerics
    if colmap.get("qty") and colmap["qty"] in out.columns:
        out[colmap["qty"]] = pd.to_numeric(out[colmap["qty"]], errors="coerce")

    # Sanitize strings (use string dtype; avoid categoricals to prevent groupby blowups)
    for key in ["device", "user", "type", "desc", "medid"]:
        c = colmap.get(key)
        if c and c in out.columns:
            out[c] = out[c].astype("string").str.strip()

    out = out.dropna(subset=[dtcol]).copy()
    out = out.sort_values(dtcol).reset_index(drop=True)

    # Engineered calendar columns
    out["__date"] = out[dtcol].dt.date
    out["__hour"] = out[dtcol].dt.hour
    out["__dow"] = out[dtcol].dt.day_name()
    return out

def fmt_hms(x) -> str:
    if pd.isna(x):
        return ""
    x = int(round(float(x)))
    h, r = divmod(x, 3600)
    m, s = divmod(r, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"

def safe_unique(df: pd.DataFrame, col: str) -> List[str]:
    if col not in df.columns:
        return []
    return sorted([x for x in df[col].dropna().astype(str).unique()])

def build_pk(df: pd.DataFrame, colmap: Dict[str, str]) -> pd.Series:
    cols = []
    for k in ["datetime","device","user","type","desc","qty","medid"]:
        c = colmap.get(k)
        if c and c in df.columns:
            cols.append(df[c].astype(str))
        else:
            cols.append(pd.Series([""], index=df.index, dtype="string"))
    arr = np.vstack([c.values for c in cols]).T
    out = [hashlib.sha1("|".join(row).encode("utf-8")).hexdigest() for row in arr]
    return pd.Series(out, index=df.index, dtype="string")

def weekly_summary(ev: pd.DataFrame, colmap: Dict[str, str]) -> pd.DataFrame:
    ts  = colmap["datetime"]; dev = colmap["device"]; usr = colmap["user"]; typ = colmap["type"]
    df = ev.copy()
    df["week"] = df[ts].dt.to_period("W-SUN").apply(lambda p: p.start_time.date())
    out = (
        df.groupby("week", observed=True)
        .agg(
            events=(typ, "count"),
            devices=(dev, "nunique"),
            techs=(usr, "nunique"),
        )
        .reset_index()
        .sort_values("week")
    )
    return out

def _non_empty_frames(frames: List[pd.DataFrame]) -> List[pd.DataFrame]:
    out = []
    for df in frames:
        if isinstance(df, pd.DataFrame) and not df.empty:
            if not df.isna().all(axis=None):
                out.append(df)
    return out

def load_upload(up) -> pd.DataFrame:
    """Read a CSV/XLSX upload to a DataFrame with deduped headers."""
    if up.name.lower().endswith(".xlsx"):
        df = pd.read_excel(up)
    else:
        try:
            df = pd.read_csv(up)
        except UnicodeDecodeError:
            up.seek(0)
            df = pd.read_csv(up, encoding="latin-1")
    return dedupe_columns(df)

# ------------------ CORE ANALYTICS (robust, observed=True) ---------------------
def build_delivery_analytics(
    ev: pd.DataFrame,
    colmap: Dict[str, str],
    idle_min: int,
    idle_max: int | None = 1800,   # 0 or None disables cap
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      data: events with next-event/gap/dwell + visit duration per row
      device_stats: per device volume + median dwell
      tech_stats: per tech volume + median walk gap
      run_stats: per (tech, run) aggregated sequences
      hourly: events by hour
      visit: per-visit (tech, device) durations
    """
    ts  = colmap["datetime"]; dev = colmap["device"]; usr = colmap["user"]; typ = colmap["type"]

    needed = [ts, dev, usr, typ]
    if colmap.get("desc")  and colmap["desc"]  in ev.columns: needed.append(colmap["desc"])
    if colmap.get("medid") and colmap["medid"] in ev.columns: needed.append(colmap["medid"])
    if colmap.get("qty")   and colmap["qty"]   in ev.columns: needed.append(colmap["qty"])
    needed = list(dict.fromkeys(needed))  # keep order, drop dupes

    # Ensure unique cols / slice and sort
    ev = ev.loc[:, ~ev.columns.duplicated()].copy()
    data = ev[needed].sort_values([usr, ts]).copy()

    # Normalize string keys early (avoid nullable bools & groupby weirdness)
    for c in [dev, usr, typ]:
        if c in data.columns:
            data[c] = data[c].astype("string").fillna("[unknown]")

    # next-event per tech
    data["__next_ts"]  = data.groupby(usr, observed=True)[ts].shift(-1)
    data["__next_dev"] = data.groupby(usr, observed=True)[dev].shift(-1)

    # gaps (seconds)
    data["__gap_s"] = (data["__next_ts"] - data[ts]).dt.total_seconds()

    # device change flag (plain bool; no NA)
    data["__device_change"] = (data[dev] != data["__next_dev"]).fillna(False)

    # walk gap: device changed and gap >= threshold (and <= idle_max if set)
    cap_ok = (data["__gap_s"] <= float(idle_max)) if (idle_max and idle_max > 0) else True
    data["__walk_gap_s"] = np.where(
        data["__device_change"] & (data["__gap_s"] >= float(idle_min)) & cap_ok,
        data["__gap_s"],
        np.nan,
    )

    # dwell: same device consecutive events
    data["__dwell_s"] = np.where(~data["__device_change"], data["__gap_s"], np.nan)

    # visit id: bump when device changes within user (NA-safe)
    prev_dev = data.groupby(usr, observed=True)[dev].shift()
    device_changed = (data[dev] != prev_dev).fillna(False)  # pure bool
    data["__visit_id"] = device_changed.astype("int8").groupby(data[usr], observed=True).cumsum()

    # visit summary
    visit = (
        data.groupby([usr, "__visit_id", dev], as_index=False, observed=True)
            .agg(start=(ts, "min"), end=(ts, "max"))
    )
    visit["visit_duration_s"] = (visit["end"] - visit["start"]).dt.total_seconds()

    # attach visit duration to rows
    data = data.merge(
        visit[[usr, "__visit_id", "visit_duration_s"]],
        on=[usr, "__visit_id"],
        how="left",
        validate="m:1",
    )

    # device stats
    device_counts = data.groupby(dev, observed=True).size().rename("events")
    device_dwell  = (
        data.loc[~data["__device_change"], ["__gap_s", dev]]
            .groupby(dev, observed=True)["__gap_s"].median()
            .rename("median_dwell_s")
    )
    device_stats = (
        pd.concat([device_counts, device_dwell], axis=1)
          .fillna(0)
          .sort_values("events", ascending=False)
          .reset_index()
    )

    # tech stats
    tech_counts = data.groupby(usr, observed=True).size().rename("events")
    tech_walk   = data.groupby(usr, observed=True)["__walk_gap_s"].median().rename("median_walk_gap_s")
    tech_stats  = (
        pd.concat([tech_counts, tech_walk], axis=1)
          .fillna(0)
          .sort_values("events", ascending=False)
          .reset_index()
    )

    # run sequences: break whenever there is a walk gap (already thresholded)
    data["__is_break"] = data["__walk_gap_s"].notna()
    data["__run_id"]   = data["__is_break"].astype("int8").groupby(data[usr], observed=True).cumsum()

    run_stats = (
        data.groupby([usr, "__run_id"], as_index=False, observed=True)
            .agg(
                start=(ts, "min"),
                end=(ts, "max"),
                n_events=(ts, "count"),
                n_devices=(dev, "nunique"),
                total_walk_s=("__walk_gap_s", lambda s: float(np.nansum(s.values))),
            )
    )
    run_stats["duration_s"] = (run_stats["end"] - run_stats["start"]).dt.total_seconds()

    # hourly rollup
    hourly = (
        data.groupby(data[ts].dt.floor("h"), observed=True)
            .size()
            .rename("events")
            .reset_index()
            .rename(columns={ts: "hour"})
    )

    return data, device_stats, tech_stats, run_stats, hourly, visit

# ------------------------- PERSISTENCE (POSTGRES via Supabase) ----------------------

@st.cache_resource
def get_engine(db_url: str, salt: str):
    """Create a cached SQLAlchemy engine. Changing db_url or salt busts the cache."""
    return create_engine(db_url, pool_pre_ping=True)

def init_db(eng):
    """Create canonical schema & indexes (fast statements)."""
    ddl = """
    CREATE TABLE IF NOT EXISTS events (
        pk     TEXT PRIMARY KEY,
        dt     TIMESTAMP WITHOUT TIME ZONE NOT NULL,
        device TEXT,
        "user" TEXT,
        "type" TEXT,
        "desc" TEXT,
        qty    DOUBLE PRECISION,
        medid  TEXT
    );
    """
    with eng.begin() as con:
        con.execute(text(ddl))

def ensure_indexes(eng, timeout_sec: int = 15):
    """
    Create/repair indexes concurrently and quickly. If DB is busy, just skip.
    """
    stmts = [
        'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_events_dt     ON events (dt)',
        'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_events_device ON events (device)',
        'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_events_user   ON events ("user")',
        'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_events_type   ON events ("type")',
    ]
    with eng.begin() as con:
        con.execute(text(f"SET LOCAL lock_timeout = '{timeout_sec}s'"))
        for s in stmts:
            try:
                con.execute(text(s))
            except Exception:
                # Non-fatal: index is busy or being built elsewhere
                pass

def refresh_materialized_views(eng):
    # No materialized views yet; keep as a harmless stub.
    return True, "Materialized views refresh: skipped (none configured)."

def _df_to_rows_canonical(df: pd.DataFrame, colmap: Dict[str, str]) -> list[dict]:
    """Map dynamic DataFrame columns into canonical SQL columns."""
    ts  = colmap["datetime"]; dev = colmap["device"]; usr = colmap["user"]; typ = colmap["type"]
    get = lambda k: (colmap.get(k) in df.columns) if colmap.get(k) else False
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "pk":    r["pk"],
            "dt":    pd.to_datetime(r[ts]).to_pydatetime() if pd.notna(r[ts]) else None,
            "device": r.get(dev, None),
            "user":   r.get(usr, None),
            "type":   r.get(typ, None),
            "desc":   r.get(colmap.get("desc", ""), None) if get("desc") else None,
            "qty":    float(r[colmap["qty"]]) if get("qty") and pd.notna(r[colmap["qty"]]) else None,
            "medid":  r.get(colmap.get("medid",""), None) if get("medid") else None,
        })
    return rows

def save_history_sql(df: pd.DataFrame, colmap: Dict[str, str], eng) -> tuple[bool, str]:
    """UPSERT by pk into Postgres (chunked for large loads)."""
    try:
        init_db(eng)
        rows = _df_to_rows_canonical(df, colmap)
        if not rows:
            return True, "No rows to save."

        upsert_sql = text("""
            INSERT INTO events (pk, dt, device, "user", "type", "desc", qty, medid)
            VALUES (:pk, :dt, :device, :user, :type, :desc, :qty, :medid)
            ON CONFLICT (pk) DO UPDATE SET
                dt=EXCLUDED.dt,
                device=EXCLUDED.device,
                "user"=EXCLUDED."user",
                "type"=EXCLUDED."type",
                "desc"=EXCLUDED."desc",
                qty=EXCLUDED.qty,
                medid=EXCLUDED.medid;
        """)

        CHUNK = 5000
        total = 0
        with eng.begin() as con:
            for i in range(0, len(rows), CHUNK):
                batch = rows[i:i+CHUNK]
                con.execute(upsert_sql, batch)
                total += len(batch)
        return True, f"Saved to Postgres: {total:,} rows (upserted by pk)."
    except Exception as e:
        return False, f"DB save error: {e}"

def load_history_sql(colmap: Dict[str, str], eng) -> pd.DataFrame:
    """Load canonical table and alias columns back to current mapping."""
    try:
        init_db(eng)
        sql = """
            SELECT pk, dt, device, "user", "type", "desc", qty, medid
            FROM events
            ORDER BY dt
        """
        raw = pd.read_sql(sql, eng)
        if raw.empty:
            return raw

        rename_back = {
            "dt": colmap["datetime"],
            "device": colmap["device"],
            "user": colmap["user"],
            "type": colmap["type"],
        }
        if colmap.get("desc"):  rename_back["desc"]  = colmap["desc"]
        if colmap.get("qty"):   rename_back["qty"]   = colmap["qty"]
        if colmap.get("medid"): rename_back["medid"] = colmap["medid"]

        out = raw.rename(columns=rename_back)
        out[colmap["datetime"]] = pd.to_datetime(out[colmap["datetime"]], errors="coerce")
        # Keep string types for groupby keys
        for c in [colmap["device"], colmap["user"], colmap["type"]]:
            if c in out.columns:
                out[c] = out[c].astype("string")
        return out
    except Exception:
        return pd.DataFrame()

# ------------------------------------------------------------------------------------

# ----------------------------- UI ------------------------------------
st.title("All Device Event Insights — Pro")

# Build engine once (cache-busted by secrets)
DB_URL = st.secrets["DB_URL"]
ENGINE_SALT = st.secrets.get("ENGINE_SALT", "")
eng = get_engine(DB_URL, ENGINE_SALT)

# ================================
# SIDEBAR: DATA SOURCE + SETTINGS
# ================================
st.sidebar.header("Data source")
data_mode = st.sidebar.radio(
    "Choose data source",
    ["Upload files", "Database only"],
    help="Use 'Database only' to analyze existing Postgres data without uploading new files."
)

# Column map (using defaults for now; add UI later if you want)
colmap: Dict[str, str] = DEFAULT_COLMAP.copy()

idle_min = st.sidebar.number_input(
    "Walk gap threshold min (seconds)",
    min_value=5, max_value=900, value=60, step=5
)
idle_max = st.sidebar.number_input(
    "Walk gap threshold max (seconds, 0 = unlimited)",
    min_value=0, max_value=7200, value=1800, step=60,
    help="Only count walk gaps ≤ this many seconds as walking. Set 0 to disable the cap."
)

st.sidebar.header("Admin")
if st.sidebar.button("🛠 Build/repair DB indexes"):
    try:
        ensure_indexes(eng)
        st.sidebar.success("Index build/repair requested (non-blocking).")
    except Exception as e:
        st.sidebar.warning(f"Index maintenance skipped: {e}")

if st.sidebar.button("🧹 Daily closeout (refresh & clear caches)"):
    ok_mv, mv_msg = refresh_materialized_views(eng)
    if ok_mv:
        st.sidebar.success(mv_msg)
    else:
        st.sidebar.warning(mv_msg)
    try:
        st.cache_data.clear()
        st.cache_resource.clear()
    except Exception:
        pass
    st.sidebar.info("Caches cleared — rerunning app...")
    st.rerun()

# ===================================
# DATA LOAD: UPLOAD OR DATABASE MODE
# ===================================
history = load_history_sql(colmap, eng)
uploads = []
if data_mode == "Upload files":
    st.sidebar.header("1) Upload")
    uploads = st.sidebar.file_uploader(
        "Drag & drop daily XLSX/CSV (one or many)",
        type=["xlsx", "csv"],
        accept_multiple_files=True
    )
    if not uploads and history.empty:
        st.info("Upload daily files or switch to 'Database only' mode.")
        st.stop()
else:
    if history.empty:
        st.warning("No records in database yet. Switch to 'Upload files' first.")
        st.stop()

# ============================
# PROCESS UPLOADS IF PROVIDED
# ============================
if uploads:
    new_files = []
    for up in uploads:
        try:
            raw = load_upload(up)
            cleaned = base_clean(raw, colmap)
            if cleaned.empty:
                st.warning(f"{up.name}: no valid rows.")
                continue
            cleaned["pk"] = build_pk(cleaned, colmap)
            new_files.append(cleaned)
        except Exception as e:
            st.error(f"Failed to read {up.name}: {e}")

    if not new_files:
        st.warning("No usable files found.")
        st.stop()

    new_ev = pd.concat(new_files, ignore_index=True)
    new_ev = new_ev.drop_duplicates(subset=["pk"]).sort_values(colmap["datetime"])

    # Merge with DB history
    frames = [history, new_ev] if not history.empty else [new_ev]
    ev_all = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    ev_all = ev_all.drop_duplicates(subset=["pk"]).sort_values(colmap["datetime"])

    # ---- Upload summary ----
    new_pks = set(new_ev["pk"])
    old_pks = set(history["pk"]) if not history.empty and "pk" in history.columns else set()
    num_new = len(new_pks - old_pks)
    num_dup = len(new_pks & old_pks)
    earliest = pd.to_datetime(ev_all[colmap["datetime"]].min())
    latest = pd.to_datetime(ev_all[colmap["datetime"]].max())

    with st.expander("📥 Upload summary", expanded=True):
        st.write(f"**Rows in this upload:** {len(new_ev):,}")
        st.write(f"- New rows vs DB: **{num_new:,}**")
        st.write(f"- Already existed (upserts): **{num_dup:,}**")
        st.write(f"**History time range:** {earliest:%Y-%m-%d %H:%M} → {latest:%Y-%m-%d %H:%M}")

    # Save to DB
    ok, msg = save_history_sql(ev_all, colmap, eng)
    st.sidebar.success(msg if ok else msg)
else:
    ev_all = history.copy()

if ev_all.empty:
    st.warning("No events available yet. Upload files or verify your DB.")
    st.stop()

# ===================
# TIME + BASIC FILTERS (BEFORE ANALYTICS)
# ===================
_min = pd.to_datetime(ev_all[colmap["datetime"]].min())
_max = pd.to_datetime(ev_all[colmap["datetime"]].max())
min_ts = _min.to_pydatetime()
max_ts = _max.to_pydatetime() if _max > _min else (min_ts + timedelta(minutes=1))

rng = st.sidebar.slider(
    "Time range",
    min_value=min_ts,
    max_value=max_ts,
    value=(min_ts, max_ts),
    format="YYYY-MM-DD HH:mm"
)

ev_time = ev_all[
    (ev_all[colmap["datetime"]] >= pd.to_datetime(rng[0])) &
    (ev_all[colmap["datetime"]] <= pd.to_datetime(rng[1]))
].copy()
if ev_time.empty:
    st.warning("No events in selected time range.")
    st.stop()

# Side filters applied BEFORE analytics so sequences are consistent
pick_devices = st.sidebar.multiselect("Devices", safe_unique(ev_time, colmap["device"]))
pick_users   = st.sidebar.multiselect("Users", safe_unique(ev_time, colmap["user"]))
pick_types   = st.sidebar.multiselect("Transaction types", safe_unique(ev_time, colmap["type"]))

if pick_devices:
    ev_time = ev_time[ev_time[colmap["device"]].isin(pick_devices)]
if pick_users:
    ev_time = ev_time[ev_time[colmap["user"]].isin(pick_users)]
if pick_types:
    ev_time = ev_time[ev_time[colmap["type"]].isin(pick_types)]

if ev_time.empty:
    st.warning("No events after applying device/user/type filters.")
    st.stop()

# ====================================
# ANALYTICS
# ====================================
data, device_stats, tech_stats, run_stats, hourly, visit = build_delivery_analytics(
    ev_time, colmap, idle_min=idle_min, idle_max=idle_max
)

# Pre-format for Drill-down
data["gap_hms"]      = data["__gap_s"].map(fmt_hms)
data["walk_gap_hms"] = data["__walk_gap_s"].map(fmt_hms)
data["dwell_hms"]    = data["__dwell_s"].map(fmt_hms)
data["visit_hms"]    = data["visit_duration_s"].map(fmt_hms)

st.success(f"Loaded {len(data):,} events for analysis.")

# Tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs(
    ["📈 Overview", "🚶 Delivery Analytics", "🧑‍🔧 Tech Comparison",
     "📦 Devices", "⏱ Hourly", "🧪 Drill-down", "🔟 Weekly Top 10",
     "🚨 Outliers", "❓ Ask the data"]
)

with tab1:
    st.subheader("Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Events", f"{len(ev_time):,}")
    c2.metric("Devices", f"{ev_time[colmap['device']].nunique():,}")
    c3.metric("Users", f"{ev_time[colmap['user']].nunique():,}")
    c4.metric("Types", f"{ev_time[colmap['type']].nunique():,}")

    week_df = weekly_summary(ev_time, colmap)
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
    st.subheader("Drill-down (exportable)")
    show_cols = [
        colmap["datetime"], colmap["user"], colmap["device"], colmap["type"],
        "gap_hms", "walk_gap_hms", "dwell_hms", "visit_hms",
        "__gap_s", "__walk_gap_s", "__dwell_s", "visit_duration_s",
        "__device_change",
    ]
    if colmap.get("desc") and colmap["desc"] in data.columns:
        show_cols.insert(4, colmap["desc"])
    if colmap.get("qty") and colmap["qty"] in data.columns:
        show_cols.insert(5, colmap["qty"])
    if colmap.get("medid") and colmap["medid"] in data.columns:
        show_cols.insert(6, colmap["medid"])
    show_cols = [c for c in show_cols if c in data.columns]

    table = data[show_cols].copy()
    st.dataframe(table, use_container_width=True, height=520)

    st.download_button(
        "Download current drill-down as CSV",
        data=table.to_csv(index=False).encode("utf-8"),
        file_name="drilldown.csv",
        mime="text/csv"
    )

    st.markdown("### Per-visit summary (continuous time at a device)")
    ts  = colmap["datetime"]; dev = colmap["device"]; usr = colmap["user"]
    visit_show = visit[[usr, dev, "start", "end", "visit_duration_s"]].copy()
    visit_show["visit_hms"] = visit_show["visit_duration_s"].map(fmt_hms)

    st.dataframe(
        visit_show[[usr, dev, "start", "end", "visit_hms", "visit_duration_s"]],
        use_container_width=True,
        height=360
    )

with tab7:
    st.subheader("Weekly Top 10 (signals)")
    digest = anomalies_top10(ev_all, data, colmap)  # use the full history frame
    if digest.empty:
        st.info("No notable anomalies in the last 7 days window.")
    else:
        st.dataframe(digest, use_container_width=True)

with tab8:
    st.subheader("Outliers")
    ow = outliers_iqr(data.dropna(subset=["__walk_gap_s"]), colmap["user"], "__walk_gap_s", "Walk gap")
    od = outliers_iqr(data.dropna(subset=["__dwell_s"]),    colmap["device"], "__dwell_s",    "Dwell")
    c1, c2 = st.columns(2)
    c1.write("By user (walk gap):")
    c1.dataframe(ow, use_container_width=True, height=320)
    c2.write("By device (dwell):")
    c2.dataframe(od, use_container_width=True, height=320)

with tab9:
    st.subheader("Ask the data ❓")
    q = st.text_input("Try e.g. 'top devices', 'longest dwell devices', 'median walk gap for Melissa', 'busiest hour'")
    if q:
        ans, tbl = qa_answer(q, ev_time, data, colmap)
        st.write(ans)
        if not tbl.empty:
            st.dataframe(tbl, use_container_width=True)

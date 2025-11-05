# app.py
# Device Event Insights — Pro (Supabase Postgres + cache-busted engine)
# - Multi-file upload (xlsx/csv)  |  DB-only resume toggle
# - Column mapping with duplicate guard
# - Safe datetime parsing; duplicate header trimming
# - Durable history in Postgres (UPSERT by pk, chunked)
# - Delivery analytics: walk gaps, dwell, visit/run durations
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

# ------------------ CORE ANALYTICS (robust, observed=True) ---------------------
# ------------------ CORE ANALYTICS (robust, observed=True) ---------------------
def build_delivery_analytics(
    ev: pd.DataFrame,
    colmap: Dict[str, str],
    idle_min: int,
    idle_max: int | None = 1800,   # ← cap long gaps (0 or None disables cap)
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
    ts  = colmap["datetime"]
    dev = colmap["device"]
    usr = colmap["user"]
    typ = colmap["type"]

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

    # --- visit id: bump when device changes within user (NA-safe) ---
    prev_dev = data.groupby(usr, observed=True)[dev].shift()
    device_changed = (data[dev] != prev_dev).fillna(False)  # pure bool
    data["__visit_id"] = device_changed.astype("int8").groupby(data[usr], observed=True).cumsum()

    # visit summary: start/end/duration for each (tech, visit_id, device)
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




def anomalies_top10(history: pd.DataFrame, data: pd.DataFrame, colmap: Dict[str,str]) -> pd.DataFrame:
    ts  = colmap["datetime"]; dev = colmap["device"]; usr = colmap["user"]; typ = colmap["type"]
    out = []
    if history.empty:
        return pd.DataFrame(columns=["rank","topic","detail","why","severity"])

    hist = history.copy()
    hist[ts] = pd.to_datetime(hist[ts], errors="coerce")
    hist = hist.dropna(subset=[ts])
    if hist.empty:
        return pd.DataFrame(columns=["rank","topic","detail","why","severity"])

    end = hist[ts].max()
    start_recent = end - pd.Timedelta(days=7)
    start_prior  = start_recent - pd.Timedelta(days=7)

    recent = hist[(hist[ts] > start_recent) & (hist[ts] <= end)]
    prior  = hist[(hist[ts] > start_prior)  & (hist[ts] <= start_recent)]

    # 1) Device volume spike
    if not recent.empty and not prior.empty:
        r_dev = recent.groupby(dev, observed=True).size().rename("recent").reset_index()
        p_dev = prior.groupby(dev, observed=True).size().rename("prior").reset_index()
        vol = r_dev.merge(p_dev, on=dev, how="left").fillna(0.0)
        vol["delta"] = vol["recent"] - vol["prior"]
        vol["pct"]   = np.where(vol["prior"]>0, (vol["recent"]-vol["prior"])/vol["prior"], np.nan)
        vol = vol.sort_values(["pct","delta"], ascending=False).head(3)
        for _, r in vol.iterrows():
            out.append({
                "topic":"Device volume spike",
                "detail":f"{r[dev]} recent {int(r['recent'])} vs prior {int(r['prior'])}",
                "why":"Sudden workload shift; check staffing/stocking cadence",
                "severity":"high" if (r["pct"]>=0.5 and r["recent"]>=50) else "med"
            })

    # 2) Techs with largest median walk gap (current filtered 'data')
    if not data.empty and usr in data.columns:
        twalk = data["__walk_gap_s"].groupby(data[usr], observed=True).median().dropna()
        if not twalk.empty:
            top_walk = twalk.sort_values(ascending=False).head(3)
            for u, s in top_walk.items():
                out.append({
                    "topic":"High walking time",
                    "detail":f"{u} median walk gap {fmt_hms(s)} ({int(s)}s)",
                    "why":"Inefficient routing or distant devices in their run",
                    "severity":"med" if s>=120 else "low"
                })

    # 3) Devices with largest median dwell (current filtered 'data')
    if not data.empty and dev in data.columns:
        dwell = (
            data.loc[~data["__device_change"], "__gap_s"]
            .groupby(data[dev], observed=True).median().dropna()
        )
        if not dwell.empty:
            top_dwell = dwell.sort_values(ascending=False).head(3)
            for d, s in top_dwell.items():
                out.append({
                    "topic":"Long dwell at device",
                    "detail":f"{d} median dwell {fmt_hms(s)} ({int(s)}s)",
                    "why":"Many refills per stop or slow transactions; check slot layout",
                    "severity":"med" if s>=60 else "low"
                })

    # 4) Busiest hours (recent)
    if not recent.empty:
        rh = recent.groupby(recent[ts].dt.floor("h")).size()
        if not rh.empty:
            hr_top = rh.sort_values(ascending=False).head(2)
            for h, n in hr_top.items():
                out.append({
                    "topic":"Rush hour",
                    "detail":f"{h:%Y-%m-%d %H:%M} had {int(n)} events",
                    "why":"Consider JIT timing / more techs in this window",
                    "severity":"med" if n>=100 else "low"
                })

    # 5) Transaction-type surges (recent vs prior)
    if not recent.empty and not prior.empty and typ in hist.columns:
        r_t = recent.groupby(typ, observed=True).size().rename("recent").reset_index()
        p_t = prior.groupby(typ, observed=True).size().rename("prior").reset_index()
        typd = r_t.merge(p_t, on=typ, how="left").fillna(0.0)
        typd["delta"] = typd["recent"]-typd["prior"]
        typd = typd.sort_values("delta", ascending=False).head(2)
        for _, r in typd.iterrows():
            out.append({
                "topic":"Transaction-type surge",
                "detail":f"{r[typ]} recent {int(r['recent'])} vs prior {int(r['prior'])}",
                "why":"Upstream demand or workflow change",
                "severity":"med" if r["delta"]>=30 else "low"
            })

    if not out:
        return pd.DataFrame(columns=["rank","topic","detail","why","severity"])
    df_out = pd.DataFrame(out)
    sev_rank = df_out["severity"].map({"high":3,"med":2,"low":1}).fillna(1)
    df_out = df_out.iloc[sev_rank.sort_values(ascending=False).index].reset_index(drop=True)
    df_out.insert(0, "rank", np.arange(1, len(df_out)+1))
    return df_out.head(10)

def outliers_iqr(data: pd.DataFrame, key_col: str, value_col: str, label: str) -> pd.DataFrame:
    """IQR-based outlier detection per key_col on value_col."""
    df = data[[key_col, value_col]].dropna().copy()
    if df.empty:
        return pd.DataFrame(columns=[key_col, value_col, "z_note"])

    def _flag(group):
        q1 = group[value_col].quantile(0.25)
        q3 = group[value_col].quantile(0.75)
        iqr = q3 - q1
        upper = q3 + (1.5 if iqr == 0 else 1.5 * iqr)
        return group[group[value_col] > upper].assign(
            z_note=f"{label}: > Q3+1.5*IQR (>{upper:.1f}s)"
        )

    out = df.groupby(key_col, dropna=True, observed=True).apply(_flag).reset_index(drop=True)
    return out

def qa_answer(question: str, ev: pd.DataFrame, data: pd.DataFrame, colmap: Dict[str,str]) -> Tuple[str, pd.DataFrame]:
    q = question.strip().lower()
    ts  = colmap["datetime"]; dev = colmap["device"]; usr = colmap["user"]; typ = colmap["type"]

    if re.search(r"\b(top|most)\b.*\bdevices?\b", q):
        t = ev.groupby(dev, observed=True).size().rename("events").reset_index().sort_values("events", ascending=False).head(10)
        return f"Top devices by event volume (showing {len(t)}):", t

    if "longest" in q and "dwell" in q:
        if data.empty:
            return "No dwell data in current filter.", pd.DataFrame()
        t = (
            data.loc[~data["__device_change"], ["__gap_s"]]
            .groupby(data[dev], observed=True).median().rename(columns={"__gap_s":"median_dwell_s"})
            .sort_values("median_dwell_s", ascending=False).head(10).reset_index()
        )
        t["median_dwell_hms"] = t["median_dwell_s"].map(fmt_hms)
        return "Devices with longest median dwell:", t

    m = re.search(r"median .*walk.* for (.+)", q)
    if m:
        name = m.group(1).strip()
        sub = data[data[usr].str.lower()==name.lower()]
        if sub.empty:
            return f"No rows found for user '{name}'.", pd.DataFrame()
        val = np.nanmedian(sub["__walk_gap_s"].values)
        return f"Median walk gap for {name}: {fmt_hms(val)} ({int(val)}s)", pd.DataFrame()

    if "hour" in q:
        t = ev.groupby(ev[ts].dt.floor("h")).size().rename("events").reset_index().rename(columns={ts:"hour"})
        if t.empty:
            return "No hourly data in current filter.", pd.DataFrame()
        top = t.sort_values("events", ascending=False).head(1).iloc[0]
        return f"Busiest hour: {top['hour']:%Y-%m-%d %H:%M} with {int(top['events'])} events.", t.sort_values("hour")

    if "which tech" in q and "median walk" in q:
        if data.empty:
            return "No walk-gap data in current filter.", pd.DataFrame()
        t = data.groupby(usr, observed=True)["__walk_gap_s"].median().reset_index().rename(columns={"__walk_gap_s":"median_walk_s"})
        t = t.sort_values("median_walk_s", ascending=False)
        top = t.iloc[0]
        return f"Highest median walk gap: {top[usr]} at {fmt_hms(top['median_walk_s'])}.", t

    tbl = ev[[ts, usr, dev, typ]].head(50)
    return ("Try asks like: 'top devices', 'longest dwell devices', "
            "'median walk gap for Melissa', 'busiest hour'."), tbl

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

def refresh_materialized_views(eng):
    # No materialized views yet; keep as a harmless stub.
    return True, "Materialized views refresh: skipped (none configured)."

    # per-statement timeout
    with eng.begin() as con:
        con.execute(text(f"SET LOCAL lock_timeout = '{timeout_sec}s'"))
        for s in stmts:
            try:
                con.execute(text(s))
            except Exception:
                # Non-fatal: index is busy or being built elsewhere
                pass

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
    if not uploads:
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
        new = base_clean(load_upload(up), colmap)
        if new.empty:
            st.warning(f"{up.name}: no valid rows.")
            continue
        new_files.append(new)

    if not new_files:
        st.warning("No usable files found.")
        st.stop()

    new_ev = pd.concat(new_files, ignore_index=True)
    new_ev = new_ev.drop_duplicates(subset=["pk"]).sort_values(colmap["datetime"])

    # Merge with DB history
    frames = [history, new_ev] if not history.empty else [new_ev]
    combined = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    combined = combined.drop_duplicates(subset=["pk"]).sort_values(colmap["datetime"])

    # ---- Upload summary ----
    new_pks = set(new_ev["pk"])
    old_pks = set(history["pk"]) if not history.empty and "pk" in history.columns else set()
    num_new = len(new_pks - old_pks)
    num_dup = len(new_pks & old_pks)
    earliest = pd.to_datetime(combined[colmap["datetime"]].min())
    latest = pd.to_datetime(combined[colmap["datetime"]].max())

    with st.expander("📥 Upload summary", expanded=True):
        st.write(f"**Rows in this upload:** {len(new_ev):,}")
        st.write(f"- New rows vs DB: **{num_new:,}**")
        st.write(f"- Already existed (upserts): **{num_dup:,}**")
        st.write(f"**History time range:** {earliest:%Y-%m-%d %H:%M} → {latest:%Y-%m-%d %H:%M}")

    # Save to DB
    ok, msg = save_history_sql(combined, colmap, eng)
    st.sidebar.success(msg if ok else msg)
    ev_all = combined.copy()

else:
    ev_all = history.copy()

# ===================
# TIME RANGE FILTER
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

# ====================================
# SMARTER ANALYTICS (KEEP FULL SEQUENCE)
# ====================================
data, device_stats, tech_stats, run_stats, hourly, visit = build_delivery_analytics(
    ev, colmap, idle_min=idle_min, idle_max=idle_max
)

# ===========
# FILTERS
# ===========
pick_devices = st.sidebar.multiselect("Devices", safe_unique(ev_time, colmap["device"]))
pick_users   = st.sidebar.multiselect("Users", safe_unique(ev_time, colmap["user"]))
pick_types   = st.sidebar.multiselect("Transaction types", safe_unique(ev_time, colmap["type"]))

mask = pd.Series(True, index=data_full.index)
if pick_devices:
    mask &= data_full[colmap["device"]].isin(pick_devices)
if pick_users:
    mask &= data_full[colmap["user"]].isin(pick_users)
if pick_types:
    mask &= data_full[colmap["type"]].isin(pick_types)

data   = data_full.loc[mask].copy()
visit  = visit_full[visit_full[colmap["user"]].isin(data[colmap["user"]].unique())]
hourly = hourly_full
device_stats = (data.groupby(colmap["device"]).size().rename("events").reset_index()
                    .sort_values("events", ascending=False))
tech_stats = (data.groupby(colmap["user"]).size().rename("events").reset_index()
                  .sort_values("events", ascending=False))
run_stats = run_stats_full[run_stats_full[colmap["user"]].isin(data[colmap["user"]].unique())]

# ============================
# CONTINUE TO DASHBOARD BUILD
# ============================
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
    digest = anomalies_top10(combined, data, colmap)
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
        ans, tbl = qa_answer(q, ev, data, colmap)
        st.write(ans)
        if not tbl.empty:
            st.dataframe(tbl, use_container_width=True)


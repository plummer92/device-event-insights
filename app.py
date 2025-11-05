# app.py
# Device Event Insights — Pro (Supabase Postgres + cache-busted engine)
# - Upload CSV/XLSX OR analyze Postgres history only
# - Column mapping UI with duplicate guard
# - Durable history (UPSERT by pk), chunked
# - Delivery analytics: walk gaps (min+max cap), dwell, visits, runs
# - Drill-down with H:MM:SS + CSV export
# - Weekly summary, anomalies (7d vs prior 7d), IQR outliers
# - Index maintenance (CONCURRENTLY) with lock timeout
# - FutureWarning-safe (observed=True; no categoricals)
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
    "device":   "Device",
    "user":     "UserName",
    "type":     "TransactionType",
    # optional:
    "desc":     "MedDescription",
    "qty":      "Quantity",
    "medid":    "MedID",
}

# Make a default mapping available immediately (prevents NameError)
# Make a default mapping available immediately (prevents NameError before mapping UI runs)
colmap: Dict[str, str] = DEFAULT_COLMAP.copy()


DEFAULT_IDLE_MIN = 30  # seconds to qualify as "walk/travel" gap

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

def load_upload(up) -> pd.DataFrame:
    """Robust reader for CSV/XLSX uploads."""
    name = up.name.lower()
    if name.endswith(".xlsx"):
        return pd.read_excel(up)
    # CSV: try utf-8 then latin-1
    try:
        up.seek(0)
        return pd.read_csv(up)
    except UnicodeDecodeError:
        up.seek(0)
        return pd.read_csv(up, encoding="latin-1")

def base_clean(df_raw: pd.DataFrame, colmap: Dict[str, str]) -> pd.DataFrame:
    out = dedupe_columns(df_raw).copy()
    dtcol = colmap["datetime"]
    if dtcol not in out.columns:
        raise ValueError(f"Mapped datetime column '{dtcol}' not found in file.")
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
    ts, dev, usr, typ = colmap["datetime"], colmap["device"], colmap["user"], colmap["type"]
    df = ev.copy()
    df["week"] = df[ts].dt.to_period("W-SUN").apply(lambda p: p.start_time.date())
    out = (
        df.groupby("week", observed=True)
          .agg(events=(typ, "count"), devices=(dev, "nunique"), techs=(usr, "nunique"))
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
def build_delivery_analytics(
    ev: pd.DataFrame,
    colmap: Dict[str, str],
    idle_min: int,
    idle_max: int | None = 1800,   # 0/None disables cap
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
    ts, dev, usr, typ = colmap["datetime"], colmap["device"], colmap["user"], colmap["type"]

    needed = [ts, dev, usr, typ]
    for k in ["desc", "medid", "qty"]:
        c = colmap.get(k)
        if c and c in ev.columns:
            needed.append(c)
    needed = list(dict.fromkeys(needed))

    ev = ev.loc[:, ~ev.columns.duplicated()].copy()
    data = ev[needed].sort_values([usr, ts]).copy()

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

    # walk gap with min+max cap
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
    device_changed = (data[dev] != prev_dev).fillna(False)
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

    # run sequences: break whenever there is a walk gap
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
    ts, dev, usr, typ = colmap["datetime"], colmap["device"], colmap["user"], colmap["type"]
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

    # 5) Transaction-type surges
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
    ts, dev, usr, typ = colmap["datetime"], colmap["device"], colmap["user"], colmap["type"]

    if re.search(r"\b(top|most)\b.*\bdevices?\b", q):
        t = ev.groupby(dev, observed=True).size().rename("events").reset_index().sort_values("events", ascending=False).head(10)
        return f"Top devices by event volume (showing {len(t)}):", t

    if "longest" in q and "dwell" in q:
        if data.empty:
            return "No dwell data in current filter.", pd.DataFrame()
        t = (
            data.loc[~data["__device_change"], ["__gap_s"]]
                .groupby(data[dev], observed=True).median().rename(columns={"__gap_s":"median_dwell_s"})
                .sort_values("median_dwell_s", descending=True if hasattr(pd, "NA") else False).head(10).reset_index()
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
    """Create/repair indexes concurrently; skip if busy."""
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
                pass  # non-fatal

def refresh_materialized_views(eng):
    # No MVs yet; placeholder
    return True, "Materialized views refresh: skipped (none configured)."

def _df_to_rows_canonical(df: pd.DataFrame, colmap: Dict[str, str]) -> list[dict]:
    ts, dev, usr, typ = colmap["datetime"], colmap["device"], colmap["user"], colmap["type"]
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
    """UPSERT by pk into Postgres (chunked)."""
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
        for c in [colmap["device"], colmap["user"], colmap["type"]]:
            if c in out.columns:
                out[c] = out[c].astype("string")
        return out
    except Exception:
        return pd.DataFrame()

# ------------------------------------------------------------------------------------

# ----------------------------- UI ------------------------------------
st.title("All Device Event Insights — Pro")

# Engine (cached)
DB_URL = st.secrets["DB_URL"]
ENGINE_SALT = st.secrets.get("ENGINE_SALT", "")
eng = get_engine(DB_URL, ENGINE_SALT)

# ================================ SIDEBAR ==============================
st.sidebar.header("Data source")
data_mode = st.sidebar.radio(
    "Choose data source",
    ["Upload files", "Database only"],
    help="Use 'Database only' to analyze existing Postgres data without uploading new files."
)

idle_min = st.sidebar.number_input(
    "Walk gap threshold min (seconds)",
    min_value=5, max_value=900, value=DEFAULT_IDLE_MIN, step=5
)
idle_max = st.sidebar.number_input(
    "Walk gap threshold max (seconds, 0 = unlimited)",
    min_value=0, max_value=7200, value=1800, step=60,
    help="Only count walk gaps ≤ this many seconds as walking. Set 0 to disable the cap."
)

st.sidebar.header("Admin")
if st.sidebar.button("🧹 Daily closeout (refresh & clear caches)"):
    ok_mv, mv_msg = refresh_materialized_views(eng)
    st.sidebar.success(mv_msg if ok_mv else mv_msg)
    try:
        st.cache_data.clear()
        st.cache_resource.clear()
    except Exception:
        pass
    st.sidebar.info("Caches cleared — rerunning app...")
    st.rerun()

if st.sidebar.button("🛠 Build/repair DB indexes"):
    try:
        ensure_indexes(eng)
        st.sidebar.success("Index build/repair requested.")
    except Exception as e:
        st.sidebar.warning(f"Index maintenance skipped: {e}")

# ===================== LOAD HISTORY (needed early) =====================
history = load_history_sql(colmap, eng)

# =================================== DATA INPUT ===================================
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
        st.warning("No records in database yet. Switch to 'Upload files' to seed data.")
        st.stop()

# Column mapping UI (only when we have an upload to inspect)
if uploads:
    # Peek first file to build mapping options
    sample_df = load_upload(uploads[0])
    sample_df = dedupe_columns(sample_df)
    st.sidebar.header("2) Map columns")
    # Start from defaults but allow override
    for k, default in DEFAULT_COLMAP.items():
        opts = list(sample_df.columns)
        sel = st.sidebar.selectbox(
            f"{k.capitalize()} column",
            options=opts,
            index=opts.index(default) if default in opts else 0,
            key=f"map_{k}",
            help="Pick the matching column from your export",
        )
        colmap[k] = sel

    # Guard: no duplicated mappings
    _selected = [v for v in colmap.values() if v]
    _dups = sorted({c for c in _selected if _selected.count(c) > 1})
    if _dups:
        st.error("You mapped the same column to multiple fields: " + ", ".join(_dups))
        st.stop()

# ============================ PROCESS UPLOADS ============================
if uploads:
    new_files = []
    for up in uploads:
        try:
            raw = load_upload(up)
            cleaned = base_clean(raw, colmap)
            if cleaned.empty:
                st.warning(f"{up.name}: no valid rows.")
                continue
            new_files.append(cleaned)
        except Exception as e:
            st.error(f"Failed to read {up.name}: {e}")

    if not new_files:
        st.warning("No usable files found.")
        st.stop()

    new_ev = pd.concat(new_files, ignore_index=True)
    # Build pk for upload rows (needed for UPSERT)
    new_ev["pk"] = build_pk(new_ev, colmap)
    new_ev = new_ev.drop_duplicates(subset=["pk"]).sort_values(colmap["datetime"])

    # Merge with DB history
    frames = [history, new_ev] if not history.empty else [new_ev]
    ev_all = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    ev_all = ev_all.drop_duplicates(subset=["pk"]).sort_values(colmap["datetime"])

    # Upload summary
    new_pks = set(new_ev["pk"])
    old_pks = set(history["pk"]) if not history.empty and "pk" in history.columns else set()
    num_new = len(new_pks - old_pks)
    num_dup = len(new_pks & old_pks)
    earliest = pd.to_datetime(ev_all[colmap["datetime"]].min())
    latest   = pd.to_datetime(ev_all[colmap["datetime"]].max())

    with st.expander("📥 Upload summary", expanded=True):
        st.write(f"**Rows in this upload:** {len(new_ev):,}")
        st.write(f"- New rows vs DB: **{num_new:,}**")
        st.write(f"- Already existed (upserts): **{num_dup:,}**")
        st.write(f"**History time range:** {earliest:%Y-%m-%d %H:%M} → {latest:%Y-%m-%d %H:%M}")

    ok, msg = save_history_sql(ev_all, colmap, eng)
    (st.sidebar.success if ok else st.sidebar.error)(msg)
else:
    ev_all = history.copy()

if ev_all.empty:
    st.warning("No events available yet. Upload files or verify your DB.")
    st.stop()

# =================== TIME RANGE FILTER ===================
_min = pd.to_datetime(ev_all[colmap["datetime"]].min())
_max = pd.to_datetime(ev_all[colmap["datetime"]].max())
min_ts = _min.to_pydatetime()
max_ts = _max.to_pydatetime() if _max > _min else (min_ts + timedelta(minutes=1))

rng = st.sidebar.slider(
    "Time range",
    min_value=min_ts, max_value=max_ts, value=(min_ts, max_ts),
    format="YYYY-MM-DD HH:mm"
)

ev_time = ev_all[
    (ev_all[colmap["datetime"]] >= pd.to_datetime(rng[0])) &
    (ev_all[colmap["datetime"]] <= pd.to_datetime(rng[1]))
].copy()
if ev_time.empty:
    st.warning("No events in selected time range.")
    st.stop()

# =================== ANALYTICS ===================
data, device_stats, tech_stats, run_stats, hourly, visit = build_delivery_analytics(
    ev_time, colmap, idle_min=idle_min, idle_max=idle_max
)

# Pre-format for Drill-down
data["gap_hms"]      = data["__gap_s"].map(fmt_hms)
data["walk_gap_hms"] = data["__walk_gap_s"].map(fmt_hms)
data["dwell_hms"]    = data["__dwell_s"].map(fmt_hms)
data["visit_hms"]    = data["visit_duration_s"].map(fmt_hms)

# =================== FILTERS (post-analytics) ===================
pick_devices = st.sidebar.multiselect("Devices", safe_unique(ev_time, colmap["device"]))
pick_users   = st.sidebar.multiselect("Users", safe_unique(ev_time, colmap["user"]))
pick_types   = st.sidebar.multiselect("Transaction types", safe_unique(ev_time, colmap["type"]))

mask = pd.Series(True, index=data.index)
if pick_devices:
    mask &= data[colmap["device"]].isin(pick_devices)
if pick_users:
    mask &= data[colmap["user"]].isin(pick_users)
if pick_types:
    mask &= data[colmap["type"]].isin(pick_types)

data_f   = data.loc[mask].copy()
visit_f  = visit[visit[colmap["user"]].isin(data_f[colmap["user"]].unique())]
hourly_f = hourly
device_stats_f = (data_f.groupby(colmap["device"]).size().rename("events").reset_index()
                     .sort_values("events", ascending=False))
tech_stats_f   = (data_f.groupby(colmap["user"]).size().rename("events").reset_index()
                     .sort_values("events", ascending=False))
run_stats_f    = run_stats[run_stats[colmap["user"]].isin(data_f[colmap["user"]].unique())]

st.success(f"Loaded {len(data_f):,} events for analysis.")

# =================== TABS ===================
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs(
    ["📈 Overview", "🚶 Delivery Analytics", "🧑‍🔧 Tech Comparison",
     "📦 Devices", "⏱ Hourly", "🧪 Drill-down", "🔟 Weekly Top 10",
     "🚨 Outliers", "❓ Ask the data"]
)

with tab1:
    st.subheader("Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Events",  f"{len(ev_time):,}")
    c2.metric("Devices", f"{ev_time[colmap['device']].nunique():,}")
    c3.metric("Users",   f"{ev_time[colmap['user']].nunique():,}")
    c4.metric("Types",   f"{ev_time[colmap['type']].nunique():,}")

    week_df = weekly_summary(ev_time, colmap)
    if not week_df.empty:
        fig = px.bar(week_df, x="week", y="events", title="Weekly events")
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Delivery Analytics (per-tech sequences)")
    c1, c2 = st.columns(2)

    hg = data_f["__walk_gap_s"].dropna()
    if not hg.empty:
        fig = px.histogram(hg, nbins=40, title="Walking/Travel gaps (seconds)")
        c1.plotly_chart(fig, use_container_width=True)
        c1.caption("X-axis = seconds between finishing a device and starting the next (≥ min & ≤ max; device changed).")

    dw = data_f.loc[~data_f["__device_change"], "__gap_s"].dropna()
    if not dw.empty:
        fig2 = px.histogram(dw, nbins=40, title="Same-device dwell gaps (seconds)")
        c2.plotly_chart(fig2, use_container_width=True)

    st.markdown("**Tip:** Use the *Drill-down* tab to inspect rows behind any long gaps.")

with tab3:
    st.subheader("Tech comparison")
    st.dataframe(tech_stats_f, use_container_width=True)
    if not tech_stats.empty:
        # If you want a bar of median walk, recompute from data_f:
        med_walk = data_f.groupby(colmap["user"], observed=True)["__walk_gap_s"].median().reset_index()
        med_walk = med_walk.rename(columns={"__walk_gap_s":"median_walk_gap_s"}).fillna(0)
        fig = px.bar(med_walk, x=colmap["user"], y="median_walk_gap_s", title="Median walk gap by tech (s)")
        st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.subheader("Devices")
    st.dataframe(device_stats_f, use_container_width=True)
    if not device_stats_f.empty:
        fig = px.bar(device_stats_f.head(25), x=colmap["device"], y="events", title="Top devices by event volume")
        st.plotly_chart(fig, use_container_width=True)

with tab5:
    st.subheader("Hourly cadence")
    st.dataframe(hourly_f, use_container_width=True)
    if not hourly_f.empty:
        fig = px.line(hourly_f, x="hour", y="events", markers=True, title="Events by hour")
        st.plotly_chart(fig, use_container_width=True)

with tab6:
    st.subheader("Drill-down (exportable)")
    show_cols = [
        colmap["datetime"], colmap["user"], colmap["device"], colmap["type"],
        "gap_hms", "walk_gap_hms", "dwell_hms", "visit_hms",
        "__gap_s", "__walk_gap_s", "__dwell_s", "visit_duration_s",
        "__device_change",
    ]
    for opt in ["desc","qty","medid"]:
        c = colmap.get(opt)
        if c and c in data_f.columns:
            show_cols.insert(4, c)  # keep details near the left
    show_cols = [c for c in show_cols if c in data_f.columns]

    table = data_f[show_cols].copy()
    st.dataframe(table, use_container_width=True, height=520)

    st.download_button(
        "Download current drill-down as CSV",
        data=table.to_csv(index=False).encode("utf-8"),
        file_name="drilldown.csv",
        mime="text/csv"
    )

    st.markdown("### Per-visit summary (continuous time at a device)")
    ts, dev, usr = colmap["datetime"], colmap["device"], colmap["user"]
    visit_show = visit_f[[usr, dev, "start", "end", "visit_duration_s"]].copy()
    visit_show["visit_hms"] = visit_show["visit_duration_s"].map(fmt_hms)
    st.dataframe(
        visit_show[[usr, dev, "start", "end", "visit_hms", "visit_duration_s"]],
        use_container_width=True,
        height=360
    )

with tab7:
    st.subheader("Weekly Top 10 (signals)")
    digest = anomalies_top10(ev_all, data_f, colmap)
    if digest.empty:
        st.info("No notable anomalies in the last 7 days window.")
    else:
        st.dataframe(digest, use_container_width=True)

with tab8:
    st.subheader("Outliers")
    ow = outliers_iqr(data_f.dropna(subset=["__walk_gap_s"]), colmap["user"], "__walk_gap_s", "Walk gap")
    od = outliers_iqr(data_f.dropna(subset=["__dwell_s"]),    colmap["device"], "__dwell_s",    "Dwell")
    c1, c2 = st.columns(2)
    c1.write("By user (walk gap):")
    c1.dataframe(ow, use_container_width=True, height=320)
    c2.write("By device (dwell):")
    c2.dataframe(od, use_container_width=True, height=320)

with tab9:
    st.subheader("Ask the data ❓")
    q = st.text_input("Try e.g. 'top devices', 'longest dwell devices', 'median walk gap for Melissa', 'busiest hour'")
    if q:
        ans, tbl = qa_answer(q, ev_time, data_f, colmap)
        st.write(ans)
        if not tbl.empty:
            st.dataframe(tbl, use_container_width=True)

# app.py — Device Event Insights (Supabase Postgres)
# - Multi-file upload (xlsx/csv) + DB-only mode
# - Safe datetime parsing; duplicate header trimming
# - Durable history in Postgres (UPSERT by pk)
# - Delivery analytics (walk gaps, dwell, per-visit)
# - Fast filters: server-side WHERE/IN + @st.cache_data
# - One-click non-blocking index builder (no startup collisions)
# - Save/Restore filter presets to Postgres
# - Pandas deprecation fixes ("H"->"h", tz check, no groupby.apply warnings)
# - Safer concat (skip empty/all-NA uploads)

from __future__ import annotations

import hashlib, io, os, re, json
from datetime import timedelta
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sqlalchemy import create_engine, text
from pandas.api.types import DatetimeTZDtype, is_datetime64_any_dtype

# ----------------------------- CONFIG ---------------------------------
st.set_page_config(page_title="Device Event Insights", layout="wide", initial_sidebar_state="expanded")

DEFAULT_COLMAP = {
    "datetime": "TransactionDateTime",
    "device":   "Device",
    "user":     "UserName",
    "type":     "TransactionType",
    "desc":     "MedDescription",   # optional
    "qty":      "Quantity",         # optional
    "medid":    "MedID",            # optional
}
DEFAULT_IDLE_MIN = 30  # seconds to qualify as a walk/travel gap

# ----------------------------- HELPERS --------------------------------
def dedupe_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()].copy()
    df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]
    return df

def _is_tz_aware(series: pd.Series) -> bool:
    try:
        return isinstance(series.dtype, DatetimeTZDtype)
    except Exception:
        return is_datetime64_any_dtype(series) and getattr(series.dt, "tz", None) is not None

def parse_datetime_series(s: pd.Series) -> pd.Series:
    out = pd.to_datetime(s, errors="coerce", utc=True)
    if _is_tz_aware(out):
        out = out.dt.tz_convert("UTC").dt.tz_localize(None)
    else:
        out = out.dt.tz_localize(None)
    return out

def fmt_hms(x) -> str:
    if pd.isna(x): return ""
    x = int(round(float(x)))
    h, r = divmod(x, 3600); m, s = divmod(r, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"

def safe_unique(df: pd.DataFrame, col: str) -> List[str]:
    if col not in df.columns: return []
    return sorted([x for x in df[col].dropna().astype(str).unique()])

def build_pk(df: pd.DataFrame, colmap: Dict[str, str]) -> pd.Series:
    cols = []
    for k in ["datetime","device","user","type","desc","qty","medid"]:
        c = colmap.get(k)
        cols.append(df[c].astype(str) if c and c in df.columns else pd.Series([""], index=df.index, dtype="string"))
    arr = np.vstack([c.values for c in cols]).T
    out = [hashlib.sha1("|".join(row).encode("utf-8")).hexdigest() for row in arr]
    return pd.Series(out, index=df.index, dtype="string")

def base_clean(df_raw: pd.DataFrame, colmap: Dict[str, str]) -> pd.DataFrame:
    out = dedupe_columns(df_raw).copy()
    dtcol = colmap["datetime"]
    if dtcol not in out.columns:
        raise ValueError(f"Mapped datetime column '{dtcol}' not found in file.")
    s = out[dtcol]
    if isinstance(s, pd.DataFrame): s = s.iloc[:,0]
    out[dtcol] = parse_datetime_series(s)
    if colmap.get("qty") and colmap["qty"] in out.columns:
        out[colmap["qty"]] = pd.to_numeric(out[colmap["qty"]], errors="coerce")
    for key in ["device","user","type","desc","medid"]:
        c = colmap.get(key)
        if c and c in out.columns:
            out[c] = out[c].astype("string").str.strip()
    out = out.dropna(subset=[dtcol]).sort_values(dtcol).reset_index(drop=True)
    out["__date"] = out[dtcol].dt.date
    out["__hour"] = out[dtcol].dt.hour
    out["__dow"]  = out[dtcol].dt.day_name()
    return out

def weekly_summary(ev: pd.DataFrame, colmap: Dict[str, str]) -> pd.DataFrame:
    ts,dev,usr,typ = colmap["datetime"], colmap["device"], colmap["user"], colmap["type"]
    df = ev.copy()
    df["week"] = df[ts].dt.to_period("W-SUN").apply(lambda p: p.start_time.date())
    return (df.groupby("week")
            .agg(events=(typ,"count"), devices=(dev,"nunique"), techs=(usr,"nunique"))
            .reset_index().sort_values("week"))

def build_delivery_analytics(
    ev: pd.DataFrame, colmap: Dict[str, str], idle_min: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ts,dev,usr,typ = colmap["datetime"], colmap["device"], colmap["user"], colmap["type"]
    needed = [ts,dev,usr,typ]
    for k in ["desc","medid","qty"]:
        if colmap.get(k) and colmap[k] in ev.columns: needed.append(colmap[k])
    needed = list(dict.fromkeys(needed))
    ev = ev.loc[:, ~ev.columns.duplicated()].copy()
    data = ev[needed].sort_values([usr, ts]).copy()

    data["__next_ts"]  = data.groupby(usr, group_keys=False)[ts].shift(-1)
    data["__next_dev"] = data.groupby(usr, group_keys=False)[dev].shift(-1)
    data["__gap_s"]    = (data["__next_ts"] - data[ts]).dt.total_seconds()
    data["__device_change"] = (data[dev] != data["__next_dev"]) & data["__next_dev"].notna()
    data["__walk_gap_s"] = np.where((data["__device_change"]) & (data["__gap_s"] >= idle_min), data["__gap_s"], np.nan)
    data["__dwell_s"]    = np.where(~data["__device_change"], data["__gap_s"], np.nan)

    change_flag = data.groupby(usr, group_keys=False)[dev].transform(lambda s: (s != s.shift()).astype(int))
    data["__visit_id"] = change_flag.groupby(data[usr]).cumsum()

    visit = (data.groupby([usr,"__visit_id",dev], as_index=False)
             .agg(start=(ts,"min"), end=(ts,"max")))
    visit["visit_duration_s"] = (visit["end"] - visit["start"]).dt.total_seconds()
    data = data.merge(visit[[usr,"__visit_id","visit_duration_s"]], on=[usr,"__visit_id"], how="left")

    device_stats = (pd.concat([
                        data.groupby(dev, dropna=False).size().rename("events"),
                        data.loc[~data["__device_change"],"__gap_s"].groupby(data[dev]).median().rename("median_dwell_s")
                    ], axis=1).fillna(0).sort_values("events", ascending=False).reset_index())

    tech_stats = (pd.concat([
                        data.groupby(usr, dropna=False).size().rename("events"),
                        data["__walk_gap_s"].groupby(data[usr]).median().rename("median_walk_gap_s")
                    ], axis=1).fillna(0).sort_values("events", ascending=False).reset_index())

    data["__is_break"] = (data["__walk_gap_s"] >= idle_min).fillna(False)
    data["__run_id"] = data.groupby(usr)["__is_break"].cumsum()
    run_stats = (data.groupby([usr,"__run_id"], as_index=False)
                 .agg(start=(ts,"min"), end=(ts,"max"), n_events=(ts,"count"),
                      n_devices=(dev,"nunique"),
                      total_walk_s=("__walk_gap_s", lambda s: np.nansum(s.values))))
    run_stats["duration_s"] = (run_stats["end"] - run_stats["start"]).dt.total_seconds()

    hourly = (data.groupby(data[ts].dt.floor("h")).size().rename("events").reset_index()
              .rename(columns={ts:"hour"}))

    return data, device_stats, tech_stats, run_stats, hourly, visit

def anomalies_top10(history: pd.DataFrame, data: pd.DataFrame, colmap: Dict[str,str]) -> pd.DataFrame:
    ts,dev,usr,typ = colmap["datetime"], colmap["device"], colmap["user"], colmap["type"]
    out = []
    if history.empty: return pd.DataFrame(columns=["rank","topic","detail","why","severity"])
    hist = history.copy()
    hist[ts] = pd.to_datetime(hist[ts], errors="coerce")
    hist = hist.dropna(subset=[ts]); if_empty = hist.empty
    if if_empty: return pd.DataFrame(columns=["rank","topic","detail","why","severity"])
    end = hist[ts].max(); start_recent = end - pd.Timedelta(days=7); start_prior = start_recent - pd.Timedelta(days=7)
    recent = hist[(hist[ts] > start_recent) & (hist[ts] <= end)]
    prior  = hist[(hist[ts] > start_prior)  & (hist[ts] <= start_recent)]
    if not recent.empty and not prior.empty:
        r_dev = recent.groupby(dev).size().rename("recent").reset_index()
        p_dev = prior.groupby(dev).size().rename("prior").reset_index()
        vol = r_dev.merge(p_dev, on=dev, how="left").fillna(0.0)
        vol["delta"] = vol["recent"] - vol["prior"]
        vol["pct"]   = np.where(vol["prior"]>0, (vol["recent"]-vol["prior"])/vol["prior"], np.nan)
        for _, r in vol.sort_values(["pct","delta"], ascending=False).head(3).iterrows():
            out.append({"topic":"Device volume spike",
                        "detail":f"{r[dev]} recent {int(r['recent'])} vs prior {int(r['prior'])}",
                        "why":"Sudden workload shift; check staffing/stocking cadence",
                        "severity":"high" if (r["pct"]>=0.5 and r["recent"]>=50) else "med"})
    if not data.empty and usr in data.columns:
        twalk = data["__walk_gap_s"].groupby(data[usr]).median().dropna()
        for u, s in twalk.sort_values(ascending=False).head(3).items():
            out.append({"topic":"High walking time","detail":f"{u} median walk gap {fmt_hms(s)} ({int(s)}s)",
                        "why":"Inefficient routing or distant devices","severity":"med" if s>=120 else "low"})
    if not data.empty and dev in data.columns:
        dwell = data.loc[~data["__device_change"],"__gap_s"].groupby(data[dev]).median().dropna()
        for d, s in dwell.sort_values(ascending=False).head(3).items():
            out.append({"topic":"Long dwell at device","detail":f"{d} median dwell {fmt_hms(s)} ({int(s)}s)",
                        "why":"Many refills per stop or slow transactions","severity":"med" if s>=60 else "low"})
    if not recent.empty:
        rh = recent.groupby(recent[ts].dt.floor("h")).size()
        for h, n in rh.sort_values(ascending=False).head(2).items():
            out.append({"topic":"Rush hour","detail":f"{h:%Y-%m-%d %H:%M} had {int(n)} events",
                        "why":"Consider JIT timing / more techs","severity":"med" if n>=100 else "low"})
    if not recent.empty and not prior.empty and typ in hist.columns:
        r_t = recent.groupby(typ).size().rename("recent").reset_index()
        p_t = prior.groupby(typ).size().rename("prior").reset_index()
        typd = r_t.merge(p_t, on=typ, how="left").fillna(0.0)
        for _, r in typd.assign(delta=lambda d: d["recent"]-d["prior"]).sort_values("delta", ascending=False).head(2).iterrows():
            out.append({"topic":"Transaction-type surge",
                        "detail":f"{r[typ]} recent {int(r['recent'])} vs prior {int(r['prior'])}",
                        "why":"Upstream demand or workflow change","severity":"med" if r['delta']>=30 else "low"})
    if not out: return pd.DataFrame(columns=["rank","topic","detail","why","severity"])
    df_out = pd.DataFrame(out)
    sev_rank = df_out["severity"].map({"high":3,"med":2,"low":1}).fillna(1)
    df_out = df_out.iloc[sev_rank.sort_values(ascending=False).index].reset_index(drop=True)
    df_out.insert(0,"rank", np.arange(1, len(df_out)+1))
    return df_out.head(10)

def outliers_iqr(data: pd.DataFrame, key_col: str, value_col: str, label: str) -> pd.DataFrame:
    df = data[[key_col, value_col]].dropna().copy()
    if df.empty: return pd.DataFrame(columns=[key_col, value_col, "z_note"])
    def _flag(group):
        q1 = group[value_col].quantile(0.25); q3 = group[value_col].quantile(0.75); iqr = q3-q1
        upper = q3 + (1.5 if iqr==0 else 1.5*iqr)
        return group[group[value_col] > upper].assign(z_note=f"{label}: > Q3+1.5*IQR (>{upper:.1f}s)")
    return df.groupby(key_col, dropna=True, group_keys=False).apply(_flag).reset_index(drop=True)

def qa_answer(question: str, ev: pd.DataFrame, data: pd.DataFrame, colmap: Dict[str,str]) -> Tuple[str, pd.DataFrame]:
    q = question.strip().lower()
    ts,dev,usr,typ = colmap["datetime"], colmap["device"], colmap["user"], colmap["type"]
    if re.search(r"\b(top|most)\b.*\bdevices?\b", q):
        t = ev.groupby(dev).size().rename("events").reset_index().sort_values("events", ascending=False).head(10)
        return f"Top devices by event volume (showing {len(t)}):", t
    if "longest" in q and "dwell" in q:
        if data.empty: return "No dwell data in current filter.", pd.DataFrame()
        t = (data.loc[~data["__device_change"],["__gap_s"]].groupby(data[dev]).median()
             .rename(columns={"__gap_s":"median_dwell_s"}).sort_values("median_dwell_s", ascending=False).head(10).reset_index())
        t["median_dwell_hms"] = t["median_dwell_s"].map(fmt_hms); return "Devices with longest median dwell:", t
    m = re.search(r"median .*walk.* for (.+)", q)
    if m:
        name = m.group(1).strip()
        sub = data[data[usr].astype(str).str.lower()==name.lower()]
        if sub.empty: return f"No rows found for user '{name}'.", pd.DataFrame()
        val = np.nanmedian(sub["__walk_gap_s"].values)
        return f"Median walk gap for {name}: {fmt_hms(val)} ({int(val)}s)", pd.DataFrame()
    if "hour" in q:
        t = ev.groupby(ev[ts].dt.floor("h")).size().rename("events").reset_index().rename(columns={ts:"hour"})
        if t.empty: return "No hourly data in current filter.", pd.DataFrame()
        top = t.sort_values("events", ascending=False).head(1).iloc[0]
        return f"Busiest hour: {top['hour']:%Y-%m-%d %H:%M} with {int(top['events'])} events.", t.sort_values("hour")
    if "which tech" in q and "median walk" in q:
        if data.empty: return "No walk-gap data in current filter.", pd.DataFrame()
        t = data.groupby(usr)["__walk_gap_s"].median().reset_index().rename(columns={"__walk_gap_s":"median_walk_s"})
        t = t.sort_values("median_walk_s", ascending=False); top = t.iloc[0]
        return f"Highest median walk gap: {top[usr]} at {fmt_hms(top['median_walk_s'])}.", t
    tbl = ev[[ts, usr, dev, typ]].head(50)
    return ("Try asks like: 'top devices', 'longest dwell devices', "
            "'median walk gap for Melissa', 'busiest hour'."), tbl

# ------------------------- PERSISTENCE (POSTGRES) ----------------------
@st.cache_resource
def get_engine(db_url: str, salt: str):
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
    CREATE TABLE IF NOT EXISTS app_prefs (
        k TEXT PRIMARY KEY,
        v JSONB
    );
    """
    with eng.begin() as con:
        for stmt in [s.strip() for s in ddl.split(";") if s.strip()]:
            con.execute(text(stmt))

def ensure_indexes(eng):
    """
    Build indexes concurrently; if busy, skip quietly.
    Trigger via sidebar button or after uploads (not at startup).
    """
    idx_stmts = [
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_events_dt      ON events (dt)",
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_events_device  ON events (device)",
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_events_user    ON events (\"user\")",
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_events_type    ON events (\"type\")",
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_events_dt_hour ON events (date_trunc('hour', dt))",
    ]
    with eng.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        conn.execute(text("SET statement_timeout = '2min'"))
        conn.execute(text("SET lock_timeout = '1500ms'"))
        lock_key = 931248712
        got = conn.execute(text("SELECT pg_try_advisory_lock(:k)"), {"k": lock_key}).scalar()
        if not got:
            return
        try:
            for stmt in idx_stmts:
                try:
                    conn.execute(text(stmt))
                except Exception as e:
                    msg = str(e).lower()
                    if "lock timeout" in msg or "locknotavailable" in msg:
                        # optional: comment next line if you don't want UI chatter
                        st.sidebar.info(f"Index build deferred (busy): {stmt.split(' IF NOT EXISTS ')[-1]}")
                        continue
                    raise
        finally:
            conn.execute(text("SELECT pg_advisory_unlock(:k)"), {"k": lock_key})

def _df_to_rows(df: pd.DataFrame, colmap: Dict[str, str]) -> list[dict]:
    ts,dev,usr,typ = colmap["datetime"], colmap["device"], colmap["user"], colmap["type"]
    get = lambda k: (colmap.get(k) in df.columns) if colmap.get(k) else False
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "pk":    r["pk"],
            "dt":    pd.to_datetime(r[ts]).to_pydatetime() if pd.notna(r[ts]) else None,
            "device": r.get(dev, None),
            "user":   r.get(usr, None),
            "type":   r.get(typ, None),
            "desc":   r.get(colmap.get("desc",""), None) if get("desc") else None,
            "qty":    float(r[colmap["qty"]]) if get("qty") and pd.notna(r[colmap["qty"]]) else None,
            "medid":  r.get(colmap.get("medid",""), None) if get("medid") else None,
        })
    return rows

def save_history_sql(df: pd.DataFrame, colmap: Dict[str, str], eng) -> tuple[bool, str]:
    try:
        rows = _df_to_rows(df, colmap)
        if not rows: return True, "No rows to save."
        upsert_sql = text("""
            INSERT INTO events (pk, dt, device, "user", "type", "desc", qty, medid)
            VALUES (:pk, :dt, :device, :user, :type, :desc, :qty, :medid)
            ON CONFLICT (pk) DO UPDATE SET
                dt=EXCLUDED.dt, device=EXCLUDED.device, "user"=EXCLUDED."user",
                "type"=EXCLUDED."type", "desc"=EXCLUDED."desc",
                qty=EXCLUDED.qty, medid=EXCLUDED.medid;
        """)
        CHUNK, total = 5000, 0
        with eng.begin() as con:
            con.execute(text("SET statement_timeout = '5min'"))
            for i in range(0, len(rows), CHUNK):
                batch = rows[i:i+CHUNK]
                con.execute(upsert_sql, batch)
                total += len(batch)
        return True, f"Saved to Postgres: {total:,} rows (upserted by pk)."
    except Exception as e:
        return False, f"DB save error: {e}"

def load_history_sql_all(colmap: Dict[str, str], eng) -> pd.DataFrame:
    sql = 'SELECT pk, dt, device, "user", "type", "desc", qty, medid FROM events ORDER BY dt'
    raw = pd.read_sql(sql, eng)
    if raw.empty: return raw
    rename_back = {"dt": colmap["datetime"], "device": colmap["device"], "user": colmap["user"], "type": colmap["type"]}
    if colmap.get("desc"):  rename_back["desc"]  = colmap["desc"]
    if colmap.get("qty"):   rename_back["qty"]   = colmap["qty"]
    if colmap.get("medid"): rename_back["medid"] = colmap["medid"]
    out = raw.rename(columns=rename_back)
    out[colmap["datetime"]] = pd.to_datetime(out[colmap["datetime"]], errors="coerce")
    return out

def load_history_sql_filtered(colmap, eng, start, end, devices=None, users=None, types=None) -> pd.DataFrame:
    ts,dev,usr,typ = colmap["datetime"], colmap["device"], colmap["user"], colmap["type"]
    cols = 'pk, dt, device, "user", "type", "desc", qty, medid'
    where, params = ["dt BETWEEN :start AND :end"], {"start": pd.Timestamp(start), "end": pd.Timestamp(end)}
    if devices: where.append("device = ANY(:devices)"); params["devices"] = list(devices)
    if users:   where.append('"user" = ANY(:users)');   params["users"]   = list(users)
    if types:   where.append('"type" = ANY(:types)');   params["types"]   = list(types)
    sql = f"SELECT {cols} FROM events WHERE " + " AND ".join(where) + " ORDER BY dt"
    raw = pd.read_sql(text(sql), eng, params=params)
    if raw.empty: return raw
    rename_back = {"dt": ts, "device": dev, "user": usr, "type": typ}
    if colmap.get("desc"):  rename_back["desc"]  = colmap["desc"]
    if colmap.get("qty"):   rename_back["qty"]   = colmap["qty"]
    if colmap.get("medid"): rename_back["medid"] = colmap["medid"]
    out = raw.rename(columns=rename_back)
    out[ts] = pd.to_datetime(out[ts], errors="coerce")
    for c in [dev, usr, typ]:
        if c in out.columns: out[c] = out[c].astype("category")
    return out

# -- Prefs (save/restore last filters) ---------------------------------
def save_prefs(eng, key: str, value: dict):
    with eng.begin() as con:
        con.execute(text("INSERT INTO app_prefs (k, v) VALUES (:k, :v) "
                         "ON CONFLICT (k) DO UPDATE SET v = EXCLUDED.v"),
                    {"k": key, "v": json.dumps(value)})

def load_prefs(eng, key: str) -> Optional[dict]:
    with eng.connect() as con:
        row = con.execute(text("SELECT v FROM app_prefs WHERE k = :k"), {"k": key}).mappings().first()
        return (row and row["v"]) and (row["v"] if isinstance(row["v"], dict) else json.loads(row["v"]))

# ----------------------------- UI ------------------------------------
st.title("All Device Event Insights — Pro")

# Build engine / schema
DB_URL = st.secrets["DB_URL"]
ENGINE_SALT = st.secrets.get("ENGINE_SALT", "")
eng = get_engine(DB_URL, ENGINE_SALT)
try:
    init_db(eng)
except Exception as e:
    st.warning(f"Initialization warning (schema): {e}")

# Sidebar: Upload or DB-only
st.sidebar.header("1) Upload")
uploads = st.sidebar.file_uploader("Drag & drop daily XLSX/CSV (one or many)",
                                   type=["xlsx","csv"], accept_multiple_files=True)
use_db_only = st.sidebar.toggle("Open with existing DB only (no upload)", value=not bool(uploads))

# Index admin
st.sidebar.divider()
if st.sidebar.button("🔧 Build/repair DB indexes"):
    try:
        ensure_indexes(eng)
        st.sidebar.success("Index build started (non-blocking).")
    except Exception as e:
        st.sidebar.info(f"Index build skipped: {e}")

# Read uploads (skip empties)
frames = []
if uploads and not use_db_only:
    for up in uploads:
        try:
            if up.name.lower().endswith(".xlsx"):
                df = pd.read_excel(up)
            else:
                try: df = pd.read_csv(up)
                except UnicodeDecodeError:
                    up.seek(0); df = pd.read_csv(up, encoding="latin-1")
            if isinstance(df, pd.DataFrame) and not df.empty and not df.isna().all(axis=None):
                frames.append(df)
            else:
                st.warning(f"{up.name}: empty or all-NA; skipped.")
        except Exception as e:
            st.error(f"Failed to read {up.name}: {e}")

# Column mapping (from either uploads or DB to get headers)
sample_df = None
if frames:
    sample_df = dedupe_columns(pd.concat(frames, ignore_index=True))
else:
    # small peek to get columns when opening DB-only
    peek = load_history_sql_all(DEFAULT_COLMAP, eng)
    if not peek.empty:
        sample_df = peek.head(100)

if sample_df is None:
    if use_db_only:
        st.warning("Database is empty. Upload files to get started.")
    else:
        st.info("Upload one or more daily exports to get started, or toggle 'DB only'.")
    st.stop()

# Column mapping UI
st.sidebar.header("2) Map columns")
colmap: Dict[str, str] = {}
opts = list(sample_df.columns)
for k, default in DEFAULT_COLMAP.items():
    sel = st.sidebar.selectbox(f"{k.capitalize()} column", options=opts,
                               index=opts.index(default) if default in opts else 0,
                               key=f"map_{k}", help="Pick the matching column from your export")
    colmap[k] = sel

_selected = [v for v in colmap.values() if v is not None]
_dups = sorted({c for c in _selected if _selected.count(c) > 1})
if _dups:
    st.error("You mapped the same column to multiple fields: " + ", ".join(_dups))
    st.stop()

# If uploads: clean, pk, merge with history, save; else DB-only
if frames and not use_db_only:
    try:
        new_ev = base_clean(sample_df, colmap)
    except Exception as e:
        st.error(f"Column mapping / cleaning error: {e}"); st.stop()
    new_ev["pk"] = build_pk(new_ev, colmap)
    history = load_history_sql_all(colmap, eng)
    if not history.empty:
        if "pk" not in history.columns:
            try:
                if colmap["datetime"] in history.columns:
                    history[colmap["datetime"]] = parse_datetime_series(history[colmap["datetime"]])
                history["pk"] = build_pk(history, colmap)
            except Exception:
                history["pk"] = pd.util.hash_pandas_object(history.astype(str), index=False).astype(str)
        for c in new_ev.columns:
            if c not in history.columns: history[c] = pd.NA
        combined = pd.concat([history, new_ev], ignore_index=True)
    else:
        combined = new_ev.copy()
    combined = combined.drop_duplicates(subset=["pk"]).reset_index(drop=True)
    ok, msg = save_history_sql(combined, colmap, eng)
    (st.sidebar.success if ok else st.sidebar.error)(msg)
    if ok:
        # optional: build indexes after write to avoid startup collisions
        try: ensure_indexes(eng)
        except Exception: pass

# Filters (saved/restore)
st.sidebar.header("3) Filters")
prefs_key = "default_filters"
saved = load_prefs(eng, prefs_key) or {}

# Range bounds from DB to avoid loading everything
bounds_df = load_history_sql_filtered(colmap, eng,
                                      start="1900-01-01", end="2999-12-31", devices=None, users=None, types=None)
if bounds_df.empty:
    st.warning("No events available yet. Upload files to get started.")
    st.stop()
ts = colmap["datetime"]
_min = pd.to_datetime(bounds_df[ts].min()); _max = pd.to_datetime(bounds_df[ts].max())
min_ts, max_ts = _min.to_pydatetime(), max(_min.to_pydatetime(), _max.to_pydatetime())
if min_ts == max_ts: max_ts = min_ts + timedelta(minutes=1)

# Restore saved filters if present
default_rng = (pd.to_datetime(saved.get("start", min_ts)).to_pydatetime(),
               pd.to_datetime(saved.get("end",   max_ts)).to_pydatetime())
rng = st.sidebar.slider("Time range", min_value=min_ts, max_value=max_ts,
                        value=default_rng, format="YYYY-MM-DD HH:mm")

# Device/user/type options (from bounds, cheap)
pick_devices = st.sidebar.multiselect("Devices", safe_unique(bounds_df, colmap["device"]),
                                      default=saved.get("devices", []))
pick_users   = st.sidebar.multiselect("Users",   safe_unique(bounds_df, colmap["user"]),
                                      default=saved.get("users", []))
pick_types   = st.sidebar.multiselect("Transaction types", safe_unique(bounds_df, colmap["type"]),
                                      default=saved.get("types", []))
idle_min = st.sidebar.number_input("Walk gap threshold (seconds)", min_value=5, max_value=900,
                                   value=int(saved.get("idle_min", DEFAULT_IDLE_MIN)), step=5)

col_b1, col_b2 = st.sidebar.columns(2)
if col_b1.button("💾 Save filters"):
    save_prefs(eng, prefs_key, {
        "start": rng[0].isoformat(), "end": rng[1].isoformat(),
        "devices": pick_devices, "users": pick_users, "types": pick_types, "idle_min": idle_min
    })
    st.sidebar.success("Saved as default filters.")
if col_b2.button("↩️ Restore filters"):
    st.experimental_rerun()

# Load filtered data fast (server-side + cache)
@st.cache_data(ttl=600)
def cached_filtered(db_url, salt, _colmap, start, end, devices, users, types):
    eng_local = create_engine(db_url, pool_pre_ping=True)
    return load_history_sql_filtered(_colmap, eng_local, start, end, devices, users, types)

ev = cached_filtered(DB_URL, ENGINE_SALT, colmap, rng[0], rng[1],
                     pick_devices or None, pick_users or None, pick_types or None)
if ev.empty:
    st.warning("No events in current filter range.")
    st.stop()

# Compute analytics
data, device_stats, tech_stats, run_stats, hourly, visit = build_delivery_analytics(ev, colmap, idle_min=idle_min)

# Pre-format for Drill-down
for k, src in [("gap_hms","__gap_s"), ("walk_gap_hms","__walk_gap_s"),
               ("dwell_hms","__dwell_s"), ("visit_hms","visit_duration_s")]:
    data[k] = data[src].map(fmt_hms)

# Tabs -----------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs(
    ["📈 Overview","🚶 Delivery Analytics","🧑‍🔧 Tech Comparison","📦 Devices",
     "⏱ Hourly","🧪 Drill-down","🔟 Weekly Top 10","🚨 Outliers","❓ Ask the data"]
)

with tab1:
    st.subheader("Overview")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Events", f"{len(ev):,}")
    c2.metric("Devices", f"{ev[colmap['device']].nunique():,}")
    c3.metric("Users",   f"{ev[colmap['user']].nunique():,}")
    c4.metric("Types",   f"{ev[colmap['type']].nunique():,}")
    week_df = weekly_summary(ev, colmap)
    if not week_df.empty:
        st.plotly_chart(px.bar(week_df, x="week", y="events", title="Weekly events"), use_container_width=True)

with tab2:
    st.subheader("Delivery Analytics (per-tech sequences)")
    c1,c2 = st.columns(2)
    hg = data["__walk_gap_s"].dropna()
    if not hg.empty:
        c1.plotly_chart(px.histogram(hg, nbins=40, title="Walking/Travel gaps (seconds)"), use_container_width=True)
        c1.caption("Seconds between finishing a device and starting the next (≥ threshold & device changed).")
    dw = data.loc[~data["__device_change"], "__gap_s"].dropna()
    if not dw.empty:
        c2.plotly_chart(px.histogram(dw, nbins=40, title="Same-device dwell gaps (seconds)"), use_container_width=True)
    st.markdown("**Tip:** Use the *Drill-down* tab to inspect rows behind any long gaps.")

with tab3:
    st.subheader("Tech comparison")
    st.dataframe(tech_stats, use_container_width=True)
    if not tech_stats.empty:
        st.plotly_chart(px.bar(tech_stats, x=colmap["user"], y="median_walk_gap_s", title="Median walk gap by tech (s)"),
                        use_container_width=True)

with tab4:
    st.subheader("Devices")
    st.dataframe(device_stats, use_container_width=True)
    if not device_stats.empty:
        st.plotly_chart(px.bar(device_stats.head(25), x=colmap["device"], y="events", title="Top devices by event volume"),
                        use_container_width=True)

with tab5:
    st.subheader("Hourly cadence")
    st.dataframe(hourly, use_container_width=True)
    if not hourly.empty:
        st.plotly_chart(px.line(hourly, x="hour", y="events", markers=True, title="Events by hour"),
                        use_container_width=True)

with tab6:
    st.subheader("Drill-down (exportable)")
    show_cols = [colmap["datetime"], colmap["user"], colmap["device"], colmap["type"],
                 "gap_hms","walk_gap_hms","dwell_hms","visit_hms",
                 "__gap_s","__walk_gap_s","__dwell_s","visit_duration_s","__device_change"]
    for k in ["desc","qty","medid"]:
        c = colmap.get(k)
        if c and c in data.columns: show_cols.insert(4, c)
    show_cols = [c for c in show_cols if c in data.columns]
    table = data[show_cols].copy()
    MAX_SHOW = 5000
    st.dataframe(table.head(MAX_SHOW), use_container_width=True, height=520)
    if len(table) > MAX_SHOW:
        st.caption(f"Showing first {MAX_SHOW:,} of {len(table):,} rows.")
    st.download_button("Download current drill-down as CSV",
                       data=table.to_csv(index=False).encode("utf-8"),
                       file_name="drilldown.csv", mime="text/csv")
    st.markdown("### Per-visit summary (continuous time at a device)")
    ts  = colmap["datetime"]; dev = colmap["device"]; usr = colmap["user"]
    visit_show = visit[[usr, dev, "start", "end", "visit_duration_s"]].copy()
    visit_show["visit_hms"] = visit_show["visit_duration_s"].map(fmt_hms)
    st.dataframe(visit_show[[usr, dev, "start", "end", "visit_hms", "visit_duration_s"]],
                 use_container_width=True, height=360)

with tab7:
    st.subheader("Weekly Top 10 (signals)")
    digest = anomalies_top10(ev, data, colmap)
    if digest.empty: st.info("No notable anomalies in the last 7 days window.")
    else:            st.dataframe(digest, use_container_width=True)

with tab8:
    st.subheader("Outliers")
    compute_outliers = st.toggle("Compute outliers (may be slow on huge ranges)", value=False)
    if compute_outliers:
        ow = outliers_iqr(data.dropna(subset=["__walk_gap_s"]), colmap["user"], "__walk_gap_s", "Walk gap")
        od = outliers_iqr(data.dropna(subset=["__dwell_s"]),    colmap["device"], "__dwell_s",    "Dwell")
        c1,c2 = st.columns(2)
        c1.write("By user (walk gap):");  c1.dataframe(ow, use_container_width=True, height=320)
        c2.write("By device (dwell):");   c2.dataframe(od, use_container_width=True, height=320)
    else:
        st.info("Toggle on to compute outliers.")

with tab9:
    st.subheader("Ask the data ❓")
    q = st.text_input("Try e.g. 'top devices', 'longest dwell devices', 'median walk gap for Melissa', 'busiest hour'")
    if q:
        ans, tbl = qa_answer(q, ev, data, colmap); st.write(ans)
        if not tbl.empty: st.dataframe(tbl, use_container_width=True)

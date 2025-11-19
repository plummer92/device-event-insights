 app.py
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
import matplotlib.pyplot as plt  # needed for pandas Styler gradients
from pandas.api.types import DatetimeTZDtype



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

# ---------- SIMPLE EXTRACTOR FOR DEVICE ACTIVITY LOG ----------

A_MIN  = r"Inventory\s*Refill\s*Point\s*Min\s*Quantity"
A_MAX  = r"Inventory\s*Par\s*Max\s*Quantity"
A_STD  = r"Inventory\s*Standard\s*Stock"

_RX_AFFECTED = re.compile(
    r"""
    ^(?P<device_prefix>[^\s]+)        # e.g., SJS11E_TWR (kept only if you want)
    .*?
    (?:Drw|Drawer|Dr)\s*(?P<drawer>[A-Za-z0-9\.]+)   # 1 or 2.1 or D3
    [\-\s]*
    (?:Pkt|Pocket)\s*(?P<pocket>[A-Za-z0-9\.]+)      # 5 or B1 etc.
    \s*\(\s*(?P<med>[A-Za-z0-9._\-]+)\s*\)\s*
    :\s*(?P<val>-?\d+(?:\.\d+)?)\s*$
    """,
    re.IGNORECASE | re.VERBOSE,
)

def _parse_affected(s: str):
    s = "" if s is None else str(s).strip()
    m = _RX_AFFECTED.search(s)
    if not m:
        # soft fallback so bad rows don’t crash
        med_m = re.search(r"\(([^)]+)\)", s)
        val_m = re.search(r":\s*(-?\d+(?:\.\d+)?)", s)
        drw_m = re.search(r"(?:Drw|Drawer|Dr)\s*([A-Za-z0-9\.]+)", s, flags=re.I)
        pkt_m = re.search(r"(?:Pkt|Pocket)\s*([A-Za-z0-9\.]+)", s, flags=re.I)
        return (
            (drw_m.group(1).upper() if drw_m else ""),
            (pkt_m.group(1).upper() if pkt_m else ""),
            (med_m.group(1).upper() if med_m else ""),
            (float(val_m.group(1)) if val_m else None),
        )
    return (
        m.group("drawer").strip().upper(),
        m.group("pocket").strip().upper(),
        m.group("med").strip().upper(),
        float(m.group("val")),
    )

def build_simple_activity_view(df: pd.DataFrame) -> pd.DataFrame:
    need = {"Device","AffectedElement","TransactionDateTime","UserName","ActivityType"}
    if not need.issubset(df.columns):
        return pd.DataFrame(columns=[
            "ts","device","drawer","pocket","med_id","username",
            "min_qty","max_qty","is_min","is_max","is_standard_stock"
        ])

    out = df[list(need)].copy().rename(columns={
        "Device":"device",
        "AffectedElement":"affected",
        "TransactionDateTime":"ts",
        "UserName":"username",
        "ActivityType":"activity_type",
    })
    out["ts"] = pd.to_datetime(out["ts"], errors="coerce")
    out["username"] = out["username"].astype(str).str.strip()
    out["device"] = out["device"].astype(str).str.strip()

    # drop ActivityType NaN or literal "nan"
    mask_valid = out["activity_type"].notna() & (out["activity_type"].astype(str).str.lower().ne("nan"))
    out = out.loc[mask_valid].copy()
    out["activity_type"] = out["activity_type"].astype(str).str.strip()

    # parse AffectedElement (drawer, pocket, med_id, qty)
    parsed = out["affected"].apply(_parse_affected)
    out[["drawer","pocket","med_id","qty"]] = pd.DataFrame(parsed.tolist(), index=out.index)
    out.drop(columns=["affected"], inplace=True)

    # identify min/max/standard stock types
    A_MIN  = r"Inventory\s*Refill\s*Point\s*Min\s*Quantity"
    A_MAX  = r"Inventory\s*Par\s*Max\s*Quantity"
    A_STD  = r"Inventory\s*Standard\s*Stock"

    out["is_min"]  = out["activity_type"].str.contains(A_MIN, case=False, na=False)
    out["is_max"]  = out["activity_type"].str.contains(A_MAX, case=False, na=False)
    out["is_standard_stock"] = out["activity_type"].str.contains(A_STD, case=False, na=False)

    out["min_qty"] = np.where(out["is_min"], out["qty"], np.nan)
    out["max_qty"] = np.where(out["is_max"], out["qty"], np.nan)

    # normalize
    for c in ("drawer","pocket","med_id"):
        out[c] = out[c].fillna("").astype(str).str.upper().str.strip()

    # combine same timestamp+slot+med into one row
    keys = ["ts","device","drawer","pocket","med_id"]
    def _first(s):
        s = s.dropna()
        return s.iloc[0] if not s.empty else None

    combined = (
        out.groupby(keys, as_index=False)
           .agg(
               username=("username", _first),
               min_qty=("min_qty", "max"),
               max_qty=("max_qty", "max"),
               is_min=("is_min", "any"),
               is_max=("is_max", "any"),
               is_standard_stock=("is_standard_stock", "any"),
           )
           .sort_values("ts")
    )

    # Final cleaned order (activity_type & qty dropped)
    cols = ["ts","device","drawer","pocket","med_id","min_qty","max_qty","username",
            "is_min","is_max","is_standard_stock"]
    return combined[cols]

def _device_base(s: str) -> str:
    s = "" if s is None else str(s).strip().upper()
    return s.split("_", 1)[0]  # SJS7ES_MAIN → SJS7ES

def _norm_slot_str(s):
    if pd.isna(s): return ""
    return str(s).strip().upper().replace(" ", "")

def _drawer_root(s: str) -> str:
    if pd.isna(s): return ""
    s = str(s).strip().upper().replace(" ", "")
    return s.split(".", 1)[0]  # '2.1' → '2'

def _norm_med(s: str) -> str:
    if pd.isna(s): return ""
    return str(s).strip().upper()

def parse_datetime_series(s: pd.Series) -> pd.Series:
    """Parse to pandas datetime (UTC-naive)."""
    out = pd.to_datetime(s, errors="coerce", utc=True)
    if isinstance(out.dtype, DatetimeTZDtype):
        out = out.dt.tz_convert("UTC").dt.tz_localize(None)
    return out

def load_upload(up) -> pd.DataFrame:
    name = up.name.lower()
    if name.endswith(".xlsx"):
        return pd.read_excel(up)
    try:
        up.seek(0)
        return pd.read_csv(up, low_memory=False)
    except UnicodeDecodeError:
        up.seek(0)
        return pd.read_csv(up, encoding="latin-1", low_memory=False)
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


def refill_trend(ev: pd.DataFrame, colmap: Dict[str, str], freq: str = "W-SUN") -> pd.DataFrame:
    """
    Build a refill-only time series and compare to total events in the same buckets.

    Returns columns:
      - period (date)
      - refill_events
      - total_events
      - refill_share (0–1 float)
    """
    ts_col = colmap["datetime"]
    typ_col = colmap["type"]

    if ts_col not in ev.columns or typ_col not in ev.columns:
        return pd.DataFrame(columns=["period", "refill_events", "total_events", "refill_share"])

    df = ev[[ts_col, typ_col]].copy()
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    df = df.dropna(subset=[ts_col])
    if df.empty:
        return pd.DataFrame(columns=["period", "refill_events", "total_events", "refill_share"])

    # Determine period bucket
    if freq.upper().startswith("W"):
        df["period"] = df[ts_col].dt.to_period(freq).apply(lambda p: p.start_time.date())
    elif freq.upper().startswith("M"):
        df["period"] = df[ts_col].dt.to_period("M").dt.to_timestamp().dt.date
    else:
        df["period"] = df[ts_col].dt.date

    # Total events per period
    total = (
        df.groupby("period", observed=True)
          .size()
          .rename("total_events")
          .reset_index()
    )

    # Refill-only events per period
    mask_refill = df[typ_col].astype(str).str.contains("refill", case=False, na=False)
    df_ref = df[mask_refill]
    if df_ref.empty:
        out = total.copy()
        out["refill_events"] = 0
        out["refill_share"] = 0.0
        return out[["period", "refill_events", "total_events", "refill_share"]]

    ref = (
        df_ref.groupby("period", observed=True)
             .size()
             .rename("refill_events")
             .reset_index()
    )

    out = total.merge(ref, on="period", how="left").fillna({"refill_events": 0})
    out["refill_events"] = out["refill_events"].astype(int)
    out["total_events"] = out["total_events"].astype(int)
    out["refill_share"] = np.where(
        out["total_events"] > 0,
        out["refill_events"] / out["total_events"],
        0.0,
    )

    return out.sort_values("period")



def _non_empty_frames(frames: List[pd.DataFrame]) -> List[pd.DataFrame]:
    out = []
    for df in frames:
        if isinstance(df, pd.DataFrame) and not df.empty:
            if not df.isna().all(axis=None):
                out.append(df)
    return out

def _fmt_int(x):
    try:
        return f"{int(round(float(x))):,}"
    except Exception:
        return "0"

def _plain_english_summary(g: pd.DataFrame) -> str:
    """Short, readable narrative for non-data folks."""
    if g.empty:
        return "No load/unload activity in the selected range."
    loads = g.loc[g["dir"].eq("load"), "orig_qty"].sum()
    unlds = g.loc[g["dir"].eq("unload"), "orig_qty"].sum()
    net = loads - unlds
    # top device by net
    by_dev = (g.groupby(["device","dir"])["orig_qty"].sum().unstack(fill_value=0))
    by_dev["net"] = by_dev.get("load",0) - by_dev.get("unload",0)
    top_dev = by_dev["net"].sort_values(ascending=False).head(1)
    dev_str = f" Top device: {top_dev.index[0]} (net {_fmt_int(top_dev.iloc[0])})." if not top_dev.empty else ""
    direction = "up" if net >= 0 else "down"
    return (f"In this period we loaded {_fmt_int(loads)} and unloaded {_fmt_int(unlds)} "
            f"(net {direction} {_fmt_int(abs(net))})." + dev_str)

def _safe_int(x, lo=-2147483648, hi=2147483647):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    try:
        v = int(round(float(x)))
        if v < lo or v > hi:
            return None
        return v
    except Exception:
        return None

def ensure_pends_minmax_columns(eng):
    alter = """
    ALTER TABLE pyxis_pends ADD COLUMN IF NOT EXISTS min_qty INTEGER;
    ALTER TABLE pyxis_pends ADD COLUMN IF NOT EXISTS max_qty INTEGER;
    """
    with eng.begin() as con:
        con.execute(text(alter))



# ---- PENDED LOADS (Pyxis DeviceActivityLog-style files) -----------------------
PEND_ACTIVITY_MATCH = r"Standard\s*Stock"   # ActivityType contains this
PEND_ACTION_MATCH   = r"Add"                # Action contains this

def _parse_pend_affected(s: str):
    """
    Parse strings like:
      'SJS11E_MAIN Drw 2.1-Pkt E1 (LORAZ2IV1): 1'
       -> med_id='LORAZ2IV1', drawer='2.1', pocket='E1', qty=1
    Robust to 'Dr', 'Drw', 'Drawer' and 'Pkt'/'Pocket'.
    """
    s = s if isinstance(s, str) else str(s)
    med_m = re.search(r"\(([^)]+)\)", s)
    med_id = med_m.group(1).strip() if med_m else None
    drw_m = re.search(r"(?:Drw|Drawer|Dr)\s*([A-Za-z0-9\.]+)", s, flags=re.I)
    drawer = drw_m.group(1) if drw_m else None
    pkt_m = re.search(r"(?:Pkt|Pocket)\s*([A-Za-z0-9\.]+)", s, flags=re.I)
    pocket = pkt_m.group(1) if pkt_m else None
    qty_m = re.search(r":\s*(-?\d+)\s*$", s)
    qty = int(qty_m.group(1)) if qty_m else None
    return med_id, drawer, pocket, qty

# ActivityType text fragments (works with your screenshot)
MIN_ACTIVITY_MATCH = r"Refill\s*Point\s*Min\s*Quantity"
MAX_ACTIVITY_MATCH = r"(?:Par|Physical)\s*Max\s*Quantity"

def _parse_slot_affected(s: str):
    """
    Parse strings like:
      'SJS7ES_MAIN Drw 2.1-Pkt B1 (FLUCO200TAB): 3'
       -> med_id='FLUCO200TAB', drawer='2.1', pocket='B1', value=3.0

    Works with Dr/Drw/Drawer, Pkt/Pocket, hyphen or space between segments,
    decimals in drawer, and optional spaces around the colon.
    """
    s = "" if s is None else str(s).strip()

    # robust regex for the whole line
    rx = re.compile(
        r"""
        ^(?P<device>[^\s]+)            # e.g. SJS7ES_MAIN (kept if you ever need it)
        .*?                            # anything until drawer
        (?:Drw|Drawer|Dr)\s*(?P<drawer>[A-Za-z0-9\.]+)   # 2.1 or 3
        [\-\s]*                        # hyphen/space separator
        (?:Pkt|Pocket)\s*(?P<pocket>[A-Za-z0-9\.]+)     # B1 / 1 / E1
        \s*\(\s*(?P<med>[A-Za-z0-9._\-]+)\s*\)\s*       # (FLUCO200TAB)
        :\s*(?P<val>-?\d+(?:\.\d+)?)\s*                 # : 3 / : 6 / : 0
        $""",
        re.IGNORECASE | re.VERBOSE,
    )

    m = rx.search(s)
    if not m:
        # fall back to your lighter extraction so we never hard-fail
        med_m = re.search(r"\(([^)]+)\)", s)
        med_id = (med_m.group(1).strip().upper() if med_m else "")
        drw_m = re.search(r"(?:Drw|Drawer|Dr)\s*([A-Za-z0-9\.]+)", s, flags=re.I)
        drawer = (drw_m.group(1).strip().upper() if drw_m else "")
        pkt_m = re.search(r"(?:Pkt|Pocket)\s*([A-Za-z0-9\.]+)", s, flags=re.I)
        pocket = (pkt_m.group(1).strip().upper() if pkt_m else "")
        val_m = re.search(r":\s*(-?\d+\.?\d*)\s*$", s)
        value = float(val_m.group(1)) if val_m else None
        return med_id, drawer, pocket, value

    med_id = m.group("med").strip().upper()
    drawer = m.group("drawer").strip().upper()
    pocket = m.group("pocket").strip().upper()
    value = float(m.group("val"))
    return med_id, drawer, pocket, value


def build_slot_config(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build latest qty_min/qty_max per (device_base, med_id, drawer, pocket)
    and also provide drawer_root for fallback merging.
    """
    need = {"Device","ActivityType","Action","AffectedElement","TransactionDateTime"}
    if not need.issubset(df.columns):
        return pd.DataFrame(columns=[
            "device_base","device","med_id","drawer","drawer_root","pocket",
            "qty_min","qty_max","ts_updated"
        ])

    base = df.copy()
    base["ts"]          = pd.to_datetime(base["TransactionDateTime"], errors="coerce")
    base["device"]      = base["Device"].astype(str).str.strip()
    base["device_base"] = base["device"].map(_device_base)

    # ---- MIN rows ----
    is_min = base["ActivityType"].astype(str).str.contains(r"Refill\s*Point\s*Min\s*Quantity", case=False, na=False)
    mins = base.loc[is_min, ["ts","device_base","AffectedElement"]].copy()
    if not mins.empty:
        parsed = mins["AffectedElement"].apply(_parse_slot_affected)  # -> (med_id, drawer, pocket, value)
        mins[["med_id","drawer","pocket","value"]] = pd.DataFrame(parsed.tolist(), index=mins.index)
        mins["med_id"]      = mins["med_id"].map(_norm_med)
        mins["drawer"]      = mins["drawer"].map(_norm_slot_str)
        mins["pocket"]      = mins["pocket"].map(_norm_slot_str)
        mins["drawer_root"] = mins["drawer"].map(_drawer_root)
        mins = (mins.dropna(subset=["ts","device_base","med_id"])
                    .sort_values("ts")
                    .groupby(["device_base","med_id","drawer","pocket"], as_index=False)
                    .last()[["device_base","med_id","drawer","pocket","value","ts"]]
                    .rename(columns={"value":"qty_min","ts":"ts_updated_min"}))
    else:
        mins = pd.DataFrame(columns=["device_base","med_id","drawer","pocket","qty_min","ts_updated_min"])

    # ---- MAX rows ----
    is_max = base["ActivityType"].astype(str).str.contains(r"(?:Par|Physical)\s*Max\s*Quantity", case=False, na=False)
    maxs = base.loc[is_max, ["ts","device_base","AffectedElement"]].copy()
    if not maxs.empty:
        parsed = maxs["AffectedElement"].apply(_parse_slot_affected)
        maxs[["med_id","drawer","pocket","value"]] = pd.DataFrame(parsed.tolist(), index=maxs.index)
        maxs["med_id"]      = maxs["med_id"].map(_norm_med)
        maxs["drawer"]      = maxs["drawer"].map(_norm_slot_str)
        maxs["pocket"]      = maxs["pocket"].map(_norm_slot_str)
        maxs["drawer_root"] = maxs["drawer"].map(_drawer_root)
        maxs = (maxs.dropna(subset=["ts","device_base","med_id"])
                    .sort_values("ts")
                    .groupby(["device_base","med_id","drawer","pocket"], as_index=False)
                    .last()[["device_base","med_id","drawer","pocket","value","ts"]]
                    .rename(columns={"value":"qty_max","ts":"ts_updated_max"}))
    else:
        maxs = pd.DataFrame(columns=["device_base","med_id","drawer","pocket","qty_max","ts_updated_max"])

    # Merge + finalize
    cfg = mins.merge(maxs, on=["device_base","med_id","drawer","pocket"], how="outer")
    cfg["ts_updated"] = cfg[["ts_updated_min","ts_updated_max"]].max(axis=1)
    cfg["device"]       = cfg["device_base"]
    cfg["drawer_root"]  = cfg["drawer"].map(_drawer_root)

    # Nullable Int64 (keeps NA)
    for c in ("qty_min","qty_max"):
        if c in cfg.columns:
            cfg[c] = pd.to_numeric(cfg[c], errors="coerce").round().astype("Int64")

    return cfg[["device_base","device","med_id","drawer","drawer_root","pocket","qty_min","qty_max","ts_updated"]]


    # --- parse "AffectedElement": SJS11W_TWR Dr 3-Pkt 1 (ENOXA100IV): 2
    def _parse_slot(s: str):
        s = s if isinstance(s, str) else str(s)
        med_m = re.search(r"\(([^)]+)\)", s)
        med_id = med_m.group(1).strip() if med_m else None
        drw_m = re.search(r"(?:Drw|Drawer|Dr)\s*([A-Za-z0-9\.]+)", s, flags=re.I)
        drawer = drw_m.group(1) if drw_m else None
        pkt_m = re.search(r"(?:Pkt|Pocket)\s*([A-Za-z0-9\.]+)", s, flags=re.I)
        pocket = pkt_m.group(1) if pkt_m else None
        val_m = re.search(r":\s*(-?\d+\.?\d*)\s*$", s)
        value = float(val_m.group(1)) if val_m else None
        return med_id, drawer, pocket, value

    # --- MIN rows
    is_min = base["ActivityType"].astype(str).str.contains(r"Refill\s*Point\s*Min\s*Quantity", case=False, na=False)
    mins = base.loc[is_min].copy()
    if not mins.empty:
        mins[["med_id","drawer","pocket","value"]] = mins["AffectedElement"].apply(_parse_slot).apply(pd.Series)
        mins["drawer"] = mins["drawer"].map(_norm_slot_str)
        mins["pocket"] = mins["pocket"].map(_norm_slot_str)
        mins = (mins.dropna(subset=["ts","device_base","med_id"])
                    .sort_values("ts")
                    .groupby(["device_base","med_id","drawer","pocket"], as_index=False)
                    .last()[["device_base","med_id","drawer","pocket","value","ts"]]
                    .rename(columns={"value":"qty_min","ts":"ts_updated_min"}))
    else:
        mins = pd.DataFrame(columns=["device_base","med_id","drawer","pocket","qty_min","ts_updated_min"])

    # --- MAX rows
    is_max = base["ActivityType"].astype(str).str.contains(r"(?:Par|Physical)\s*Max\s*Quantity", case=False, na=False)
    maxs = base.loc[is_max].copy()
    if not maxs.empty:
        maxs[["med_id","drawer","pocket","value"]] = maxs["AffectedElement"].apply(_parse_slot).apply(pd.Series)
        maxs["drawer"] = maxs["drawer"].map(_norm_slot_str)
        maxs["pocket"] = maxs["pocket"].map(_norm_slot_str)
        maxs = (maxs.dropna(subset=["ts","device_base","med_id"])
                    .sort_values("ts")
                    .groupby(["device_base","med_id","drawer","pocket"], as_index=False)
                    .last()[["device_base","med_id","drawer","pocket","value","ts"]]
                    .rename(columns={"value":"qty_max","ts":"ts_updated_max"}))
    else:
        maxs = pd.DataFrame(columns=["device_base","med_id","drawer","pocket","qty_max","ts_updated_max"])

    cfg = mins.merge(maxs, on=["device_base","med_id","drawer","pocket"], how="outer")
    cfg["ts_updated"] = cfg[["ts_updated_min","ts_updated_max"]].max(axis=1)
    cfg["device"] = cfg["device_base"]  # optional display
    return cfg[["device_base","device","med_id","drawer","pocket","qty_min","qty_max","ts_updated"]]


    # helpers
    def _norm_keys(x):
        return (str(x).strip().upper().replace(" ", ""))

    # --- MIN rows ---
    is_min = base["ActivityType"].astype(str).str.contains(MIN_ACTIVITY_MATCH, case=False, na=False)
    mins = base.loc[is_min, ["Device","AffectedElement","ts"]].copy()
    if not mins.empty:
        parsed = mins["AffectedElement"].apply(_parse_slot_affected)
        mins[["med_id","drawer","pocket","value"]] = pd.DataFrame(parsed.tolist(), index=mins.index)
        mins = mins.dropna(subset=["ts","med_id"])
        mins["device"] = mins["Device"].astype(str)
        # normalize keys
        for c in ("device","med_id","drawer","pocket"):
            mins[c] = mins[c].apply(_norm_keys)
        # take latest by ts
        mins = (mins.sort_values("ts")
                    .groupby(["device","med_id","drawer","pocket"], as_index=False)
                    .last()[["device","med_id","drawer","pocket","value","ts"]]
                    .rename(columns={"value":"qty_min","ts":"ts_updated_min"}))
    else:
        mins = pd.DataFrame(columns=["device","med_id","drawer","pocket","qty_min","ts_updated_min"])

    # --- MAX rows ---
    is_max = base["ActivityType"].astype(str).str.contains(MAX_ACTIVITY_MATCH, case=False, na=False)
    maxs = base.loc[is_max, ["Device","AffectedElement","ts"]].copy()
    if not maxs.empty:
        parsed = maxs["AffectedElement"].apply(_parse_slot_affected)
        maxs[["med_id","drawer","pocket","value"]] = pd.DataFrame(parsed.tolist(), index=maxs.index)
        maxs = maxs.dropna(subset=["ts","med_id"])
        maxs["device"] = maxs["Device"].astype(str)
        for c in ("device","med_id","drawer","pocket"):
            maxs[c] = maxs[c].apply(_norm_keys)
        maxs = (maxs.sort_values("ts")
                    .groupby(["device","med_id","drawer","pocket"], as_index=False)
                    .last()[["device","med_id","drawer","pocket","value","ts"]]
                    .rename(columns={"value":"qty_max","ts":"ts_updated_max"}))
    else:
        maxs = pd.DataFrame(columns=["device","med_id","drawer","pocket","qty_max","ts_updated_max"])

    # merge + dtype
    cfg = mins.merge(maxs, on=["device","med_id","drawer","pocket"], how="outer")
    cfg["ts_updated"] = cfg[["ts_updated_min","ts_updated_max"]].max(axis=1)

    # use nullable Int64 (keeps NA cleanly)
    for c in ("qty_min","qty_max"):
        if c in cfg.columns:
            cfg[c] = pd.to_numeric(cfg[c], errors="coerce").round().astype("Int64")

    return cfg[["device","med_id","drawer","pocket","qty_min","qty_max","ts_updated"]]


def build_pyxis_pends_from_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a DeviceActivityLog-style dataframe with columns:
      ['Device','ActivityType','Action','AffectedElement','TransactionDateTime','UserName','UserID',
       optional: 'DispensingDeviceName','Area']
    return tidy pended-load rows with parsed fields.
    """
    needed = {"Device","ActivityType","Action","AffectedElement","TransactionDateTime"}
    if not needed.issubset(set(df.columns)):
        return pd.DataFrame(columns=[
            "ts","device","med_id","drawer","pocket","qty",
            "AffectedElement","DispensingDeviceName","Area","username","userid"
        ])

    mask = (
        df["ActivityType"].astype(str).str.contains(PEND_ACTIVITY_MATCH, case=False, na=False)
        & df["Action"].astype(str).str.contains(PEND_ACTION_MATCH, case=False, na=False)
    )
    sub = df.loc[mask].copy()
    if sub.empty:
        return pd.DataFrame(columns=[
            "ts","device","med_id","drawer","pocket","qty",
            "AffectedElement","DispensingDeviceName","Area","username","userid"
        ])

    parsed = sub["AffectedElement"].apply(_parse_pend_affected)
    sub[["med_id","drawer","pocket","qty"]] = pd.DataFrame(parsed.tolist(), index=sub.index)

    sub["ts"]       = pd.to_datetime(sub["TransactionDateTime"], errors="coerce")
    sub["device"]   = sub["Device"].astype(str).str.strip()
    sub["username"] = sub.get("UserName", pd.Series(index=sub.index)).astype(str).str.strip()
    sub["userid"]   = sub.get("UserID", pd.Series(index=sub.index)).astype(str).str.strip()

    # Optional columns present in your CSV
    if "DispensingDeviceName" not in sub: sub["DispensingDeviceName"] = pd.NA
    if "Area" not in sub:                  sub["Area"] = pd.NA

    # Canonical order + drop unparseable timestamps/devices/med_id
    out = sub[[
        "ts","device","med_id","drawer","pocket","qty",
        "AffectedElement","DispensingDeviceName","Area","username","userid"
    ]].dropna(subset=["ts","device","med_id"]).reset_index(drop=True)

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
        # use visit duration as a per-row dwell metric (seconds)
    data["dwell_sec"] = data["visit_duration_s"].fillna(0)


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

def build_refill_audit(df_in: pd.DataFrame, colmap: Dict[str, str]) -> pd.DataFrame:
    """
    Per-user refill audit metrics, using the dataframe that HAS dwell_sec
    (in your case, this is likely data_f from delivery analytics).

    Returns:
        user, refill_count, median_dwell_sec, avg_dwell_sec, risk_score
    """
    if df_in is None or df_in.empty:
        return pd.DataFrame(
            columns=["user", "refill_count", "median_dwell_sec", "avg_dwell_sec", "risk_score"]
        )

    df = df_in.copy()

    # Map your actual names
    user_col = colmap.get("user", "UserName")
    type_col = colmap.get("type", "TransactionType")

    # We assume dwell is in 'dwell_sec' in data_f – change here if named differently
    dwell_col = "dwell_sec"
    if dwell_col not in df.columns:
        return pd.DataFrame(
            columns=["user", "refill_count", "median_dwell_sec", "avg_dwell_sec", "risk_score"]
        )

    df = df.rename(
        columns={
            user_col: "user",
            type_col: "type",
            dwell_col: "dwell_sec",
        },
        errors="ignore",
    )

    # Adjust to match your actual refill transaction strings
    REFILL_TYPES = {
    "REFILL",
    "REFILL-LOAD",
    "REFILL LOAD",
    "LOAD",
    "REFILL RETURN",
    }

    df_refill = df[df["type"].isin(REFILL_TYPES)].copy()
    df_refill = df_refill[df_refill["dwell_sec"] > 0]


    if df_refill.empty:
        return pd.DataFrame(
            columns=["user", "refill_count", "median_dwell_sec", "avg_dwell_sec", "risk_score"]
        )

    per_user = (
        df_refill
        .groupby("user", as_index=False)
        .agg(
            refill_count=("type", "size"),
            median_dwell_sec=("dwell_sec", "median"),
            avg_dwell_sec=("dwell_sec", "mean"),
        )
    )

    # Volume score 0–1
    max_refills = per_user["refill_count"].max()
    if max_refills == 0:
        max_refills = 1
    per_user["volume_score"] = per_user["refill_count"] / max_refills

    # Speed score 0–1 (fast = short dwell)
    max_median = per_user["median_dwell_sec"].max()
    if max_median == 0:
        max_median = 1
    per_user["speed_score"] = (max_median - per_user["median_dwell_sec"]) / max_median
    per_user["speed_score"] = per_user["speed_score"].clip(lower=0)

    # Combined risk 0–100
    per_user["risk_score"] = (
        0.6 * per_user["speed_score"] + 0.4 * per_user["volume_score"]
    ) * 100
    per_user["risk_score"] = per_user["risk_score"].round(0).astype(int)

    per_user = per_user.sort_values("risk_score", ascending=False)

    return per_user[
        ["user", "refill_count", "median_dwell_sec", "avg_dwell_sec", "risk_score"]
    ]


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

        # Only rows where we stayed on the same device
        sub = data.loc[~data["__device_change"], [dev, "__gap_s"]].copy()
        sub["__gap_s"] = pd.to_numeric(sub["__gap_s"], errors="coerce")

        if sub["__gap_s"].dropna().empty:
            return "No dwell data in current filter.", pd.DataFrame()

        t = (
            sub.groupby(dev, observed=True)["__gap_s"]
               .median()
               .rename("median_dwell_s")
               .reset_index()
               .sort_values("median_dwell_s", ascending=False)
               .head(10)
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
# ==== Load/Unload Insights START ====

# Event name aliases (tune to your export naming as needed)
LOAD_ALIASES = {"LOAD", "RESTOCK", "ADD", "INITIAL_LOAD"}
UNLOAD_ALIASES = {"UNLOAD", "REMOVE", "PULL", "RETURN_TO_STOCK", "WASTE_PULL"}

def _classify_direction(evt_type: str) -> str:
    if not evt_type:
        return "other"
    t = str(evt_type).strip().upper()
    if t in LOAD_ALIASES: return "load"
    if t in UNLOAD_ALIASES: return "unload"
    if "LOAD" in t or "RESTOCK" in t or "ADD" in t: return "load"
    if any(x in t for x in ["UNLOAD", "REMOVE", "PULL", "RETURN", "WASTE"]): return "unload"
    return "other"
def _build_load_unload_base(df: pd.DataFrame, colmap: Dict[str, str]) -> pd.DataFrame:
    dt = colmap["datetime"]
    med = colmap.get("desc") or colmap.get("medid") or "MedID"
    typ = colmap["type"]
    dev = colmap.get("device", "Device")
    qty = colmap.get("qty")

    cols = [c for c in [dt, med, typ, dev, qty] if c and c in df.columns]
    g = df[cols].copy()

    g[dt] = pd.to_datetime(g[dt], errors="coerce")
    g["ts"] = g[dt]  # keep original timestamp for extremes
    g["dir"] = g[typ].map(_classify_direction)
    g = g[g["dir"].isin(["load","unload"])]

    if qty and qty in g.columns:
        g["orig_qty"] = pd.to_numeric(g[qty], errors="coerce").fillna(1).astype(float)
        g["qty_signed"] = g["orig_qty"].where(g["dir"].eq("load"), -g["orig_qty"])
        g["events"] = 1
    else:
        g["orig_qty"] = 1.0
        g["qty_signed"] = g["dir"].map({"load": 1.0, "unload": -1.0})
        g["events"] = 1

    g["date"] = g[dt].dt.date
    g["hour"] = g[dt].dt.hour
    g["weekday"] = g[dt].dt.day_name()
    return g.rename(columns={med: "med", dev: "device"})

def _agg_daily(g: pd.DataFrame) -> pd.DataFrame:
    daily = (g.groupby(["date","med","dir"])
               .agg(events=("events","sum"), qty=("qty_signed","sum"))
               .reset_index())
    pivot = (daily.pivot_table(index=["date","med"], columns="dir", values="qty",
                               aggfunc="sum", fill_value=0)
                  .reset_index().rename_axis(None, axis=1))
    for c in ("load","unload"):
        if c not in pivot: pivot[c] = 0.0
    pivot["net"] = pivot["load"] + pivot["unload"]  # unload is negative
    pivot = pivot.sort_values(["med","date"])
    pivot["roll7"]  = pivot.groupby("med")["net"].transform(lambda s: s.rolling(7,  min_periods=1).sum())
    pivot["roll30"] = pivot.groupby("med")["net"].transform(lambda s: s.rolling(30, min_periods=1).sum())
    return pivot

def _top_movers(daily_pivot: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    if daily_pivot is None or daily_pivot.empty:
        return pd.DataFrame(columns=["med", "net_cur", "net_prev", "delta", "pct"])

    m = daily_pivot.copy()
    m["date"] = pd.to_datetime(m["date"], errors="coerce")
    m = m.dropna(subset=["date"])
    if m.empty:
        return pd.DataFrame(columns=["med", "net_cur", "net_prev", "delta", "pct"])

    # Ensure numeric nets
    for col in ("net", "load", "unload"):
        if col in m.columns:
            m[col] = pd.to_numeric(m[col], errors="coerce").fillna(0.0)

    last_date = m["date"].max()
    cur_start = last_date - pd.Timedelta(days=window - 1)
    prev_start = cur_start - pd.Timedelta(days=window)
    prev_end = cur_start - pd.Timedelta(days=1)

    cur = m[(m["date"] >= cur_start) & (m["date"] <= last_date)].groupby("med")["net"].sum()
    prev = m[(m["date"] >= prev_start) & (m["date"] <= prev_end)].groupby("med")["net"].sum()

    out = (
        cur.fillna(0.0).to_frame("net_cur")
        .join(prev.fillna(0.0).to_frame("net_prev"), how="outer")
        .fillna(0.0)
        .astype({"net_cur": "float64", "net_prev": "float64"})
    )
    out["delta"] = out["net_cur"] - out["net_prev"]

    # Safe percentage: np.where avoids object dtypes; division-by-zero → np.nan
    out["pct"] = np.where(out["net_prev"] != 0.0, out["delta"] / out["net_prev"], np.nan).astype("float64")

    return out.sort_values("delta", ascending=False).reset_index()

def _plot_overall_timeseries(daily_pivot: pd.DataFrame, med_filter=None):
    d = daily_pivot.copy()
    if med_filter: d = d[d["med"].isin(med_filter)]
    ts = d.groupby("date")[["load","unload","net"]].sum().reset_index()
    return px.line(ts, x="date", y=["load","unload","net"], markers=True,
                   title="Loaded vs Unloaded vs Net (All Meds)")

def _plot_med_timeseries(daily_pivot: pd.DataFrame, med: str):
    d = daily_pivot[daily_pivot["med"].eq(med)].sort_values("date")
    return px.line(d, x="date", y=["load","unload","net","roll7","roll30"], markers=True,
                   title=f"{med} — Load/Unload/Net with 7/30-day rolls")

def _heatmap_by_hour_weekday(g: pd.DataFrame):
    if g.empty:
        return px.imshow([[0]], title="No data")
    p = g.groupby(["weekday","hour"])["qty_signed"].mean().reset_index()
    order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    p["weekday"] = pd.Categorical(p["weekday"], categories=order, ordered=True)
    mat = p.pivot(index="weekday", columns="hour", values="qty_signed").reindex(order)
    return px.imshow(mat, aspect="auto", title="Avg Signed Qty by Hour × Weekday (loads=+, unloads=−)")

def _refill_heatmap(ref: pd.DataFrame, ts_col: str):
    """
    Simple count-based heatmap: how many refill events by weekday × hour.
    """
    if ref is None or ref.empty or ts_col not in ref.columns:
        return px.imshow([[0]], title="No refill data")

    df = ref[[ts_col]].copy()
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    df = df.dropna(subset=[ts_col])
    if df.empty:
        return px.imshow([[0]], title="No refill data")

    df["weekday"] = df[ts_col].dt.day_name()
    df["hour"] = df[ts_col].dt.hour

    order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    grp = df.groupby(["weekday","hour"]).size().reset_index(name="refills")
    grp["weekday"] = pd.Categorical(grp["weekday"], categories=order, ordered=True)

    mat = grp.pivot(index="weekday", columns="hour", values="refills").reindex(order)
    mat = mat.fillna(0)

    return px.imshow(mat, aspect="auto", title="Refills by Hour × Weekday")

def _by_device(g: pd.DataFrame) -> pd.DataFrame:
    base = (g.groupby(["device","dir"]).agg(events=("events","sum"), qty=("qty_signed","sum")).reset_index())
    pv = (base.pivot_table(index="device", columns="dir", values="qty", aggfunc="sum", fill_value=0)
               .reset_index().rename_axis(None, axis=1))
    for c in ("load","unload"):

        if c not in pv: pv[c] = 0.0
    pv["net"] = pv["load"] + pv["unload"]
    evt = (base.pivot_table(index="device", columns="dir", values="events", aggfunc="sum", fill_value=0)
                 .reset_index().rename_axis(None, axis=1))
    for c in ("load","unload"):
        if c not in evt: evt[c] = 0
    evt = evt.rename(columns={"load":"load_events","unload":"unload_events"})
    return pv.merge(evt, on="device", how="left").sort_values("net", ascending=False)

def _med_device_breakdown(g: pd.DataFrame, med: str) -> pd.DataFrame:
    sub = g[g["med"] == med]
    if sub.empty:
        return pd.DataFrame(columns=["device","load","unload","net","load_events","unload_events"])
    return _by_device(sub)

def _extremes_by_group(g: pd.DataFrame, group_cols) -> pd.DataFrame:
    """
    For each group (e.g., ["med"] or ["med","device"]), return:
      - load_qty_max / load_ts_at_max / load_device_at_max
      - load_qty_min / load_ts_at_min / load_device_at_min
      - unload_qty_max / unload_ts_at_max / unload_device_at_max
      - unload_qty_min / unload_ts_at_min / unload_device_at_min
    """
    if isinstance(group_cols, str):
        group_cols = [group_cols]
    else:
        group_cols = list(group_cols)

    if g is None or g.empty:
        cols = group_cols + [
            "load_qty_max","load_ts_at_max","load_device_at_max",
            "load_qty_min","load_ts_at_min","load_device_at_min",
            "unload_qty_max","unload_ts_at_max","unload_device_at_max",
            "unload_qty_min","unload_ts_at_min","unload_device_at_min",
        ]
        return pd.DataFrame(columns=cols)

    df = g.copy()
    for c in group_cols + ["orig_qty","ts","device","dir"]:
        if c not in df.columns:
            df[c] = pd.NA
    df["orig_qty"] = pd.to_numeric(df["orig_qty"], errors="coerce").fillna(1.0)
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce")

    def _pick(df_all: pd.DataFrame, sign: str) -> pd.DataFrame:
        sub = df_all[df_all["dir"].eq(sign)]
        if sub.empty:
            cols = group_cols + [
                f"{sign}_qty_max", f"{sign}_ts_at_max", f"{sign}_device_at_max",
                f"{sign}_qty_min", f"{sign}_ts_at_min", f"{sign}_device_at_min",
            ]
            return pd.DataFrame(columns=cols)

        max_rows = (
            sub.sort_values(["orig_qty","ts"], ascending=[False, True])
               .groupby(group_cols, as_index=False, sort=False)
               .head(1)[group_cols + ["orig_qty","ts","device"]]
               .rename(columns={
                   "orig_qty": f"{sign}_qty_max",
                   "ts": f"{sign}_ts_at_max",
                   "device": f"{sign}_device_at_max"
               })
        )
        min_rows = (
            sub.sort_values(["orig_qty","ts"], ascending=[True, True])
               .groupby(group_cols, as_index=False, sort=False)
               .head(1)[group_cols + ["orig_qty","ts","device"]]
               .rename(columns={
                   "orig_qty": f"{sign}_qty_min",
                   "ts": f"{sign}_ts_at_min",
                   "device": f"{sign}_device_at_min"
               })
        )
        return pd.merge(max_rows, min_rows, on=group_cols, how="outer")

    loads   = _pick(df, "load")
    unloads = _pick(df, "unload")
    out = pd.merge(loads, unloads, on=group_cols, how="outer")

    desired = group_cols + [
        "load_qty_max","load_ts_at_max","load_device_at_max",
        "load_qty_min","load_ts_at_min","load_device_at_min",
        "unload_qty_max","unload_ts_at_max","unload_device_at_max",
        "unload_qty_min","unload_ts_at_min","unload_device_at_min",
    ]
    for c in desired:
        if c not in out.columns:
            out[c] = pd.NA
    return out[desired]


    def _pick_ext_sign(df_all: pd.DataFrame, sign: str) -> pd.DataFrame:
        sub = df_all[df_all["dir"].eq(sign)]
        if sub.empty:
            cols = group_cols + [
                f"{sign}_qty_max", f"{sign}_ts_at_max", f"{sign}_device_at_max",
                f"{sign}_qty_min", f"{sign}_ts_at_min", f"{sign}_device_at_min",
            ]
            return pd.DataFrame(columns=cols)
        # largest qty per group
        max_rows = (
            sub.sort_values(["orig_qty","ts"], ascending=[False, True])
               .groupby(group_cols, as_index=False, sort=False)
               .head(1)[group_cols + ["orig_qty","ts","device"]]
               .rename(columns={
                   "orig_qty": f"{sign}_qty_max",
                   "ts": f"{sign}_ts_at_max",
                   "device": f"{sign}_device_at_max"
               })
        )
        # smallest qty per group
        min_rows = (
            sub.sort_values(["orig_qty","ts"], ascending=[True, True])
               .groupby(group_cols, as_index=False, sort=False)
               .head(1)[group_cols + ["orig_qty","ts","device"]]
               .rename(columns={
                   "orig_qty": f"{sign}_qty_min",
                   "ts": f"{sign}_ts_at_min",
                   "device": f"{sign}_device_at_min"
               })
        )
        return pd.merge(max_rows, min_rows, on=group_cols, how="outer")

    loads   = _pick_ext_sign(df, "load")
    unloads = _pick_ext_sign(df, "unload")

    out = pd.merge(loads, unloads, on=group_cols, how="outer")

    desired = group_cols + [
        "load_qty_max","load_ts_at_max","load_device_at_max",
        "load_qty_min","load_ts_at_min","load_device_at_min",
        "unload_qty_max","unload_ts_at_max","unload_device_at_max",
        "unload_qty_min","unload_ts_at_min","unload_device_at_min",
    ]
    for c in desired:
        if c not in out.columns:
            out[c] = pd.NA
    return out[desired]


    def _pick_ext(df_sub: pd.DataFrame, sign: str) -> pd.DataFrame:
        sub = df_sub[df_sub["dir"].eq(sign)]
        if sub.empty:
            cols = group_cols + [
                f"{sign}_qty_max", f"{sign}_ts_at_max", f"{sign}_device_at_max",
                f"{sign}_qty_min", f"{sign}_ts_at_min", f"{sign}_device_at_min",
            ]
            return pd.DataFrame(columns=cols)

        # Max rows (largest qty)
        max_rows = (
            sub.sort_values(["orig_qty","ts"], ascending=[False, True])
               .groupby(group_cols, as_index=False, sort=False)
               .head(1)[group_cols + ["orig_qty","ts","device"]]
               .rename(columns={
                   "orig_qty": f"{sign}_qty_max",
                   "ts": f"{sign}_ts_at_max",
                   "device": f"{sign}_device_at_max"
               })
        )
        # Min rows (smallest qty)
        min_rows = (
            sub.sort_values(["orig_qty","ts"], ascending=[True, True])
               .groupby(group_cols, as_index=False, sort=False)
               .head(1)[group_cols + ["orig_qty","ts","device"]]
               .rename(columns={
                   "orig_qty": f"{sign}_qty_min",
                   "ts": f"{sign}_ts_at_min",
                   "device": f"{sign}_device_at_min"
               })
        )
        return pd.merge(max_rows, min_rows, on=group_cols, how="outer")

    loads   = _pick_ext(df, "load")
    unloads = _pick_ext(df, "unload")

    # Outer-join the two sides; keep group columns even if one side is empty
    out = pd.merge(loads, unloads, on=group_cols, how="outer")

    # Consistent column order
    desired = group_cols + [
        "load_qty_max","load_ts_at_max","load_device_at_max",
        "load_qty_min","load_ts_at_min","load_device_at_min",
        "unload_qty_max","unload_ts_at_max","unload_device_at_max",
        "unload_qty_min","unload_ts_at_min","unload_device_at_min",
    ]
    for c in desired:
        if c not in out.columns:
            out[c] = pd.NA
    return out[desired]


    def _agg_ext(df, sign: str):
        if df.empty:
            cols = group_cols + [f"{sign}_qty_max", f"{sign}_ts_at_max", f"{sign}_device_at_max",
                                 f"{sign}_qty_min", f"{sign}_ts_at_min", f"{sign}_device_at_min"]
            return pd.DataFrame(columns=cols)
        df["orig_qty"] = pd.to_numeric(df["orig_qty"], errors="coerce").fillna(1.0)
        grp = df.groupby(group_cols, as_index=False)
        idx_max = grp["orig_qty"].idxmax()
        idx_min = grp["orig_qty"].idxmin()
        r_max = df.loc[idx_max, group_cols + ["orig_qty","ts","device"]].rename(
            columns={"orig_qty":f"{sign}_qty_max","ts":f"{sign}_ts_at_max","device":f"{sign}_device_at_max"})
        r_min = df.loc[idx_min, group_cols + ["orig_qty","ts","device"]].rename(
            columns={"orig_qty":f"{sign}_qty_min","ts":f"{sign}_ts_at_min","device":f"{sign}_device_at_min"})
        return pd.merge(r_max, r_min, on=group_cols, how="outer")

    a = _agg_ext(loads, "load"); b = _agg_ext(unloads, "unload")
    if a.empty and b.empty:
        cols = group_cols + ["load_qty_max","load_ts_at_max","load_device_at_max",
                             "load_qty_min","load_ts_at_min","load_device_at_min",
                             "unload_qty_max","unload_ts_at_max","unload_device_at_max",
                             "unload_qty_min","unload_ts_at_min","unload_device_at_min"]
        return pd.DataFrame(columns=cols)
    return pd.merge(a, b, on=group_cols, how="outer")

def _time_of_day_profile(g: pd.DataFrame) -> pd.DataFrame:
    prof = (g.groupby(["hour","dir"])["qty_signed"].mean().reset_index())
    pv = (prof.pivot_table(index="hour", columns="dir", values="qty_signed", aggfunc="mean", fill_value=0)
               .reset_index().rename_axis(None, axis=1))
    for c in ("load","unload"):
        if c not in pv: pv[c] = 0.0
    pv["net"] = pv["load"] + pv["unload"]
    return pv.sort_values("hour")


def _flag_outliers(daily_pivot: pd.DataFrame) -> pd.DataFrame:
    if daily_pivot.empty:
        return pd.DataFrame(columns=["date","med","net","z"])
    d = daily_pivot.copy()
    def _z(s):
        std = s.std(ddof=0)
        return (s - s.mean()) / (std if std else 1)
    d["z"] = d.groupby("med")["net"].transform(_z)
    return d.loc[d["z"].abs() >= 2.5, ["date","med","net","z"]].sort_values("z", ascending=False)

def build_load_unload_section(df: pd.DataFrame, colmap: Dict[str,str]):
    if df is None or df.empty:
        st.info("No events in current filter.")
        return

    # Base + daily
    try:
        g = _build_load_unload_base(df, colmap)   # includes ts, device, med, dir, orig_qty, qty_signed, hour, weekday
        daily = _agg_daily(g)                     # per-day rollups
    except Exception as e:
        st.error(f"Could not build load/unload views: {e}")
        return

    # Filters
    meds_all = sorted(daily["med"].unique().tolist())
    devs_all = sorted(g["device"].dropna().unique().tolist())

    st.markdown("#### Simple view (for everyone)")
    ctop1, ctop2, ctop3, ctop4 = st.columns(4)
    # Totals for KPIs (use orig_qty so it's in units when qty exists; else = events)
    tot_load  = g.loc[g["dir"].eq("load"),   "orig_qty"].sum()
    tot_unld  = g.loc[g["dir"].eq("unload"), "orig_qty"].sum()
    tot_net   = tot_load - tot_unld
    n_devices = g["device"].nunique()

    ctop1.metric("Loaded (units)",   _fmt_int(tot_load))
    ctop2.metric("Unloaded (units)", _fmt_int(tot_unld))
    ctop3.metric("Net",              _fmt_int(tot_net), delta=None)
    ctop4.metric("Devices active",   _fmt_int(n_devices))

    st.caption(_plain_english_summary(g))

    # Controls for friendly view
    c1, c2, c3 = st.columns([2,2,1])
    with c1:
        pick_meds = st.multiselect("Focus meds (optional)", meds_all, default=[])
    with c2:
        pick_devs = st.multiselect("Focus devices (optional)", devs_all, default=[])
    with c3:
        window = st.selectbox("Compare window (days)", [7,14,30], index=1)

    g_view = g.copy()
    if pick_meds:
        g_view = g_view[g_view["med"].isin(pick_meds)]
        daily_view = daily[daily["med"].isin(pick_meds)]
    else:
        daily_view = daily.copy()
    if pick_devs:
        g_view = g_view[g_view["device"].isin(pick_devs)]

    # Tabs: By Device | By Med @ Device | Timeline | Advanced
    t1, t2, t3, t4 = st.tabs(["📦 By Device", "🧪 By Med @ Device", "📈 Timeline", "⚙️ Advanced"])

    # --- BY DEVICE ---
    with t1:
        st.markdown("##### Loads vs Unloads by Device")
        # qty by device & dir
        by_dev_qty = (g_view.groupby(["device","dir"])["orig_qty"].sum().reset_index())
        if by_dev_qty.empty:
            st.info("No device activity for the current filters.")
        else:
            # Stacked bar (top 20 devices by |net|)
            net_dev = (by_dev_qty.pivot_table(index="device", columns="dir", values="orig_qty", aggfunc="sum", fill_value=0)
                                  .reset_index())
            for c in ("load","unload"):
                if c not in net_dev: net_dev[c] = 0.0
            net_dev["net"] = net_dev["load"] - net_dev["unload"]
            top = (net_dev.reindex(net_dev["net"].abs().sort_values(ascending=False).index)
                          .head(20))
            fig = px.bar(top, x="device", y=["load","unload"], title="Top devices (loads vs unloads, units)")
            st.plotly_chart(fig, use_container_width=True)

            # Friendly table
            show = top[["device","load","unload","net"]].copy()
            show = show.sort_values("net", ascending=False)
            try:
                st.write(show.style.format({"load":"{:.0f}","unload":"{:.0f}","net":"{:+.0f}"}))
            except Exception:
                st.dataframe(show, use_container_width=True)

    # --- BY MED @ DEVICE (matrix) ---
    with t2:
        st.markdown("##### Which meds moved on which devices")
        # limit to top devices to keep the matrix readable
        dev_priority = (g_view.groupby("device")["orig_qty"].sum()
                              .sort_values(ascending=False).head(12).index.tolist())
        sub = g_view[g_view["device"].isin(dev_priority)]
        mat = (sub.groupby(["device","med","dir"])["orig_qty"].sum().reset_index())
        if mat.empty:
            st.info("No movement for the selected filters.")
        else:
            # Pivot to net units per (device, med)
            mat_p = (mat.pivot_table(index=["device","med"], columns="dir", values="orig_qty", aggfunc="sum", fill_value=0)
                        .reset_index())
            for c in ("load","unload"):
                if c not in mat_p: mat_p[c] = 0.0
            mat_p["net"] = mat_p["load"] - mat_p["unload"]
            # Show as a tidy table (device grouped)
            st.dataframe(mat_p.sort_values(["device","net"], ascending=[True, False]),
                         use_container_width=True, height=420)

    # --- TIMELINE (friendly) ---
    with t3:
        st.markdown("##### Trend over time")
        dview = daily_view.copy()
        if dview.empty:
            st.info("No daily activity with the current filters.")
        else:
            fig = _plot_overall_timeseries(dview, med_filter=pick_meds or None)
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("###### Movers (easy reading)")
            tm = _top_movers(dview, window=window).head(15)
            if tm.empty:
                st.info("No movers in this compare window.")
            else:
                # human-friendly formatting
                tm_disp = tm.copy()
                for c in ("net_cur","net_prev","delta"):
                    if c in tm_disp.columns:
                        tm_disp[c] = tm_disp[c].map(_fmt_int)
                if "pct" in tm_disp.columns:
                    tm_disp["pct"] = tm["pct"].map(lambda x: f"{x:.1%}" if pd.notna(x) else "")
                st.dataframe(tm_disp, use_container_width=True)

    # --- ADVANCED (keep your original power tools) ---
    with t4:
        st.markdown("##### Outliers & Heatmap (advanced)")
        st.markdown("Outlier Days (|z| ≥ 2.5 on net)")
        outs = _flag_outliers(daily_view)
        st.dataframe(outs, use_container_width=True, height=240)
        st.markdown("When do pulls spike? (hour × weekday)")
        st.plotly_chart(_heatmap_by_hour_weekday(g_view), use_container_width=True)

        st.markdown("##### Extremes")
        # per-med extremes in plain table
        ext_med = _extremes_by_group(g_view, group_cols=["med"])
        keep_cols = [c for c in ["med",
                                 "load_qty_max","load_ts_at_max","load_device_at_max",
                                 "unload_qty_max","unload_ts_at_max","unload_device_at_max"] if c in ext_med.columns]
        if keep_cols:
            st.dataframe(ext_med[keep_cols].sort_values("load_qty_max", ascending=False).head(30),
                         use_container_width=True, height=260)

        # Per-Med timeseries drilldown
        if meds_all:
            med_ts = st.selectbox("Per-Med timeseries", meds_all, index=0, key="med_ts_simple")
            st.plotly_chart(_plot_med_timeseries(daily_view, med_ts), use_container_width=True)

# ==== Load/Unload Insights END ====

def build_refill_efficiency_section(ev: pd.DataFrame, data: pd.DataFrame, colmap: Dict[str, str]):
    """
    Refill-focused view + abuse watchlist.

    - Only uses events where TransactionType contains 'refill'
    - Per-tech + per-(device, med) stats
    - Flags hyper-frequent, tiny refills as a 'watchlist'
    """
    if ev is None or ev.empty:
        st.info("No events available in this range.")
        return

    ts_col  = colmap["datetime"]
    dev_col = colmap["device"]
    usr_col = colmap["user"]
    typ_col = colmap["type"]

    qty_col = colmap.get("qty")
    med_col = None
    if colmap.get("medid") and colmap["medid"] in ev.columns:
        med_col = colmap["medid"]
    elif colmap.get("desc") and colmap["desc"] in ev.columns:
        med_col = colmap["desc"]

    needed = [ts_col, dev_col, usr_col, typ_col]
    if med_col:
        needed.append(med_col)
    if qty_col and qty_col in ev.columns:
        needed.append(qty_col)

    df = ev[needed].copy()
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    df = df.dropna(subset=[ts_col])
    if df.empty:
        st.info("No usable timestamps in this range.")
        return

    # Refill-only
    mask = df[typ_col].astype(str).str.contains("refill", case=False, na=False)
    df = df[mask].copy()
    if df.empty:
        st.info("No transactions with 'refill' in the type field in this time range.")
        return

    # Normalize quantities
    has_real_qty = bool(qty_col and qty_col in df.columns)
    if has_real_qty:
        df[qty_col] = pd.to_numeric(df[qty_col], errors="coerce")
        df["qty_units"] = df[qty_col].fillna(1.0)
    else:
        df["qty_units"] = 1.0

    df = df.sort_values(ts_col)

    # Overall span (for per-day rates)
    start_ts = df[ts_col].min()
    end_ts   = df[ts_col].max()
    days = max((end_ts - start_ts).days, 1)

    total_refills = int(len(df))
    total_units   = float(df["qty_units"].sum())
    unique_techs  = df[usr_col].nunique()
    unique_devs   = df[dev_col].nunique()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Refill events", f"{total_refills:,}")
    c2.metric("Refill units (approx.)", _fmt_int(total_units))
    c3.metric("Techs refilling", f"{unique_techs:,}")
    c4.metric("Devices touched", f"{unique_devs:,}")

    st.caption(
        f"Window: {start_ts:%Y-%m-%d %H:%M} → {end_ts:%Y-%m-%d %H:%M} "
        f"({days} day{'s' if days != 1 else ''})"
    )

    # ------------ Per-tech summary ------------
    st.markdown("### Per-tech refill summary")
    tech_df = (
        df.groupby(usr_col, observed=True)
          .agg(
              refills=(typ_col, "size"),
              devices=(dev_col, "nunique"),
              units=("qty_units", "sum"),
          )
          .reset_index()
          .sort_values("refills", ascending=False)
    )
    tech_df["refills_per_day"] = tech_df["refills"] / float(days)
    try:
        st.dataframe(
            tech_df.style.format(
                {
                    "refills": "{:,.0f}",
                    "devices": "{:,.0f}",
                    "units": "{:,.0f}",
                    "refills_per_day": "{:,.2f}",
                }
            ),
            use_container_width=True,
            height=300,
        )
    except Exception:
        st.dataframe(tech_df, use_container_width=True, height=300)

    # ------------ Per-(device, med) detail + abuse watchlist ------------
    st.markdown("### Refill patterns by device / med")

    # Grouping key: device + optional med
    group_cols = [dev_col]
    label_cols = [dev_col]
    if med_col:
        group_cols.append(med_col)
        label_cols.append(med_col)

    # Compute intra-refill gaps (hours) within each (device, [med]) group
    df["gap_h"] = (
        df.groupby(group_cols)[ts_col]
          .diff()
          .dt.total_seconds()
          .div(3600.0)
    )

    combo = (
        df.groupby(group_cols, observed=True)
          .agg(
              n_refills=(typ_col, "size"),
              total_units=("qty_units", "sum"),
              avg_units=("qty_units", "mean"),
              median_gap_h=("gap_h", lambda s: float(np.nanmedian(s)) if s.notna().any() else np.nan),
          )
          .reset_index()
    )
    combo["refills_per_day"] = combo["n_refills"] / float(days)

    # Nicer column names
    rename_map = {
        dev_col: "device",
    }
    if med_col:
        rename_map[med_col] = "med"
    combo = combo.rename(columns=rename_map)

    st.dataframe(
        combo.sort_values("n_refills", ascending=False),
        use_container_width=True,
        height=320,
    )

    # ------------ Refill Abuse Watchlist ------------
    st.markdown("### Refill Abuse Watchlist 🧨")

    MIN_REFILLS = 5          # at least this many refills in the window
    MAX_MEDIAN_GAP_H = 8.0   # median gap between refills ≤ 8 hours
    MAX_AVG_UNITS = 3.0      # 'tiny top-ups' if we have real qty

    if has_real_qty:
        watch = combo[
            (combo["n_refills"] >= MIN_REFILLS)
            & (combo["median_gap_h"].notna())
            & (combo["median_gap_h"] <= MAX_MEDIAN_GAP_H)
            & (combo["avg_units"] <= MAX_AVG_UNITS)
        ].copy()
    else:
        # No real quantity; just use frequency + tight gaps
        watch = combo[
            (combo["n_refills"] >= MIN_REFILLS)
            & (combo["median_gap_h"].notna())
            & (combo["median_gap_h"] <= MAX_MEDIAN_GAP_H)
        ].copy()

    if watch.empty:
        st.info(
            "No (device, med) pairs look like obvious 'dribble refills' in this window "
            f"(≥{MIN_REFILLS} refills with median gap ≤{MAX_MEDIAN_GAP_H:.0f}h)."
        )
    else:
        watch = watch.sort_values(["n_refills", "median_gap_h"], ascending=[False, True])

        explain = (
            f"Flagged if **n_refills ≥ {MIN_REFILLS}** and **median gap ≤ {MAX_MEDIAN_GAP_H:.0f} hours**"
        )
        if has_real_qty:
            explain += f" and **avg_units ≤ {MAX_AVG_UNITS}**."

        st.caption(explain)

        show_cols = ["device"]
        if "med" in watch.columns:
            show_cols.append("med")
        show_cols += ["n_refills", "refills_per_day", "avg_units", "median_gap_h", "total_units"]

        try:
            st.dataframe(
                watch[show_cols].style.format(
                    {
                        "n_refills": "{:,.0f}",
                        "refills_per_day": "{:,.2f}",
                        "avg_units": "{:,.1f}",
                        "median_gap_h": "{:,.1f}",
                        "total_units": "{:,.0f}",
                    }
                ),
                use_container_width=True,
                height=320,
            )
        except Exception:
            st.dataframe(watch[show_cols], use_container_width=True, height=320)

    # ------------ Time-of-day view (refills only) ------------
    st.markdown("### When do refills happen? (time of day)")
    df["hour"] = df[ts_col].dt.hour
    by_hour = (
        df.groupby("hour", observed=True)
          .agg(
              refills=(typ_col, "size"),
              units=("qty_units", "sum"),
          )
          .reset_index()
          .sort_values("hour")
    )
    fig = px.bar(
        by_hour,
        x="hour",
        y="refills",
        title="Refills by hour of day",
    )
    st.plotly_chart(fig, use_container_width=True)




# ------------------------- PERSISTENCE (POSTGRES via Supabase) ----------------------


def _safe_int(x, lo=-2147483648, hi=2147483647):
    """Coerce numeric values into safe Postgres integer range or None."""
    if pd.isna(x):
        return None
    try:
        v = int(float(x))
        return max(lo, min(hi, v))
    except Exception:
        return None

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

    CREATE TABLE IF NOT EXISTS pyxis_activity_simple (
        ts                TIMESTAMP WITHOUT TIME ZONE NOT NULL,
        device            TEXT NOT NULL,
        drawer            TEXT NOT NULL,
        pocket            TEXT NOT NULL,
        med_id            TEXT NOT NULL,
        username          TEXT,
        min_qty           INTEGER,
        max_qty           INTEGER,
        is_min            BOOLEAN,
        is_max            BOOLEAN,
        is_standard_stock BOOLEAN,
        PRIMARY KEY (ts, device, drawer, pocket, med_id)
    );


    CREATE TABLE IF NOT EXISTS pyxis_pends (
        ts               TIMESTAMP WITHOUT TIME ZONE NOT NULL,
        device           TEXT NOT NULL,
        med_id           TEXT NOT NULL,
        drawer           TEXT NOT NULL DEFAULT '',
        pocket           TEXT NOT NULL DEFAULT '',
        qty              INTEGER,
        min_qty          INTEGER,
        max_qty          INTEGER,
        affected_element TEXT,
        dispensing_name  TEXT,
        area             TEXT,
        username         TEXT,
        userid           TEXT,
        CONSTRAINT pyxis_pends_pk PRIMARY KEY (ts, device, med_id, drawer, pocket)
    );

    -- Refill thresholds/capacity, MED-FIRST with optional overrides
    CREATE TABLE IF NOT EXISTS pyxis_thresholds (
        med_id   TEXT NOT NULL,
        device   TEXT NOT NULL DEFAULT '*',  -- '*' = applies to all devices
        drawer   TEXT NOT NULL DEFAULT '',   -- ''  = any drawer
        pocket   TEXT NOT NULL DEFAULT '',   -- ''  = any pocket
        min_qty  INTEGER,
        max_qty  INTEGER,
        CONSTRAINT pyxis_thresholds_pk PRIMARY KEY (med_id, device, drawer, pocket)
    );
    """
    with eng.begin() as con:
        con.execute(text(ddl))

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
                dt     = EXCLUDED.dt,
                device = EXCLUDED.device,
                "user" = EXCLUDED."user",
                "type" = EXCLUDED."type",
                "desc" = EXCLUDED."desc",
                qty    = EXCLUDED.qty,
                medid  = EXCLUDED.medid;
        """)

        CHUNK = 1000  # smaller batch
        total = 0
        with eng.begin() as con:
            con.execute(text("SET LOCAL statement_timeout = '120s'"))
            con.execute(text("SET LOCAL synchronous_commit = OFF"))
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

def upsert_activity_simple(eng, df_simple: pd.DataFrame) -> int:
    """
    UPSERT rows from build_simple_activity_view(df) into pyxis_activity_simple.
    PK: (ts, device, drawer, pocket, med_id)
    """
    if df_simple is None or df_simple.empty:
        return 0

    df = df_simple.copy()

    # Coerce/clean types expected by SQL
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    df = df.dropna(subset=["ts", "device", "drawer", "pocket", "med_id"])

    for c in ("min_qty", "max_qty"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").round().astype("Int64")

    # Shape rows for execute-many
    rows = df[[
        "ts","device","drawer","pocket","med_id",
        "username","min_qty","max_qty","is_min","is_max","is_standard_stock"
    ]].to_dict(orient="records")

    sql = text("""
        INSERT INTO pyxis_activity_simple
          (ts, device, drawer, pocket, med_id,
           username, min_qty, max_qty, is_min, is_max, is_standard_stock)
        VALUES
          (:ts, :device, :drawer, :pocket, :med_id,
           :username, :min_qty, :max_qty, :is_min, :is_max, :is_standard_stock)
        ON CONFLICT (ts, device, drawer, pocket, med_id)
        DO UPDATE SET
          username          = COALESCE(EXCLUDED.username, pyxis_activity_simple.username),
          min_qty           = COALESCE(EXCLUDED.min_qty, pyxis_activity_simple.min_qty),
          max_qty           = COALESCE(EXCLUDED.max_qty, pyxis_activity_simple.max_qty),
          is_min            = COALESCE(EXCLUDED.is_min, pyxis_activity_simple.is_min),
          is_max            = COALESCE(EXCLUDED.is_max, pyxis_activity_simple.is_max),
          is_standard_stock = COALESCE(EXCLUDED.is_standard_stock, pyxis_activity_simple.is_standard_stock);
    """)

    with eng.begin() as con:
        con.execute(text("SET LOCAL statement_timeout = '120s'"))
        con.execute(sql, rows)

    return len(rows)


def upsert_pyxis_pends(eng, df_pends: pd.DataFrame) -> int:
    if df_pends is None or df_pends.empty:
        return 0

    # make sure columns exist (safe to call always)
    ensure_pends_minmax_columns(eng)

    df = df_pends.copy()

    # Normalize NOT NULL fields
    for c in ("drawer", "pocket"):
        df[c] = df.get(c, "").fillna("").astype(str)

     # Ensure numeric
    for c in ("min_qty", "max_qty", "qty"):
        if c not in df.columns:
            df[c] = np.nan

    # Normalize numeric columns strictly to ints or None
    for c in ("qty", "min_qty", "max_qty"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df[c] = df[c].apply(_safe_int)


    rows = (
        df.rename(columns={"DispensingDeviceName": "dispensing_name", "Area": "area"})
          .to_dict(orient="records")
    )

    sql = text("""
        INSERT INTO pyxis_pends
        (ts, device, med_id, drawer, pocket, qty, min_qty, max_qty,
         affected_element, dispensing_name, area, username, userid)
        VALUES
        (:ts, :device, :med_id, :drawer, :pocket, :qty, :min_qty, :max_qty,
         :AffectedElement, :dispensing_name, :area, :username, :userid)
        ON CONFLICT (ts, device, med_id, drawer, pocket)
        DO UPDATE SET
          qty = EXCLUDED.qty,
          min_qty = COALESCE(EXCLUDED.min_qty, pyxis_pends.min_qty),
          max_qty = COALESCE(EXCLUDED.max_qty, pyxis_pends.max_qty),
          affected_element = EXCLUDED.affected_element,
          dispensing_name  = EXCLUDED.dispensing_name,
          area             = EXCLUDED.area,
          username         = EXCLUDED.username,
          userid           = EXCLUDED.userid;
    """)

    with eng.begin() as con:
        con.execute(text("SET LOCAL statement_timeout = '120s'"))
        con.execute(sql, rows)

    return len(rows)

def ensure_indexes(eng, timeout_sec: int = 15):
    """Create/repair indexes concurrently; uses AUTOCOMMIT because CONCURRENTLY forbids a txn."""
    stmts = [
        # events
        'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_events_dt     ON events (dt)',
        'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_events_device ON events (device)',
        'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_events_user   ON events ("user")',
        'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_events_type   ON events ("type")',
        # pends
        'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_pends_ts      ON pyxis_pends (ts)',
        'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_pends_dev_med ON pyxis_pends (device, med_id, ts DESC)',
        # thresholds
        'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_thresh_med    ON pyxis_thresholds (med_id)',
        'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_thresh_full   ON pyxis_thresholds (med_id, device, drawer, pocket)',
        # activity_simple
        'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_pas_ts       ON pyxis_activity_simple (ts)',
        'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_pas_device   ON pyxis_activity_simple (device)',
        'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_pas_med      ON pyxis_activity_simple (med_id)',
        'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_pas_slot     ON pyxis_activity_simple (device, drawer, pocket)',]

    with eng.connect().execution_options(isolation_level="AUTOCOMMIT") as con:
        try:
            con.execute(text(f"SET lock_timeout = '{timeout_sec}s'"))
            con.execute(text(f"SET statement_timeout = '{max(timeout_sec*2, 30)}s'"))
        except Exception:
            pass

        for s in stmts:
            try:
                con.execute(text(s))
            except Exception:
                pass
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

# --- SAVE (uploads only) ---
if uploads:
    # Save only new rows (huge speedup; avoids hammering indexes)
    if history.empty or "pk" not in history.columns:
        to_save = new_ev
    else:
        old_pks = set(history["pk"])
        to_save = new_ev[~new_ev["pk"].isin(old_pks)].copy()

    if to_save.empty:
        st.sidebar.info("No new rows to save.")
    else:
        ok, msg = save_history_sql(to_save, colmap, eng)
        (st.sidebar.success if ok else st.sidebar.error)(msg)
else:
    # Database-only mode; nothing to write
    pass

# =================== TIME RANGE FILTER ===================
# ============================ MERGE & SAVE ============================
if uploads:
    # new_ev was already created above from uploaded files
    frames = []
    if isinstance(history, pd.DataFrame) and not history.empty:
        frames.append(history)
    frames.append(new_ev)

    ev_all = (
        pd.concat(frames, ignore_index=True)
          .drop_duplicates(subset=["pk"])
          .sort_values(colmap["datetime"])
    )

    # Upload summary
    new_pks = set(new_ev["pk"])
    old_pks = set(history["pk"]) if (isinstance(history, pd.DataFrame) and "pk" in history.columns) else set()
    num_new = len(new_pks - old_pks)
    num_dup = len(new_pks & old_pks)
    earliest = pd.to_datetime(ev_all[colmap["datetime"]].min())
    latest   = pd.to_datetime(ev_all[colmap["datetime"]].max())

    with st.expander("📥 Upload summary", expanded=True):
        st.write(f"**Rows in this upload:** {len(new_ev):,}")
        st.write(f"- New rows vs DB: **{num_new:,}**")
        st.write(f"- Already existed (upserts): **{num_dup:,}**")
        st.write(f"**History time range:** {earliest:%Y-%m-%d %H:%M} → {latest:%Y-%m-%d %H:%M}")

    # --- SAVE (uploads only): write only the delta ---
    to_save = new_ev if not old_pks else new_ev[~new_ev["pk"].isin(old_pks)].copy()
    if to_save.empty:
        st.sidebar.info("No new rows to save.")
    else:
        ok, msg = save_history_sql(to_save, colmap, eng)
        (st.sidebar.success if ok else st.sidebar.error)(msg)

else:
    # Database-only mode; analyze what’s already in Postgres
    ev_all = history.copy()

# Guard: stop early if nothing to analyze
if not isinstance(ev_all, pd.DataFrame) or ev_all.empty:
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
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11, tab12, tab13 = st.tabs(
    [
        "📈 Overview",
        "🚶 Delivery Analytics",
        "🧑‍🔧 Tech Comparison",
        "📦 Devices",
        "⏱ Hourly",
        "🧪 Drill-down",
        "🔟 Weekly Top 10",
        "🚨 Outliers",
        "❓ Ask the data",
        "📥 Load/Unload",
        "💊 Refill Efficiency",
        "🧷 Pended Loads",
        "🧩 Slot Config (DB)",
    ]
)


# ---------- TAB 1: OVERVIEW ----------
with tab1:
    st.subheader("Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Events",  f"{len(ev_time):,}")
    c2.metric("Devices", f"{ev_time[colmap['device']].nunique():,}")
    c3.metric("Users",   f"{ev_time[colmap['user']].nunique():,}")
    c4.metric("Types",   f"{ev_time[colmap['type']].nunique():,}")

    # Weekly events bar
    week_df = weekly_summary(ev_time, colmap)
    if not week_df.empty:
        fig = px.bar(week_df, x="week", y="events", title="Weekly events")
        st.plotly_chart(fig, use_container_width=True)

    # Refill transaction comparison (weekly & monthly)
    ref_week = refill_trend(ev_time, colmap, freq="W-SUN")
    ref_month = refill_trend(ev_time, colmap, freq="M")

    with st.expander("Refill volume vs other weeks/months", expanded=False):
        if ref_week.empty and ref_month.empty:
            st.info("No transactions with 'refill' in the type field in this time range.")
        else:
            c_rw, c_rm = st.columns(2)

            # ---- Weekly comparison ----
            if not ref_week.empty:
                fig_rw = px.bar(
                    ref_week,
                    x="period",
                    y="refill_events",
                    title="Weekly refill count",
                )
                c_rw.plotly_chart(fig_rw, use_container_width=True)

                # Last week vs prior week
                if len(ref_week) >= 2:
                    last = ref_week.iloc[-1]
                    prev = ref_week.iloc[-2]
                    last_val = int(last["refill_events"])
                    prev_val = int(prev["refill_events"])
                    delta = last_val - prev_val
                    pct = (delta / prev_val * 100.0) if prev_val != 0 else None
                    delta_label = f"{delta:+,}" + (f" ({pct:+.1f}%)" if pct is not None else "")

                    c_rw.metric(
                        "Last week vs prior week",
                        f"{last_val:,} refills",
                        delta=delta_label,
                    )

            # ---- Monthly comparison ----
            if not ref_month.empty:
                fig_rm = px.bar(
                    ref_month,
                    x="period",
                    y="refill_events",
                    title="Monthly refill count",
                )
                c_rm.plotly_chart(fig_rm, use_container_width=True)

            # Raw weekly table (easy export)
            if not ref_week.empty:
                st.markdown("**Weekly refill table**")
                st.dataframe(ref_week, use_container_width=True)

# ---------- TAB 2: DELIVERY ANALYTICS ----------
with tab2:
    st.subheader("Delivery Analytics (per-tech sequences)")
    c1, c2 = st.columns(2)

    hg = data_f["__walk_gap_s"].dropna()
    if not hg.empty:
        fig = px.histogram(hg, nbins=40, title="Walking/Travel gaps (seconds)")
        c1.plotly_chart(fig, use_container_width=True)
        c1.caption(
            "X-axis = seconds between finishing a device and starting the next "
            "(≥ min & ≤ max; device changed)."
        )

    dw = data_f.loc[~data_f["__device_change"], "__gap_s"].dropna()
    if not dw.empty:
        fig2 = px.histogram(dw, nbins=40, title="Same-device dwell gaps (seconds)")
        c2.plotly_chart(fig2, use_container_width=True)

    st.markdown("**Tip:** Use the *Drill-down* tab to inspect rows behind any long gaps.")

# ---------- TAB 3: TECH COMPARISON ----------
with tab3:
    st.subheader("Tech comparison")
    st.dataframe(tech_stats_f, use_container_width=True)
    if not tech_stats.empty:
        # Bar of median walk, recomputed from data_f:
        med_walk = (
            data_f.groupby(colmap["user"], observed=True)["__walk_gap_s"]
            .median()
            .reset_index()
        )
        med_walk = med_walk.rename(columns={"__walk_gap_s": "median_walk_gap_s"}).fillna(0)
        fig = px.bar(
            med_walk,
            x=colmap["user"],
            y="median_walk_gap_s",
            title="Median walk gap by tech (s)",
        )
        st.plotly_chart(fig, use_container_width=True)

# ---------- TAB 4: DEVICES ----------
with tab4:
    st.subheader("Devices")
    st.dataframe(device_stats_f, use_container_width=True)
    if not device_stats_f.empty:
        fig = px.bar(
            device_stats_f.head(25),
            x=colmap["device"],
            y="events",
            title="Top devices by event volume",
        )
        st.plotly_chart(fig, use_container_width=True)

# ---------- TAB 5: HOURLY ----------
with tab5:
    st.subheader("Hourly cadence")
    st.dataframe(hourly_f, use_container_width=True)
    if not hourly_f.empty:
        fig = px.line(hourly_f, x="hour", y="events", markers=True, title="Events by hour")
        st.plotly_chart(fig, use_container_width=True)

# ---------- TAB 6: DRILL-DOWN ----------
with tab6:
    st.subheader("Drill-down (exportable)")
    show_cols = [
        colmap["datetime"],
        colmap["user"],
        colmap["device"],
        colmap["type"],
        "gap_hms",
        "walk_gap_hms",
        "dwell_hms",
        "visit_hms",
        "__gap_s",
        "__walk_gap_s",
        "__dwell_s",
        "visit_duration_s",
        "__device_change",
    ]
    for opt in ["desc", "qty", "medid"]:
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
        mime="text/csv",
    )

    st.markdown("### Per-visit summary (continuous time at a device)")
    ts, dev, usr = colmap["datetime"], colmap["device"], colmap["user"]
    visit_show = visit_f[[usr, dev, "start", "end", "visit_duration_s"]].copy()
    visit_show["visit_hms"] = visit_show["visit_duration_s"].map(fmt_hms)
    st.dataframe(
        visit_show[[usr, dev, "start", "end", "visit_hms", "visit_duration_s"]],
        use_container_width=True,
        height=360,
    )

# ---------- TAB 7: WEEKLY TOP 10 ----------
with tab7:
    st.subheader("Weekly Top 10 (signals)")
    digest = anomalies_top10(ev_all, data_f, colmap)
    if digest.empty:
        st.info("No notable anomalies in the last 7 days window.")
    else:
        st.dataframe(digest, use_container_width=True)

# ---------- TAB 8: OUTLIERS ----------
with tab8:
    st.subheader("Outliers")
    ow = outliers_iqr(
        data_f.dropna(subset=["__walk_gap_s"]),
        colmap["user"],
        "__walk_gap_s",
        "Walk gap",
    )
    od = outliers_iqr(
        data_f.dropna(subset=["__dwell_s"]),
        colmap["device"],
        "__dwell_s",
        "Dwell",
    )
    c1, c2 = st.columns(2)
    c1.write("By user (walk gap):")
    c1.dataframe(ow, use_container_width=True, height=320)
    c2.write("By device (dwell):")
    c2.dataframe(od, use_container_width=True, height=320)

# ---------- TAB 9: ASK THE DATA ----------
with tab9:
    st.subheader("Ask the data ❓")
    q = st.text_input(
        "Try e.g. 'top devices', 'longest dwell devices', 'median walk gap for Melissa', 'busiest hour'"
    )
    if q:
        ans, tbl = qa_answer(q, ev_time, data_f, colmap)
        st.write(ans)
        if not tbl.empty:
            st.dataframe(tbl, use_container_width=True)

# ---------- TAB 10: LOAD / UNLOAD ----------
with tab10:
    st.subheader("Load / Unload Insights")
    build_load_unload_section(ev_time, colmap)

# ---------- TAB 11: REFILL EFFICIENCY ----------
with tab11:
    st.subheader("Refill Efficiency")
    build_refill_efficiency_section(ev_time, data_f, colmap)

    st.markdown("---")
    st.subheader("Refill Audit – High Volume & Short Dwell")

    # ✅ use data_f, which has dwell_sec
    refill_audit = build_refill_audit(data_f, colmap)


    if refill_audit.empty:
        st.info("No refill dwell-time data available for this period.")
    else:
        # Threshold controls
        c1, c2 = st.columns(2)

        with c1:
            min_refills = st.slider(
                "Minimum refills in selected period",
                min_value=int(refill_audit["refill_count"].min()),
                max_value=int(refill_audit["refill_count"].max()),
                value=int(refill_audit["refill_count"].quantile(0.5)),  # default ~median
                step=1,
            )

        with c2:
            max_median_dwell = st.slider(
                "Max median dwell (seconds)",
                min_value=int(refill_audit["median_dwell_sec"].min()),
                max_value=int(refill_audit["median_dwell_sec"].max()),
                value=int(refill_audit["median_dwell_sec"].quantile(0.25)),  # fast quartile
                step=5,
            )

        # Filter likely “rushing” users
        candidates = refill_audit[
            (refill_audit["refill_count"] >= min_refills)
            & (refill_audit["median_dwell_sec"] <= max_median_dwell)
        ].copy()

        st.markdown("### 📋 Audit candidates")
        st.dataframe(
            candidates,
            use_container_width=True,
        )

        # Drill-down to “what they refilled”
        if not candidates.empty:
            st.markdown("### 🔍 Drill-down: refills to audit")

            user_col = colmap.get("user", "UserName")
            type_col = colmap.get("type", "TransactionType")
            dt_col = colmap.get("datetime", "TransactionDateTime")
            device_col = colmap.get("device", "Device")
            desc_col = colmap.get("desc", "MedDescription")
            qty_col = colmap.get("qty", "Quantity")
            medid_col = colmap.get("medid", "MedID")

            selected_user = st.selectbox(
                "Choose a colleague to review",
                options=candidates["user"].tolist(),
            )

            REFILL_TYPES = {
                "REFILL",
                "REFILL-LOAD",
                "REFILL LOAD",
                "LOAD",
                "REFILL RETURN",
            }

            df_user = ev_time.copy()

            # Filter to this user's refill transactions
            df_user = df_user[
                (df_user[user_col] == selected_user)
                & (df_user[type_col].isin(REFILL_TYPES))
            ].copy()

            # Sort newest first if datetime present
            if dt_col in df_user.columns:
                df_user = df_user.sort_values(dt_col, ascending=False)

            dwell_col = "dwell_sec" if "dwell_sec" in df_user.columns else None

            cols_to_show = [
                c for c in [
                    dt_col,
                    device_col,
                    medid_col,
                    desc_col,
                    qty_col,
                    dwell_col,
                ]
                if c is not None and c in df_user.columns
            ]

            st.dataframe(
                df_user[cols_to_show],
                use_container_width=True,
            )


# ---------- TAB 12: PENDED / THRESHOLD ACTIVITY ----------
# ---------- TAB 12: PENDED / THRESHOLD ACTIVITY ----------
with tab12:
    st.subheader("Pended / Threshold Activity (simple view)")

    up = st.file_uploader("Upload DeviceActivityLog CSV", type=["csv", "xlsx"])
    if not up:
        st.info("Upload your DeviceActivityLog and I’ll show the parsed view.")
    else:
        raw = (
            pd.read_excel(up)
            if up.name.lower().endswith(".xlsx")
            else pd.read_csv(up, low_memory=False)
        )
        raw = dedupe_columns(raw)

        view = build_simple_activity_view(raw)
        if view.empty:
            st.warning("No parsable rows found (check column names).")
        else:
            st.dataframe(view.head(300), use_container_width=True, height=480)

            c1, c2 = st.columns(2)
            with c1:
                if st.button(
                    "💾 Save parsed snapshot to Postgres", type="primary", key="save_simple"
                ):
                    try:
                        init_db(eng)  # ensures table exists
                        n = upsert_activity_simple(eng, view)
                        st.success(f"Saved {n:,} rows into pyxis_activity_simple (UPSERT).")
                    except Exception as e:
                        st.error(f"Save failed: {e}")

            with c2:
                st.download_button(
                    "Download parsed sheet (CSV)",
                    data=view.to_csv(index=False).encode("utf-8"),
                    file_name="device_activity_parsed.csv",
                    mime="text/csv",
                )

# ---------- TAB 13: SLOT CONFIG (DB VIEW) ----------
with tab13:
    st.subheader("Slot Configuration from Database")

    # 1) Current config (latest row per slot)
    st.markdown("### 📌 Current Min/Max by Slot")

    try:
        init_db(eng)  # safe to call again

        sql_current = """
        WITH latest AS (
          SELECT *,
                 ROW_NUMBER() OVER (
                   PARTITION BY device, drawer, pocket, med_id
                   ORDER BY ts DESC
                 ) AS rn
          FROM pyxis_activity_simple
        )
        SELECT
            ts,
            device,
            drawer,
            pocket,
            med_id,
            min_qty,
            max_qty,
            is_standard_stock,
            username
        FROM latest
        WHERE rn = 1
        ORDER BY device, drawer, pocket, med_id;
        """

        cfg = pd.read_sql(sql_current, eng)

        if cfg.empty:
            st.info("No rows found in pyxis_activity_simple yet. Upload some configs in the Pended Loads tab first.")
        else:
            c1, c2 = st.columns(2)
            with c1:
                dev_filter = st.multiselect(
                    "Filter by device (optional)",
                    sorted(cfg["device"].dropna().unique().tolist())
                )
            with c2:
                med_filter = st.multiselect(
                    "Filter by Med ID (optional)",
                    sorted(cfg["med_id"].dropna().unique().tolist())
                )

            view_cfg = cfg.copy()
            if dev_filter:
                view_cfg = view_cfg[view_cfg["device"].isin(dev_filter)]
            if med_filter:
                view_cfg = view_cfg[view_cfg["med_id"].isin(med_filter)]

            st.dataframe(
                view_cfg,
                use_container_width=True,
                height=420,
            )

            st.download_button(
                "Download current slot config as CSV",
                data=view_cfg.to_csv(index=False).encode("utf-8"),
                file_name="pyxis_slot_config_current.csv",
                mime="text/csv",
            )

    except Exception as e:
        st.error(f"Error loading slot config from database: {e}")

    # 2) Bad configs (quick QA list)
    st.markdown("### 🚨 Configs to Review")

    try:
        sql_bad = """
        WITH cur AS (
          SELECT *
          FROM (
            SELECT *,
                   ROW_NUMBER() OVER (
                     PARTITION BY device, drawer, pocket, med_id
                     ORDER BY ts DESC
                   ) AS rn
            FROM pyxis_activity_simple
          ) x
          WHERE rn = 1
        )
        SELECT
            ts,
            device,
            drawer,
            pocket,
            med_id,
            min_qty,
            max_qty,
            is_standard_stock,
            username
        FROM cur
        WHERE
            -- min > max
            (min_qty IS NOT NULL AND max_qty IS NOT NULL AND min_qty > max_qty)
         OR -- negative or obviously bad values
            (COALESCE(min_qty, 0) < 0 OR COALESCE(max_qty, 0) < 0)
         OR -- standard stock but missing caps
            (is_standard_stock IS TRUE AND (min_qty IS NULL OR max_qty IS NULL))
        ORDER BY ts DESC, device, drawer, pocket, med_id;
        """

        bad = pd.read_sql(sql_bad, eng)

        if bad.empty:
            st.success("No obviously bad configs found based on these rules. 🎉")
        else:
            st.dataframe(
                bad,
                use_container_width=True,
                height=320,
            )
            st.caption(
                "These rows have min > max, negative values, or standard stock with missing min/max."
            )

    except Exception as e:
        st.error(f"Error loading QA list: {e}")



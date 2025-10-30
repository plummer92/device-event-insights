# app.py
# Device Event Insights — Delivery Analytics & Drill-down
# - Fixes: duplicate column names, Timestamp slider errors, safe datetime parsing
# - Built for daily XLSX uploads with weekly summary and on-click drill-down

import io
from datetime import timedelta
from typing import Dict, Tuple, List

import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st

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

EVENT_LABELS = {
    "Refill": {"kind": "refill"},
    "Verify Inventory": {"kind": "verify"},
    "Unload": {"kind": "unload"},
    "Load": {"kind": "load"},
}

# idle gap (seconds) considered "walk/travel" to next device
DEFAULT_IDLE_MIN = 30

# ----------------------------- HELPERS --------------------------------


def dedupe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure column names are unique: if exact dupes exist, keep first."""
    # Fast path: drop exact duplicate names
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()].copy()

    # Also normalize whitespace/casing a bit (no rename collisions)
    # (Keep original names for now, only strip whitespace)
    df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]
    return df


def parse_datetime_series(s: pd.Series) -> pd.Series:
    """Parse to pandas datetime; coerce errors; ensure tz-naive for consistency."""
    out = pd.to_datetime(s, errors="coerce", utc=True)
    # Make tz-naive for consistent comparison/plotting
    if pd.api.types.is_datetime64tz_dtype(out):
        out = out.dt.tz_convert("UTC").dt.tz_localize(None)
    return out


def base_clean(df_raw: pd.DataFrame, colmap: Dict[str, str]) -> pd.DataFrame:
    """
    1) Dedup columns
    2) Parse datetime
    3) Standardize types
    """
    out = dedupe_columns(df_raw).copy()

    # Handle possible duplicate datetime label returning a DataFrame
    dtcol = colmap["datetime"]
    s = out[dtcol]
    if isinstance(s, pd.DataFrame):
        # if multiple columns with same *displayed* label, take first
        s = s.iloc[:, 0]
    out[dtcol] = parse_datetime_series(s)

    # optional numerics
    if colmap.get("qty") and colmap["qty"] in out.columns:
        out[colmap["qty"]] = pd.to_numeric(out[colmap["qty"]], errors="coerce")

    # sanitize strings
    for key in ["device", "user", "type", "desc", "medid"]:
        c = colmap.get(key)
        if c and c in out.columns:
            out[c] = out[c].astype("string").str.strip()

    # drop rows with no datetime
    out = out.dropna(subset=[dtcol]).copy()
    out = out.sort_values(dtcol).reset_index(drop=True)

    # Add engineered columns
    out["__date"] = out[dtcol].dt.date
    out["__hour"] = out[dtcol].dt.hour
    out["__dow"] = out[dtcol].dt.day_name()

    return out


def build_delivery_analytics(
    ev: pd.DataFrame, colmap: Dict[str, str], idle_min: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compute per-tech sequences, walk gaps, device and tech stats, and hourly view.
    Returns:
      data: events with next-event/gap annotations
      device_stats: per device volume and median dwell
      tech_stats: per tech totals and median walk gap
      run_stats: per (tech, run group) sequences
      hourly: hourly counts
    """
    ts = colmap["datetime"]
    dev = colmap["device"]
    user = colmap["user"]
    typ = colmap["type"]

    needed = [ts, dev, user, typ]
    if colmap.get("desc"):
        needed.append(colmap["desc"])
    if colmap.get("medid"):
        needed.append(colmap["medid"])
    if colmap.get("qty"):
        needed.append(colmap["qty"])

    data = ev[needed].sort_values([user, ts]).copy()

    # Next-event per tech
    data["__next_ts"] = data.groupby(user)[ts].shift(-1)
    data["__next_dev"] = data.groupby(user)[dev].shift(-1)

    # Gaps (seconds) and whether they changed device
    data["__gap_s"] = (data["__next_ts"] - data[ts]).dt.total_seconds()
    data["__device_change"] = (data[dev] != data["__next_dev"]) & data["__next_dev"].notna()

    # Walking/travel gap: large gap + device change
    data["__walk_gap_s"] = np.where(
        (data["__device_change"]) & (data["__gap_s"] >= idle_min),
        data["__gap_s"],
        np.nan,
    )

    # Device stats
    cnt = data.groupby(dev).size().rename("events")
    dwell = (
        data.loc[~data["__device_change"], "__gap_s"]
        .groupby(data[dev])
        .median()
        .rename("median_dwell_s")
    )
    device_stats = pd.concat([cnt, dwell], axis=1).fillna(0).sort_values("events", ascending=False).reset_index()

    # Tech stats
    tcnt = data.groupby(user).size().rename("events")
    twalk = data["__walk_gap_s"].groupby(data[user]).median().rename("median_walk_gap_s")
    tech_stats = pd.concat([tcnt, twalk], axis=1).fillna(0).sort_values("events", ascending=False).reset_index()

    # Run sequences (simple: consecutive actions until a long idle+device change)
    data["__is_break"] = (data["__walk_gap_s"] >= idle_min).fillna(False)
    data["__run_id"] = data.groupby(user)["__is_break"].cumsum()
    run_stats = (
        data.groupby([user, "__run_id"])
        .agg(
            start=(ts, "min"),
            end=(ts, "max"),
            n_events=(ts, "count"),
            n_devices=(dev, "nunique"),
            total_walk_s=("__walk_gap_s", lambda s: np.nansum(s.values)),
        )
        .reset_index()
    )
    run_stats["duration_s"] = (run_stats["end"] - run_stats["start"]).dt.total_seconds()

    # Hourly
    hourly = data.groupby(data[ts].dt.floor("H")).size().rename("events").reset_index().rename(columns={ts: "hour"})

    return data, device_stats, tech_stats, run_stats, hourly


def weekly_summary(ev: pd.DataFrame, colmap: Dict[str, str]) -> pd.DataFrame:
    ts = colmap["datetime"]
    dev = colmap["device"]
    user = colmap["user"]
    typ = colmap["type"]

    df = ev.copy()
    df["week"] = df[ts].dt.to_period("W-SUN").apply(lambda p: p.start_time.date())
    out = (
        df.groupby("week")
        .agg(
            events=(typ, "count"),
            devices=(dev, "nunique"),
            techs=(user, "nunique"),
        )
        .reset_index()
        .sort_values("week")
    )
    return out


def safe_unique(df: pd.DataFrame, col: str) -> List[str]:
    if col not in df.columns:
        return []
    return sorted([x for x in df[col].dropna().astype(str).unique()])


# ----------------------------- UI ------------------------------------


st.title("All Device Event Insights — Pro")

# 1) Upload
st.sidebar.header("1) Upload")
up = st.sidebar.file_uploader("Drag & drop daily XLSX/CSV", type=["xlsx", "csv"])
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
    opts = [c for c in df_raw.columns]
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

# 3) Filters (with fixed slider)
st.sidebar.header("3) Filters")

# Slider requires native datetime
_min = pd.to_datetime(ev[colmap["datetime"]].min())
_max = pd.to_datetime(ev[colmap["datetime"]].max())
min_ts = _min.to_pydatetime()
max_ts = _max.to_pydatetime()
if min_ts == max_ts:
    max_ts = min_ts + timedelta(minutes=1)

rng = st.sidebar.slider(
    "Time range",
    min_value=min_ts,
    max_value=max_ts,
    value=(min_ts, max_ts),
    format="YYYY-MM-DD HH:mm",
)

# other filters
pick_devices = st.sidebar.multiselect("Devices", safe_unique(ev, colmap["device"]))
pick_users = st.sidebar.multiselect("Users", safe_unique(ev, colmap["user"]))
pick_types = st.sidebar.multiselect("Transaction types", safe_unique(ev, colmap["type"]))

# Apply mask (convert picked datetimes back to pandas for comparison)
mask = (
    (ev[colmap["datetime"]] >= pd.to_datetime(rng[0])) &
    (ev[colmap["datetime"]] <= pd.to_datetime(rng[1]))
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

idle_min = st.sidebar.number_input("Walk gap threshold (seconds)", min_value=5, max_value=900, value=DEFAULT_IDLE_MIN, step=5)

# Compute analytics
data, device_stats, tech_stats, run_stats, hourly = build_delivery_analytics(ev, colmap, idle_min=idle_min)

# Tabs (make sure counts match)
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["📈 Overview", "🚶 Delivery Analytics", "🧑‍🔧 Tech Comparison",
     "📦 Devices", "⏱ Hourly", "🧪 Drill-down"]
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
    # Walking gaps histogram
    hg = data["__walk_gap_s"].dropna()
    if not hg.empty:
        fig = px.histogram(hg, nbins=40, title="Walking/Travel gaps (seconds)")
        c1.plotly_chart(fig, use_container_width=True)
        c1.caption("X-axis = seconds between finishing a device and starting the next device (≥ threshold & device changed).")

    # Dwell (same-device) gaps histogram
    dw = data.loc[~data["__device_change"], "__gap_s"].dropna()
    if not dw.empty:
        fig2 = px.histogram(dw, nbins=40, title="Same-device dwell gaps (seconds)")
        c2.plotly_chart(fig2, use_container_width=True)

    st.markdown("**Pro tip:** Click the *Drill-down* tab and filter by user/device to inspect the exact rows behind any long gap.")

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
    # Show essential columns + gap annotations
    show_cols = [
        colmap["datetime"], colmap["user"], colmap["device"], colmap["type"],
        "__gap_s", "__walk_gap_s", "__device_change"
    ]
    if colmap.get("desc") and colmap["desc"] in data.columns:
        show_cols.append(colmap["desc"])
    if colmap.get("qty") and colmap["qty"] in data.columns:
        show_cols.append(colmap["qty"])
    if colmap.get("medid") and colmap["medid"] in data.columns:
        show_cols.append(colmap["medid"])

    table = data[show_cols].copy()
    st.dataframe(table, use_container_width=True, height=500)

    # CSV export for validation
    csv = table.to_csv(index=False).encode("utf-8")
    st.download_button("Download current drill-down as CSV", data=csv, file_name="drilldown.csv", mime="text/csv")

# Footer
st.caption("Slider uses native Python datetimes to avoid Timestamp type errors; duplicate column names are auto-deduped.")

# app.py
# Device Event Insights — Pro
# - Dedupes duplicate columns on upload
# - Safe datetime parsing + native datetime slider
# - Delivery analytics (walk gaps, dwell)
# - Drill-down with H:MM:SS, per-visit durations, CSV exports

from datetime import timedelta
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import re
import os

LOCAL_HISTORY_FILE = "event_history.parquet"
if os.path.exists(LOCAL_HISTORY_FILE):
    history = pd.read_parquet(LOCAL_HISTORY_FILE)
else:
    history = pd.DataFrame()
    # Deduplicate against history
if not history.empty:
    combined = pd.concat([history, to_save], ignore_index=True)
    combined = combined.drop_duplicates(subset=["pk"])
else:
    combined = to_save.copy()

# Save back to local parquet file
combined.to_parquet(LOCAL_HISTORY_FILE, index=False)

# Update history in memory
history = combined

st.success(f"Added {len(to_save)} new rows. Total history: {len(history)} rows.")




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

DEFAULT_IDLE_MIN = 30  # seconds to consider as a walking/travel gap

# ----------------------------- HELPERS --------------------------------

def dedupe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure column names are unique by dropping exact dupes; trim whitespace."""
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()].copy()
    df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]
    return df

def parse_datetime_series(s: pd.Series) -> pd.Series:
    """Parse to pandas datetime (UTC, tz-naive for consistency)."""
    out = pd.to_datetime(s, errors="coerce", utc=True)
    if pd.api.types.is_datetime64tz_dtype(out):
        out = out.dt.tz_convert("UTC").dt.tz_localize(None)
    return out

def base_clean(df_raw: pd.DataFrame, colmap: Dict[str, str]) -> pd.DataFrame:
    out = dedupe_columns(df_raw).copy()

    # Handle possible duplicate-named datetime columns (DataFrame-like)
    dtcol = colmap["datetime"]
    s = out[dtcol]
    if isinstance(s, pd.DataFrame):
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

    out = out.dropna(subset=[dtcol]).copy()
    out = out.sort_values(dtcol).reset_index(drop=True)

    # engineered calendar columns
    out["__date"] = out[dtcol].dt.date
    out["__hour"] = out[dtcol].dt.hour
    out["__dow"] = out[dtcol].dt.day_name()
    return out

def build_delivery_analytics(
    ev: pd.DataFrame, colmap: Dict[str, str], idle_min: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      data: events with next-event/gap annotations
      device_stats: per device volume and median dwell
      tech_stats: per tech totals and median walk gap
      run_stats: per (tech, run group) sequences
      hourly: hourly counts
    """
    ts  = colmap["datetime"]
    dev = colmap["device"]
    usr = colmap["user"]
    typ = colmap["type"]

    needed = [ts, dev, usr, typ]
    if colmap.get("desc") and colmap["desc"] in ev.columns: needed.append(colmap["desc"])
    if colmap.get("medid") and colmap["medid"] in ev.columns: needed.append(colmap["medid"])
    if colmap.get("qty")   and colmap["qty"]   in ev.columns: needed.append(colmap["qty"])

    data = ev[needed].sort_values([usr, ts]).copy()

    # Next-event per tech
    data["__next_ts"]  = data.groupby(usr)[ts].shift(-1)
    data["__next_dev"] = data.groupby(usr)[dev].shift(-1)

    # Gaps (seconds) and device change
    data["__gap_s"] = (data["__next_ts"] - data[ts]).dt.total_seconds()
    data["__device_change"] = (data[dev] != data["__next_dev"]) & data["__next_dev"].notna()

    # Walking/travel gap: large gap + device change
    data["__walk_gap_s"] = np.where(
        (data["__device_change"]) & (data["__gap_s"] >= idle_min),
        data["__gap_s"],
        np.nan,
    )

    # Dwell (same-device) gap
    data["__dwell_s"] = np.where(~data["__device_change"], data["__gap_s"], np.nan)

    # Visit ID: increments whenever device changes for a given tech
    data["__visit_id"] = (
        data.groupby(usr)[dev]
            .apply(lambda x: (x != x.shift()).cumsum())
            .reset_index(level=0, drop=True)
    )

    # Visit summary: start/end/duration for each (tech, visit_id, device)
    visit = (
        data.groupby([usr, "__visit_id", dev])
            .agg(start=(ts, "min"), end=(ts, "max"))
            .reset_index()
    )
    visit["visit_duration_s"] = (visit["end"] - visit["start"]).dt.total_seconds()

    # Attach visit duration back onto each event row
    data = data.merge(
        visit[[usr, "__visit_id", "visit_duration_s"]],
        on=[usr, "__visit_id"],
        how="left"
    )

    # Device stats
    cnt = data.groupby(dev).size().rename("events")
    dwell = (
        data.loc[~data["__device_change"], "__gap_s"]
        .groupby(data[dev])
        .median()
        .rename("median_dwell_s")
    )
    device_stats = (
        pd.concat([cnt, dwell], axis=1)
        .fillna(0)
        .sort_values("events", ascending=False)
        .reset_index()
    )

    # Tech stats
    tcnt = data.groupby(usr).size().rename("events")
    twalk = data["__walk_gap_s"].groupby(data[usr]).median().rename("median_walk_gap_s")
    tech_stats = (
        pd.concat([tcnt, twalk], axis=1)
        .fillna(0)
        .sort_values("events", ascending=False)
        .reset_index()
    )

    # Run sequences (simple heuristic)
    data["__is_break"] = (data["__walk_gap_s"] >= idle_min).fillna(False)
    data["__run_id"] = data.groupby(usr)["__is_break"].cumsum()
    run_stats = (
        data.groupby([usr, "__run_id"])
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
    hourly = (
        data.groupby(data[ts].dt.floor("H"))
        .size().rename("events").reset_index()
        .rename(columns={ts: "hour"})
    )

    return data, device_stats, tech_stats, run_stats, hourly, visit

def weekly_summary(ev: pd.DataFrame, colmap: Dict[str, str]) -> pd.DataFrame:
    ts  = colmap["datetime"]
    dev = colmap["device"]
    usr = colmap["user"]
    typ = colmap["type"]

    df = ev.copy()
    df["week"] = df[ts].dt.to_period("W-SUN").apply(lambda p: p.start_time.date())
    out = (
        df.groupby("week")
        .agg(
            events=(typ, "count"),
            devices=(dev, "nunique"),
            techs=(usr, "nunique"),
        )
        .reset_index()
        .sort_values("week")
    )
    return out
def _dt_floor_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s).dt.floor("D")

def anomalies_top10(ev_all: pd.DataFrame, data: pd.DataFrame, colmap: Dict[str,str]) -> pd.DataFrame:
    """
    Build a Top 10 'look-into-this' list using last 7 days vs prior 7 days.
    Requires combined history (your app already builds 'combined' before filtering).
    """
    if ev_all.empty:
        return pd.DataFrame(columns=["rank","topic","detail","why","severity"])

    ts  = colmap["datetime"]; dev = colmap["device"]; usr = colmap["user"]; typ = colmap["type"]

    df = ev_all.copy()
    df[ts] = pd.to_datetime(df[ts], errors="coerce")
    df = df.dropna(subset=[ts])

    end = df[ts].max().floor("D") + pd.Timedelta(days=1)
    start = end - pd.Timedelta(days=14)
    cur0  = end - pd.Timedelta(days=7)

    recent = df[(df[ts] >= cur0) & (df[ts] < end)].copy()
    prior  = df[(df[ts] >= start) & (df[ts] < cur0)].copy()

    out = []

    # 1) Devices with biggest volume spikes
    if not recent.empty and not prior.empty:
        r_dev = recent.groupby(dev).size().rename("recent").reset_index()
        p_dev = prior.groupby(dev).size().rename("prior").reset_index()
        vol = r_dev.merge(p_dev, on=dev, how="left").fillna(0.0)
        vol["delta"] = vol["recent"] - vol["prior"].replace(0, np.nan)
        vol["pct"] = np.where(vol["prior"]>0, (vol["recent"]-vol["prior"])/vol["prior"], np.nan)
        vol = vol.sort_values(["pct","delta"], ascending=False).head(3)
        for _, r in vol.iterrows():
            out.append({
                "topic":"Device volume spike",
                "detail":f"{r[dev]} recent {int(r['recent'])} vs prior {int(r['prior'])}",
                "why":"Sudden workload shift; check staffing/stocking cadence",
                "severity": "high" if (r["pct"]>=0.5 and r["recent"]>=50) else "med"
            })

    # 2) Techs with largest median walk gap
    if not data.empty:
        twalk = data["__walk_gap_s"].groupby(data[colmap["user"]]).median().dropna()
        if not twalk.empty:
            top_walk = twalk.sort_values(ascending=False).head(3)
            for u, s in top_walk.items():
                out.append({
                    "topic":"High walking time",
                    "detail":f"{u} median walk gap {int(s)}s",
                    "why":"Inefficient routing or distant devices in their run",
                    "severity": "med" if s>=120 else "low"
                })

    # 3) Devices with largest median dwell (same-device gap)
    if not data.empty:
        dwell = (
            data.loc[~data["__device_change"], "__gap_s"]
            .groupby(data[colmap["device"]]).median().dropna()
        )
        if not dwell.empty:
            top_dwell = dwell.sort_values(ascending=False).head(3)
            for d, s in top_dwell.items():
                out.append({
                    "topic":"Long dwell at device",
                    "detail":f"{d} median dwell {int(s)}s",
                    "why":"Many refills per stop or slow transactions; check slot layout",
                    "severity":"med" if s>=60 else "low"
                })

    # 4) Hours with unusually high load (events per hour)
    if not recent.empty:
        rh = recent.groupby(recent[ts].dt.floor("H")).size()
        if not rh.empty:
            hr_top = rh.sort_values(ascending=False).head(2)
            for h, n in hr_top.items():
                out.append({
                    "topic":"Rush hour",
                    "detail":f"{h:%Y-%m-%d %H:%M} had {int(n)} events",
                    "why":"Consider JIT timing / more techs in this window",
                    "severity":"med" if n>=100 else "low"
                })

    # 5) Transaction-type surges (verify+refill vs loads/unloads/outdates)
    if not recent.empty and not prior.empty:
        r_t = recent.groupby(typ).size().rename("recent").reset_index()
        p_t = prior.groupby(typ).size().rename("prior").reset_index()
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

    # Rank and trim
    if not out:
        return pd.DataFrame(columns=["rank","topic","detail","why","severity"])
    df_out = pd.DataFrame(out)
    # simple severity ranking
    sev_rank = df_out["severity"].map({"high":3,"med":2,"low":1}).fillna(1)
    df_out = df_out.iloc[sev_rank.sort_values(ascending=False).index].reset_index(drop=True)
    df_out.insert(0, "rank", np.arange(1, len(df_out)+1))
    return df_out.head(10)

def qa_answer(question: str, ev: pd.DataFrame, data: pd.DataFrame, colmap: Dict[str,str]) -> Tuple[str, pd.DataFrame]:
    """
    Lightweight Q&A without an LLM. Pattern-matches common asks and returns
    a short answer + a table backing it up. Extend with new patterns anytime.
    """
    q = question.strip().lower()
    ts  = colmap["datetime"]; dev = colmap["device"]; usr = colmap["user"]; typ = colmap["type"]

    # 1) "top devices" (by events)
    if re.search(r"\b(top|most)\b.*\bdevices?\b", q):
        t = ev.groupby(dev).size().rename("events").reset_index().sort_values("events", ascending=False).head(10)
        ans = f"Top devices by event volume (showing {len(t)}):"
        return ans, t

    # 2) "longest dwell devices"
    if "longest" in q and "dwell" in q:
        if data.empty:
            return "No dwell data in current filter.", pd.DataFrame()
        t = (
            data.loc[~data["__device_change"], ["__gap_s"]]
            .groupby(data[dev]).median().rename(columns={"__gap_s":"median_dwell_s"})
            .sort_values("median_dwell_s", ascending=False).head(10).reset_index()
        )
        t["median_dwell_hms"] = t["median_dwell_s"].map(fmt_hms)
        return "Devices with longest median dwell:", t

    # 3) "median walk gap for <tech>"
    m = re.search(r"median .*walk.* for (.+)", q)
    if m:
        name = m.group(1).strip()
        sub = data[data[usr].str.lower()==name.lower()]
        if sub.empty:
            return f"No rows found for user '{name}'.", pd.DataFrame()
        val = np.nanmedian(sub["__walk_gap_s"].values)
        return f"Median walk gap for {name}: {fmt_hms(val)} ({int(val)}s)", pd.DataFrame()

    # 4) "events by hour" / "busiest hour"
    if "hour" in q:
        t = ev.groupby(ev[ts].dt.floor("H")).size().rename("events").reset_index().rename(columns={ts:"hour"})
        if t.empty:
            return "No hourly data in current filter.", pd.DataFrame()
        top = t.sort_values("events", ascending=False).head(1).iloc[0]
        ans = f"Busiest hour: {top['hour']:%Y-%m-%d %H:%M} with {int(top['events'])} events."
        return ans, t.sort_values("hour")

    # 5) "which tech has the highest/lowest median walk gap"
    if "which tech" in q and "median walk" in q:
        if data.empty:
            return "No walk-gap data in current filter.", pd.DataFrame()
        t = data.groupby(usr)["__walk_gap_s"].median().reset_index().rename(columns={"__walk_gap_s":"median_walk_s"})
        t = t.sort_values("median_walk_s", ascending=False)
        top = t.iloc[0]
        ans = f"Highest median walk gap: {top[usr]} at {fmt_hms(top['median_walk_s'])}."
        return ans, t

    # default fallback
    # return basic overview
    ans = "Try asks like: 'top devices', 'longest dwell devices', 'median walk gap for Alice', 'busiest hour'."
    tbl = ev[[ts, usr, dev, typ]].head(50)
    return ans, tbl


def safe_unique(df: pd.DataFrame, col: str) -> List[str]:
    if col not in df.columns:
        return []
    return sorted([x for x in df[col].dropna().astype(str).unique()])

def fmt_hms(x) -> str:
    if pd.isna(x):
        return ""
    x = int(round(float(x)))
    h, r = divmod(x, 3600)
    m, s = divmod(r, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"

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

# 3) Filters (native datetime slider)
st.sidebar.header("3) Filters")

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

pick_devices = st.sidebar.multiselect("Devices", safe_unique(ev, colmap["device"]))
pick_users   = st.sidebar.multiselect("Users", safe_unique(ev, colmap["user"]))
pick_types   = st.sidebar.multiselect("Transaction types", safe_unique(ev, colmap["type"]))
idle_min = st.sidebar.number_input("Walk gap threshold (seconds)", min_value=5, max_value=900, value=DEFAULT_IDLE_MIN, step=5)

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

# Compute analytics
data, device_stats, tech_stats, run_stats, hourly, visit = build_delivery_analytics(ev, colmap, idle_min=idle_min)

# Pre-format H:MM:SS fields for drill-down
data["gap_hms"]      = data["__gap_s"].map(fmt_hms)
data["walk_gap_hms"] = data["__walk_gap_s"].map(fmt_hms)
data["dwell_hms"]    = data["__dwell_s"].map(fmt_hms)
data["visit_hms"]    = data["visit_duration_s"].map(fmt_hms)

# Tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(
    ["📈 Overview", "🚶 Delivery Analytics", "🧑‍🔧 Tech Comparison",
     "📦 Devices", "⏱ Hourly", "🧪 Drill-down", "🔟 Weekly Top 10", "❓ Ask the data"]
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
        c1.caption("X-axis = seconds between finishing a device and starting the next (≥ threshold & device changed).")

    # Dwell (same-device) gaps histogram
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

    # Columns to show in the table, with both raw and formatted times
    show_cols = [
        colmap["datetime"], colmap["user"], colmap["device"], colmap["type"],
        # human-friendly time fields:
        "gap_hms", "walk_gap_hms", "dwell_hms", "visit_hms",
        # raw seconds (for sorting/export/audits):
        "__gap_s", "__walk_gap_s", "__dwell_s", "visit_duration_s",
        "__device_change",
    ]
    if colmap.get("desc") and colmap["desc"] in data.columns:
        show_cols.insert(4, colmap["desc"])  # put description right after type
    if colmap.get("qty") and colmap["qty"] in data.columns:
        show_cols.insert(5, colmap["qty"])
    if colmap.get("medid") and colmap["medid"] in data.columns:
        show_cols.insert(6, colmap["medid"])

    show_cols = [c for c in show_cols if c in data.columns]
    table = data[show_cols].copy()

    st.dataframe(table, use_container_width=True, height=520)
    csv = table.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download current drill-down as CSV",
        data=csv,
        file_name="drilldown.csv",
        mime="text/csv"
    )

    st.markdown("### Per-visit summary (time per continuous stop at a device)")
    ts  = colmap["datetime"]; dev = colmap["device"]; usr = colmap["user"]
    visit_show = visit[[usr, dev, "start", "end", "visit_duration_s"]].copy()
    visit_show["visit_hms"] = visit_show["visit_duration_s"].map(fmt_hms)

    st.dataframe(
        visit_show[[usr, dev, "start", "end", "visit_hms", "visit_duration_s"]],
        use_container_width=True,
        height=360
    )
    csv2 = visit_show.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download visit summary as CSV",
        data=csv2,
        file_name="visit_summary.csv",
        mime="text/csv"
    )

st.caption("Slider uses native Python datetimes; duplicate column names are auto-deduped. Drill-down shows H:MM:SS plus raw seconds for auditability.")
with tab7:
    st.subheader("Weekly Top 10 (last 7d vs prior 7d)")
    # 'combined' already exists earlier (history + new). Use it for anomalies.
    try:
        top10 = anomalies_top10(combined, data, colmap)
        if top10.empty:
            st.info("Not enough history yet to compute a weekly digest.")
        else:
            st.dataframe(top10, use_container_width=True, height=480)
            dl = top10.to_csv(index=False).encode("utf-8")
            st.download_button("Download Top 10 as CSV", dl, "weekly_top10.csv", "text/csv")
    except Exception as e:
        st.error(f"Could not build Top 10: {e}")

with tab8:
    st.subheader("Ask the data")
    q = st.text_input("Type a question (e.g., 'top devices', 'median walk gap for Melissa')")
    if q:
        ans, tbl = qa_answer(q, ev, data, colmap)
        st.write(ans)
        if not tbl.empty:
            st.dataframe(tbl, use_container_width=True, height=420)
            st.download_button(
                "Download answer table as CSV",
                tbl.to_csv(index=False).encode("utf-8"),
                "qa_answer.csv",
                "text/csv"
            )
    else:
        st.info("Examples: 'top devices', 'longest dwell devices', 'median walk gap for <name>', 'busiest hour'.")


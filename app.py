import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import datetime as dt
from io import BytesIO
from pathlib import Path

# Optional Google Sheets (requires secrets)
USE_SHEETS = True

st.set_page_config(page_title="Device Event Insights — v3.1", layout="wide")
st.title("📊 All Device Event Insights — v3.1")
st.caption("Events + Staffing + Carousel (JIT) Drops • Role-group mapping • SLA thresholds • Google Sheets history")

# ------------------ Helpers ------------------
def detect_cols(df):
    colmap = {"datetime": None, "device": None, "type": None, "element": None}
    for c in df.columns:
        lc = str(c).lower()
        if colmap["datetime"] is None and any(k in lc for k in ["date", "time", "timestamp"]):
            colmap["datetime"] = c
        if colmap["device"] is None and any(k in lc for k in ["device","host","asset","endpoint","cabinet","station"]):
            colmap["device"] = c
        if colmap["type"] is None and any(k in lc for k in ["transactiontype","trans type","type","event"]):
            colmap["type"] = c
        if colmap["element"] is None and any(k in lc for k in ["element","med","item","ndc","drug"]):
            colmap["element"] = c
    if colmap["datetime"] is None:
        df["_RowTime"] = pd.to_datetime(pd.RangeIndex(len(df)), unit="s", origin="unix")
        colmap["datetime"] = "_RowTime"
    if colmap["device"] is None:
        df["Device"] = "Unknown"
        colmap["device"] = "Device"
    if colmap["type"] is None:
        df["TransactionType"] = "Unknown"
        colmap["type"] = "TransactionType"
    return colmap

@st.cache_data(show_spinner=False)
def load_events(xlsx_file):
    df = pd.read_excel(xlsx_file, sheet_name=0, dtype=str)
    df.columns = [str(c).strip() for c in df.columns]
    colmap = detect_cols(df)
    df[colmap["datetime"]] = pd.to_datetime(df[colmap["datetime"]], errors="coerce")
    df["__date"] = df[colmap["datetime"]].dt.date
    df["__hour"] = df[colmap["datetime"]].dt.hour
    df["__dow"] = df[colmap["datetime"]].dt.day_name()
    return df, colmap

# ------------------ Upload Events ------------------
events_file = st.file_uploader("Upload daily **Events** Excel (.xlsx)", type=["xlsx"], key="events")
if not events_file:
    st.info("Upload your daily events report to begin.")
    st.stop()

events_df, colmap = load_events(events_file)

with st.sidebar:
    st.header("Filters")
    valid_times = events_df[colmap["datetime"]].dropna()
    if len(valid_times) == 0:
        min_dt = dt.datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        max_dt = dt.datetime.now().replace(hour=23, minute=59, second=0, microsecond=0)
    else:
        min_dt, max_dt = valid_times.min().to_pydatetime(), valid_times.max().to_pydatetime()

    time_rng = st.slider("Date/time range",
        min_value=min_dt, max_value=max_dt,
        value=(min_dt, max_dt), format="YYYY-MM-DD HH:mm")
    mask_time = events_df[colmap["datetime"]].between(pd.to_datetime(time_rng[0]), pd.to_datetime(time_rng[1]))

    devs = sorted(events_df[colmap["device"]].fillna("Unknown").unique())
    dev_sel = st.multiselect("Device(s)", devs, default=devs)
    mask_dev = events_df[colmap["device"]].isin(dev_sel)

    types = sorted(events_df[colmap["type"]].fillna("Unknown").unique())
    type_sel = st.multiselect("Transaction Type(s)", types, default=types)
    mask_type = events_df[colmap["type"]].isin(type_sel)

ev = events_df[mask_time & mask_dev & mask_type].copy()

# ------------------ Upload JIT Drops ------------------
st.subheader("📦 Carousel (JIT) Drops (optional)")
jit_file = st.file_uploader("Upload JIT Drops Excel (.xlsx)", type=["xlsx"], key="jit")
jit_df = pd.DataFrame()
jit_map = {"time": None, "device": None, "qty": None, "location": None}

if jit_file:
    raw = pd.read_excel(jit_file, sheet_name=0)
    raw.columns = [str(c).strip() for c in raw.columns]
    for c in raw.columns:
        lc = c.lower()
        if jit_map["time"] is None and any(k in lc for k in ["time","drop","datetime"]): jit_map["time"] = c
        if jit_map["device"] is None and any(k in lc for k in ["device","cabinet","pyxis","station"]): jit_map["device"] = c
        if jit_map["location"] is None and any(k in lc for k in ["unit","location","area"]): jit_map["location"] = c
        if jit_map["qty"] is None and any(k in lc for k in ["qty","quantity","count"]): jit_map["qty"] = c
    if not jit_map["time"]:
        st.error("Couldn't detect a JIT 'time' column. Please include a column like 'Drop Time'.")
    else:
        jit_df = raw.copy()
        jit_df["__jit_time"] = pd.to_datetime(jit_df[jit_map["time"]], errors="coerce")
        jit_df["__jit_hour"] = jit_df["__jit_time"].dt.hour
        if not jit_map["qty"]:
            jit_df["__qty"] = 1.0
        else:
            jit_df["__qty"] = pd.to_numeric(jit_df[jit_map["qty"]], errors="coerce").fillna(0.0)

# ------------------ Upload Staffing ------------------
st.subheader("👥 Staffing (optional)")
st.write("Upload staffing CSV with columns: `role, person, start, end`. Extra columns like `location, device_group` are welcome.")
staff_file = st.file_uploader("Upload staffing CSV", type=["csv"], key="staff")
staff_df = pd.DataFrame()
if staff_file:
    staff_df = pd.read_csv(staff_file)
    for c in ["start","end"]:
        if c in staff_df.columns:
            staff_df[c] = pd.to_datetime(staff_df[c], errors="coerce")
    if "device_group" not in staff_df.columns:
        staff_df["device_group"] = "Central"

# ------------------ Device → Role Group Map ------------------
st.subheader("🔎 Device/Location → Role Group Map (Runner / IV / Central)")
st.write("Upload a mapping CSV with columns: `device_id` (or `friendly_name`/`location`), `role_group` (e.g., Runner/IV/Central).")
map_file = st.file_uploader("Upload device_group_map.csv (optional)", type=["csv"], key="map")
map_df = pd.DataFrame()
if map_file:
    map_df = pd.read_csv(map_file)
    map_df.columns = [c.strip().lower() for c in map_df.columns]

# ------------------ KPIs ------------------
c1,c2,c3,c4 = st.columns(4)
c1.metric("Events (filtered)", int(len(ev)))
c2.metric("Devices (filtered)", ev[colmap["device"]].nunique())
if not ev.empty:
    c3.metric("Date range", f"{ev[colmap['datetime']].min():%Y-%m-%d} → {ev[colmap['datetime']].max():%Y-%m-%d}")
c4.metric("Types", ev[colmap["type"]].nunique())

# ------------------ Tabs ------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Events Over Time", "🖥️ Device Utilization", "🧾 Types",
    "📅 Day/Hour Heatmap", "📦 JIT Drops", "🔗 Overlay"
])

with tab1:
    st.subheader("Events by Day / Hour")
    by_day = ev.groupby("__date").size().reset_index(name="count")
    if not by_day.empty:
        st.plotly_chart(px.line(by_day, x="__date", y="count", markers=True), use_container_width=True)
    by_hour = ev.groupby("__hour").size().reset_index(name="count")
    if not by_hour.empty:
        st.plotly_chart(px.bar(by_hour, x="__hour", y="count"), use_container_width=True)

with tab2:
    st.subheader("Top Devices")
    dev_counts = ev.groupby(colmap["device"]).size().reset_index(name="count").sort_values("count", ascending=False).head(40)
    st.dataframe(dev_counts, use_container_width=True)
    if not dev_counts.empty:
        st.plotly_chart(px.bar(dev_counts, x="count", y=colmap["device"], orientation="h"), use_container_width=True)

with tab3:
    st.subheader("Transaction Types")
    typ_counts = ev.groupby(colmap["type"]).size().reset_index(name="count").sort_values("count", ascending=False)
    st.dataframe(typ_counts, use_container_width=True)
    if not typ_counts.empty:
        st.plotly_chart(px.bar(typ_counts, x="count", y=colmap["type"], orientation="h"), use_container_width=True)

with tab4:
    st.subheader("Heatmap: Day of Week × Hour")
    heat = ev.pivot_table(index="__dow", columns="__hour", values=colmap["device"], aggfunc="count", fill_value=0)
    if not heat.empty:
        heat = heat.reindex(["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])
        st.plotly_chart(px.imshow(heat, aspect="auto", text_auto=True), use_container_width=True)
    st.dataframe(heat, use_container_width=True)

with tab5:
    st.subheader("JIT Drops Summary")
    if jit_file is None or jit_df.empty:
        st.info("Upload a JIT Drops Excel to view summary.")
    else:
        jit_hour = jit_df.groupby("__jit_hour")["__qty"].sum().reindex(range(0,24)).fillna(0).reset_index()
        jit_hour.columns = ["hour","drops"]
        st.dataframe(jit_hour, use_container_width=True)
        st.plotly_chart(px.bar(jit_hour, x="hour", y="drops", title="JIT Drops by Hour"), use_container_width=True)

with tab6:
    st.subheader("Overlay: Events + JIT + Staffing (Hourly) with Role Groups & SLAs")

    # Events hourly
    ev_hour = ev.groupby("__hour").size().reindex(range(0,24), fill_value=0).reset_index()
    ev_hour.columns = ["hour","events"]

    # JIT hourly
    jit_hour = None
    if not jit_df.empty:
        jit_hour = jit_df.groupby("__jit_hour")["__qty"].sum().reindex(range(0,24)).fillna(0).reset_index()
        jit_hour.columns = ["hour","jit_drops"]

    # Staffing hourly
    staff_hour_total = None
    if not staff_df.empty and {"start","end"}.issubset(staff_df.columns):
        staff_hour_total = pd.DataFrame({"hour": range(0,24), "staff_total": 0.0})
        for _, r in staff_df.iterrows():
            if pd.isna(r.get("start")) or pd.isna(r.get("end")): continue
            s = pd.to_datetime(r["start"]); e = pd.to_datetime(r["end"])
            s_h, e_h = int(s.hour), int(e.hour)
            if e.date() > s.date() or e_h < s_h:
                hours = list(range(s_h,24)) + list(range(0,e_h+1))
            else:
                hours = list(range(s_h, e_h+1))
            staff_hour_total.loc[staff_hour_total["hour"].isin(hours), "staff_total"] += 1.0

    merged = ev_hour.copy()
    if jit_hour is not None: merged = merged.merge(jit_hour, on="hour", how="left")
    if staff_hour_total is not None: merged = merged.merge(staff_hour_total, on="hour", how="left")
    merged = merged.fillna(0)

    st.plotly_chart(px.line(merged, x="hour", y=[c for c in ["events","jit_drops","staff_total"] if c in merged.columns], markers=True), use_container_width=True)
    st.dataframe(merged, use_container_width=True)


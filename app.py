import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import BytesIO
import datetime as dt
from pathlib import Path

st.set_page_config(page_title="Device Event Insights — Pro", layout="wide")

HISTORY_DIR = Path("./history")
HISTORY_DIR.mkdir(exist_ok=True)
HISTORY_FILE = HISTORY_DIR / "events_history.csv.gz"

st.title("📊 All Device Event Insights — Pro")
st.caption("Upload your daily report. Build historical trends. Overlay staffing vs. workload. Get recommendations.")

@st.cache_data(show_spinner=False)
def load_excel(file):
    df = pd.read_excel(file, sheet_name=0, dtype=str)
    df.columns = [str(c).strip() for c in df.columns]

    colmap = {"datetime": None, "device": None, "type": None, "element": None, "fractional": None}
    for c in df.columns:
        lc = c.lower()
        if colmap["datetime"] is None and ("date" in lc or "time" in lc or "timestamp" in lc):
            colmap["datetime"] = c
        if colmap["device"] is None and ("device" in lc or "host" in lc or "asset" in lc or "endpoint" in lc):
            colmap["device"] = c
        if colmap["type"] is None and ("transactiontype" in lc or "type" in lc or "event" in lc):
            colmap["type"] = c
        if colmap["element"] is None and ("element" in lc):
            colmap["element"] = c
        if colmap["fractional"] is None and ("fraction" in lc):
            colmap["fractional"] = c

    if colmap["datetime"] is not None:
        df[colmap["datetime"]] = pd.to_datetime(df[colmap["datetime"]], errors="coerce")
    else:
        st.warning("Couldn't detect a datetime column. Please ensure a column like 'TransactionDateTime' exists.")
        df["_RowTime"] = pd.to_datetime(pd.RangeIndex(len(df)), unit="s", origin="unix")
        colmap["datetime"] = "_RowTime"

    # Derived fields
    dtc = colmap["datetime"]
    df["__date"] = df[dtc].dt.date
    df["__hour"] = df[dtc].dt.hour
    df["__dow"] = df[dtc].dt.day_name()

    # Ensure canonical columns exist
    if colmap["device"] is None:
        df["Device"] = "Unknown"
        colmap["device"] = "Device"
    if colmap["type"] is None:
        df["TransactionType"] = "Unknown"
        colmap["type"] = "TransactionType"
    return df, colmap

def kpi_card(label, value, delta=None):
    vstr = f"{value:,}" if isinstance(value, (int, float)) else value
    st.metric(label, vstr, delta=delta)

def recommend(df, colmap):
    out = []
    hour_counts = df.groupby("__hour").size().rename("count").reset_index()
    if not hour_counts.empty:
        top_hours = hour_counts.sort_values("count", ascending=False).head(3)
        low_hours = hour_counts.sort_values("count", ascending=True).head(3)
        out.append(
            "**Peak hours (top 3):** "
            + ", ".join(f"{int(h)}:00 ({int(c)} evts)" for h, c in zip(top_hours['__hour'], top_hours['count']))
        )
        out.append(
            "**Quiet hours (bottom 3):** "
            + ", ".join(f"{int(h)}:00 ({int(c)} evts)" for h, c in zip(low_hours['__hour'], low_hours['count']))
        )

    dev_col = colmap["device"]
    dev_counts = df.groupby(dev_col).size().rename("count").reset_index()
    if not dev_counts.empty:
        median_load = dev_counts["count"].median()
        high = dev_counts.sort_values("count", ascending=False).head(5)
        under = dev_counts[dev_counts["count"] < 0.6 * median_load].sort_values("count").head(5)
        out.append("**Heaviest devices (top 5):** " + ", ".join(f"{d} ({int(c)})" for d, c in zip(high[dev_col], high["count"])))
        if not under.empty:
            out.append("**Potentially underutilized (<60% of median volume):** " + ", ".join(f"{d} ({int(c)})" for d, c in zip(under[dev_col], under["count"])))

    dow_counts = df.groupby("__dow").size().rename("count").reset_index()
    if not dow_counts.empty:
        mean = dow_counts["count"].mean()
        std = dow_counts["count"].std(ddof=0) or 1.0
        dow_counts["z"] = (dow_counts["count"] - mean) / std
        spikes = dow_counts[dow_counts["z"] > 1.0].sort_values("z", ascending=False)
        if not spikes.empty:
            out.append("**Day-of-week spikes (>1σ):** " + ", ".join(f"{d} (z={z:.1f})" for d, z in zip(spikes['__dow'], spikes['z'])))

    typ_col = colmap["type"]
    type_counts = df.groupby(typ_col).size().rename("count").reset_index().sort_values("count", ascending=False)
    if not type_counts.empty:
        top_types = ", ".join(f"{t} ({int(c)})" for t, c in zip(type_counts[typ_col].head(5), type_counts["count"].head(5)))
        out.append("**Top transaction types:** " + top_types)

    if not out:
        out.append("No specific recommendations yet; upload a larger time window to identify patterns.")
    return out

uploaded = st.file_uploader("Upload today's Excel file (.xlsx)", type=["xlsx"], key="upload_today")

if uploaded is None:
    st.info("Upload today's report to get started. Then you can append to history for trending and day-over-day insights.")
    st.stop()

df, colmap = load_excel(uploaded)

# Sidebar filters
with st.sidebar:
    st.header("Filters")
    # Time range slider (convert pandas -> python datetimes)
    valid_times = df[colmap["datetime"]].dropna()
    if valid_times.empty:
        min_dt = dt.datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        max_dt = dt.datetime.now().replace(hour=23, minute=59, second=0, microsecond=0)
    else:
        min_dt = valid_times.min().to_pydatetime()
        max_dt = valid_times.max().to_pydatetime()

    rng = st.slider(
        "Date/time range",
        min_value=min_dt,
        max_value=max_dt,
        value=(min_dt, max_dt),
        format="YYYY-MM-DD HH:mm",
    )
    mask_time = df[colmap["datetime"]].between(pd.to_datetime(rng[0]), pd.to_datetime(rng[1]))

    # Device filter
    devs = sorted(df[colmap["device"]].fillna("Unknown").unique().tolist())
    dev_sel = st.multiselect("Device(s)", options=devs, default=devs)
    mask_dev = df[colmap["device"]].isin(dev_sel)

    # Transaction type filter
    typs = sorted(df[colmap["type"]].fillna("Unknown").unique().tolist())
    typ_sel = st.multiselect("Transaction Type(s)", options=typs, default=typs)
    mask_typ = df[colmap["type"]].isin(typ_sel)

df_f = df[mask_time & mask_dev & mask_typ].copy()

# ===== Historical storage controls =====
st.subheader("🗂️ History & Comparisons")
left_hist, right_hist = st.columns([2,1])
with left_hist:
    st.write("Append today's filtered dataset to local history to build trends and day-over-day comparisons. History is stored at **./history/events_history.csv.gz** next to the app.")
with right_hist:
    append_now = st.button("➕ Append to History")
    clear_hist = st.button("🧹 Clear History")

history_dir = Path("./history")
history_dir.mkdir(exist_ok=True)
history_file = history_dir / "events_history.csv.gz"

if clear_hist and history_file.exists():
    history_file.unlink()
    st.success("History cleared.")

if append_now:
    upload_day = pd.to_datetime(df_f[colmap["datetime"]].min()).date() if not df_f.empty else dt.date.today()
    to_save = df_f.copy()
    to_save["__upload_day"] = pd.to_datetime(upload_day)
    header = not history_file.exists()
    to_save.to_csv(history_file, mode="at" if history_file.exists() else "wt", index=False, compression="gzip", header=header)
    st.success(f"Appended {len(to_save):,} rows to history for day {upload_day}.")

# Load history if exists
history_df = None
if history_file.exists():
    try:
        history_df = pd.read_csv(history_file, compression="gzip", low_memory=False, parse_dates=[colmap["datetime"], "__upload_day"])
    except Exception:
        history_df = pd.read_csv(history_file, compression="gzip", low_memory=False)
        if colmap["datetime"] in history_df.columns:
            history_df[colmap["datetime"]] = pd.to_datetime(history_df[colmap["datetime"]], errors="coerce")
        if "__upload_day" in history_df.columns:
            history_df["__upload_day"] = pd.to_datetime(history_df["__upload_day"], errors="coerce")

# KPIs
left, mid, right, right2 = st.columns(4)
with left:
    kpi_card("Events (filtered)", int(len(df_f)))
with mid:
    kpi_card("Devices (filtered)", df_f[colmap["device"]].nunique())
with right:
    kpi_card("Date range", f"{df_f[colmap['datetime']].min():%Y-%m-%d} → {df_f[colmap['datetime']].max():%Y-%m-%d}")
with right2:
    kpi_card("Transaction Types", df_f[colmap["type"]].nunique())

# Day-over-day comparison
if history_df is not None and not history_df.empty:
    st.caption("Day-over-day snapshot uses **__upload_day** to compare today vs. yesterday in history.")
    today_tag = pd.to_datetime(df_f[colmap["datetime"]].min()).date() if not df_f.empty else dt.date.today()
    yesterday_tag = today_tag - dt.timedelta(days=1)
    today_rows = history_df[history_df["__upload_day"].dt.date == today_tag]
    yest_rows = history_df[history_df["__upload_day"].dt.date == yesterday_tag]
    if not today_rows.empty and not yest_rows.empty:
        delta_events = len(today_rows) - len(yest_rows)
        st.info(f"**Day-over-day change:** {len(today_rows):,} events today vs {len(yest_rows):,} yesterday → Δ {delta_events:+,}")
    else:
        st.info("Append more days to history to enable day-over-day comparisons.")

st.divider()

# ===== Staffing overlay =====
st.subheader("👥 Staffing Overlay (optional)")
st.write("Upload a staffing CSV (columns: `role, person, start, end`). Times should be parsable datetimes, e.g., `2025-10-28 07:00`.")

staff_file = st.file_uploader("Upload staffing CSV", type=["csv"], key="staff")
staff_df = None
if staff_file is not None:
    staff_df = pd.read_csv(staff_file)
    for c in ["start", "end"]:
        if c in staff_df.columns:
            staff_df[c] = pd.to_datetime(staff_df[c], errors="coerce")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Volume Over Time", "🖥️ Device Utilization", "🧾 Transaction Types", "📅 Day/Hour Heatmap", "📊 Staffing vs. Workload"])

with tab1:
    st.subheader("Events by Day (filtered)")
    by_day = df_f.groupby("__date").size().reset_index(name="count")
    if not by_day.empty:
        fig = px.line(by_day, x="__date", y="count", markers=True)
        st.plotly_chart(fig, use_container_width=True)
    st.subheader("Events by Hour (filtered)")
    by_hour = df_f.groupby("__hour").size().reset_index(name="count")
    if not by_hour.empty:
        fig = px.bar(by_hour, x="__hour", y="count")
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Top Devices by Volume (filtered)")
    dev_counts = df_f.groupby(colmap["device"]).size().reset_index(name="count").sort_values("count", ascending=False).head(25)
    if not dev_counts.empty:
        fig = px.bar(dev_counts, x="count", y=colmap["device"], orientation="h")
        st.plotly_chart(fig, use_container_width=True)
    st.dataframe(dev_counts, use_container_width=True)

with tab3:
    st.subheader("Transaction Type Distribution (filtered)")
    typ_counts = df_f.groupby(colmap["type"]).size().reset_index(name="count").sort_values("count", ascending=False)
    if not typ_counts.empty:
        fig = px.bar(typ_counts, x="count", y=colmap["type"], orientation="h")
        st.plotly_chart(fig, use_container_width=True)
    st.dataframe(typ_counts, use_container_width=True)

with tab4:
    st.subheader("Heatmap: Day of Week × Hour (filtered)")
    heat = df_f.pivot_table(index="__dow", columns="__hour", values=colmap["device"], aggfunc="count", fill_value=0)
    if not heat.empty:
        heat = heat.reindex(["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])
        fig = px.imshow(heat, aspect="auto", text_auto=True)
        st.plotly_chart(fig, use_container_width=True)
    st.dataframe(heat, use_container_width=True)

with tab5:
    st.subheader("Staffing vs. Workload (hourly)")
    workload_hour = df_f.groupby("__hour").size().reindex(range(0,24), fill_value=0).reset_index()
    workload_hour.columns = ["hour", "events"]

    if staff_df is not None and not staff_df.empty and "start" in staff_df.columns and "end" in staff_df.columns:
        staff_hour = pd.DataFrame({"hour": range(0,24)})
        staff_hour["staff_count"] = 0
        for _, row in staff_df.iterrows():
            if pd.isna(row.get("start")) or pd.isna(row.get("end")):
                continue
            start_h = int(pd.to_datetime(row["start"]).hour)
            end_h = int(pd.to_datetime(row["end"]).hour)
            if end_h < start_h:  # overnight
                hours = list(range(start_h, 24)) + list(range(0, end_h+1))
            else:
                hours = list(range(start_h, end_h+1))
            staff_hour.loc[staff_hour["hour"].isin(hours), "staff_count"] += 1

        merged = pd.merge(workload_hour, staff_hour, on="hour", how="left")
        fig = px.line(merged, x="hour", y=["events","staff_count"], markers=True)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(merged, use_container_width=True)
    else:
        st.info("Upload a staffing CSV to see the overlay. A template is included in the download.")

st.divider()
st.subheader("🔎 Recommendations")
for rec in recommend(df_f, colmap):
    st.markdown("- " + rec)

st.download_button(
    "Download filtered data as CSV",
    data=df_f.to_csv(index=False).encode("utf-8"),
    file_name="filtered_device_events.csv",
    mime="text/csv"
)

def to_excel_bytes(dfs: dict) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        for name, d in dfs.items():
            d.to_excel(writer, sheet_name=name[:31], index=False)
    return output.getvalue()

if st.checkbox("Build an Excel summary workbook", value=False):
    summaries = {}
    summaries["by_day"] = df_f.groupby(["__date"]).size().reset_index(name="count")
    summaries["by_hour"] = df_f.groupby(["__hour"]).size().reset_index(name="count")
    summaries["by_device"] = df_f.groupby([colmap["device"]]).size().reset_index(name="count").sort_values("count", ascending=False)
    summaries["by_type"] = df_f.groupby([colmap["type"]]).size().reset_index(name="count").sort_values("count", ascending=False)
    summaries["dow_hour"] = df_f.groupby(["__dow", "__hour"]).size().reset_index(name="count")
    xls = to_excel_bytes(summaries)
    st.download_button("Download Excel summary", data=xls, file_name="device_event_summary.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import datetime as dt
from io import BytesIO

# ====== Settings ======
APP_VER = "v4.1 — Delivery Analytics + Weekly Summary + Drilldown (with header de-dupe)"
USE_SHEETS = True          # set False if you don’t want Google Sheets history
DEFAULT_IDLE_MIN = 15      # minutes of no activity to split delivery runs
DEFAULT_DRILL_MARGIN = 10  # minutes of context on each side when showing source events

st.set_page_config(page_title=f"Device Event Insights — {APP_VER}", layout="wide")
st.title(f"📊 All Device Event Insights — {APP_VER}")
st.caption("Delivery cycle times, walking gaps, device/med trends, and a drilldown inspector. Includes header de-duplication + debug expander.")

# ------------------ Helpers ------------------
def _dedupe_columns(cols):
    """Make duplicate column headers unique: 'User', 'User' -> 'User', 'User_1'."""
    seen = {}
    new = []
    for c in [str(x).strip() for x in cols]:
        if c in seen:
            seen[c] += 1
            new.append(f"{c}_{seen[c]}")
        else:
            seen[c] = 0
            new.append(c)
    return new

def detect_cols(df):
    """
    Heuristics to find key columns in the daily export.
    """
    colmap = {"datetime": None, "device": None, "type": None, "element": None, "user": None, "desc": None}
    for c in df.columns:
        lc = str(c).lower()
        if colmap["datetime"] is None and any(k in lc for k in ["date", "time", "timestamp"]):
            colmap["datetime"] = c
        if colmap["device"] is None and any(k in lc for k in ["device","host","asset","endpoint","cabinet","station","pyxis"]):
            colmap["device"] = c
        if colmap["type"] is None and any(k in lc for k in ["transactiontype","trans type","type","event"]):
            colmap["type"] = c
        if colmap["element"] is None and any(k in lc for k in ["element","med","item","ndc","drug","code"]):
            colmap["element"] = c
        if colmap["desc"] is None and any(k in lc for k in ["desc","name","title","drug","med"]):
            colmap["desc"] = c
        if colmap["user"] is None and any(k in lc for k in ["user","operator","tech","technician","employee"]):
            colmap["user"] = c

    # Required fallbacks
    if colmap["datetime"] is None:
        df["_RowTime"] = pd.to_datetime(pd.RangeIndex(len(df)), unit="s", origin="unix")
        colmap["datetime"] = "_RowTime"
    if colmap["device"] is None:
        df["Device"] = "Unknown"
        colmap["device"] = "Device"
    if colmap["type"] is None:
        df["TransactionType"] = "Unknown"
        colmap["type"] = "TransactionType"
    if colmap["user"] is None:
        df["User"] = "Unknown"
        colmap["user"] = "User"
    return colmap

@st.cache_data(show_spinner=False)
def load_events(xlsx_file):
    df = pd.read_excel(xlsx_file, sheet_name=0, dtype=str)

    # 🔧 make headers unique to avoid "label is not unique" errors
    df.columns = _dedupe_columns(df.columns)

    colmap = detect_cols(df)
    # coerce time and add helpers
    df[colmap["datetime"]] = pd.to_datetime(df[colmap["datetime"]], errors="coerce")
    df = df.dropna(subset=[colmap["datetime"]]).copy()
    df["__date"] = df[colmap["datetime"]].dt.date
    df["__hour"] = df[colmap["datetime"]].dt.hour
    df["__dow"] = df[colmap["datetime"]].dt.day_name()
    return df, colmap

def _seconds_to_hhmmss(s):
    if pd.isna(s): return ""
    s = int(max(0, s))
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    return f"{h:d}:{m:02d}:{sec:02d}" if h > 0 else f"{m:d}:{sec:02d}"

def build_delivery_analytics(ev, colmap, idle_min=DEFAULT_IDLE_MIN):
    """
    Core engine:
      - Per tech: sort by time; compute per-row delta to next row for same tech
      - Dwell at device = time until next event (same tech) IF next device is different
      - Walk time (between devices) = same as dwell estimate (gap after device A before device B)
      - Delivery runs: split by idle gaps >= idle_min
    """
    ts = colmap["datetime"]; dev = colmap["device"]; user = colmap["user"]

    needed = [ts, dev, user, colmap["type"]]
    if colmap.get("desc"): needed.append(colmap["desc"])
    data = ev[needed].sort_values([user, ts]).copy()

    # next-event per tech
    data["__next_ts"]  = data.groupby(user)[ts].shift(-1)
    data["__next_dev"] = data.groupby(user)[dev].shift(-1)

    # gap (sec) to next event for same tech
    data["gap_sec"] = (data["__next_ts"] - data[ts]).dt.total_seconds()
    data["gap_sec"] = data["gap_sec"].fillna(0).clip(lower=0)

    # dwell at this device (approx): only if next event is a different device
    data["dwell_sec"] = np.where(
        (data["__next_dev"].notna()) & (data["__next_dev"] != data[dev]),
        data["gap_sec"],
        np.nan
    )

    # use the same as “walk” (between-device delay); store timestamps/hours for drilldown
    data["walk_sec"]   = data["dwell_sec"]
    data["start_time"] = data[ts]
    data["end_time"]   = data["__next_ts"]
    data["hour"]       = data["start_time"].dt.hour

    # delivery runs: new run when idle gap >= idle_min (per tech)
    cutoff = idle_min * 60
    data["__new_run"] = (data["gap_sec"] >= cutoff).astype(int)
    data["run_id"] = data.groupby(user)["__new_run"].cumsum()

    # ---- Aggregations ----
    # 1) Device-level: average dwell + volume
    device_stats = data.groupby(dev).agg(
        events=("gap_sec","count"),
        avg_dwell_sec=("dwell_sec","mean"),
        p75_dwell_sec=("dwell_sec", lambda x: np.nanpercentile(x.dropna(), 75) if x.notna().any() else np.nan),
        p90_dwell_sec=("dwell_sec", lambda x: np.nanpercentile(x.dropna(), 90) if x.notna().any() else np.nan),
    ).reset_index()
    device_stats["avg_dwell_hhmmss"] = device_stats["avg_dwell_sec"].apply(_seconds_to_hhmmss)
    device_stats["p75_hhmmss"] = device_stats["p75_dwell_sec"].apply(_seconds_to_hhmmss)
    device_stats["p90_hhmmss"] = device_stats["p90_dwell_sec"].apply(_seconds_to_hhmmss)
    device_stats = device_stats.sort_values(["avg_dwell_sec","events"], ascending=[False, False])

    # 2) Tech-level: walking time and dwell
    tech_stats = data.groupby(user).agg(
        events=("gap_sec","count"),
        avg_walk_sec=("walk_sec","mean"),
        total_walk_min=("walk_sec", lambda x: np.nansum(x)/60.0),
        avg_dwell_sec=("dwell_sec","mean"),
    ).reset_index()
    tech_stats["avg_walk_hhmmss"] = tech_stats["avg_walk_sec"].apply(_seconds_to_hhmmss)
    tech_stats["avg_dwell_hhmmss"] = tech_stats["avg_dwell_sec"].apply(_seconds_to_hhmmss)
    tech_stats = tech_stats.sort_values("total_walk_min", ascending=False)

    # 3) Delivery runs per tech
    run_stats = data.groupby([user,"run_id"]).agg(
        start=("start_time", "min"),
        end=("start_time", "max"),
        devices_visited=(dev, lambda x: x.nunique()),
        events=("gap_sec","count")
    ).reset_index()
    run_stats["duration_min"] = (run_stats["end"] - run_stats["start"]).dt.total_seconds()/60.0

    # 4) Hourly volume
    hourly = ev.groupby("__hour").size().reindex(range(0,24), fill_value=0).reset_index()
    hourly.columns = ["hour","events"]

    return data, device_stats, tech_stats, run_stats, hourly

def get_source_slice(ev, colmap, tech, start_ts, end_ts, margin_min=DEFAULT_DRILL_MARGIN):
    """Return raw source rows around a walk gap for the chosen tech with time margins."""
    ts = colmap["datetime"]; user = colmap["user"]
    if pd.isna(start_ts) or pd.isna(end_ts):
        return pd.DataFrame()
    left  = pd.to_datetime(start_ts) - pd.Timedelta(minutes=margin_min)
    right = pd.to_datetime(end_ts)   + pd.Timedelta(minutes=margin_min)
    sl = ev[(ev[user]==tech) & (ev[ts] >= left) & (ev[ts] <= right)].copy()
    return sl.sort_values(ts)

# ------------------ Upload Events ------------------
events_file = st.file_uploader("Upload daily **Events** Excel (.xlsx)", type=["xlsx"], key="events")
if not events_file:
    st.info("Upload your daily events report to begin.")
    st.stop()

events_df, colmap = load_events(events_file)

# 🔍 Debug visibility (so you can confirm detected columns & headers)
with st.expander("Show detected columns (debug)"):
    st.write("Detected colmap:", colmap)
    st.write("All headers:", list(events_df.columns))

# ------------------ Sidebar Filters ------------------
with st.sidebar:
    st.header("Filters")
    valid_times = events_df[colmap["datetime"]].dropna()
    min_dt = valid_times.min().to_pydatetime()
    max_dt = valid_times.max().to_pydatetime()
    time_rng = st.slider(
        "Date/time range",
        min_value=min_dt, max_value=max_dt,
        value=(min_dt, max_dt), format="YYYY-MM-DD HH:mm"
    )
    mask_time = events_df[colmap["datetime"]].between(pd.to_datetime(time_rng[0]), pd.to_datetime(time_rng[1]))

    devs = sorted(events_df[colmap["device"]].fillna("Unknown").unique())
    dev_sel = st.multiselect("Device(s)", devs, default=devs)
    mask_dev = events_df[colmap["device"]].isin(dev_sel)

    types = sorted(events_df[colmap["type"]].fillna("Unknown").unique())
    type_sel = st.multiselect("Transaction Type(s)", types, default=types)
    mask_type = events_df[colmap["type"]].isin(type_sel)

    users = sorted(events_df[colmap["user"]].fillna("Unknown").unique())
    user_sel = st.multiselect("Technician(s)", users, default=users)
    mask_user = events_df[colmap["user"]].isin(user_sel)

    idle_min = st.number_input("Idle gap to split delivery runs (minutes)", min_value=5, max_value=60, value=DEFAULT_IDLE_MIN, step=1)

ev = events_df[mask_time & mask_dev & mask_type & mask_user].copy()

# ------------------ KPI strip ------------------
c1,c2,c3,c4 = st.columns(4)
c1.metric("Events (filtered)", int(len(ev)))
c2.metric("Devices (filtered)", ev[colmap["device"]].nunique())
c3.metric("Technicians", ev[colmap["user"]].nunique())
if not ev.empty:
    c4.metric("Date range", f"{ev[colmap['datetime']].min():%Y-%m-%d} → {ev[colmap['datetime']].max():%Y-%m-%d}")

# ------------------ Tabs ------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Overview", "🚶 Delivery Analytics", "🧑‍🔧 Tech Comparison", "🏥 Device Rankings", "📅 Weekly Summary"
])

with tab1:
    st.subheader("Events over Time")
    by_hour = ev.groupby("__hour").size().reset_index(name="count")
    st.plotly_chart(px.bar(by_hour, x="__hour", y="count", title="Events by Hour"), use_container_width=True)

    by_day = ev.groupby("__date").size().reset_index(name="count")
    st.plotly_chart(px.line(by_day, x="__date", y="count", markers=True, title="Events by Day"), use_container_width=True)

# ---------- Delivery Analytics Engine ----------
if ev.empty:
    st.warning("No events in current filter range.")
    st.stop()

data, device_stats, tech_stats, run_stats, hourly = build_delivery_analytics(ev, colmap, idle_min=idle_min)

with tab2:
    st.subheader("Delivery Analytics (per tech sequence)")

    # WALKING HISTOGRAM + TABLE (with timestamps & hour)
    cA, cB = st.columns(2)
    with cA:
        st.markdown("**Walking gaps (approx.)** — time between finishing one device and starting the next device for the same tech.")
        walk = data.dropna(subset=["walk_sec"]).copy()
        if not walk.empty:
            walk["walk_min"] = walk["walk_sec"]/60.0
            st.plotly_chart(
                px.histogram(walk, x="walk_sec", nbins=50, color=colmap["user"], barmode="overlay",
                             title="Histogram: Walking + between-device delay (seconds)"),
                use_container_width=True
            )
        walk_table = (
            walk[[colmap["user"], colmap["device"], "__next_dev", "start_time", "end_time", "hour", "walk_sec"]]
            .assign(walk_hhmmss=lambda d: d["walk_sec"].apply(_seconds_to_hhmmss))
            .rename(columns={colmap["user"]:"tech", colmap["device"]:"device", "__next_dev":"next_device"})
            .reset_index(drop=True)
        )
        st.dataframe(walk_table, use_container_width=True)

    # DELIVERY RUNS
    with cB:
        st.markdown("**Delivery runs** — split when idle gap ≥ selected threshold.")
        if not run_stats.empty:
            st.plotly_chart(
                px.box(run_stats, x=colmap["user"], y="duration_min", points="all",
                       title="Run duration (minutes) by tech"),
                use_container_width=True
            )
        st.dataframe(run_stats.sort_values(["start"]).rename(columns={colmap["user"]:"tech"}), use_container_width=True)

    st.divider()
    st.markdown("### 🔎 Drilldown Inspector (click-like filtering)")
    st.caption("Pick a tech + gap threshold and optionally specific devices/hours. We’ll show matching long gaps and the raw source rows around any selected gap.")
    cc1, cc2, cc3, cc4 = st.columns(4)
    with cc1:
        drill_tech = st.selectbox("Technician", sorted(ev[colmap["user"]].unique()), index=0 if ev[colmap["user"]].nunique()>0 else None)
    with cc2:
        min_gap = st.number_input("Min gap (seconds)", min_value=0, max_value=3600, value=200, step=10)
    with cc3:
        dev_filter = st.selectbox("Device (optional)", ["(any)"] + sorted(ev[colmap["device"]].unique()))
    with cc4:
        hour_filter = st.selectbox("Hour (optional)", ["(any)"] + list(range(24)))

    # filter walk_table to candidates
    filt = (walk_table["tech"]==drill_tech) & (walk_table["walk_sec"]>=min_gap)
    if dev_filter != "(any)":
        filt &= (walk_table["device"]==dev_filter)
    if hour_filter != "(any)":
        filt &= (walk_table["hour"]==hour_filter)

    drill_candidates = walk_table[filt].copy()
    drill_candidates["label"] = drill_candidates.apply(
        lambda r: f"{r['tech']} | {r['device']} ➜ {r['next_device']} | {r['start_time']:%H:%M:%S} → {r['end_time']:%H:%M:%S} | {int(r['walk_sec'])}s",
        axis=1
    )

    st.write(f"Matches: **{len(drill_candidates)}**")
    st.dataframe(drill_candidates, use_container_width=True)

    chosen = None
    if not drill_candidates.empty:
        chosen_label = st.selectbox("Choose a gap to inspect", drill_candidates["label"].tolist())
        chosen = drill_candidates.loc[drill_candidates["label"]==chosen_label].iloc[0]

    if chosen is not None:
        st.markdown("#### Source rows around this gap")
        src = get_source_slice(
            ev, colmap,
            tech=chosen["tech"],
            start_ts=chosen["start_time"],
            end_ts=chosen["end_time"],
            margin_min=DEFAULT_DRILL_MARGIN
        )
        if src.empty:
            st.info("No source rows found in the window.")
        else:
            # show compact, ordered columns first
            preferred = [colmap["datetime"], colmap["user"], colmap["device"], colmap["type"]]
            if colmap.get("desc"): preferred.append(colmap["desc"])
            rest = [c for c in src.columns if c not in preferred]
            ordered = src[preferred + rest].sort_values(colmap["datetime"]).reset_index(drop=True)
            st.dataframe(ordered, use_container_width=True)

with tab3:
    st.subheader("Technician Comparison")
    if not tech_stats.empty:
        st.dataframe(
            tech_stats[[colmap["user"], "events", "avg_walk_hhmmss", "avg_dwell_hhmmss", "total_walk_min"]]
            .rename(columns={colmap["user"]:"tech"}).sort_values("total_walk_min", ascending=False),
            use_container_width=True
        )
        st.plotly_chart(
            px.bar(tech_stats.sort_values("total_walk_min", ascending=False),
                   x="total_walk_min", y=colmap["user"], orientation="h",
                   title="Total walking / between-device time (minutes)"),
            use_container_width=True
        )

with tab4:
    st.subheader("Device Rankings — Volume & Time")
    if not device_stats.empty:
        st.dataframe(
            device_stats[[colmap["device"], "events", "avg_dwell_hhmmss", "p75_hhmmss", "p90_hhmmss"]]
            .rename(columns={colmap["device"]:"device"})
            .head(100),
            use_container_width=True
        )
        left, right = st.columns(2)
        with left:
            st.plotly_chart(
                px.bar(device_stats.head(30), x="events", y=colmap["device"], orientation="h",
                       title="Top devices by volume (events)"),
                use_container_width=True
            )
        with right:
            top_slow = device_stats.sort_values("avg_dwell_sec", ascending=False).head(30)
            st.plotly_chart(
                px.bar(top_slow, x="avg_dwell_sec", y=colmap["device"], orientation="h",
                       title="Slowest devices by avg dwell (sec)"),
                use_container_width=True
            )

    st.divider()
    st.markdown("### Transaction Type Split per Device")
    # pivot: device × type counts
    pivot = ev.pivot_table(index=colmap["device"], columns=colmap["type"], values=colmap["datetime"], aggfunc="count", fill_value=0)
    st.dataframe(pivot.sort_index(ascending=True).head(200), use_container_width=True)

    if len(pivot.columns) > 0:
        stacked = pivot.reset_index().melt(id_vars=colmap["device"], var_name="transaction_type", value_name="count")
        st.plotly_chart(px.bar(stacked, x="count", y=colmap["device"], color="transaction_type", orientation="h",
                               title="Device × Transaction Type (stacked)"), use_container_width=True)

    st.markdown("### Med Description Trends")
    if colmap.get("desc"):
        # top meds overall and by type
        med_counts = ev.groupby([colmap["desc"], colmap["type"]]).size().reset_index(name="count")
        st.dataframe(med_counts.sort_values("count", ascending=False).head(200), use_container_width=True)

        # Per device top meds (optional heavy), so show a selector
        dev_choice = st.selectbox("Choose a device to view top meds", ["(all)"] + sorted(ev[colmap["device"]].unique()))
        if dev_choice != "(all)":
            dev_med = ev[ev[colmap["device"]]==dev_choice].groupby([colmap["desc"], colmap["type"]]).size().reset_index(name="count")
            st.plotly_chart(
                px.bar(dev_med.sort_values("count", ascending=False).head(30),
                       x="count", y=colmap["desc"], color=colmap["type"], orientation="h",
                       title=f"Top meds at {dev_choice}"),
                use_container_width=True
            )
    else:
        st.info("No medication description column detected. If your export has one (e.g., 'Med Description'), it will appear here automatically.")

# ------------------ Weekly Summary (Google Sheets History) ------------------
with tab5:
    st.subheader("Weekly Summary & History")
    st.write("Each day, append your **daily metrics**. This tab aggregates the last 7 calendar days from the history sheet.")

    # Build a 1-row daily summary from current filtered data
    today = pd.to_datetime(ev[colmap["datetime"]].dt.date.min())
    _summary = {
        "date": today.date().isoformat(),
        "events": int(len(ev)),
        "unique_devices": int(ev[colmap["device"]].nunique()),
        "unique_techs": int(ev[colmap["user"]].nunique()),
        "avg_walk_sec": float(np.nanmean(data["walk_sec"])) if np.isfinite(np.nanmean(data["walk_sec"])) else np.nan,
        "avg_dwell_sec": float(np.nanmean(data["dwell_sec"])) if np.isfinite(np.nanmean(data["dwell_sec"])) else np.nan,
        "p90_device_dwell_sec": float(np.nanpercentile(device_stats["avg_dwell_sec"].dropna(), 90)) if device_stats["avg_dwell_sec"].notna().any() else np.nan,
    }
    daily_df = pd.DataFrame([_summary])
    daily_df["avg_walk_mm:ss"] = daily_df["avg_walk_sec"].apply(_seconds_to_hhmmss)
    daily_df["avg_dwell_mm:ss"] = daily_df["avg_dwell_sec"].apply(_seconds_to_hhmmss)
    st.markdown("**Today’s summary (from current upload/filters):**")
    st.dataframe(daily_df, use_container_width=True)

    if USE_SHEETS:
        st.divider()
        st.markdown("### Google Sheets History")

        try:
            import gspread
            from google.oauth2.service_account import Credentials

            secrets = st.secrets
            svc_section = secrets.get("gcp_service_account", None)
            gs_section = secrets.get("gsheets", None)

            if (svc_section is None) or (gs_section is None):
                st.info("Add Streamlit **Secrets** for Google Sheets to enable history (see README).")
            else:
                scopes = ["https://www.googleapis.com/auth/spreadsheets",
                          "https://www.googleapis.com/auth/drive"]
                creds = Credentials.from_service_account_info(dict(svc_section), scopes=scopes)
                gc = gspread.authorize(creds)

                workbook_url = gs_section.get("workbook_url")
                sheet_name   = gs_section.get("events_sheet", "events_history")
                sh = gc.open_by_url(workbook_url)
                try:
                    ws = sh.worksheet(sheet_name)
                except Exception:
                    ws = sh.add_worksheet(sheet_name, rows=1000, cols=20)
                    ws.append_row(list(daily_df.columns))

                c1, c2 = st.columns(2)
                with c1:
                    if st.button("Append today’s summary to Google Sheets"):
                        existing = ws.get_all_records()
                        if not existing:
                            ws.clear()
                            ws.append_row(list(daily_df.columns))
                        ws.append_row([str(v) for v in daily_df.iloc[0].tolist()])
                        st.success("Appended today’s summary ✅")

                with c2:
                    if st.button("Load last 7 days from Google Sheets"):
                        recs = ws.get_all_records()
                        if recs:
                            hist = pd.DataFrame(recs)
                            hist["date"] = pd.to_datetime(hist["date"], errors="coerce").dt.date
                            end = dt.date.today()
                            start = end - dt.timedelta(days=6)
                            last7 = hist[(hist["date"] >= start) & (hist["date"] <= end)].copy()
                            if not last7.empty:
                                st.dataframe(last7, use_container_width=True)
                                plot_df = last7.copy()
                                plot_df["date"] = pd.to_datetime(plot_df["date"])
                                plot_df["avg_walk_sec"] = pd.to_numeric(plot_df["avg_walk_sec"], errors="coerce")
                                plot_df["avg_dwell_sec"] = pd.to_numeric(plot_df["avg_dwell_sec"], errors="coerce")
                                colsel = ["events","unique_devices","unique_techs"]
                                st.plotly_chart(px.line(plot_df, x="date", y=colsel, markers=True,
                                                        title="Weekly trend: Volume, Devices, Techs"),
                                                use_container_width=True)
                                st.plotly_chart(px.line(plot_df, x="date", y=["avg_walk_sec","avg_dwell_sec"], markers=True,
                                                        title="Weekly trend: Avg Walk vs Avg Dwell (sec)"),
                                                use_container_width=True)
                            else:
                                st.info("No rows in the last 7 days yet. Append today first.")
                        else:
                            st.info("No history yet. Append today first.")
        except Exception as e:
            st.error(f"Google Sheets error: {e}")
    else:
        st.info("History disabled (USE_SHEETS=False). Enable and add Secrets to persist & summarize weekly.")


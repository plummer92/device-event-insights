import io
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# =========================
# Utils & constants
# =========================

st.set_page_config(page_title="Device Event Insights", layout="wide")

DEFAULT_IDLE_MIN = 10          # split "runs" if idle gap >= this
DEFAULT_DRILL_MARGIN = 10      # minutes around a slice for audit

def ensure_unique_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate columns by label, keeping the first occurrence."""
    return df.loc[:, ~df.columns.duplicated()].copy()

def parse_datetime_series(s: pd.Series):
    """Best-effort parse to datetime; coerce errors to NaT."""
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    return pd.to_datetime(s, errors="coerce", infer_datetime_format=True)

def seconds(s: pd.Series) -> pd.Series:
    return s.dt.total_seconds()

def time_hms(total_sec: float) -> str:
    if pd.isna(total_sec):
        return "—"
    total_sec = int(total_sec)
    return f"{total_sec//3600}:{(total_sec%3600)//60:02d}:{total_sec%60:02d}"

# =========================
# Data loading
# =========================

st.title("📊 Device Event Insights")

st.sidebar.header("1) Upload daily file")
upl = st.sidebar.file_uploader("Drag your XLSX here", type=["xlsx"])
if not upl:
    st.info("Upload today’s **Excel (.xlsx)** export to begin.")
    st.stop()

try:
    df_raw = pd.read_excel(upl, engine="openpyxl")
except Exception as e:
    st.error(f"Could not read Excel: {e}")
    st.stop()

df_raw = ensure_unique_columns(df_raw)

if df_raw.empty:
    st.warning("The uploaded file has no rows.")
    st.stop()

# =========================
# Column mapping UI
# =========================

st.sidebar.header("2) Map columns")
cols = list(df_raw.columns)

def pick(label, fallback_candidates):
    """Try to auto-pick a column; otherwise default to first."""
    for c in fallback_candidates:
        for real in cols:
            if real.strip().lower() == c.lower():
                return real
    return cols[0]

datetime_col = st.sidebar.selectbox("Date/Time column", cols, index=cols.index(pick("datetime", ["DateTime", "Timestamp", "Date Time", "EventDateTime", "Event Time"])))
device_col   = st.sidebar.selectbox("Device column",   cols, index=cols.index(pick("device",   ["Device", "DeviceName", "Cabinet", "Device Name"])))
user_col     = st.sidebar.selectbox("User/Tech column",cols, index=cols.index(pick("user",     ["UserName", "User", "Technician", "Operator"])))
type_col     = st.sidebar.selectbox("Transaction Type",cols, index=cols.index(pick("type",     ["TransactionType", "Type", "Action", "EventType"])))

# optional
med_id_col   = st.sidebar.selectbox("Med ID (optional)", ["<none>"] + cols, index=0)
desc_col     = st.sidebar.selectbox("Med Description (optional)", ["<none>"] + cols, index=0)
qty_col      = st.sidebar.selectbox("Quantity (optional)", ["<none>"] + cols, index=0)

colmap = {
    "datetime": datetime_col,
    "device": device_col,
    "user": user_col,
    "type": type_col,
}
if med_id_col != "<none>": colmap["med_id"] = med_id_col
if desc_col   != "<none>": colmap["desc"]   = desc_col
if qty_col    != "<none>": colmap["quantity"]= qty_col

# guard: duplicate label issue (e.g., two columns both named "UserName")
if df_raw.columns.duplicated().any():
    st.warning("Duplicate column labels detected; keeping first instance of each.")
    df_raw = ensure_unique_columns(df_raw)

# =========================
# Basic cleaning + typed cols
# =========================

def base_clean(df: pd.DataFrame, colmap: dict) -> pd.DataFrame:
    keep = [colmap["datetime"], colmap["device"], colmap["user"], colmap["type"]]
    if colmap.get("med_id"):   keep.append(colmap["med_id"])
    if colmap.get("desc"):     keep.append(colmap["desc"])
    if colmap.get("quantity"): keep.append(colmap["quantity"])
    out = df[keep].copy()

    # Parse datetime
    out[colmap["datetime"]] = parse_datetime_series(out[colmap["datetime"]])
    out = out.dropna(subset=[colmap["datetime"]])

    # Normalize strings
    for c in [colmap["device"], colmap["user"], colmap["type"]]:
        out[c] = out[c].astype(str).str.strip()

    # Quantity numeric
    if colmap.get("quantity"):
        out[colmap["quantity"]] = pd.to_numeric(out[colmap["quantity"]], errors="coerce")

    # Lowercase type for matching later
    out["__type_norm"] = out[colmap["type"]].str.lower()

    return out

ev = base_clean(df_raw, colmap)

if ev.empty:
    st.error("No valid rows after cleaning. Check the column mapping.")
    st.stop()

# =========================
# Filters
# =========================

st.sidebar.header("3) Filters")

min_ts, max_ts = ev[colmap["datetime"]].min(), ev[colmap["datetime"]].max()
rng = st.sidebar.slider(
    "Time range",
    value=(pd.to_datetime(min_ts), pd.to_datetime(max_ts)),
    min_value=pd.to_datetime(min_ts),
    max_value=pd.to_datetime(max_ts),
    format="YYYY-MM-DD HH:mm"
)

devices = sorted(ev[colmap["device"]].dropna().unique().tolist())
users   = sorted(ev[colmap["user"]].dropna().unique().tolist())
types   = sorted(ev[colmap["type"]].dropna().unique().tolist())

pick_devices = st.sidebar.multiselect("Devices", devices, default=devices)
pick_users   = st.sidebar.multiselect("Users", users, default=users)
pick_types   = st.sidebar.multiselect("Transaction Types", types, default=types)

idle_min = st.sidebar.number_input("Split runs if idle gap ≥ (minutes)", min_value=3, max_value=60, value=DEFAULT_IDLE_MIN, step=1)

st.sidebar.markdown("---")
st.sidebar.subheader("SLA & Alerts")
use_p90 = st.sidebar.checkbox("Use device p90 dwell as SLA", value=True)
dwell_thresh = st.sidebar.number_input("Manual SLA dwell (sec) if not using p90", min_value=30, max_value=3600, value=300, step=10)
st.sidebar.session_state["min_gap_for_alert"] = st.sidebar.number_input("Alert: min walking gap (sec)", min_value=60, max_value=1800, value=200, step=10)

st.sidebar.markdown("---")
st.sidebar.subheader("Verify→Refill Pairing")
verify_defaults = ["Verify Inventory", "Verify Stock"]
refill_defaults = ["Refill", "Load"]  # include Load if your system logs refills as "Load"
verify_labels = st.sidebar.multiselect("Verify labels", options=sorted(set([*verify_defaults, *types])), default=verify_defaults)
refill_labels = st.sidebar.multiselect("Refill labels", options=sorted(set([*refill_defaults, *types])), default=refill_defaults)
max_pair_gap_sec = st.sidebar.slider("Max time between Verify→Refill (sec)", min_value=15, max_value=600, value=180, step=5)

# Apply filters
mask = (
    (ev[colmap["datetime"]] >= rng[0]) &
    (ev[colmap["datetime"]] <= rng[1]) &
    (ev[colmap["device"]].isin(pick_devices)) &
    (ev[colmap["user"]].isin(pick_users)) &
    (ev[colmap["type"]].isin(pick_types))
)
ev = ev.loc[mask].copy().sort_values(colmap["datetime"])

if ev.empty:
    st.warning("No events in current filter range.")
    st.stop()

# =========================
# Core analytics
# =========================

def build_delivery_analytics(ev: pd.DataFrame, colmap: dict, idle_min: int = DEFAULT_IDLE_MIN):
    """
    Compute dwell vs walk per-tech sequences, hourly events, run grouping.
    dwell_sec: gap to next event on SAME device by same user
    walk_sec : gap to next event on DIFFERENT device by same user
    """
    ts   = colmap["datetime"]; dev  = colmap["device"]; user = colmap["user"]; typ = colmap["type"]
    desc = colmap.get("desc")

    needed = [ts, dev, user, typ]
    if desc: needed.append(desc)
    data = ev[needed].sort_values([user, ts]).copy()

    # next-event per tech
    data["__next_ts"]  = data.groupby(user)[ts].shift(-1)
    data["__next_dev"] = data.groupby(user)[dev].shift(-1)

    gaps = seconds(data["__next_ts"] - data[ts])  # seconds to next event same tech
    same_dev = (data[dev] == data["__next_dev"])

    data["dwell_sec"] = np.where(same_dev, gaps, np.nan)
    data["walk_sec"]  = np.where(~same_dev, gaps, np.nan)

    # Hourly
    hourly = (
        data.assign(hour=data[ts].dt.hour)
            .groupby("hour").size().reset_index(name="events")
            .sort_values("hour")
    )

    # Per-device dwell stats
    device_stats = (
        data.groupby(dev).agg(
            events=(ts, "count"),
            avg_dwell_sec=("dwell_sec", "mean"),
            p90_dwell_sec=("dwell_sec", lambda x: np.percentile(x.dropna(), 90) if x.notna().any() else np.nan),
            avg_walk_sec=("walk_sec","mean")
        ).reset_index()
    )

    # Per-tech (walking load etc.)
    tech_stats = (
        data.groupby(user).agg(
            events=(ts, "count"),
            avg_dwell_sec=("dwell_sec", "mean"),
            total_walk_sec=("walk_sec", "sum")
        ).reset_index()
    )
    tech_stats["total_walk_min"] = (tech_stats["total_walk_sec"].fillna(0) / 60.0).round(1)

    # Runs per tech (split by idle threshold)
    idle_sec = idle_min * 60
    data["__gap_prev"] = seconds(data[ts] - data.groupby(user)[ts].shift(1))
    data["__new_run"] = (data["__gap_prev"].fillna(idle_sec + 1) >= idle_sec).astype(int)
    data["run_id"] = data.groupby(user)["__new_run"].cumsum()

    run_stats = (
        data.groupby([user, "run_id"]).agg(
            start=(ts, "min"),
            end=(ts, "max"),
            devices_visited=(dev, lambda x: x.nunique())
        ).reset_index()
    )
    run_stats["duration_min"] = seconds(run_stats["end"] - run_stats["start"]) / 60.0

    return data, device_stats, tech_stats, run_stats, hourly

def get_source_slice(ev: pd.DataFrame, colmap: dict, tech: str, start_ts: pd.Timestamp, end_ts: pd.Timestamp, margin_min: int = DEFAULT_DRILL_MARGIN):
    ts = colmap["datetime"]; user = colmap["user"]
    start = pd.to_datetime(start_ts) - pd.Timedelta(minutes=margin_min)
    end   = pd.to_datetime(end_ts)   + pd.Timedelta(minutes=margin_min)
    mask = (ev[user] == tech) & (ev[ts] >= start) & (ev[ts] <= end)
    return ev.loc[mask].copy()

def build_verify_refill_cycles(ev: pd.DataFrame, colmap: dict, verify_labels, refill_labels, max_gap_sec=180):
    """
    Pair VERIFY → next REFILL for same tech+device+med_id within window.
    Quantity comes from the Refill row (qty col if available).
    """
    ts   = colmap["datetime"]; dev  = colmap["device"]; user = colmap["user"]
    typ  = colmap["type"];     desc = colmap.get("desc"); mid  = colmap.get("med_id")
    qtyc = colmap.get("quantity")

    work = ev.copy()
    if mid is None:
        work["__MedID_fallback"] = work.get(desc, "Unknown")
        mid = "__MedID_fallback"

    if qtyc:
        work["_qty_num"] = pd.to_numeric(work[qtyc], errors="coerce")
    else:
        work["_qty_num"] = np.nan

    vset = [v.lower() for v in verify_labels]
    rset = [r.lower() for r in refill_labels]
    work["is_verify"] = work[typ].astype(str).str.strip().str.lower().isin(vset)
    work["is_refill"] = work[typ].astype(str).str.strip().str.lower().isin(rset)

    work = work.loc[work["is_verify"] | work["is_refill"], [ts, dev, user, typ, mid] + ([desc] if desc else []) + ["_qty_num"]]
    if work.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    work = work.sort_values([user, dev, mid, ts])

    grp = [user, dev, mid]
    # carry forward last verify ts within group
    work["last_verify_ts"] = work.groupby(grp).apply(lambda g: g[ts].where(g["is_verify"]).ffill()).reset_index(level=grp, drop=True)

    pairs = work[work["is_refill"] & work["last_verify_ts"].notna()].copy()
    pairs["delta_sec"] = seconds(pairs[ts] - pairs["last_verify_ts"])
    pairs = pairs[(pairs["delta_sec"] >= 0) & (pairs["delta_sec"] <= max_gap_sec)]

    pairs = pairs.sort_values(grp + ["last_verify_ts", ts])
    pairs["__seq"] = pairs.groupby(grp + ["last_verify_ts"]).cumcount()
    pairs = pairs[pairs["__seq"] == 0].copy()

    cycles = pd.DataFrame({
        "tech":  pairs[user],
        "device": pairs[dev],
        "med_id": pairs[mid],
        "verify_ts": pairs["last_verify_ts"],
        "refill_ts": pairs[ts],
        "cycle_sec": pairs["delta_sec"].astype(float),
        "qty": pairs["_qty_num"],
        "hour": pairs["last_verify_ts"].dt.hour
    })
    if desc and desc in work.columns:
        latest_desc = work.dropna(subset=[desc]).sort_values(ts).groupby([mid])[desc].last()
        cycles = cycles.merge(latest_desc.rename("med_desc"), left_on="med_id", right_index=True, how="left")
    else:
        cycles["med_desc"] = np.nan

    # aggregates
    c_dev = cycles.groupby("device").agg(
        cycles=("cycle_sec","count"),
        avg_cycle_sec=("cycle_sec","mean"),
        p90_cycle_sec=("cycle_sec", lambda x: np.percentile(x, 90) if len(x)>0 else np.nan),
        total_qty=("qty","sum")
    ).reset_index()

    c_tech = cycles.groupby("tech").agg(
        cycles=("cycle_sec","count"),
        avg_cycle_sec=("cycle_sec","mean"),
        p90_cycle_sec=("cycle_sec", lambda x: np.percentile(x, 90) if len(x)>0 else np.nan),
        total_qty=("qty","sum")
    ).reset_index()

    c_med = cycles.groupby(["med_id","med_desc"]).agg(
        cycles=("cycle_sec","count"),
        avg_cycle_sec=("cycle_sec","mean"),
        p90_cycle_sec=("cycle_sec", lambda x: np.percentile(x, 90) if len(x)>0 else np.nan),
        total_qty=("qty","sum")
    ).reset_index()

    return cycles, c_dev, c_tech, c_med

def build_top_alerts(ev, data, device_stats, tech_stats, run_stats, hourly, colmap,
                     dwell_thresh_sec, use_p90, min_gap_for_alert=200,
                     verify_labels=None, refill_labels=None):
    alerts = []
    def add(priority, category, entity, metric, value, threshold=None, details="", qf=None):
        alerts.append({
            "Priority": priority,
            "Category": category,
            "Entity": entity,
            "Metric": metric,
            "Value": value,
            "Threshold": threshold if threshold is not None else "",
            "Details": details,
            "QuickFilterHint": qf if qf else ""
        })

    ts=colmap["datetime"]; dev=colmap["device"]; user=colmap["user"]; typ=colmap["type"]; desc=colmap.get("desc")
    qtyc=colmap.get("quantity"); midc=colmap.get("med_id")

    # 1) Devices over SLA
    tmp = device_stats.copy()
    tmp["sla"] = np.where(use_p90, tmp["p90_dwell_sec"], dwell_thresh_sec)
    slow = tmp[tmp["avg_dwell_sec"] >= tmp["sla"]].sort_values("avg_dwell_sec", ascending=False).head(3)
    for _, r in slow.iterrows():
        add(1, "Device over SLA", r[dev], "avg_dwell_sec",
            round(float(r["avg_dwell_sec"]),1), round(float(r["sla"]),1),
            "Avg dwell exceeds SLA", qf=f"Device={r[dev]}")

    # Build walk list (biggest gaps)
    big_gaps = data.dropna(subset=["walk_sec"]).copy()
    big_gaps = big_gaps[big_gaps["walk_sec"] >= min_gap_for_alert].sort_values("walk_sec", ascending=False).head(3)
    for _, r in big_gaps.iterrows():
        ent = f"{r[user]} @ {r[dev]} → {r['__next_dev']}"
        add(2, "Walking gap", ent, "walk_sec", int(r["walk_sec"]), min_gap_for_alert,
            f"{r[ts]:%H:%M}→{r['__next_ts']:%H:%M}", qf=f"Tech={r[user]}, Device={r[dev]}")

    # Techs with most walking time
    top_walk = tech_stats.sort_values("total_walk_min", ascending=False).head(3)
    for _, r in top_walk.iterrows():
        add(3, "Tech walking load", r[user], "total_walk_min", round(float(r["total_walk_min"]),1), None,
            "High cumulative between-device time", qf=f"Tech={r[user]}")

    # Hourly spikes
    hr = hourly.copy()
    if len(hr)>1 and hr["events"].std(ddof=0)>0:
        mu=hr["events"].mean(); sd=hr["events"].std(ddof=0)
        hr["z"]=(hr["events"]-mu)/sd
        spikes = hr[hr["z"]>1.5].sort_values("z", ascending=False).head(3)
        for _, r in spikes.iterrows():
            add(4, "Hourly spike", f"Hour={int(r['hour'])}", "events", int(r["events"]), "z>1.5",
                "Unusual surge", qf=f"Hour={int(r['hour'])}")

    # Long runs
    long_runs = run_stats.sort_values("duration_min", ascending=False).head(3)
    for _, r in long_runs.iterrows():
        ent=f"{r[user]} run #{int(r['run_id'])}"
        add(5, "Long delivery run", ent, "duration_min", round(float(r["duration_min"]),1), None,
            f"{r['start']:%H:%M}→{r['end']:%H:%M}, devices={int(r['devices_visited'])}",
            qf=f"Tech={r[user]}")

    # Med quantity heavy hitters
    if qtyc and midc and desc:
        ev["_qty_num_alert"] = pd.to_numeric(ev[qtyc], errors="coerce").fillna(0)
        meds = ev.groupby([midc, desc]).agg(total_qty=("_qty_num_alert","sum")).reset_index() \
                 .sort_values("total_qty", ascending=False).head(3)
        for _, r in meds.iterrows():
            add(6, "Med quantity spike", f"{r[midc]} — {r[desc]}", "total_qty", int(r["total_qty"]))

    # Transaction type slowdowns
    by_type = data.groupby(typ).agg(avg_dwell=("dwell_sec","mean")).reset_index()
    by_type = by_type[by_type["avg_dwell"].notna()].sort_values("avg_dwell", ascending=False).head(3)
    for _, r in by_type.iterrows():
        add(7, "Transaction slowdown", r[typ], "avg_dwell_sec", round(float(r["avg_dwell"]),1))

    # Data quality
    if qtyc: 
        mis_qty = int(ev[qtyc].isna().sum())
        if mis_qty>0:
            add(9, "Data quality", "Quantity", "missing_rows", mis_qty, None, "Rows without Quantity")
    if midc:
        mis_mid = int(ev[midc].isna().sum())
        if mis_mid>0:
            add(9, "Data quality", "MedID", "missing_rows", mis_mid, None, "Rows without MedID")

    # High-volume devices (top decile)
    dev_events = ev.groupby(dev).size().reset_index(name="events")
    if len(dev_events)>0:
        p90v = np.percentile(dev_events["events"], 90)
        hot = dev_events[dev_events["events"]>=p90v].sort_values("events", ascending=False).head(3)
        for _, r in hot.iterrows():
            add(10, "Device volume surge", r[dev], "events", int(r["events"]), int(p90v), "Top decile today", qf=f"Device={r[dev]}")

    # Worst verify→refill cycles
    if verify_labels is not None and refill_labels is not None:
        try:
            cycles_for_alerts, _, _, _ = build_verify_refill_cycles(ev, colmap, verify_labels, refill_labels, max_gap_sec=180)
            if not cycles_for_alerts.empty:
                worst = cycles_for_alerts.sort_values("cycle_sec", ascending=False).head(3)
                for _, r in worst.iterrows():
                    ent = f"{r['tech']} @ {r['device']} • {r.get('med_desc', r['med_id'])}"
                    add(2, "Verify→Refill cycle", ent, "cycle_sec", round(float(r["cycle_sec"]),1), 180,
                        f"{r['verify_ts']:%H:%M}→{r['refill_ts']:%H:%M}",
                        qf=f"Tech={r['tech']}, Device={r['device']}")
        except Exception:
            pass

    out = pd.DataFrame(alerts)
    if not out.empty:
        out = out.sort_values(["Priority","Value"], ascending=[True, False]).head(10).reset_index(drop=True)
    return out

# =========================
# Compute analytics
# =========================

data, device_stats, tech_stats, run_stats, hourly = build_delivery_analytics(ev, colmap, idle_min=idle_min)

# =========================
# Tabs
# =========================

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📈 Overview", "🚶 Delivery Analytics", "🧑‍🔧 Tech Comparison",
    "🏥 Device Rankings", "💊 Med Trends", "🚨 Top 10 Alerts", "🔁 Verify→Refill Cycles"
])

with tab1:
    st.subheader("Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Events", int(len(ev)))
    c2.metric("Devices", ev[colmap["device"]].nunique())
    c3.metric("Users",   ev[colmap["user"]].nunique())
    c4.metric("Window", f"{rng[0]:%Y-%m-%d %H:%M} → {rng[1]:%Y-%m-%d %H:%M}")

    if not hourly.empty:
        st.plotly_chart(px.bar(hourly, x="hour", y="events", title="Events by hour"), use_container_width=True)

    st.markdown("#### Latest 200 events")
    pref = [colmap["datetime"], colmap["user"], colmap["device"], colmap["type"]]
    if colmap.get("desc"): pref.append(colmap["desc"])
    if colmap.get("med_id"): pref.append(colmap["med_id"])
    if colmap.get("quantity"): pref.append(colmap["quantity"])
    others = [c for c in ev.columns if c not in pref+["__type_norm"]]
    st.dataframe(ev[pref+others].sort_values(colmap["datetime"], ascending=False).head(200), use_container_width=True)

with tab2:
    st.subheader("Delivery Analytics")
    st.markdown("Dwell = time to next event on same device by same user. Walk = time to next event on a different device.")

    st.markdown("##### Dwell vs Walk (sec) — sample points")
    show = data[["dwell_sec","walk_sec", colmap["user"], colmap["device"], colmap["datetime"], "__next_ts","__next_dev"]].copy().head(500)
    st.dataframe(show, use_container_width=True)

    st.markdown("##### Top walking gaps (longest)")
    top_walk = data.dropna(subset=["walk_sec"]).sort_values("walk_sec", ascending=False).head(25).copy()
    if not top_walk.empty:
        top_walk["gap_hh:mm:ss"] = top_walk["walk_sec"].apply(time_hms)
        st.dataframe(top_walk[[colmap["user"], colmap["device"], "__next_dev", colmap["datetime"], "__next_ts", "walk_sec","gap_hh:mm:ss"]], use_container_width=True)

    st.markdown("##### Run stats (split by idle gap)")
    st.dataframe(run_stats.sort_values(["duration_min"], ascending=False), use_container_width=True)

with tab3:
    st.subheader("Tech Comparison")
    t = tech_stats.copy()
    t["avg_dwell_sec"] = t["avg_dwell_sec"].round(1)
    st.dataframe(t.sort_values(["total_walk_min","events"], ascending=[False, False]), use_container_width=True)
    try:
        st.plotly_chart(px.bar(t, x=colmap["user"], y="total_walk_min", title="Total walking time by tech (min)"), use_container_width=True)
    except Exception:
        pass

with tab4:
    st.subheader("Device Rankings")
    d = device_stats.copy()
    d["avg_dwell_sec"] = d["avg_dwell_sec"].round(1)
    d["p90_dwell_sec"] = d["p90_dwell_sec"].round(1)
    d["avg_walk_sec"]  = d["avg_walk_sec"].round(1)
    st.dataframe(d.sort_values(["avg_dwell_sec","events"], ascending=[False, False]), use_container_width=True)
    try:
        st.plotly_chart(px.bar(d.sort_values("avg_dwell_sec", ascending=False).head(25), x=colmap["device"], y="avg_dwell_sec", title="Slowest average dwell by device"), use_container_width=True)
    except Exception:
        pass

with tab5:
    st.subheader("Med Trends (by MedID + Description + Quantity)")
    if colmap.get("med_id") and colmap.get("desc") and colmap.get("quantity"):
        tmp = ev.copy()
        tmp["_qty_num"] = pd.to_numeric(tmp[colmap["quantity"]], errors="coerce").fillna(0)
        med_agg = tmp.groupby([colmap["med_id"], colmap["desc"]]).agg(
            events=(colmap["datetime"], "count"),
            total_qty=("_qty_num","sum")
        ).reset_index().sort_values(["total_qty","events"], ascending=False)
        st.dataframe(med_agg.head(200), use_container_width=True)
        try:
            st.plotly_chart(px.bar(med_agg.head(30), x=colmap["desc"], y="total_qty", title="Top quantities"), use_container_width=True)
        except Exception:
            pass
    else:
        st.info("To enable med trends, map **Med ID**, **Description**, and **Quantity** in the sidebar.")

with tab6:
    st.subheader("🚨 Top 10 Alerts — things to investigate")
    alerts = build_top_alerts(
        ev=ev, data=data, device_stats=device_stats, tech_stats=tech_stats,
        run_stats=run_stats, hourly=hourly, colmap=colmap,
        dwell_thresh_sec=dwell_thresh, use_p90=use_p90,
        min_gap_for_alert=st.sidebar.session_state.get("min_gap_for_alert", 200),
        verify_labels=verify_labels, refill_labels=refill_labels
    )
    if alerts.empty:
        st.success("No red flags based on current filters/thresholds. 🙌")
    else:
        st.dataframe(alerts, use_container_width=True)
        st.download_button(
            "Download alerts (CSV)",
            alerts.to_csv(index=False).encode("utf-8"),
            file_name="top10_alerts.csv",
            mime="text/csv"
        )
    st.caption("Tip: Use **QuickFilterHint** (e.g., `Tech=…`, `Device=…`, `Hour=…`) to drill in with the sidebar filters.")

with tab7:
    st.subheader("🔁 Verify → Refill Cycles (same Tech + Device + Med within window)")
    cycles, c_dev, c_tech, c_med = build_verify_refill_cycles(
        ev, colmap,
        verify_labels=verify_labels,
        refill_labels=refill_labels,
        max_gap_sec=max_pair_gap_sec
    )

    if cycles.empty:
        st.info("No cycles detected with the current labels and time window.")
    else:
        # KPIs
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Cycles detected", int(len(cycles)))
        c2.metric("Avg cycle (sec)", round(float(np.nanmean(cycles["cycle_sec"])),1))
        c3.metric("P90 cycle (sec)", round(float(np.nanpercentile(cycles["cycle_sec"].dropna(), 90)),1) if cycles["cycle_sec"].notna().any() else "—")
        c4.metric("Total qty (from Refill)", int(pd.to_numeric(cycles["qty"], errors="coerce").fillna(0).sum()))

        # Outliers
        st.markdown("#### Longest cycles (top 25)")
        top_out = cycles.sort_values("cycle_sec", ascending=False).head(25).copy()
        top_out["cycle_hh:mm:ss"] = top_out["cycle_sec"].apply(time_hms)
        st.dataframe(top_out[["tech","device","med_id","med_desc","verify_ts","refill_ts","cycle_sec","cycle_hh:mm:ss","qty","hour"]],
                     use_container_width=True)
        st.download_button(
            "Download cycles (CSV)",
            cycles.to_csv(index=False).encode("utf-8"),
            file_name="verify_refill_cycles.csv",
            mime="text/csv"
        )

        # Charts
        left, right = st.columns(2)
        with left:
            try:
                st.plotly_chart(
                    px.box(cycles, x="device", y="cycle_sec", points="all", title="Cycle time by device (sec)"),
                    use_container_width=True
                )
            except Exception:
                pass
        with right:
            try:
                st.plotly_chart(
                    px.box(cycles, x="tech", y="cycle_sec", points="all", title="Cycle time by tech (sec)"),
                    use_container_width=True
                )
            except Exception:
                pass

        # Aggregates
        st.markdown("#### Per-device cycle stats")
        if not c_dev.empty:
            c_dev = c_dev.assign(
                avg_cycle_sec=lambda d: d["avg_cycle_sec"].round(1),
                p90_cycle_sec=lambda d: d["p90_cycle_sec"].round(1)
            ).sort_values(["avg_cycle_sec","cycles"], ascending=[False, False])
            st.dataframe(c_dev, use_container_width=True)
            st.download_button(
                "Download per-device cycle stats (CSV)",
                c_dev.to_csv(index=False).encode("utf-8"),
                file_name="cycles_by_device.csv",
                mime="text/csv"
            )

        st.markdown("#### Per-tech cycle stats")
        if not c_tech.empty:
            c_tech = c_tech.assign(
                avg_cycle_sec=lambda d: d["avg_cycle_sec"].round(1),
                p90_cycle_sec=lambda d: d["p90_cycle_sec"].round(1)
            ).sort_values(["avg_cycle_sec","cycles"], ascending=[False, False])
            st.dataframe(c_tech, use_container_width=True)
            st.download_button(
                "Download per-tech cycle stats (CSV)",
                c_tech.to_csv(index=False).encode("utf-8"),
                file_name="cycles_by_tech.csv",
                mime="text/csv"
            )

        st.markdown("#### Med (ID + Description) cycle stats")
        if not c_med.empty:
            c_med = c_med.assign(
                avg_cycle_sec=lambda d: d["avg_cycle_sec"].round(1),
                p90_cycle_sec=lambda d: d["p90_cycle_sec"].round(1)
            ).sort_values(["avg_cycle_sec","cycles"], ascending=[False, False])
            st.dataframe(c_med.head(200), use_container_width=True)
            st.download_button(
                "Download cycles by med (CSV)",
                c_med.to_csv(index=False).encode("utf-8"),
                file_name="cycles_by_med.csv",
                mime="text/csv"
            )

        # Click-to-audit
        st.markdown("#### Audit a cycle’s source rows")
        if not top_out.empty:
            pick_idx = st.number_input("Row index from 'Longest cycles' (0-based)", min_value=0, max_value=len(top_out)-1, value=0, step=1)
            chosen = top_out.iloc[int(pick_idx)]
            src = get_source_slice(
                ev, colmap,
                tech=chosen["tech"],
                start_ts=chosen["verify_ts"],
                end_ts=chosen["refill_ts"],
                margin_min=DEFAULT_DRILL_MARGIN
            )
            if src.empty:
                st.info("No source rows found in the context window.")
            else:
                pref = [colmap["datetime"], colmap["user"], colmap["device"], colmap["type"]]
                if colmap.get("desc"): pref.append(colmap["desc"])
                rest = [c for c in src.columns if c not in pref]
                view = src[pref+rest].sort_values(colmap["datetime"]).reset_index(drop=True)
                st.dataframe(view, use_container_width=True)
                st.download_button(
                    "Download this cycle’s source slice (CSV)",
                    view.to_csv(index=False).encode("utf-8"),
                    file_name="cycle_source_slice.csv",
                    mime="text/csv"
                )

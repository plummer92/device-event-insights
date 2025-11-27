###############################################
# DEVICE EVENT INSIGHTS (NEON VERSION)
# Full Trend-Building Analytics Dashboard
# Python 3.12 / Streamlit Cloud
###############################################

import streamlit as st
import pandas as pd
import numpy as np
import hashlib
import psycopg2
from psycopg2.extras import execute_batch
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(page_title="Device Event Insights", layout="wide")

###########################################################
#                 DATABASE CONNECTION
###########################################################
DB_URL = st.secrets["neon"]["db_url"]

def get_conn():
    return psycopg2.connect(DB_URL)


###########################################################
#                 CLEANING FUNCTIONS
###########################################################
def generate_pk(row):
    """Stable unique hash for deduplication."""
    return hashlib.sha256("|".join(str(v) for v in row.values).encode()).hexdigest()


def clean_dataframe(df):
    """Normalize uploaded file → final schema that matches DB."""
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower()

    colmap = {
        "username": "user_name",
        "user": "user_name",
        "employee": "user_name",

        "device": "device",

        "medid": "med_id",
        "med id": "med_id",

        "description": "med_desc",
        "desc": "med_desc",

        "type": "event_type",
        "transaction type": "event_type",

        "datetime": "dt",
        "transaction datetime": "dt",
        "transaction date and time": "dt",

        "quantity": "qty",
        "qty": "qty",

        "beg": "beginning_qty",
        "beginning": "beginning_qty",
        "begin": "beginning_qty",

        "end": "ending_qty",
        "ending": "ending_qty"
    }

    df = df.rename(columns=colmap)

    required = [
        "user_name", "device", "med_id", "med_desc",
        "event_type", "dt", "qty", "beginning_qty", "ending_qty"
    ]

    for c in required:
        if c not in df.columns:
            df[c] = None

    # Datetime handling
    df["dt"] = pd.to_datetime(df["dt"], errors="coerce")
    df["dt"] = df["dt"].astype(str).where(df["dt"].notna(), None)

    # Numeric cleanup
    for c in ["qty", "beginning_qty", "ending_qty"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.where(pd.notna(df), None)
    df["pk"] = df.apply(lambda r: generate_pk(r), axis=1)

    return df


###########################################################
#                 INSERT INTO DATABASE
###########################################################
def insert_batch(df):
    conn = get_conn()
    cur = conn.cursor()

    sql = """
        INSERT INTO events (
            pk, user_name, device, med_id, med_desc,
            event_type, dt, qty, beginning_qty, ending_qty
        )
        VALUES (
            %(pk)s, %(user_name)s, %(device)s, %(med_id)s, %(med_desc)s,
            %(event_type)s, %(dt)s, %(qty)s, %(beginning_qty)s, %(ending_qty)s
        )
        ON CONFLICT (pk) DO NOTHING;
    """

    rows = df.to_dict("records")
    for i in range(0, len(rows), 5000):
        execute_batch(cur, sql, rows[i:i + 5000], page_size=len(rows[i:i + 5000]))
        conn.commit()

    cur.close()
    conn.close()


###########################################################
#       PROCESSING ENGINE: TIME, WALK, REFILL PAIRS
###########################################################

def normalize_event_type(raw):
    if raw is None:
        return None

    r = str(raw).strip().lower()
    if "cancel" in r: return "cancelled"
    if "verify" in r: return "verify"
    if "refill" in r: return "refill"
    if "load" in r: return "load"
    if "unload" in r: return "unload"
    if "outdate" in r: return "outdate"
    return r


def prepare_events(df):
    df = df.copy()
    df.columns = df.columns.str.lower()

    required = [
        "dt","user_name","device","event_type",
        "med_id","med_desc","qty","beginning_qty","ending_qty","pk"
    ]
    for c in required:
        if c not in df.columns:
            df[c] = None

    df["dt"] = pd.to_datetime(df["dt"], errors="coerce")
    df["event_type_clean"] = df["event_type"].apply(normalize_event_type)
    df = df.sort_values(["user_name", "dt"]).reset_index(drop=True)

    return df


def compute_time_deltas(df):
    df = df.copy()
    df["next_dt"] = df.groupby("user_name")["dt"].shift(-1)
    df["next_device"] = df.groupby("user_name")["device"].shift(-1)

    df["time_delta_sec"] = (df["next_dt"] - df["dt"]).dt.total_seconds()
    df["time_delta_sec"] = df["time_delta_sec"].fillna(0)
    df.loc[df["time_delta_sec"] < 0, "time_delta_sec"] = 0

    return df


IDLE_CUTOFF_SEC = 10 * 60  # 10 minutes

def compute_dwell_walk(df):
    df = df.copy()
    df["dwell_sec"] = 0
    df["walk_sec"] = 0

    mask_valid = df["event_type_clean"] != "cancelled"

    for idx in df.index:
        if not mask_valid[idx]:
            continue

        delta = df.at[idx, "time_delta_sec"]
        if delta <= 0:
            continue

        if df.at[idx, "device"] == df.at[idx, "next_device"]:
            if delta <= IDLE_CUTOFF_SEC:
                df.at[idx, "dwell_sec"] = delta
        else:
            df.at[idx, "walk_sec"] = delta

    return df


def compute_refill_pairs(df):
    df = df.copy()
    df["refill_pair_id"] = None
    df["refill_pair_sec"] = 0
    df["is_refill_primary"] = False

    counter = 1

    for user, grp in df.groupby("user_name"):
        grp_idx = grp.index

        last_verify_idx = None
        last_verify_dt = None

        for idx in grp_idx:
            etype = df.at[idx, "event_type_clean"]

            if etype == "verify":
                last_verify_idx = idx
                last_verify_dt = df.at[idx, "dt"]

            elif etype == "refill" and last_verify_idx is not None:
                refill_dt = df.at[idx, "dt"]

                pair_id = f"{user}_{counter}"
                counter += 1

                df.at[last_verify_idx, "refill_pair_id"] = pair_id
                df.at[idx, "refill_pair_id"] = pair_id

                sec = (refill_dt - last_verify_dt).total_seconds()
                sec = max(sec, 0)
                df.at[idx, "refill_pair_sec"] = sec
                df.at[idx, "is_refill_primary"] = True

                last_verify_idx = None
                last_verify_dt = None

    return df


def process_events(df_raw):
    df = prepare_events(df_raw)
    df = compute_time_deltas(df)
    df = compute_dwell_walk(df)
    df = compute_refill_pairs(df)

    df["total_machine_sec"] = df["dwell_sec"]
    df["total_walk_sec"] = df["walk_sec"]

    return df


###########################################################
#              LOAD FROM DATABASE (DATE FILTER)
###########################################################

def load_events_from_db(start_date, end_date):
    """Query Neon for events in date range."""
    conn = get_conn()
    cur = conn.cursor()

    sql = """
        SELECT pk, user_name, device, med_id, med_desc,
               event_type, dt, qty, beginning_qty, ending_qty
        FROM events
        WHERE dt::date BETWEEN %s AND %s
        ORDER BY user_name, dt;
    """
    cur.execute(sql, (start_date, end_date))
    rows = cur.fetchall()

    cols = [
        "pk","user_name","device","med_id","med_desc",
        "event_type","dt","qty","beginning_qty","ending_qty"
    ]

    df = pd.DataFrame(rows, columns=cols)
    df["dt"] = pd.to_datetime(df["dt"], errors="coerce")

    cur.close()
    conn.close()

    if df.empty:
        return df

    return process_events(df)


###########################################################
#                    SIDEBAR UI
###########################################################

st.sidebar.header("📤 Upload & Database Controls")

uploaded = st.sidebar.file_uploader("Upload Device Event Report", type=["csv","xlsx"])

if uploaded:
    if uploaded.name.endswith(".xlsx"):
        df_raw = pd.read_excel(uploaded)
    else:
        df_raw = pd.read_csv(uploaded)

    df_clean = clean_dataframe(df_raw)
    st.sidebar.success(f"Loaded {len(df_clean):,} rows.")

    if st.sidebar.button("💾 Save to Neon Database"):
        with st.spinner("Saving to Neon…"):
            insert_batch(df_clean)
        st.sidebar.success("Saved to DB!")

# Date Range Filter
st.sidebar.markdown("---")
st.sidebar.subheader("📅 Date Range")

default_end = datetime.today().date()
default_start = default_end - timedelta(days=7)

start_date = st.sidebar.date_input("Start Date", default_start)
end_date = st.sidebar.date_input("End Date", default_end)

load_btn = st.sidebar.button("📥 Load Data From Database")


###########################################################
#                 ALWAYS-VISIBLE TABS
###########################################################
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 Overview",
    "💊 Refill Efficiency",
    "🧑‍🔧 Technician Metrics",
    "🏥 Device Metrics",
    "🚶 Walking Time",
    "🧱 Machine Dwell Time",
    "📋 Raw Events"
])


###########################################################
# If no data loaded → show message in all tabs
###########################################################
if not load_btn:
    with tab1: st.info("Upload a file or load data from DB to begin.")
    with tab2: st.info("Upload a file or load data from DB to begin.")
    with tab3: st.info("Upload a file or load data from DB to begin.")
    with tab4: st.info("Upload a file or load data from DB to begin.")
    with tab5: st.info("Upload a file or load data from DB to begin.")
    with tab6: st.info("Upload a file or load data from DB to begin.")
    with tab7: st.info("Upload a file or load data from DB to begin.")
    st.stop()


###########################################################
#          LOAD & PROCESS DATA FOR ANALYTICS
###########################################################
df = load_events_from_db(start_date, end_date)

if df.empty:
    with tab1: st.warning("No data found for this date range.")
    with tab2: st.warning("No data found for this date range.")
    with tab3: st.warning("No data found for this date range.")
    with tab4: st.warning("No data found for this date range.")
    with tab5: st.warning("No data found for this date range.")
    with tab6: st.warning("No data found for this date range.")
    with tab7: st.warning("No data found for this date range.")
    st.stop()


###########################################################
#                     TAB 1: OVERVIEW
###########################################################
with tab1:
    st.header("📊 Overview Dashboard")

    total_events = len(df)
    total_refills = df["is_refill_primary"].sum()
    avg_walk = df["walk_sec"].mean()
    avg_dwell = df["dwell_sec"].mean()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Events", f"{total_events:,}")
    c2.metric("Refills", f"{total_refills:,}")
    c3.metric("Avg Walking", f"{avg_walk:.1f} sec")
    c4.metric("Avg Dwell", f"{avg_dwell:.1f} sec")

    st.markdown("---")
    st.subheader("Event Volume by Event Type")

    vol = df.groupby("event_type_clean").size().reset_index(name="count")
    fig = px.bar(vol, x="event_type_clean", y="count", title="Event Counts by Type")
    st.plotly_chart(fig, use_container_width=True)


###########################################################
#                 TAB 2: REFILL EFFICIENCY
###########################################################
with tab2:
    st.header("💊 Refill Efficiency")

    df_ref = df[df["is_refill_primary"] == True]

    if df_ref.empty:
        st.info("No refill events found.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Refills", f"{len(df_ref):,}")
        c2.metric("Avg Refill Time", f"{df_ref['refill_pair_sec'].mean():.1f} sec")
        c3.metric("Median Refill", f"{df_ref['refill_pair_sec'].median():.1f} sec")
        c4.metric("Max Refill", f"{df_ref['refill_pair_sec'].max():.1f} sec")

        st.markdown("### Refill Time by Technician")
        tech = (
            df_ref.groupby("user_name")["refill_pair_sec"]
            .mean().reset_index().sort_values("refill_pair_sec")
        )
        fig = px.bar(tech, x="user_name", y="refill_pair_sec", title="Avg Refill Time per Tech")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.subheader("📋 All Refill Events")
        st.dataframe(df_ref, hide_index=True, use_container_width=True)


###########################################################
#              TAB 3: TECHNICIAN METRICS
###########################################################
with tab3:
    st.header("🧑‍🔧 Technician Metrics")

    tech = (
        df.groupby("user_name")
        .agg(
            events=("pk","count"),
            walk_sec=("walk_sec","sum"),
            dwell_sec=("dwell_sec","sum"),
            refills=("is_refill_primary","sum")
        )
        .reset_index()
    )

    st.dataframe(tech, use_container_width=True)

    fig = px.bar(tech, x="user_name", y="events", title="Total Events per Technician")
    st.plotly_chart(fig, use_container_width=True)


###########################################################
#              TAB 4: DEVICE METRICS
###########################################################
with tab4:
    st.header("🏥 Device Metrics")

    dev = (
        df.groupby("device")
        .agg(
            events=("pk","count"),
            avg_dwell=("dwell_sec","mean"),
            avg_walk=("walk_sec","mean")
        ).reset_index()
    )

    st.dataframe(dev, use_container_width=True)

    fig = px.bar(dev, x="device", y="avg_dwell", title="Average Dwell Time per Device")
    st.plotly_chart(fig, use_container_width=True)


###########################################################
#              TAB 5: WALKING TIME
###########################################################
with tab5:
    st.header("🚶 Walking Time Analytics")

    fig = px.histogram(df, x="total_walk_sec", nbins=40, title="Distribution of Walking Times")
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(df[["dt","user_name","device","total_walk_sec"]], hide_index=True)


###########################################################
#              TAB 6: MACHINE DWELL TIME
###########################################################
with tab6:
    st.header("🧱 Machine Dwell Times")

    fig = px.histogram(df, x="total_machine_sec", nbins=40, title="Distribution of Machine Dwell")
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(df[["dt","user_name","device","total_machine_sec"]], hide_index=True)


###########################################################
#              TAB 7: RAW EVENT BROWSER
###########################################################
with tab7:
    st.header("📋 Raw Events")

    st.dataframe(df, hide_index=True, use_container_width=True)

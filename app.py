###############################################
# DEVICE EVENT INSIGHTS (NEON PRO VERSION)
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

# Page Config
st.set_page_config(
    page_title="RxTrack: Device Analytics", 
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for "Leadership" look
st.markdown("""
    <style>
    .big-font { font-size:24px !important; font-weight: bold; }
    .metric-card { background-color: #f0f2f6; padding: 15px; border-radius: 10px; border: 1px solid #e0e0e0; }
    </style>
    """, unsafe_allow_html=True)

###########################################################
#                 DATABASE CONNECTION
###########################################################
# Uses st.cache_resource to avoid reconnecting on every rerun
@st.cache_resource
def get_db_connection():
    try:
        return psycopg2.connect(st.secrets["neon"]["db_url"])
    except Exception as e:
        st.error(f"❌ Connection to Neon DB failed: {e}")
        return None

###########################################################
#                 CLEANING FUNCTIONS
###########################################################
def generate_pk(row):
    """Stable unique hash for deduplication."""
    # Convert entire row to string, handle NaNs
    row_str = "|".join(str(v) for v in row.values)
    return hashlib.sha256(row_str.encode()).hexdigest()

def clean_dataframe(df):
    df = df.copy()
    
    # Normalize columns
    df.columns = df.columns.str.strip().str.lower()
    
    colmap = {
        "username": "user_name", "user": "user_name", "employee": "user_name",
        "device": "device",
        "medid": "med_id", "med id": "med_id",
        "meddescription": "med_desc", "med description": "med_desc", "description": "med_desc", "desc": "med_desc",
        "transactiontype": "event_type", "type": "event_type",
        "transactiondatetime": "dt", "datetime": "dt", "timestamp": "dt",
        "quantity": "qty", "qty": "qty",
        "beg": "beginning_qty", "beginning": "beginning_qty",
        "end": "ending_qty",
    }
    
    df = df.rename(columns=colmap)
    
    required_cols = [
        "user_name", "device", "med_id", "med_desc", 
        "event_type", "dt", "qty", 
        "beginning_qty", "ending_qty"
    ]
    
    for col in required_cols:
        if col not in df.columns:
            df[col] = None

    # Parse dt
    df["dt"] = pd.to_datetime(df["dt"], errors="coerce")
    
    # Remove rows with no date
    df = df.dropna(subset=["dt"])

    # Numeric conversion
    for c in ["qty", "beginning_qty", "ending_qty"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    # Convert to appropriate types for SQL
    df["dt"] = df["dt"].astype(str) # Send as string to PG, let driver handle
    df["pk"] = df.apply(lambda r: generate_pk(r), axis=1)
    
    return df

###########################################################
#                 INSERT INTO DATABASE
###########################################################
def insert_batch(df):
    conn = get_db_connection()
    if not conn: return
    
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
    
    progress_bar = st.progress(0)
    total_rows = len(rows)
    batch_size = 2000
    
    try:
        for i in range(0, total_rows, batch_size):
            batch = rows[i:i + batch_size]
            execute_batch(cur, sql, batch, page_size=len(batch))
            conn.commit()
            progress_bar.progress(min((i + batch_size) / total_rows, 1.0))
        st.success(f"✅ Successfully processed {total_rows} records.")
    except Exception as e:
        st.error(f"Error inserting data: {e}")
        conn.rollback()
    finally:
        cur.close()
        # Do not close conn here as it is cached

###########################################################
#        LOGIC ENGINE (Cached for Performance)
###########################################################

def normalize_event_type(raw):
    if raw is None: return None
    r = str(raw).strip().lower()
    if "cancel" in r: return "cancelled"
    if "verify" in r: return "verify"
    if "refill" in r: return "refill"
    if "load" in r: return "load"
    if "unload" in r: return "unload"
    if "outdate" in r: return "outdate"
    return r

@st.cache_data(show_spinner=False)
def process_analytics_logic(df):
    """
    Runs the heavy lifting of walk times and refill pairing.
    Cached so it doesn't rerun on simple UI interaction.
    """
    df = df.copy()
    
    # 1. Sort
    df["dt"] = pd.to_datetime(df["dt"])
    df["event_type_clean"] = df["event_type"].apply(normalize_event_type)
    df = df.sort_values(["user_name", "dt"]).reset_index(drop=True)

    # 2. Time Deltas & Walking
    df["next_dt"] = df.groupby("user_name")["dt"].shift(-1)
    df["next_device"] = df.groupby("user_name")["device"].shift(-1)
    df["time_delta_sec"] = (df["next_dt"] - df["dt"]).dt.total_seconds().fillna(0).clip(lower=0)

    # Logic: Dwell vs Walk
    IDLE_CUTOFF_SEC = 600 # 10 minutes
    
    # Vectorized approach for speed
    mask_same_device = (df["device"] == df["next_device"])
    mask_valid_time = (df["time_delta_sec"] <= IDLE_CUTOFF_SEC) & (df["time_delta_sec"] > 0)
    
    df["dwell_sec"] = np.where(mask_same_device & mask_valid_time, df["time_delta_sec"], 0)
    df["walk_sec"] = np.where((~mask_same_device) & mask_valid_time, df["time_delta_sec"], 0)

    # 3. Refill Pairs (Complex logic, keep iterative but optimized)
    # Using a list accumulator is faster than updating DataFrame cell-by-cell
    
    refill_indices = []
    
    for user, grp in df.groupby("user_name"):
        last_verify_idx = None
        last_verify_dt = None
        
        # Iterate over tuples for speed
        for row in grp.itertuples():
            etype = row.event_type_clean
            
            if etype == "verify":
                last_verify_idx = row.Index
                last_verify_dt = row.dt
            elif etype == "refill" and last_verify_idx is not None:
                sec = (row.dt - last_verify_dt).total_seconds()
                if sec >= 0:
                    refill_indices.append({
                        "verify_idx": last_verify_idx,
                        "refill_idx": row.Index,
                        "sec": sec,
                        "user": user
                    })
                last_verify_idx = None # Reset
    
    # Apply Refill Data
    df["is_refill_primary"] = False
    df["refill_pair_sec"] = 0.0
    
    if refill_indices:
        refill_meta = pd.DataFrame(refill_indices)
        df.loc[refill_meta["refill_idx"], "is_refill_primary"] = True
        df.loc[refill_meta["refill_idx"], "refill_pair_sec"] = refill_meta["sec"].values

    return df

###########################################################
#               LOAD FROM DB
###########################################################

@st.cache_data(ttl=300) # Cache for 5 mins
def load_data(start_date, end_date):
    conn = get_db_connection()
    if not conn: return pd.DataFrame()
    
    query = """
        SELECT pk, user_name, device, med_id, med_desc, 
               event_type, dt, qty, beginning_qty, ending_qty
        FROM events 
        WHERE dt::date BETWEEN %s AND %s
    """
    
    try:
        df = pd.read_sql(query, conn, params=(start_date, end_date))
        if df.empty: return df
        return process_analytics_logic(df)
    except Exception as e:
        st.error(f"Database Read Error: {e}")
        return pd.DataFrame()

###########################################################
#               UI & DASHBOARD
###########################################################

# SIDEBAR
# --- SIDEBAR COMMAND CENTER ---
with st.sidebar:
    st.image("https://img.icons8.com/color/96/medical-history.png", width=50)
    st.title("RxTrack Admin")
    
    # 1. Persistent Date Logic (The "Anchor")
    # This ensures dates stay locked when switching to the Tardies or Student pages
    n_events, n_pharm, n_sched, n_att, min_db, max_db = get_stats_range()
    
    if 'start_date' not in st.session_state:
        st.session_state.start_date = max_db - timedelta(days=7)
    if 'end_date' not in st.session_state:
        st.session_state.end_date = max_db

    st.markdown("### 📅 Time Filter")
    d_col1, d_col2 = st.columns(2)
    start_date = d_col1.date_input("Start", st.session_state.start_date)
    end_date = d_col2.date_input("End", st.session_state.end_date)
    
    # Update the "shared clipboard" so sub-pages can see these dates
    st.session_state.start_date = start_date
    st.session_state.end_date = end_date

    st.markdown("### ⚙️ Settings")
    hourly_rate = st.number_input("Tech Avg Hourly Rate ($)", value=22.50, step=0.50)
    
    st.markdown("---")
    
    # 2. Universal Data Ingest
    # Keeping this in the sidebar allows you to upload data from ANY page
    st.markdown("### 📤 Data Upload")
    u_type = st.selectbox("Report Type:", [
        "Daily Transaction Report", "Staff Schedule", "Attendance Tracking",
        "Pharmacy Workflow Report", "Inventory Audit (Prices)"
    ])
    uploaded = st.file_uploader(f"Upload {u_type}", type=["csv","xlsx"])
    
    if uploaded:
        if st.button(f"Process & Staging"):
            try:
                # Reuse your existing cleaning functions based on selection
                if u_type == "Daily Transaction Report":
                    df_raw = pd.read_excel(uploaded) if uploaded.name.endswith(".xlsx") else pd.read_csv(uploaded)
                    df_clean = clean_dataframe(df_raw)
                    # SQL Insert logic here...
                
                st.success(f"✅ {u_type} processed!")
                st.cache_data.clear() # Refresh all pages with new data
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")
# MAIN LOADING
df = load_data(start_date, end_date)

if df.empty:
    st.info("👋 Welcome to RxTrack. No data found for selected dates. Please upload a report or adjust dates.")
    st.stop()

# --- HEADER STATS ---
st.markdown(f"## Executive Summary ({start_date} to {end_date})")

total_walk_hrs = df["walk_sec"].sum() / 3600
estimated_loss = total_walk_hrs * hourly_rate
avg_refill_time = df[df["is_refill_primary"]]["refill_pair_sec"].mean()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Events", f"{len(df):,}", delta="Volume")
col2.metric("Refill Efficiency", f"{avg_refill_time:.1f} sec", delta_color="inverse", help="Time between Verify and Refill")
col3.metric("Walking Time (Hours)", f"{total_walk_hrs:.1f} hrs", delta_color="inverse")
col4.metric("Est. Walking Cost", f"${estimated_loss:,.2f}", delta_color="inverse", help="Labor cost based on walk time")

st.markdown("---")

# --- TABS ---
tab_leads, tab_tech, tab_device, tab_data = st.tabs([
    "📈 Leadership View", "🧑‍⚕️ Technician Deep Dive", "🏥 Device Utilization", "💾 Data Explorer"
])

with tab_leads:
    st.markdown("### 📉 Labor Efficiency Trends")
    
    # Aggregate by Day
    daily = df.groupby(df['dt'].dt.date).agg(
        total_walk=('walk_sec', 'sum'),
        total_dwell=('dwell_sec', 'sum'),
        total_events=('pk', 'count')
    ).reset_index()
    
    daily['walk_hours'] = daily['total_walk'] / 3600
    
    fig_trend = px.area(daily, x='dt', y='walk_hours', 
                        title="Daily Lost Labor Hours (Walking)",
                        labels={'walk_hours': 'Hours Walking', 'dt': 'Date'},
                        color_discrete_sequence=['#FF4B4B'])
    st.plotly_chart(fig_trend, use_container_width=True)
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Event Type Breakdown")
        vol = df['event_type_clean'].value_counts().reset_index()
        vol.columns = ['Type', 'Count']
        fig_pie = px.pie(vol, names='Type', values='Count', hole=0.4)
        st.plotly_chart(fig_pie, use_container_width=True)
        
    with c2:
        st.markdown("#### Top 5 High-Traffic Devices")
        top_dev = df['device'].value_counts().head(5).reset_index()
        top_dev.columns = ['Device', 'Events']
        fig_bar = px.bar(top_dev, x='Events', y='Device', orientation='h')
        st.plotly_chart(fig_bar, use_container_width=True)

with tab_tech:
    st.markdown("### Staff Performance & Outliers")
    
    tech_stats = df.groupby('user_name').agg(
        events=('pk', 'count'),
        avg_refill_speed=('refill_pair_sec', lambda x: x[x>0].mean() if len(x[x>0]) > 0 else 0),
        total_walk_sec=('walk_sec', 'sum')
    ).reset_index()
    
    tech_stats['walk_hours'] = tech_stats['total_walk_sec'] / 3600
    
    # Scatter plot: Speed vs Volume
    fig_scatter = px.scatter(
        tech_stats, 
        x='events', 
        y='avg_refill_speed',
        size='walk_hours',
        hover_name='user_name',
        title="Technician Efficiency Matrix (Size = Walk Time)",
        labels={'events': 'Total Activity Volume', 'avg_refill_speed': 'Avg Refill Time (Sec)'},
        color='walk_hours',
        color_continuous_scale='Viridis'
    )
    # Add lines for averages
    fig_scatter.add_hline(y=tech_stats['avg_refill_speed'].mean(), line_dash="dash", annotation_text="Avg Speed")
    fig_scatter.add_vline(x=tech_stats['events'].mean(), line_dash="dash", annotation_text="Avg Volume")
    
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    st.dataframe(tech_stats.sort_values('events', ascending=False), use_container_width=True)

with tab_device:
    st.markdown("### Device Load Balancing")
    
    dev_stats = df.groupby('device').agg(
        utilization_min=('dwell_sec', lambda x: x.sum()/60),
        unique_users=('user_name', 'nunique')
    ).reset_index()
    
    fig_dev = px.bar(dev_stats.sort_values('utilization_min', ascending=False).head(15),
                     x='device', y='utilization_min',
                     color='unique_users',
                     title="Top 15 Devices by Active Utilization Time (Minutes)")
    st.plotly_chart(fig_dev, use_container_width=True)

with tab_data:
    st.markdown("### Raw Event Data")
    st.dataframe(
        df[["dt", "user_name", "device", "event_type", "med_desc", "qty", "refill_pair_sec"]].sort_values("dt", ascending=False),
        use_container_width=True
    )

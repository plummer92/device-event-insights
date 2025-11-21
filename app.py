# ============================================================
# Device Event Insights — Pro (FULL PATCHED VERSION)
# ============================================================

from __future__ import annotations
import streamlit as st
import pandas as pd
import numpy as np
import hashlib
import io
import json
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text

st.set_page_config(
    page_title="Device Event Insights — Pro",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------
# CONFIG
# ----------------------------------------
DEFAULT_COLMAP = {
    "datetime": "TransactionDateTime",
    "device": "Device",
    "user": "UserName",
    "type": "TransactionType",
    "desc": "MedDescription",
    "qty": "Quantity",
    "medid": "MedID",
}

# IMPORTANT: canonical field (our internal name) → CSV header name
colmap = DEFAULT_COLMAP.copy()

DEFAULT_IDLE_MIN = 30

# Pull DB URL from Streamlit Secrets
DB_URL = st.secrets["DB_URL"]
ENGINE_SALT = st.secrets.get("ENGINE_SALT", "v1")

# Build engine
eng = create_engine(DB_URL, pool_pre_ping=True, connect_args={"sslmode": "require"})

# ============================================================
# DATABASE INITIALIZATION (DDL)
# ============================================================

def init_db(eng):
    ddl = """
    -----------------------------------------------------------
    -- MAIN EVENTS TABLE (Pyxis Devices)
    -----------------------------------------------------------
    CREATE TABLE IF NOT EXISTS events (
        pk     TEXT PRIMARY KEY,
        dt     TIMESTAMP WITHOUT TIME ZONE NOT NULL,
        device TEXT,
        "user" TEXT,
        "type" TEXT,
        description TEXT,
        qty    DOUBLE PRECISION,
        medid  TEXT
    );

    -----------------------------------------------------------
    -- PYXIS ACTIVITY SIMPLE (Min/Max)
    -----------------------------------------------------------
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

    -----------------------------------------------------------
    -- PYXIS PENDED LOADS
    -----------------------------------------------------------
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

    -----------------------------------------------------------
    -- THRESHOLDS / STANDARD STOCK OVERRIDES
    -----------------------------------------------------------
    CREATE TABLE IF NOT EXISTS pyxis_thresholds (
        med_id   TEXT NOT NULL,
        device   TEXT NOT NULL DEFAULT '*',
        drawer   TEXT NOT NULL DEFAULT '',
        pocket   TEXT NOT NULL DEFAULT '',
        min_qty  INTEGER,
        max_qty  INTEGER,
        CONSTRAINT pyxis_thresholds_pk PRIMARY KEY (med_id, device, drawer, pocket)
    );

    -----------------------------------------------------------
    -- CAROUSEL EVENTS (NEW TABLE)
    -----------------------------------------------------------
    CREATE TABLE IF NOT EXISTS carousel_events (
        pk         TEXT PRIMARY KEY,
        ts         TIMESTAMP,
        queue_id   TEXT,
        priority   TEXT,
        medid      TEXT,
        description TEXT,
        dest       TEXT,
        user_name  TEXT,
        qty        NUMERIC,
        raw_row    JSONB
    );
    """

    with eng.begin() as con:
        con.execute(text(ddl))

# ALWAYS run initialization
init_db(eng)

# ============================================================
# SECTION 3 — Utility Functions (Normalization + PK Builders)
# ============================================================

import hashlib
import numpy as np

# -------------------------------
# Replace pandas NA/NaN with None
# -------------------------------
def clean_na(df):
    return df.where(pd.notnull(df), None)


# -------------------------------
# Safe datetime parser
# -------------------------------
def safe_ts(x):
    try:
        return pd.to_datetime(x, errors="coerce")
    except:
        return None


# ------------------------------------------------------------
# BUILD PK FOR PYXIS EVENTS
# Uses FULL uniqueness (dt, dev, user, type, description, qty, medid)
# ------------------------------------------------------------
def build_pk_pyxis(row, colmap):
    parts = [
        str(row.get(colmap["datetime"], "")),
        str(row.get(colmap["device"], "")),
        str(row.get(colmap["user"], "")),
        str(row.get(colmap["type"], "")),
        str(row.get(colmap.get("desc", ""), "")),
        str(row.get(colmap.get("qty", ""), "")),
        str(row.get(colmap.get("medid", ""), "")),
    ]
    return hashlib.sha1("_".join(parts).encode("utf-8")).hexdigest()


# ------------------------------------------------------------
# BUILD PK FOR CAROUSEL EVENTS
# Uses full row uniqueness
# ------------------------------------------------------------
def build_pk_carousel(r):
    parts = [
        str(r.get("Date / Time", "")),
        str(r.get("TranQueueID", "")),
        str(r.get("Item ID", "")),
        str(r.get("Quantity", "")),
        str(r.get("User", "")),
        str(r.get("Priority", "")),
        str(r.get("Description", "")),
        str(r.get("Destination", "")),
    ]
    return hashlib.sha1("_".join(parts).encode("utf-8")).hexdigest()


# ------------------------------------------------------------
# Convert a Pyxis DataFrame → canonical rows for DB events table
# ------------------------------------------------------------
def df_to_pyxis_rows(df: pd.DataFrame, colmap: Dict[str, str]):
    df = clean_na(df)

    ts = colmap["datetime"]
    dev = colmap["device"]
    usr = colmap["user"]
    typ = colmap["type"]

    rows = []
    for _, r in df.iterrows():
        dt_val = safe_ts(r.get(ts))

        pk_val = build_pk_pyxis(r, colmap)

        rows.append({
            "pk": pk_val,
            "dt": dt_val.to_pydatetime() if dt_val is not None and not pd.isna(dt_val) else None,
            "device": r.get(dev),
            "user": r.get(usr),
            "type": r.get(typ),
            "description": r.get(colmap.get("desc", ""), None),
            "qty": float(r.get(colmap.get("qty", ""), None)) if r.get(colmap.get("qty", "")) not in (None, "", np.nan) else None,
            "medid": r.get(colmap.get("medid", ""), None),
        })
    return rows


# ------------------------------------------------------------
# Convert Carousel DF → canonical rows for DB carousel_events table
# ------------------------------------------------------------
def df_to_carousel_rows(df: pd.DataFrame):
    df = clean_na(df)

    rows = []
    for _, r in df.iterrows():
        ts_val = safe_ts(r.get("Date / Time"))
        pk_val = build_pk_carousel(r)

        rows.append({
            "pk": pk_val,
            "ts": ts_val.to_pydatetime() if ts_val is not None and not pd.isna(ts_val) else None,
            "queue_id": r.get("TranQueueID"),
            "priority": r.get("Priority"),
            "medid": r.get("Item ID"),
            "description": r.get("Description"),
            "dest": r.get("Destination"),
            "user_name": r.get("User"),
            "qty": r.get("Quantity"),
            "raw_row": json.dumps(r.to_dict())
        })

    return rows


# ------------------------------------------------------------
# Read CSV or Excel file (auto-detect)
# ------------------------------------------------------------
def load_any_file(upload):
    name = upload.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(upload)
    elif name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(upload)
    else:
        st.error(f"Unsupported file type: {upload.name}")
        return None


# ------------------------------------------------------------
# Normalize Pyxis column names using user's mapping
# ------------------------------------------------------------
def normalize_columns(df, colmap):
    rename_map = {}
    for k, v in colmap.items():
        if v in df.columns:
            rename_map[v] = k
    return df.rename(columns=rename_map)

# ============================================================
# SECTION 4 — PYXIS UPLOAD PIPELINE
# ============================================================

def process_uploaded_file(upload, colmap):
    """
    Loads a CSV/XLSX, normalizes columns via colmap,
    computes PKs, returns canonical rows for `events` table.
    """

    df = load_any_file(upload)
    if df is None or df.empty:
        st.warning(f"⚠ File {upload.name} is empty or invalid.")
        return []

    # Normalize user-facing column names → our canonical names
    df_norm = normalize_columns(df, colmap)

    # Build PKs + canonical DB rows
    rows = df_to_pyxis_rows(df_norm, colmap)
    return rows


def save_events_to_db(engine, rows: list[dict]):
    """
    Inserts many Pyxis rows (events) with UPSERT.
    All NA/None issues are already resolved in df_to_pyxis_rows().
    """

    if not rows:
        return 0

    sql = text("""
        INSERT INTO events
          (pk, dt, device, "user", "type", "desc", qty, medid)
        VALUES
          (:pk, :dt, :device, :user, :type, :description, :qty, :medid)
        ON CONFLICT (pk) DO UPDATE SET
          dt = EXCLUDED.dt,
          device = EXCLUDED.device,
          "user" = EXCLUDED."user",
          "type" = EXCLUDED."type",
          "desc" = EXCLUDED."desc",
          qty = EXCLUDED.qty,
          medid = EXCLUDED.medid;
    """)

    with engine.begin() as con:
        con.execute(text("SET LOCAL statement_timeout = '120s'"))
        con.execute(sql, rows)

    return len(rows)


# ------------------------------------------------------------
# PYXIS PENDED ITEMS UPLOAD (Tab 12)
# ------------------------------------------------------------
def upload_pended_items(upload):
    df = load_any_file(upload)
    if df is None or df.empty:
        st.error("Uploaded file is empty.")
        return 0

    df = clean_na(df)

    # Required columns
    req = ["Timestamp", "Device", "Med ID", "Drawer", "Pocket"]
    for c in req:
        if c not in df.columns:
            st.error(f"Missing required column: {c}")
            return 0

    rows = []
    for _, r in df.iterrows():
        ts = safe_ts(r.get("Timestamp"))
        pk_parts = [
            str(ts),
            str(r.get("Device")),
            str(r.get("Med ID")),
            str(r.get("Drawer")),
            str(r.get("Pocket")),
        ]
        pk = hashlib.sha1("_".join(pk_parts).encode("utf-8")).hexdigest()

        rows.append({
            "ts": ts.to_pydatetime() if ts and not pd.isna(ts) else None,
            "device": r.get("Device"),
            "med_id": r.get("Med ID"),
            "drawer": r.get("Drawer"),
            "pocket": r.get("Pocket"),
            "qty": r.get("Qty"),
            "min_qty": r.get("MinQty"),
            "max_qty": r.get("MaxQty"),
            "affected_element": r.get("Affected Element"),
            "dispensing_name": r.get("Dispensing Name"),
            "area": r.get("Area"),
            "username": r.get("User"),
            "userid": r.get("UserID"),
        })

    sql = text("""
        INSERT INTO pyxis_pends (
            ts, device, med_id, drawer, pocket, qty,
            min_qty, max_qty, affected_element, dispensing_name,
            area, username, userid
        )
        VALUES (
            :ts, :device, :med_id, :drawer, :pocket, :qty,
            :min_qty, :max_qty, :affected_element, :dispensing_name,
            :area, :username, :userid
        )
        ON CONFLICT (ts, device, med_id, drawer, pocket) DO NOTHING;
    """)

    with eng.begin() as con:
        con.execute(sql, rows)

    return len(rows)

# ============================================================
# SECTION 5 — CAROUSEL UPLOAD PIPELINE
# ============================================================

def save_carousel_to_db(engine, rows: list[dict]):
    """
    Inserts many Carousel rows with UPSERT.
    Assumes rows were produced by df_to_carousel_rows().
    """
    if not rows:
        return 0

    sql = text("""
        INSERT INTO carousel_events
          (pk, ts, queue_id, priority, medid, description, dest, user_name, qty, raw_row)
        VALUES
          (:pk, :ts, :queue_id, :priority, :medid, :description, :dest, :user_name, :qty, CAST(:raw_row AS JSONB))
        ON CONFLICT (pk) DO NOTHING;
    """)

    with engine.begin() as con:
        con.execute(text("SET LOCAL statement_timeout = '120s'"))
        con.execute(sql, rows)

    return len(rows)


def process_carousel_upload(upload):
    """
    Takes a CSV from the user → loads → cleans → builds PKs → returns DB-ready rows.
    """
    df = load_any_file(upload)
    if df is None or df.empty:
        st.error("Uploaded Carousel file is empty or invalid.")
        return []

    # Columns MUST match the raw report
    required = [
        "TranQueueID",
        "Priority",
        "Date / Time",
        "Item ID",
        "Description",
        "Destination",
        "User",
        "Quantity",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"❌ Carousel file missing required columns: {missing}")
        return []

    # Convert "Date / Time" to datetime
    df["Date / Time"] = pd.to_datetime(df["Date / Time"], errors="coerce")

    # Replace NA/NaN with None
    df = clean_na(df)

    # Build rows for DB
    rows = df_to_carousel_rows(df)
    return rows
# ============================================================
# SECTION 6 — CAROUSEL TAB (GOES BETWEEN TAB 2 AND TAB 3)
# ============================================================

with tab_carousel:
    st.header("🎠 Carousel Transactions")

    st.markdown("Upload the `TransactionDetailReport_*.csv` export from your Carousel system.")

    uploaded_carousel = st.file_uploader(
        "Upload Carousel CSV",
        type=["csv"],
        key="carousel_csv_uploader",
        help="This file must include TranQueueID, Priority, Date / Time, Item ID, Description, Destination, User, Quantity."
    )

    if uploaded_carousel:

        # 1) PROCESS FILE
        with st.spinner("Processing Carousel CSV..."):
            rows = process_carousel_upload(uploaded_carousel)

        if not rows:
            st.error("No valid rows found for upload.")
        else:
            st.success(f"Loaded {len(rows):,} rows from CSV.")

            # Preview first few rows
            df_preview = pd.DataFrame(rows).drop(columns=["raw_row"], errors="ignore")
            st.markdown("### Preview (first 100 rows)")
            st.dataframe(df_preview.head(100), use_container_width=True, height=350)

            # 2) Save to database
            if st.button("📥 Insert Carousel Data Into Database"):
                try:
                    with st.spinner("Writing Carousel data to database..."):
                        inserted = save_carousel_to_db(eng, rows)

                    st.success(f"Successfully inserted {inserted:,} Carousel rows.")
                    st.cache_data.clear()
                    st.experimental_rerun()

                except Exception as e:
                    st.error(f"Carousel upload failed: {e}")

    # NOTE: You will later add analytics here if you want (heatmaps, user counts, etc.)
# ============================================================
# SECTION 7 — CAROUSEL ANALYTICS (inside tab_carousel)
# ============================================================

st.markdown("---")
st.subheader("📊 Carousel Analytics")

# Try loading carousel data
try:
    df_car = pd.read_sql("SELECT * FROM carousel_events ORDER BY ts DESC", eng)

    if df_car.empty:
        st.info("No Carousel data found yet. Upload a CSV above to get started.")
    else:
        # Clean df for visuals
        df_car["ts"] = pd.to_datetime(df_car["ts"], errors="coerce")
        df_car["date"] = df_car["ts"].dt.date
        df_car["hour"] = df_car["ts"].dt.hour

        # ============================
        # FILTER BAR
        # ============================
        with st.expander("🔎 Filters"):
            c1, c2, c3, c4 = st.columns(4)

            # Date Range
            min_ts = df_car["ts"].min().date()
            max_ts = df_car["ts"].max().date()
            date_range = c1.date_input(
                "Date range",
                (min_ts, max_ts),
                min_value=min_ts,
                max_value=max_ts,
            )

            # Priority filter
            priority_filter = c2.multiselect(
                "Priority",
                sorted(df_car["priority"].dropna().unique().tolist())
            )

            # User filter
            user_filter = c3.multiselect(
                "User",
                sorted(df_car["user_name"].dropna().unique().tolist())
            )

            # Destination filter
            dest_filter = c4.multiselect(
                "Destination",
                sorted(df_car["dest"].dropna().unique().tolist())
            )

        # Apply filters
        df_view = df_car.copy()

        # Date range filter
        if isinstance(date_range, tuple) and len(date_range) == 2:
            start, end = date_range
            df_view = df_view[(df_view["date"] >= start) & (df_view["date"] <= end)]

        if priority_filter:
            df_view = df_view[df_view["priority"].isin(priority_filter)]
        if user_filter:
            df_view = df_view[df_view["user_name"].isin(user_filter)]
        if dest_filter:
            df_view = df_view[df_view["dest"].isin(dest_filter)]

        # ============================
        # KPI CARDS
        # ============================
        st.markdown("### 📌 Key Metrics")
        k1, k2, k3, k4 = st.columns(4)

        k1.metric("Total Transactions", f"{len(df_view):,}")
        k2.metric("Med Count (unique)", df_view["medid"].nunique())
        k3.metric("Users Involved", df_view["user_name"].nunique())
        k4.metric("Avg Qty per Transaction", f"{df_view['qty'].mean():.2f}")

        # ============================
        # TOP MEDS
        # ============================
        st.markdown("### 🥇 Top Meds by Volume")
        top_meds = (
            df_view.groupby("description", dropna=True)["qty"]
            .sum()
            .sort_values(ascending=False)
            .head(20)
        )
        st.bar_chart(top_meds)

        # ============================
        # TOP USERS
        # ============================
        st.markdown("### 👤 Top Users by Transactions")
        top_users = (
            df_view.groupby("user_name", dropna=True)
            .size()
            .sort_values(ascending=False)
            .head(20)
        )
        st.bar_chart(top_users)

        # ============================
        # HOURLY HEATMAP
        # ============================
        st.markdown("### ⏱ Hourly Activity Heatmap")
        heat = (
            df_view.groupby(["date", "hour"]).size().unstack(fill_value=0)
        )

        st.dataframe(heat.style.background_gradient(cmap="Blues"), use_container_width=True)

        # ============================
        # PRIORITY BREAKDOWN
        # ============================
        st.markdown("### 🚨 Priority Breakdown")
        pr = df_view["priority"].value_counts()
        st.bar_chart(pr)

        # ============================
        # DESTINATION BREAKDOWN
        # ============================
        st.markdown("### 📦 Destination Breakdown")
        dest_counts = df_view["dest"].value_counts()
        st.bar_chart(dest_counts)

        # ============================
        # DOWNLOAD FILTERED CSV
        # ============================
        st.markdown("### 📥 Download Filtered Results")
        st.download_button(
            "Download CSV",
            df_view.to_csv(index=False).encode("utf-8"),
            file_name="carousel_filtered_export.csv",
            mime="text/csv",
        )

except Exception as e:
    st.error(f"Error loading Carousel analytics: {e}")

tab1, tab2, tab_carousel, tab_cross, tab_heatmap, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11, tab12 = st.tabs(
    [
        "📈 Overview",
        "🚶 Delivery Analytics",
        "🎠 Carousel",
        "🔗 Cross-System",
        "🗺 Heatmaps",
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
    ]
)

# ============================================================
# SECTION 8 — CROSS-SYSTEM ANALYTICS (Carousel + Pyxis)
# ============================================================

with tab_cross:
    st.header("🔗 Cross-System Analytics")
    st.markdown(
        "Analyze medication movement between **Carousel → Pyxis → Patient** "
        "to detect par-level issues, shortages, workload patterns, and routing problems."
    )

    # ------------------------------------------------------------
    # Load data safely
    # ------------------------------------------------------------
    try:
        df_ev = pd.read_sql("SELECT * FROM events", eng)
        df_car = pd.read_sql("SELECT * FROM carousel_events", eng)
    except Exception as e:
        st.error(f"Database load error: {e}")
        st.stop()

    if df_ev.empty or df_car.empty:
        st.info("Upload both Pyxis and Carousel data to activate cross-system analytics.")
        st.stop()

    # Normalize timestamps
    df_ev["dt"] = pd.to_datetime(df_ev["dt"], errors="coerce")
    df_car["ts"] = pd.to_datetime(df_car["ts"], errors="coerce")

    # ------------------------------------------------------------
    # 1. Identify Pyxis Refills that required Carousel Pulls
    # ------------------------------------------------------------
    st.subheader("📦 Carousel → Pyxis Refill Matching")

    # For a refill we look for:
    # same medid AND carousel pull within 0–8 hours BEFORE the refill
    df_refills = df_ev[df_ev["type"] == "Refill"].copy()

    df_merge = df_refills.merge(
        df_car,
        left_on="medid",
        right_on="medid",
        how="inner",
        suffixes=("_pyxis", "_car")
    )

    # time diff in hours
    df_merge["lag_hours"] = (df_merge["dt"] - df_merge["ts"]) / pd.Timedelta(hours=1)

    # keep only positive lags within window
    df_matched = df_merge[(df_merge["lag_hours"] >= 0) & (df_merge["lag_hours"] <= 8)]

    st.metric("Matched Carousel→Pyxis Refills", f"{len(df_matched):,}")

    if len(df_matched):
        st.dataframe(
            df_matched[[
                "dt", "device", "user", "qty_pyxis",
                "ts", "priority", "user_name", "qty_car",
                "medid", "lag_hours"
            ]].rename(columns={
                "dt": "pyxis_time",
                "ts": "carousel_time",
                "user": "pyxis_user",
                "user_name": "carousel_user",
                "qty_pyxis": "pyxis_qty",
                "qty_car": "carousel_qty",
            }),
            use_container_width=True,
            height=350
        )

        # Avg lag analysis
        st.markdown("### ⏱ Average Carousel→Pyxis Delay")
        st.metric("Avg Hours", f"{df_matched['lag_hours'].mean():.2f}")

    else:
        st.info("No Carousel pulls matched to Pyxis refills within 8 hours.")

    # ------------------------------------------------------------
    # 2. Top meds generating cross-system workload
    # ------------------------------------------------------------
    st.markdown("---")
    st.subheader("🥇 Meds With Highest Cross-System Traffic")

    top_cross = (
        df_matched.groupby("medid")
        .size()
        .sort_values(ascending=False)
        .head(15)
    )

    st.bar_chart(top_cross)

    # ------------------------------------------------------------
    # 3. User workload comparison
    # ------------------------------------------------------------
    st.markdown("---")
    st.subheader("👥 Cross-System User Workload")

    # Pyxis users
    pyxis_users = (
        df_matched.groupby("pyxis_user")
        .size()
        .sort_values(ascending=False)
        .head(10)
    )

    st.markdown("### 🧑 Pyxis Refills Triggered by Carousel Pulls")
    st.bar_chart(pyxis_users)

    # Carousel users
    car_users = (
        df_matched.groupby("carousel_user")
        .size()
        .sort_values(ascending=False)
        .head(10)
    )

    st.markdown("### 🎠 Carousel Picks Causing Downstream Pyxis Refills")
    st.bar_chart(car_users)

    # ------------------------------------------------------------
    # 4. Bounce-back meds (carousel → pyxis → carousel → pyxis)
    # ------------------------------------------------------------
    st.markdown("---")
    st.subheader("🔁 Bounce-Back Meds (High Instability)")

    # Any med appearing > X times in matching list
    bounce = top_cross[top_cross >= 8]  # threshold

    if len(bounce):
        st.warning("These meds bounce between systems (possible bad par level or usage pattern):")
        st.write(bounce)
    else:
        st.info("No unstable meds detected.")

    # ------------------------------------------------------------
    # 5. Download matched dataset
    # ------------------------------------------------------------
    st.markdown("---")
    st.subheader("📥 Download Full Cross-System Dataset")

    st.download_button(
        "Download Matched Carousel→Pyxis CSV",
        df_matched.to_csv(index=False).encode("utf-8"),
        file_name="cross_system_matched.csv",
        mime="text/csv"
    )

# ============================================================
# SECTION 9 — HEATMAP DASHBOARD
# ============================================================

with tab_heatmap:
    st.header("🗺 Heatmap Dashboard")
    st.markdown("Visualize workload intensity across **hours, days, users, and devices**.")

    # Load data
    try:
        df_ev = pd.read_sql("SELECT * FROM events", eng)
        df_car = pd.read_sql("SELECT * FROM carousel_events", eng)
    except Exception as e:
        st.error(f"Database load error: {e}")
        st.stop()

    if df_ev.empty and df_car.empty:
        st.info("Upload Pyxis and Carousel data to view heatmaps.")
        st.stop()

    # ============================
    # Normalize timestamps
    # ============================
    if not df_ev.empty:
        df_ev["dt"] = pd.to_datetime(df_ev["dt"], errors="coerce")
        df_ev["date"] = df_ev["dt"].dt.date
        df_ev["day"] = df_ev["dt"].dt.day_name()
        df_ev["hour"] = df_ev["dt"].dt.hour

    if not df_car.empty:
        df_car["ts"] = pd.to_datetime(df_car["ts"], errors="coerce")
        df_car["date"] = df_car["ts"].dt.date
        df_car["day"] = df_car["ts"].dt.day_name()
        df_car["hour"] = df_car["ts"].dt.hour

    # ============================================================
    # 1. Carousel Hourly Heatmap
    # ============================================================
    st.subheader("🎠 Carousel — Hourly Heatmap")

    if not df_car.empty:
        heat_car = df_car.groupby(["day", "hour"]).size().unstack(fill_value=0)
        st.dataframe(
            heat_car.style.background_gradient(cmap="Blues"),
            use_container_width=True,
            height=350
        )
    else:
        st.info("No Carousel records.")

    # ============================================================
    # 2. Pyxis Hourly Heatmap
    # ============================================================
    st.subheader("📦 Pyxis — Hourly Heatmap")

    if not df_ev.empty:
        heat_pyxis = df_ev.groupby(["day", "hour"]).size().unstack(fill_value=0)
        st.dataframe(
            heat_pyxis.style.background_gradient(cmap="Greens"),
            use_container_width=True,
            height=350
        )
    else:
        st.info("No Pyxis records.")

    # ============================================================
    # 3. Combined Pressure Map (Carousel → Pyxis)
    # ============================================================
    st.subheader("🔥 Combined Pressure — Carousel → Pyxis Flow")

    if not df_car.empty and not df_ev.empty:
        combined = pd.concat([
            df_car.assign(source="Carousel"),
            df_ev.assign(source="Pyxis")
        ], ignore_index=True)

        pressure = combined.groupby(["source", "day", "hour"]).size().unstack(fill_value=0)

        st.dataframe(
            pressure.style.background_gradient(cmap="Oranges"),
            use_container_width=True,
            height=350
        )
    else:
        st.info("Not enough data to compute flow pressure.")

    # ============================================================
    # 4. Device-Level Heatmap (Pyxis)
    # ============================================================
    st.subheader("🏥 Pyxis Device Activity Heatmap")

    if not df_ev.empty:
        dev_heat = (
            df_ev.groupby(["device", "hour"])
            .size()
            .unstack(fill_value=0)
        )
        st.dataframe(
            dev_heat.style.background_gradient(cmap="Purples"),
            use_container_width=True,
            height=350
        )
    else:
        st.info("No Pyxis device events found.")

    # ============================================================
    # 5. User Workload Heatmap (Carousel + Pyxis)
    # ============================================================
    st.subheader("🧑 User Workload Heatmap (All Systems)")

    if not df_ev.empty or not df_car.empty:
        # Standardize user column names
        df_ev["user_norm"] = df_ev["user"]
        df_car["user_norm"] = df_car["user_name"]

        users = pd.concat([
            df_ev[["user_norm", "hour"]],
            df_car[["user_norm", "hour"]]
        ], ignore_index=True)

        user_heat = users.groupby(["user_norm", "hour"]).size().unstack(fill_value=0)

        st.dataframe(
            user_heat.style.background_gradient(cmap="Reds"),
            use_container_width=True,
            height=350
        )
    else:
        st.info("No user data found.")








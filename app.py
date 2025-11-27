import streamlit as st
import pandas as pd
import hashlib
import psycopg2
from psycopg2.extras import execute_batch

st.set_page_config(page_title="Device Event Loader", layout="wide")

# -------------------------------------------------------
# DATABASE CONNECTION
# -------------------------------------------------------
DB_URL = st.secrets["neon"]["db_url"]

def get_conn():
    return psycopg2.connect(DB_URL)


# -------------------------------------------------------
# UTILITIES
# -------------------------------------------------------
def generate_pk(row):
    row_string = "|".join(str(v) for v in row.values)
    return hashlib.sha256(row_string.encode()).hexdigest()


def clean_dataframe(df):
    df = df.copy()

    # Normalize columns
    df.columns = df.columns.str.strip().str.lower()

    # Map all possible source column names → your final schema
    colmap = {
        "username": "user_name",
        "user": "user_name",
        "employee": "user_name",

        "device": "device",

        "medid": "med_id",
        "med id": "med_id",
        "medication id": "med_id",

        "description": "med_desc",
        "desc": "med_desc",
        "med description": "med_desc",

        "type": "event_type",
        "transaction type": "event_type",

        "datetime": "dt",
        "date": "dt",
        "time": "dt",
        "transaction date": "dt",
        "transaction time": "dt",

        "qty": "qty",
        "quantity": "qty",

        "beginning": "beginning_qty",
        "begin": "beginning_qty",
        "beginning qty": "beginning_qty",

        "end": "ending_qty",
        "ending": "ending_qty",
        "end qty": "ending_qty"
    }

    df = df.rename(columns=colmap)

    # Ensure required columns exist
    required_cols = [
        "user_name",
        "device",
        "med_id",
        "med_desc",
        "event_type",
        "dt",
        "qty",
        "beginning_qty",
        "ending_qty"
    ]

    for col in required_cols:
        if col not in df.columns:
            df[col] = None

    # Convert dt to TEXT safely
    df["dt"] = pd.to_datetime(df["dt"], errors="coerce")
    df["dt"] = df["dt"].astype(str).where(df["dt"].notna(), None)

    # Convert numeric fields safely
    for c in ["qty", "beginning_qty", "ending_qty"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Replace NaN with None
    df = df.where(pd.notna(df), None)

    # Generate PK
    df["pk"] = df.apply(lambda r: generate_pk(r), axis=1)

    return df

def insert_batch(df):
    conn = get_conn()
    cur = conn.cursor()

    sql = """
        INSERT INTO events (
            pk,
            user_name,
            device,
            med_id,
            med_desc,
            event_type,
            dt,
            qty,
            beginning_qty,
            ending_qty
        )
        VALUES (
            %(pk)s,
            %(user_name)s,
            %(device)s,
            %(med_id)s,
            %(med_desc)s,
            %(event_type)s,
            %(dt)s,
            %(qty)s,
            %(beginning_qty)s,
            %(ending_qty)s
        )
        ON CONFLICT (pk) DO NOTHING;
    """

    rows = df.to_dict("records")
    batch_size = 5000

    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]
        execute_batch(cur, sql, batch, page_size=len(batch))
        conn.commit()

    cur.close()
    conn.close()
def build_processed_events(df):
    df = df.copy()

    # ---------------------------------------
    # 1) Standardize event types
    # ---------------------------------------
    df["event_type_clean"] = df["event_type"].str.upper().str.strip()

    CANCELLED_KEYWORDS = ["CANCELLED"]
    IGNORE_TYPES = ["** NO MEDICATION ACTIVITY **"]

    def clean_type(x):
        if x is None:
            return None

        txt = str(x).upper().strip()

        if any(k in txt for k in CANCELLED_KEYWORDS):
            return "CANCELLED"

        if txt in ["VERIFY INVENTORY", "VERIFY INVENTORY (VERIFY COUNT AT OR BELOW MIN)"]:
            return "VERIFY"

        if txt == "REFILL":
            return "REFILL"

        if txt == "LOAD" or txt == "LOAD/REFILL":
            return "LOAD"

        if txt == "UNLOAD":
            return "UNLOAD"

        if txt == "OUTDATE":
            return "OUTDATE"

        if txt in IGNORE_TYPES:
            return "IGNORE"

        return txt  # leave anything unknown untouched

    df["etype"] = df["event_type_clean"].apply(clean_type)

    # ---------------------------------------
    # 2) Filter IGNORE events completely
    # But keep CANCELLED (per your rule)
    # ---------------------------------------
    df = df[df["etype"] != "IGNORE"].copy()


    # ---------------------------------------
    # 3) Build chronological sort key
    # ---------------------------------------
    df["dt"] = pd.to_datetime(df["dt"], errors="coerce")
    df = df.sort_values(["user_name", "dt"], ascending=[True, True]).reset_index(drop=True)


    # ---------------------------------------
    # 4) Walking time + dwell time logic
    # ---------------------------------------
    df["prev_dt"] = df.groupby("user_name")["dt"].shift(1)
    df["prev_device"] = df.groupby("user_name")["device"].shift(1)

    # Time between events in seconds
    df["time_delta_sec"] = (df["dt"] - df["prev_dt"]).dt.total_seconds()

    # Walking vs dwell
    df["walking_sec"] = df.apply(
        lambda r: r["time_delta_sec"] if r["device"] != r["prev_device"] else 0,
        axis=1
    )

    df["dwell_sec"] = df.apply(
        lambda r: r["time_delta_sec"] if r["device"] == r["prev_device"] else 0,
        axis=1
    )

    # Cancelled events = keep in dataset but do not count
    df.loc[df["etype"] == "CANCELLED", ["walking_sec", "dwell_sec"]] = 0


    # ---------------------------------------
    # 5) Refill pairing
    # Pairs: VERIFY → REFILL only when same device & user
    # ---------------------------------------
    df["verify_pk"] = None
    df["refill_pair_sec"] = 0

    # Loop per user/device groups
    for (user, device), group in df.groupby(["user_name", "device"]):
        indexes = group.index.tolist()
        events = group["etype"].tolist()
        times = group["dt"].tolist()

        last_verify_time = None
        last_verify_index = None

        for idx, etype in zip(indexes, events):
            if etype == "VERIFY":
                last_verify_time = df.loc[idx, "dt"]
                last_verify_index = idx

            elif etype == "REFILL":
                if last_verify_time is not None:
                    diff = (df.loc[idx, "dt"] - last_verify_time).total_seconds()

                    if diff >= 0 and diff < 600:  # 10-minute window
                        df.loc[idx, "verify_pk"] = df.loc[last_verify_index, "pk"]
                        df.loc[idx, "refill_pair_sec"] = diff

                last_verify_time = None
                last_verify_index = None

    return df

# =====================================================
# PROCESSING ENGINE (FINAL VERSION)
# =====================================================

import pandas as pd
import numpy as np

# -----------------------------------------------
# 1. Standardize/clean event types
# -----------------------------------------------
def normalize_event_type(raw):
    if raw is None:
        return None

    r = str(raw).strip().lower()

    # Cancelled (kept in dataset but excluded later)
    if "cancel" in r:
        return "cancelled"

    # Standard types
    if "verify" in r:
        return "verify"
    if "refill" in r:
        return "refill"
    if "load" in r:
        return "load"
    if "unload" in r:
        return "unload"
    if "outdate" in r or "out date" in r:
        return "outdate"
    if "empty" in r:
        return "emptyreturn"
    if "count" in r:
        return "count"

    return r


# -----------------------------------------------
# 2. Clean and structure the raw dataset
# -----------------------------------------------
def prepare_events(df):
    df = df.copy()

    # Lowercase column names
    df.columns = df.columns.str.lower()

    # Enforce required fields
    req = ["dt","user_name","device","event_type","medid","description","qty","beg","end","pk"]
    for c in req:
        if c not in df.columns:
            df[c] = None

    # Clean dt → datetime
    df["dt"] = pd.to_datetime(df["dt"], errors="coerce")

    # Standardize event types
    df["event_type_clean"] = df["event_type"].apply(normalize_event_type)

    # Sort properly
    df = df.sort_values(["user_name","dt"]).reset_index(drop=True)

    return df


# -----------------------------------------------
# 3. Calculate per-event time deltas
# -----------------------------------------------
def compute_time_deltas(df):
    df = df.copy()

    df["next_dt"] = df.groupby("user_name")["dt"].shift(-1)
    df["next_device"] = df.groupby("user_name")["device"].shift(-1)

    df["time_delta_sec"] = (df["next_dt"] - df["dt"]).dt.total_seconds()

    # Negative deltas → set to 0
    df["time_delta_sec"] = df["time_delta_sec"].fillna(0)
    df.loc[df["time_delta_sec"] < 0, "time_delta_sec"] = 0

    return df


# -----------------------------------------------
# 4. Compute dwell + walking time
# -----------------------------------------------
IDLE_CUTOFF_SEC = 10 * 60  # 10 minutes

def compute_dwell_walk(df):
    df = df.copy()

    df["dwell_sec"] = 0
    df["walk_sec"] = 0

    mask_valid = df["event_type_clean"] != "cancelled"

    for idx in df.index:
        if not mask_valid[idx]:
            continue  # cancelled gets 0 everything

        delta = df.at[idx, "time_delta_sec"]

        # no next event
        if delta <= 0:
            continue

        this_dev = df.at[idx, "device"]
        next_dev = df.at[idx, "next_device"]

        # SAME device → dwell (if <= cutoff)
        if this_dev == next_dev:
            if delta <= IDLE_CUTOFF_SEC:
                df.at[idx, "dwell_sec"] = delta
            # else → idle, ignore

        else:
            # DIFFERENT device → walking (always)
            df.at[idx, "walk_sec"] = delta

    return df


# -----------------------------------------------
# 5. Pair Verify + Refill events
# -----------------------------------------------
def compute_refill_pairs(df):
    df = df.copy()

    df["refill_pair_id"] = None
    df["refill_pair_sec"] = 0
    df["is_refill_primary"] = False

    pair_counter = 1

    for user, grp in df.groupby("user_name"):
        grp_idx = grp.index

        last_verify_idx = None
        last_verify_dt = None

        for idx in grp_idx:
            etype = df.at[idx, "event_type_clean"]

            # verify event
            if etype == "verify":
                last_verify_idx = idx
                last_verify_dt = df.at[idx, "dt"]

            # refill event
            elif etype == "refill" and last_verify_idx is not None:
                refill_dt = df.at[idx, "dt"]

                pair_id = f"{user}_{pair_counter}"
                pair_counter += 1

                df.at[last_verify_idx, "refill_pair_id"] = pair_id
                df.at[idx, "refill_pair_id"] = pair_id

                # refill duration = dt difference
                if last_verify_dt is not None:
                    sec = (refill_dt - last_verify_dt).total_seconds()
                    if sec < 0:
                        sec = 0
                    df.at[idx, "refill_pair_sec"] = sec
                    df.at[idx, "is_refill_primary"] = True

                last_verify_idx = None
                last_verify_dt = None

    return df


# -----------------------------------------------
# MASTER PROCESSOR (call this once per upload)
# -----------------------------------------------
def process_events(df_raw):
    df = prepare_events(df_raw)
    df = compute_time_deltas(df)
    df = compute_dwell_walk(df)
    df = compute_refill_pairs(df)

    # Final cleanup
    df["total_machine_sec"] = df["dwell_sec"]
    df["total_walk_sec"] = df["walk_sec"]

    return df



# -------------------------------------------------------
# STREAMLIT UI
# -------------------------------------------------------
st.title("📥 Device Event Loader (Clean Reset, Neon + Psycopg2)")

uploaded = st.file_uploader("Upload Device Event Report (CSV or Excel)", type=["csv", "xlsx"])

if uploaded:
    st.success("File uploaded successfully!")

    # Load raw file
    if uploaded.name.endswith(".xlsx"):
        df_raw = pd.read_excel(uploaded)
    else:
        df_raw = pd.read_csv(uploaded)

    st.write(f"**Raw rows:** {len(df_raw):,}")

    # Clean data
    df_clean = clean_dataframe(df_raw)
    st.write(f"**Cleaned rows:** {len(df_clean):,}")

    # Debug dt values (now in correct location)
    st.write("Example dt values:", df_clean["dt"].head(10).tolist())
    st.write("DT dtype:", str(df_clean["dt"].dtype))
    st.write("Rows with dt = None:", df_clean[df_clean["dt"].isna()].head(10))

    # Preview
    st.dataframe(df_clean.head(20), use_container_width=True)

    if st.button("🚀 Save to Neon Database"):
        with st.spinner("Saving rows to Neon…"):
            insert_batch(df_clean)
        st.success("✅ Upload complete and saved to Neon!")

# =====================================================
# TAB: REFILL EFFICIENCY (Charts + Tables)
# =====================================================

import plotly.express as px
import plotly.graph_objects as go

def build_refill_efficiency_tab(df):
    st.subheader("💊 Refill Efficiency Overview")

    # Only include primary refill rows (paired refills)
    df_refill = df[df["is_refill_primary"] == True].copy()

    if df_refill.empty:
        st.info("No refill events found for this time range.")
        return

    # High-level metrics
    total_refills = len(df_refill)
    avg_refill = df_refill["refill_pair_sec"].mean()
    median_refill = df_refill["refill_pair_sec"].median()
    max_refill = df_refill["refill_pair_sec"].max()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Refills", f"{total_refills:,}")
    c2.metric("Avg Refill Time", f"{avg_refill:.1f} sec")
    c3.metric("Median Refill Time", f"{median_refill:.1f} sec")
    c4.metric("Longest Refill", f"{max_refill:.1f} sec")

    st.markdown("---")

    # ============================
    # Refill Time by Technician
    # ============================
    st.subheader("⏱️ Refill Time by Technician")

    tech_stats = (
        df_refill.groupby("user_name")
        .agg(
            total_refills=("refill_pair_sec", "count"),
            avg_refill_sec=("refill_pair_sec", "mean"),
            median_refill_sec=("refill_pair_sec", "median")
        )
        .reset_index()
    )

    fig_tech = px.bar(
        tech_stats,
        x="user_name",
        y="avg_refill_sec",
        title="Average Refill Duration per Technician",
        labels={"avg_refill_sec": "Avg Refill Time (sec)", "user_name": "Technician"},
    )
    st.plotly_chart(fig_tech, use_container_width=True)

    st.markdown("---")

    # ============================
    # Refill Time by Device
    # ============================
    st.subheader("🏥 Refill Time by Device")

    dev_stats = (
        df_refill.groupby("device")
        .agg(
            refill_count=("refill_pair_sec", "count"),
            avg_refill_sec=("refill_pair_sec", "mean")
        )
        .reset_index()
    )

    fig_dev = px.bar(
        dev_stats.sort_values("avg_refill_sec", ascending=False),
        x="device",
        y="avg_refill_sec",
        title="Average Refill Duration per Device",
        labels={"avg_refill_sec": "Avg Refill Time (sec)", "device": "Device"},
    )
    st.plotly_chart(fig_dev, use_container_width=True)

    st.markdown("---")

    # ============================
    # Outlier Refills (Long Times)
    # ============================
    st.subheader("🚨 Longest Refill Times (Outliers)")

    df_outliers = df_refill.sort_values("refill_pair_sec", ascending=False).head(15)

    st.dataframe(
        df_outliers[
            ["dt", "user_name", "device", "medid", "description", "refill_pair_sec"]
        ],
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("---")

    # ============================
    # Full Refill Table
    # ============================
    st.subheader("📋 All Refills")

    st.dataframe(
        df_refill[
            ["dt", "user_name", "device", "medid", "description",
             "qty", "beg", "end", "refill_pair_sec", "refill_pair_id"]
        ],
        use_container_width=True,
        hide_index=True,
    )


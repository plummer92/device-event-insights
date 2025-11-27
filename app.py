import streamlit as st
import pandas as pd
import hashlib
import psycopg2
from psycopg2.extras import execute_batch
import plotly.express as px

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
    """Stable row hash for deduplication."""
    row_string = "|".join(str(v) for v in row.values)
    return hashlib.sha256(row_string.encode()).hexdigest()


def clean_dataframe(df):
    df = df.copy()

    # Normalize source column names
    df.columns = df.columns.str.strip().str.lower()

    rename_map = {
        "username": "user_name",
        "user": "user_name",
        "userid": "user_name",

        "device": "device",

        "medid": "medid",
        "med id": "medid",

        "meddescription": "description",
        "description": "description",
        "med description": "description",

        "transactiontype": "event_type",
        "transaction type": "event_type",
        "type": "event_type",

        "transactiondatetime": "dt",
        "transaction date and time": "dt",
        "transaction date": "dt",

        "quantity": "qty",
        "qty": "qty",

        "beg": "beg",
        "beginning": "beg",
        "beginning qty": "beg",

        "end": "end",
        "ending": "end",
        "end qty": "end"
    }

    df = df.rename(columns=rename_map)

    # Required columns guaranteed
    required = ["user_name","device","medid","description","event_type","dt","qty","beg","end"]
    for col in required:
        if col not in df.columns:
            df[col] = None

    # Normalize dt
    df["dt"] = pd.to_datetime(df["dt"], errors="coerce")

    # Normalize event type → lowercase
    df["event_type"] = (
        df["event_type"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    # Numeric conversions
    for c in ["qty","beg","end"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.where(pd.notna(df), None)

    # Generate PK
    df["pk"] = df.apply(lambda r: generate_pk(r), axis=1)

    return df


def insert_batch(df):
    """Insert cleaned rows into Neon using psycopg2."""
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


# -------------------------------------------------------
# PROCESSING ENGINE
# -------------------------------------------------------

IDLE_CUTOFF_SEC = 10 * 60  # 10 minutes


def normalize_event_type(raw):
    if raw is None:
        return None

    r = str(raw).strip().lower()

    if "cancel" in r:
        return "cancelled"
    if "verify" in r:
        return "verify"
    if "refill" in r:
        return "refill"
    if "load" in r:
        return "load"
    if "unload" in r:
        return "unload"
    if "outdate" in r:
        return "outdate"
    if "empty" in r:
        return "emptyreturn"
    if "count" in r:
        return "count"

    return r


def prepare_events(df):
    df = df.copy()
    df.columns = df.columns.str.lower()

    # enforce required columns
    needed = [
        "dt", "user_name", "device", "event_type",
        "med_id", "med_desc", "qty",
        "beginning_qty", "ending_qty", "pk"
    ]
    for c in needed:
        if c not in df.columns:
            df[c] = None

    df["dt"] = pd.to_datetime(df["dt"], errors="coerce")
    df["event_type_clean"] = df["event_type"].apply(normalize_event_type)
    df = df.sort_values(["user_name", "dt"])
    return df.reset_index(drop=True)


def compute_time_deltas(df):
    df = df.copy()
    df["next_dt"] = df.groupby("user_name")["dt"].shift(-1)
    df["next_device"] = df.groupby("user_name")["device"].shift(-1)

    df["time_delta_sec"] = (df["next_dt"] - df["dt"]).dt.total_seconds()
    df["time_delta_sec"] = df["time_delta_sec"].fillna(0)
    df.loc[df["time_delta_sec"] < 0, "time_delta_sec"] = 0

    return df


def compute_dwell_walk(df):
    df = df.copy()

    df["dwell_sec"] = 0
    df["walk_sec"] = 0

    valid_mask = df["event_type_clean"] != "cancelled"

    for idx in df.index:
        if not valid_mask[idx]:
            continue

        delta = df.at[idx, "time_delta_sec"]
        if delta <= 0:
            continue

        same_device = df.at[idx, "device"] == df.at[idx, "next_device"]

        if same_device:
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

    pair_counter = 1

    for user, grp in df.groupby("user_name"):
        last_verify_idx = None
        last_verify_dt = None

        for idx in grp.index:
            etype = df.at[idx, "event_type_clean"]

            if etype == "verify":
                last_verify_idx = idx
                last_verify_dt = df.at[idx, "dt"]

            elif etype == "refill" and last_verify_idx is not None:
                refill_dt = df.at[idx, "dt"]

                pair_id = f"{user}_{pair_counter}"
                pair_counter += 1

                df.at[last_verify_idx, "refill_pair_id"] = pair_id
                df.at[idx, "refill_pair_id"] = pair_id

                if last_verify_dt is not None:
                    sec = (refill_dt - last_verify_dt).total_seconds()
                    if sec < 0:
                        sec = 0
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


# -------------------------------------------------------
# REFILL EFFICIENCY TAB
# -------------------------------------------------------

def build_refill_efficiency_tab(df):
    st.subheader("💊 Refill Efficiency Overview")

    df_refill = df[df["is_refill_primary"] == True].copy()
    if df_refill.empty:
        st.info("No refill events found.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Refills", len(df_refill))
    c2.metric("Avg Refill Time", f"{df_refill['refill_pair_sec'].mean():.1f} sec")
    c3.metric("Median Refill Time", f"{df_refill['refill_pair_sec'].median():.1f} sec")
    c4.metric("Longest Refill", f"{df_refill['refill_pair_sec'].max():.1f} sec")

    st.markdown("---")

    # Technician comparison
    st.subheader("⏱️ Refill Time by Technician")
    tech_stats = (
        df_refill.groupby("user_name")
        .agg(avg_refill_sec=("refill_pair_sec", "mean"))
        .reset_index()
    )
    fig = px.bar(
        tech_stats,
        x="user_name",
        y="avg_refill_sec",
        title="Average Refill Duration per Technician",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Device comparison
    st.subheader("🏥 Refill Time by Device")
    dev_stats = (
        df_refill.groupby("device")
        .agg(avg_refill_sec=("refill_pair_sec", "mean"))
        .reset_index()
    )
    fig2 = px.bar(
        dev_stats,
        x="device",
        y="avg_refill_sec",
        title="Average Refill Duration per Device",
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")

    # Outliers
    st.subheader("🚨 Longest Refill Times")
    st.dataframe(
        df_refill.sort_values("refill_pair_sec", ascending=False).head(20),
        use_container_width=True,
        hide_index=True,
    )


# -------------------------------------------------------
# STREAMLIT UI
# -------------------------------------------------------

st.title("📥 Device Event Loader (Neon + Analytics Engine)")

uploaded = st.file_uploader("Upload Device Event Report", type=["csv", "xlsx"])

if uploaded:
    st.success("File uploaded")

    if uploaded.name.endswith(".xlsx"):
        df_raw = pd.read_excel(uploaded)
    else:
        df_raw = pd.read_csv(uploaded)

    st.write(f"Raw rows: {len(df_raw):,}")

    df_clean = clean_dataframe(df_raw)
    st.write(f"Cleaned rows: {len(df_clean):,}")

    st.dataframe(df_clean.head(20), use_container_width=True)

    if st.button("Save to Neon"):
        with st.spinner("Saving…"):
            insert_batch(df_clean)
        st.success("Saved to Neon!")

    st.markdown("---")
    st.header("📊 Analytics (local processing only)")

    df_proc = process_events(df_clean)

    tab1, = st.tabs(["💊 Refill Efficiency"])
    with tab1:
        build_refill_efficiency_tab(df_proc)

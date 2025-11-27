import streamlit as st
import pandas as pd
import hashlib
import psycopg2
from psycopg2.extras import execute_batch

st.set_page_config(page_title="Device Event Loader", layout="wide")

# -------------------------------
# DB CONNECTION
# -------------------------------
DB_CONN = st.secrets["neon"]["conn"]

def get_conn():
    return psycopg2.connect(DB_CONN)

# -------------------------------
# UTILITIES
# -------------------------------
def generate_pk(row):
    """Create a stable SHA-256 key for deduplication."""
    row_string = "|".join(str(v) for v in row.values)
    return hashlib.sha256(row_string.encode()).hexdigest()

def clean_dataframe(df):
    df = df.copy()

    # Standardize column names
    df.columns = df.columns.str.strip().str.lower()

    # Rename known columns
    colmap = {
        "datetime": "dt",
        "time": "dt",
        "device": "device",
        "user": "user_name",
        "type": "event_type",
        "desc": "description",
        "description": "description",
        "qty": "qty",
        "medid": "medid",
        "med id": "medid",
    }

    df = df.rename(columns=colmap)

    # Keep only the DB columns
    keep = ["dt", "device", "user_name", "event_type", "description", "qty", "medid"]
    df = df[[c for c in keep if c in df.columns]]

    # Convert datetime
    if "dt" in df.columns:
        df["dt"] = pd.to_datetime(df["dt"], errors="coerce")

    df = df.where(pd.notna(df), None)

    # Generate PK last
    df["pk"] = df.apply(lambda r: generate_pk(r), axis=1)

    return df

def insert_batch(df):
    """Insert into Postgres in large batches (Neon paid tier)."""
    conn = get_conn()
    cur = conn.cursor()

    sql = """
        INSERT INTO events (pk, dt, device, user_name, event_type, description, qty, medid)
        VALUES (%(pk)s, %(dt)s, %(device)s, %(user_name)s, %(event_type)s, %(description)s, %(qty)s, %(medid)s)
        ON CONFLICT (pk) DO NOTHING;
    """

    rows = df.to_dict("records")
    batch_size = 5000  # safe for paid Neon

    for i in range(0, len(rows), batch_size):
        batch = rows[i:i+batch_size]
        execute_batch(cur, sql, batch, page_size=len(batch))
        conn.commit()

    cur.close()
    conn.close()

# -------------------------------
# STREAMLIT UI
# -------------------------------
st.title("📥 Device Event Loader (Fresh Reset)")

uploaded = st.file_uploader("Upload Device Event Report (CSV or Excel)", type=["csv", "xlsx"])

if uploaded:
    st.success("File uploaded! Preview below:")

    # Load file
    if uploaded.name.endswith(".xlsx"):
        df_raw = pd.read_excel(uploaded)
    else:
        df_raw = pd.read_csv(uploaded)

    st.write("Raw rows:", len(df_raw))

    # Clean
    df_clean = clean_dataframe(df_raw)
    st.write("Cleaned rows:", len(df_clean))

    st.dataframe(df_clean.head(20))

    # Save button
    if st.button("🚀 Save to Neon Database"):
        insert_batch(df_clean)
        st.success("Upload complete and saved to Neon!")

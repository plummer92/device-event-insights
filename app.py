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

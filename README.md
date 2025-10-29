# All Device Event Insights — Pro (Streamlit)

**New features:**
- **Historical storage**: Append each day’s upload to a local history file (`./history/events_history.csv.gz`).
- **Day-over-day comparison**: See changes in volume versus yesterday.
- **Staffing overlay**: Upload a staffing CSV to overlay hourly workload vs. staffing counts.

## Staffing CSV Template
Save a CSV like this (included as `staffing_template.csv`):
```csv
role,person,start,end
IV Tech,Liz,2025-10-28 06:00,2025-10-28 14:00
Runner,Ben,2025-10-28 07:00,2025-10-28 15:00
Pharmacist,Melissa,2025-10-28 09:00,2025-10-28 17:00
```

## Run locally
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

**Notes**
- History is stored in a `history` folder alongside the app. Use the buttons to append or clear.
- Day-over-day comparison uses the `__upload_day` tag (the min datetime in your upload). Append at least two days to see a delta.
- The staffing overlay counts staff per hour inclusive of shift endpoints (simple model). We can add roles, weighting, and shift rules if needed.

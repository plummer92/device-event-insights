# Device Event Insights — v3.1

Adds:
- JIT device/location → role group mapping (Runner/IV/Central)
- SLA thresholds (events-per-staff-hour, JIT-per-staff-hour)
- Google Sheets history (append & preview)

## Mapping CSV
Columns (any of these keys to match): device_id, friendly_name, location
Required: role_group (Runner / IV / Central)

Example:
device_id,friendly_name,location,role_group
PYX-4E-01,Pyxis 4th East A,4th East,Runner
IVC-ROOM-01,IV Compounder 1,IV Room,IV
CENT-WS-01,Central Workstation 1,Central Pharmacy,Central

## Staffing CSV
Required: role, person, start, end
Optional: location, device_group (Runner/IV/Central), notes

## Google Sheets (Streamlit Cloud)
1. Create a Service Account in Google Cloud (download JSON).
2. In Streamlit Cloud → Manage app → Settings → Secrets, paste:

[gcp_service_account]
type = "service_account"
project_id = "..."
private_key_id = "..."
private_key = "-----BEGIN PRIVATE KEY-----\\n...\\n-----END PRIVATE KEY-----\\n"
client_email = "your-sa@your-project.iam.gserviceaccount.com"
client_id = "..."
token_uri = "https://oauth2.googleapis.com/token"

[gsheets]
workbook_url = "https://docs.google.com/spreadsheets/d/XXXXXXXX/edit"
events_sheet = "events_history"

3. Share the Google Sheet with the service account's client_email.

# Device Event Insights — v4.0

## What’s new
- **Delivery Analytics**: 
  - Walking gaps (between-device delay per tech)
  - Delivery runs (split by idle gap threshold, default 15 min)
  - Tech comparison (total walking time, avg dwell)
  - Device rankings (avg dwell, p75/p90, volume)
- **Weekly Summary**:
  - Append today’s metrics to Google Sheets
  - Load last 7 days and plot trends

## Daily workflow
1) Upload your **All Device Events** XLSX.
2) (Sidebar) choose devices/types/techs and set **idle gap** minutes.
3) See tabs:
   - Overview
   - Delivery Analytics
   - Tech Comparison
   - Device Rankings
   - Weekly Summary (append to Sheets, then load last 7 days)

## Google Sheets (Streamlit Cloud Secrets)
In **Manage app → Settings → Secrets**, add:

[gcp_service_account]
type = "service_account"
project_id = "..."
private_key_id = "..."
private_key = "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n"
client_email = "your-sa@your-project.iam.gserviceaccount.com"
client_id = "..."
token_uri = "https://oauth2.googleapis.com/token"

[gsheets]
workbook_url = "https://docs.google.com/spreadsheets/d/XXXXXXXX/edit"
events_sheet = "events_history"

Then share the Google Sheet with the **client_email**.


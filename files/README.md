# AquaSure - n8n + GenAI Automation

Automated water quality reporting and alerting for Maharashtra (MPCB / NWMP data).

## Problem statement
Regulators and public health officials do not open dashboards every day. They need an automatic plain-English summary of water quality and an early warning when contamination risks appear. Two n8n workflows turn KPI data into that communication.

## Files
| File | Purpose |
|---|---|
| `workflow_1_daily_summary.json` | Problem I: daily AI summary |
| `workflow_2_risk_alert.json` | Problem II: risk alert automation |
| `sql/kpi_views.sql` | `vw_daily_kpi` used by workflow 1 |
| `sample_outputs.md` | Example input KPIs, summary and alert |
| `screenshots/` | Add your n8n canvas and Telegram/email screenshots here |

## Workflow 1 - Daily AI Environmental Summary
1. **Daily Trigger (Cron)** runs at 08:00 (`0 8 * * *`).
2. **Fetch Latest KPIs** (Postgres) reads the newest row of `vw_daily_kpi`: total samples, % non-compliant, average BOD and DO, high-risk stations, low-DO stations, fecal coliform exceedances, district with most high-risk stations.
3. **Build Prompt** (Code) inserts the KPIs into a fixed prompt.
4. **Claude Summary** (HTTP Request to the Anthropic Messages API) writes the summary.
5. **Extract Text** (Code) pulls the text out of the API response.
6. **Send Telegram** delivers the message (swap for Gmail or Slack if you prefer).

## Workflow 2 - Water Quality Risk Alert
1. **Daily Trigger (Cron)** at 09:00.
2. **Fetch Recent Data** reads the last 14 days of `dw_water_quality_clean`.
3. **Apply Alert Rules** (Code) checks five conditions. Thresholds sit in one `T` object at the top of the node:
   - % non-compliant rises by 5 points or more week on week
   - a district's average BOD is above 3 mg/L
   - dissolved oxygen is below 5 mg/L
   - fecal coliform is above 500 MPN/100 mL
   - the same station is non-compliant on 3 consecutive days
4. **Any Alert?** (IF) stops the run quietly when nothing fires.
5. **Build Alert Prompt**, **Claude Explains**, **Extract Text** produce the explanation.
6. **Send Telegram** delivers the alert.

## Trigger logic
Both flows use a Schedule Trigger with a cron expression. Workflow 2 only sends a message when at least one rule is true, so officials are not flooded with "all clear" messages.

## Data source connection
PostgreSQL credential named "AquaSure Postgres" pointing at the database holding `dw_water_quality_clean` and `vw_daily_kpi`. For a database-free demo, replace the Postgres node with *Read Binary File* + *Extract From File (CSV)* on `dw_water_quality_clean.csv`; the Code nodes work with the same field names.

## GenAI usage (Input -> Prompt -> Output)
- **Input:** KPI numbers (workflow 1) or the rule results plus supporting station/district figures as JSON (workflow 2).
- **Prompt:** A system prompt fixes the role (environmental analyst for MPCB), forbids invented numbers and requires causes to be labelled "possible". The user prompt lists the data and sets a strict length and format (summary: status, two concerns, one action; alert: What happened / Possible causes / Suggested regulatory action).
- **Output:** A short plain-English message. GenAI does the reasoning over the numbers (what matters most, plausible causes, next step); the rules and numbers come from SQL and code, not from the model.

## Setup
1. Import both JSON files in n8n (Workflows > Import from file).
2. Create the Postgres and Telegram credentials and select them in the nodes marked `REPLACE_ME`.
3. Set `ANTHROPIC_API_KEY` as an n8n environment variable (or switch the header to an n8n credential). Check the model name in the HTTP node against your account.
4. Run `sql/kpi_views.sql`, then execute each workflow manually once and screenshot the result.

Note: the workflows were validated for JSON structure and the alert-rule JavaScript was tested against the August 2025 data, but they have not been imported into a live n8n instance. If an import warning appears for a node version, re-add that node in the UI and copy the settings.

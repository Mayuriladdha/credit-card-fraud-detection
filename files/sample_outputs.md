# Sample outputs

These examples show the output format. The numbers are real KPIs computed from NWMP_August2025_MPCB_0.csv
(sampling date 6 Aug 2025, the busiest day). The text was written to illustrate the format; replace it with
screenshots of your own live n8n runs before submitting.

## Workflow 1 - Daily summary (input KPIs for 6 Aug 2025)

| KPI | Value |
|---|---|
| Total samples | 60 |
| % non-compliant (classified stations) | 87.2% |
| Average BOD | 15.1 mg/L |
| Average DO | 5.0 mg/L |
| High-risk stations | 19 |
| Stations with DO < 5 mg/L | 21 |
| Fecal coliform > 500 | 2 |
| District with most high-risk stations | Thane (11) |

**Example AI summary**

> On 6 August 2025, 60 water samples were analysed across Maharashtra. About 87% of samples at classified stations did not meet their designated-use criteria, and average BOD was 15.1 mg/L against a safe level of about 3 mg/L. Thane had the most high-risk stations (11 of 19 statewide), with severe oxygen depletion in industrial drains. Two samples exceeded the fecal coliform limit. Recommended action: prioritise inspection of industrial effluent discharges in Thane and review sewage treatment near the Mithi River in Mumbai.

## Workflow 2 - Alert (rule output for data up to 11 Aug 2025)

Rules fired: 15 districts with average BOD above 3 mg/L; 57 stations with DO below 5 mg/L; 4 samples with fecal coliform above 500. The 3-consecutive-days rule did not fire because each station has only one sample in this file. The week-on-week rule did not fire because non-compliance fell from 95.1% to 86.9%.

**Example alert**

> WATER QUALITY ALERT
> What happened: Average BOD exceeds 3 mg/L in 15 districts, led by Thane (23.4), Mumbai (15.7) and Pune (11.0). 57 stations show DO below 5 mg/L, and several sit at 0.3 mg/L, including Mithi River (Mumbai) and Tarapur MIDC Nalla (Thane). Four samples exceed 500 fecal coliform, with Nagpur stations at 1600.
> Possible causes: Untreated or partly treated industrial effluent in Thane's MIDC drains; sewage inflow into urban creeks and the Mithi River; low flow concentrating pollutants.
> Suggested regulatory action: Inspect and sample upstream discharges at Tarapur MIDC Nalla within 48 hours; start effluent source tracing for the Mithi River and Nagpur stations 5979 and 5980.

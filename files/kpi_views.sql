-- Views used by the n8n workflows. Run after dw_water_quality_clean exists (see Section 2 of the report).
CREATE OR REPLACE VIEW vw_daily_kpi AS
SELECT w.sampling_date,
       COUNT(*) FILTER (WHERE w.bod IS NOT NULL)                                  AS total_samples,
       ROUND(100.0 * COUNT(*) FILTER (WHERE w.non_compliant)
             / NULLIF(COUNT(*) FILTER (WHERE w.non_compliant IS NOT NULL), 0), 1) AS pct_non_compliant,
       ROUND(AVG(w.bod), 2)                                                       AS avg_bod,
       ROUND(AVG(w.dissolved_oxygen), 2)                                          AS avg_do,
       COUNT(*) FILTER (WHERE w.risk_category = 'High')                           AS high_risk_stations,
       COUNT(*) FILTER (WHERE w.dissolved_oxygen < 5)                             AS low_do_stations,
       COUNT(*) FILTER (WHERE w.fecal_coliform > 500)                             AS fecal_coliform_exceedances,
       (SELECT x.district FROM dw_water_quality_clean x
         WHERE x.sampling_date = w.sampling_date AND x.risk_category = 'High'
         GROUP BY x.district ORDER BY COUNT(*) DESC, x.district LIMIT 1)          AS district_highest_risk
FROM dw_water_quality_clean w
GROUP BY w.sampling_date;

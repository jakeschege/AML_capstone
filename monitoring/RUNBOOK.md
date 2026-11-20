# Simulated monitoring runbook

1. Run `python monitoring/simulate_drift.py`.
2. Exit code `1` means a **simulated** drift signal (e.g. county mix / education skew).
3. Inspect `monitoring/outputs/drift_report.html` or `drift_summary.json`.
4. Investigate data collection or population shift.
5. **Do not auto-retrain.** Champion replacement is a deliberate packaging export after research review.

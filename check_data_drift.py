import warnings
import pandas as pd
from datetime import datetime
from evidently import Report
from evidently.presets import DataDriftPreset

# Suppress warnings
warnings.filterwarnings("ignore")

# 1. Load CSVs
reference_df = pd.read_csv(r"C:\Users\Amogh\Documents\mlops\hear_desease_project\training_reference.csv")
current_df = pd.read_csv(r"C:\Users\Amogh\Documents\mlops\hear_desease_project\current_data.csv")

print(f"[INFO] Loaded {len(reference_df)} reference rows and {len(current_df)} current log rows.")

# 2. Strip GCP Cloud Logging prefixes
current_df.columns = [
    col.replace("jsonPayload.input_features.", "").replace("jsonPayload.", "")
    for col in current_df.columns
]

# 3. Align Feature Columns
common_columns = [col for col in reference_df.columns if col in current_df.columns]
exclude_cols = ["request_id", "timestamp", "latency_ms", "model_used"]
common_columns = [c for c in common_columns if c not in exclude_cols]

print(f"[INFO] Evaluating drift across {len(common_columns)} matching columns.")

reference_subset = reference_df[common_columns].copy()
current_subset = current_df[common_columns].copy()

# Fill missing values
reference_subset = reference_subset.fillna(reference_subset.mode().iloc[0])
current_subset = current_subset.fillna(reference_subset.mode().iloc[0])

# Align data types
for col in common_columns:
    if reference_subset[col].dtype != current_subset[col].dtype:
        reference_subset[col] = reference_subset[col].astype(str)
        current_subset[col] = current_subset[col].astype(str)

# 4. Run Drift Report
report = Report(metrics=[
    DataDriftPreset()
])

# Generate snapshot result
snapshot = report.run(
    current_data=current_subset,
    reference_data=reference_subset
)

# 5. Save HTML Output via Snapshot / Render
output_file = f"drift_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"

# Modern Evidently versions save HTML from the snapshot result or report object
if hasattr(snapshot, "save_html"):
    snapshot.save_html(output_file)
elif hasattr(snapshot, "get_html"):
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(snapshot.get_html())
else:
    # Fallback to saving summary as a clean CSV / DataFrame drift summary table
    df_drift = report.as_dataframe() if hasattr(report, "as_dataframe") else pd.DataFrame()
    csv_file = output_file.replace(".html", ".csv")
    df_drift.to_csv(csv_file)
    print(f"[INFO] Report saved as DataFrame CSV summary: {csv_file}")

print(f"[SUCCESS] Execution complete! Output file created.")
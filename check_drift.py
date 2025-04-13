# Drift detection pipeline for comparing training and live datasets

import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

# Load reference and current datasets
ref_df = pd.read_csv("data/final_processed_data.csv")  # model training data
curr_df = pd.read_csv("data/final_data.csv")  # current/live data

# Rename columns to match schema
rename_map = {
    "User_ID": "userId",
    "Movie_Name": "movieId",
    "Rating": "rating",
    "Timestamp_y": "timestamp",
    "Timestamp_x": "timestamp",
}
ref_df.rename(columns=rename_map, inplace=True)
curr_df.rename(columns=rename_map, inplace=True)

# Drop unnecessary timestamp columns if present
ref_df.drop(columns=["Timestamp_x", "Timestamp_y"], errors="ignore", inplace=True)
curr_df.drop(columns=["Timestamp_x", "Timestamp_y"], errors="ignore", inplace=True)

# Handle missing values in rating
ref_df["rating"] = ref_df["rating"].astype(float).fillna(ref_df["rating"].mean())
curr_df["rating"] = curr_df["rating"].astype(float).fillna(curr_df["rating"].mean())

# Convert timestamps to UNIX time, fix duplicate columns if any
for i, df in enumerate([ref_df, curr_df]):
    df = df.loc[:, ~df.columns.duplicated()]  # Remove duplicate column names
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["timestamp"] = df["timestamp"].fillna(pd.Timestamp("1970-01-01"))
    df["timestamp"] = df["timestamp"].astype("int64") // 10**9

    # Assign cleaned df back to reference or current
    if i == 0:
        ref_df = df
    else:
        curr_df = df

# Select matching columns for drift comparison
common_cols = ["userId", "movieId", "rating", "timestamp"]
ref_df = ref_df[common_cols]
curr_df = curr_df[common_cols]

# Run Evidently drift detection
report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=ref_df, current_data=curr_df)

# Print results
result = report.as_dict()["metrics"][0]["result"]
print(f"\n⚠️ Drift Detected: {result['dataset_drift']}")
print(
    f"📊 {result['number_of_drifted_columns']} / {result['number_of_columns']} columns drifted"
)

# Save visual report
report.save_html("drift_report.html")
print("\n📁 Drift report saved to: drift_report.html")

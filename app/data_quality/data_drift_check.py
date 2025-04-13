# Data Quality Component: Drift Detection (modular function) using evidently


from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

def check_drift(reference_df, current_df):
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_df, current_data=current_df)
    result = report.as_dict()

    if result["metrics"][0]["result"]["dataset_drift"]:
        print("⚠️ Data Drift Detected!")
    return result

# Standalone Schema Validation Script

import pandas as pd
from app.data_quality.schema_validation import validate_ratings_schema

# Change the file to your processed output
df = pd.read_csv("data/final_processed_data.csv")

# Fix column names to match expected schema
df.rename(
    columns={
        "User_ID": "userId",
        "Movie_Name": "movieId",
        "Rating": "rating",
        "Timestamp_y": "timestamp",
    },
    inplace=True,
)

# Optional: drop unused column
df.drop(columns=["Timestamp_x"], errors="ignore", inplace=True)

# Fill missing ratings with column mean
df["rating"] = df["rating"].astype(float)
df["rating"] = df["rating"].fillna(df["rating"].mean())

df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
df["timestamp"] = df["timestamp"].fillna(pd.Timestamp("1970-01-01"))
df["timestamp"] = df["timestamp"].astype("int64") // 10**9  # Convert to UNIX seconds


# Run schema validation
if validate_ratings_schema(df):
    print("✅ Schema is valid")
else:
    print("❌ Schema validation failed. Check logs.")

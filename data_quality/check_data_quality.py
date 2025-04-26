# Standalone Schema Validation Script

import pandas as pd
from data_quality.schema_validation import validate_ratings_schema
from app.logger import logger
from pathlib import Path


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent  # Resolves to Movie-Recommendation/
    df = pd.read_csv(root / "data" / "final_processed_data.csv")

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
    df["timestamp"] = (
        df["timestamp"].astype("int64") // 10**9
    )  # Convert to UNIX seconds

    # Run schema validation
    if validate_ratings_schema(df):
        logger.info("✅ Schema is valid")
    else:
        logger.info("❌ Schema validation failed. Check logs.")

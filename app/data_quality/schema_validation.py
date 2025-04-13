# Data Quality Component: Schema Validation using pandera

import pandera as pa
from pandera import Column, DataFrameSchema, Check

rating_schema = DataFrameSchema(
    {
        "userId": Column(int),
        "movieId": Column(object),
        "rating": Column(float, Check.in_range(0.5, 5.0)),
        "timestamp": Column(int),
    }
)


def validate_ratings_schema(df):
    try:
        rating_schema.validate(df)
        return True
    except pa.errors.SchemaError as e:
        print(f"[Schema ERROR] {e}")
        return False

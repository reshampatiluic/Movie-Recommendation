# live_ctr_wtr_calculator.py
from kafka_raw_reader import read_kafka_raw_messages
from kafka_cleaned_stream_processor import clean_raw_message
import requests
import time

# Connect to Kafka
consumer = read_kafka_raw_messages()

# Initialize Counters
total_recommendations = 0
total_clicks = 0
total_watch_5min = 0
user_seen = 0

# --- Main Loop ---
for msg in consumer:
    raw_value = msg.value
    clean_data = clean_raw_message(raw_value)

    if clean_data:
        user_id = clean_data['user_id']
        movie_name = clean_data['movie_name']
        minute_watched = clean_data['minute']

        # Step 1: Ask Recommendation Server for Recommendations
        try:
            response = requests.get(f"http://127.0.0.1:5050/recommendations/{user_id}")
            if response.status_code == 200:
                recommended_movies = response.json().get("recommended_movies", [])

                total_recommendations += len(recommended_movies)
                user_seen += 1

                # Step 2: Check if watched movie was recommended
                if movie_name in recommended_movies:
                    total_clicks += 1

                    # Step 3: Check if user watched >5 minutes
                    if minute_watched >= 5:
                        total_watch_5min += 1

        except Exception as e:
            print(f"❌ Exception fetching recommendations for User {user_id}: {e}")

        # --- Print metrics every 10 users ---
        if user_seen % 10 == 0:
            ctr = (total_clicks / total_recommendations) * 100 if total_recommendations > 0 else 0
            wtr = (total_watch_5min / total_recommendations) * 100 if total_recommendations > 0 else 0
            print(f"📈 Metrics | CTR: {ctr:.2f}% | WTR: {wtr:.2f}% | Users Seen: {user_seen}")

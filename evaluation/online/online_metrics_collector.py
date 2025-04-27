import json
import time
import random
import requests
from kafka import KafkaConsumer
from prometheus_client import start_http_server, Gauge

# Connect to Kafka
consumer = KafkaConsumer(
    'movielog2',
    bootstrap_servers=['localhost:9092'],
    auto_offset_reset='earliest',  # start from beginning
    enable_auto_commit=True,
    group_id='online-metrics',
    value_deserializer=lambda x: x.decode('utf-8')
)

# Start a Prometheus HTTP server for exposing metrics
start_http_server(9000)  # metrics will be available at localhost:9000/metrics

# Define Prometheus Gauges
ctr_gauge = Gauge('ctr_percent', 'Click Through Rate (%)')
wtr_gauge = Gauge('wtr_percent', 'Watch Through Rate (%)')

# Initialize local counters
total_recommendations = 0
total_clicks = 0
total_watch_5_min = 0
users_seen = 0

SERVER_URL = "http://127.0.0.1:8000/recommendations/"

# Main loop
for message in consumer:
    try:
        # Read raw Kafka message
        raw_log = message.value
        parts = raw_log.strip().split(",")

        if len(parts) != 3:
            continue

        timestamp, user_id, request = parts
        user_id = int(user_id)

        # Only process movie watch events
        if "/data/m/" not in request:
            continue

        # Clean movie name
        movie_part = request.split("/data/m/")[1]
        movie_name = movie_part.split("/")[0].replace("+", " ").replace("_", " ").lower()

        # Fetch recommendation from our server
        try:
            response = requests.get(SERVER_URL + str(user_id))
            if response.status_code != 200:
                continue
            recommended_movies = response.json().get('recommendations', [])
        except Exception as e:
            print(f"❌ Exception fetching recommendations for User {user_id}: {e}")
            continue

        # Update counters
        total_recommendations += len(recommended_movies)
        users_seen += 1

        recommended_titles = [m.lower() for m in recommended_movies]

        if any(movie_name in title for title in recommended_titles):
            total_clicks += 1

            # Simulate 80% chance of watching >=5min if clicked
            if random.random() < 0.8:
                total_watch_5_min += 1

        # Calculate metrics
        ctr = (total_clicks / total_recommendations) * 100 if total_recommendations > 0 else 0
        wtr = (total_watch_5_min / total_recommendations) * 100 if total_recommendations > 0 else 0

        # Update Prometheus Gauges
        ctr_gauge.set(ctr)
        wtr_gauge.set(wtr)

        # Periodically print metrics
        if users_seen % 10 == 0:
            print(f"📈 Metrics | CTR: {ctr:.2f}% | WTR: {wtr:.2f}% | Users Seen: {users_seen}")

    except Exception as e:
        print(f"❌ General Exception: {e}")
        continue

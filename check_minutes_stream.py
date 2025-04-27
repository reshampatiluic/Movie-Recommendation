from kafka import KafkaConsumer
import json
import re

# Connect to your Kafka server and topic
consumer = KafkaConsumer(
    'movielog2',  # <-- your topic name
    bootstrap_servers='localhost:9092',
    auto_offset_reset='earliest',  # read from earliest
    value_deserializer=lambda m: m.decode('utf-8')
)

# Pattern to match movie watch events like /data/m/<movie_name>/<minute>.mpg
pattern = re.compile(r"GET /data/m/(?P<movie_name>.+)/(?P<minute>\d+)\.mpg")

print("\n🔎 Listening for movie watch events... looking for minute >= 5 activity:\n")

# Counters
count_total = 0
count_min5plus = 0

for message in consumer:
    try:
        value = message.value
        parts = value.strip().split(",")
        
        if len(parts) >= 3:
            timestamp, user_id, request = parts[0], parts[1], parts[2]
            match = pattern.search(request)
            
            if match:
                movie_name = match.group("movie_name").replace("+", " ")
                minute = int(match.group("minute"))
                
                count_total += 1
                if minute >= 5:
                    count_min5plus += 1
                    print(f"✅ Minute >=5 | User: {user_id}, Movie: {movie_name}, Minute: {minute}")
                else:
                    print(f"⏳ Minute <5  | User: {user_id}, Movie: {movie_name}, Minute: {minute}")

                # Every 50 messages, print a small summary
                if count_total % 50 == 0:
                    print(f"\nSummary so far: {count_min5plus}/{count_total} messages had minute >=5\n")

    except Exception as e:
        print(f"❌ Error parsing message: {e}")

# online/telemetry.py
# Add at the top of the file
import logging

logging.basicConfig(
    filename="telemetry_debug.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


import json
import os
from datetime import datetime
import threading


class TelemetryCollector:
    """Class for collecting and managing telemetry data for online evaluation"""

    def __init__(self, data_dir="telemetry"):
        """Initialize the telemetry collector"""
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

        # Data structures to store telemetry
        self.recommendation_log = {}  # Format: {user_id: [rec_events]}
        self.watch_log = {}  # Format: {user_id: {movie_id: [watch_events]}}
        self.rating_log = {}  # Format: {user_id: {movie_id: rating_event}}

        # Load existing data if available
        self._load_data()

        # Lock for thread safety
        self.lock = threading.Lock()

    def _load_data(self):
        """Load existing telemetry data from disk"""
        try:
            rec_file = os.path.join(self.data_dir, "recommendations.json")
            if os.path.exists(rec_file):
                with open(rec_file, "r") as f:
                    self.recommendation_log = json.load(f)

            watch_file = os.path.join(self.data_dir, "watches.json")
            if os.path.exists(watch_file):
                with open(watch_file, "r") as f:
                    self.watch_log = json.load(f)

            rating_file = os.path.join(self.data_dir, "ratings.json")
            if os.path.exists(rating_file):
                with open(rating_file, "r") as f:
                    self.rating_log = json.load(f)

            print(
                f"Loaded existing telemetry data: {len(self.recommendation_log)} users with recommendations"
            )
        except Exception as e:
            print(f"Error loading telemetry data: {str(e)}")

    def save_data(self):
        """Save telemetry data to disk"""
        with self.lock:
            try:
                # Save recommendation logs
                with open(
                    os.path.join(self.data_dir, "recommendations.json"), "w"
                ) as f:
                    json.dump(self.recommendation_log, f)

                # Save watch logs
                with open(os.path.join(self.data_dir, "watches.json"), "w") as f:
                    json.dump(self.watch_log, f)

                # Save rating logs
                with open(os.path.join(self.data_dir, "ratings.json"), "w") as f:
                    json.dump(self.rating_log, f)

                print(f"Telemetry data saved at {datetime.now().isoformat()}")
            except Exception as e:
                print(f"Error saving telemetry data: {str(e)}")

    def log_recommendation(self, user_id, recommendations):
        """Log a recommendation event"""
        """Log a recommendation event"""
        logging.debug(f"log_recommendation called for user {user_id}")
        print(f"DEBUG: Logging recommendation for user {user_id}")
        with self.lock:
            # Convert to strings for JSON serialization
            user_id = str(user_id)
            recommendations = [str(r) for r in recommendations]

            # Create event
            event = {
                "timestamp": datetime.now().isoformat(),
                "recommendations": recommendations,
            }

            # Add to log
            if user_id not in self.recommendation_log:
                self.recommendation_log[user_id] = []

            self.recommendation_log[user_id].append(event)

            # Save data periodically (every 100 recommendation events)
            recs_count = sum(len(recs) for recs in self.recommendation_log.values())
            if recs_count % 100 == 0:
                self.save_data()
            self.save_data()

    def log_watch(self, user_id, movie_id, minute):
        """Log a watch event"""
        logging.debug(f"log_watch called for user {user_id}, movie {movie_id}")
        print(f"DEBUG: Logging watch for user {user_id}, movie {movie_id}")
        with self.lock:
            # Convert to strings for JSON serialization
            user_id = str(user_id)
            movie_id = str(movie_id)

            # Create event
            event = {"timestamp": datetime.now().isoformat(), "minute": minute}

            # Add to log
            if user_id not in self.watch_log:
                self.watch_log[user_id] = {}

            if movie_id not in self.watch_log[user_id]:
                self.watch_log[user_id][movie_id] = []

            self.watch_log[user_id][movie_id].append(event)

            # Save data periodically (every 100 watch events)
            watches_count = sum(len(movies) for user, movies in self.watch_log.items())
            if watches_count % 100 == 0:
                self.save_data()
            self.save_data()

    def log_rating(self, user_id, movie_id, rating):
        """Log a rating event"""
        logging.debug(f"log_rating called for user {user_id}, movie {movie_id}")
        print(f"DEBUG: Logging rating for user {user_id}, movie {movie_id}")

        with self.lock:
            # Convert to strings for JSON serialization
            user_id = str(user_id)
            movie_id = str(movie_id)

            # Create event
            event = {"timestamp": datetime.now().isoformat(), "rating": rating}

            # Add to log
            if user_id not in self.rating_log:
                self.rating_log[user_id] = {}

            self.rating_log[user_id][movie_id] = event

            # Save data periodically (every 50 rating events)
            ratings_count = sum(len(movies) for user, movies in self.rating_log.items())
            if ratings_count % 50 == 0:
                self.save_data()
            self.save_data()


# Create a singleton instance
telemetry_collector = TelemetryCollector()

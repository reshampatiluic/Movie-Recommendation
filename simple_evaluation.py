"""
Simple script to demonstrate online evaluation for the movie recommendation system.
This script:
1. Simulates telemetry data (recommendations, watches, ratings)
2. Computes evaluation metrics
3. Generates a report
"""

import json
import os
import random
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np

# Create directories for data
os.makedirs("simulated_telemetry", exist_ok=True)
os.makedirs("evaluation_results", exist_ok=True)


def generate_sample_data(num_users=100, num_days=14):
    """Generate simulated telemetry data for demonstration"""
    users = [f"user_{i}" for i in range(num_users)]
    movies = [f"movie_{i}" for i in range(500)]

    # Initialize data structures
    recommendations = {}
    watches = {}
    ratings = {}

    # Generate data over the past num_days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=num_days)

    # For each user
    for user in users:
        recommendations[user] = []
        watches[user] = {}
        ratings[user] = {}

        # Generate 1-5 recommendation events per user
        num_rec_events = random.randint(1, 5)
        for _ in range(num_rec_events):
            # Random timestamp within the date range
            days_ago = random.randint(0, num_days)
            timestamp = (end_date - timedelta(days=days_ago)).isoformat()

            # Generate 10-20 recommendations
            rec_movies = random.sample(movies, random.randint(10, 20))

            # Add recommendation event
            recommendations[user].append(
                {"timestamp": timestamp, "recommendations": rec_movies}
            )

            # Simulate some watches and ratings from these recommendations
            for movie in rec_movies:
                # 30% chance user watches the movie
                if random.random() < 0.3:
                    watches[user][movie] = []

                    # Generate 1-60 minutes of watching
                    max_minute = random.randint(1, 60)
                    for minute in range(1, max_minute + 1):
                        watches[user][movie].append(
                            {"timestamp": timestamp, "minute": minute}
                        )

                    # 50% chance user rates the movie if they watched it
                    if random.random() < 0.5:
                        ratings[user][movie] = {
                            "timestamp": timestamp,
                            "rating": random.randint(1, 5),
                        }

    # Save the simulated data
    with open("simulated_telemetry/recommendations.json", "w") as f:
        json.dump(recommendations, f, indent=2)

    with open("simulated_telemetry/watches.json", "w") as f:
        json.dump(watches, f, indent=2)

    with open("simulated_telemetry/ratings.json", "w") as f:
        json.dump(ratings, f, indent=2)

    return recommendations, watches, ratings


def compute_ctr(recommendations, watches):
    """Compute Click-Through Rate"""
    clicks = 0
    total_recommendations = 0

    for user_id, user_recommendations in recommendations.items():
        for rec_event in user_recommendations:
            recommended_movies = rec_event["recommendations"]
            total_recommendations += len(recommended_movies)

            if user_id in watches:
                for movie_id in recommended_movies:
                    if movie_id in watches[user_id]:
                        clicks += 1

    if total_recommendations == 0:
        return 0

    return (clicks / total_recommendations) * 100


def compute_wtr(recommendations, watches, min_watch_minutes=5):
    """Compute Watch-Through Rate"""
    significant_watches = 0
    total_recommendations = 0

    for user_id, user_recommendations in recommendations.items():
        for rec_event in user_recommendations:
            recommended_movies = rec_event["recommendations"]
            total_recommendations += len(recommended_movies)

            if user_id in watches:
                for movie_id in recommended_movies:
                    if movie_id in watches[user_id]:
                        max_minute = max(
                            event["minute"] for event in watches[user_id][movie_id]
                        )
                        if max_minute >= min_watch_minutes:
                            significant_watches += 1

    if total_recommendations == 0:
        return 0

    return (significant_watches / total_recommendations) * 100


def compute_avg_rating(recommendations, ratings):
    """Compute average rating for recommended movies"""
    total_ratings = 0
    sum_ratings = 0

    for user_id, user_recommendations in recommendations.items():
        for rec_event in user_recommendations:
            recommended_movies = rec_event["recommendations"]

            if user_id in ratings:
                for movie_id in recommended_movies:
                    if movie_id in ratings[user_id]:
                        sum_ratings += ratings[user_id][movie_id]["rating"]
                        total_ratings += 1

    if total_ratings == 0:
        return 0

    return sum_ratings / total_ratings


def generate_report(recommendations, watches, ratings):
    """Generate evaluation report"""
    ctr = compute_ctr(recommendations, watches)
    wtr = compute_wtr(recommendations, watches)
    avg_rating = compute_avg_rating(recommendations, ratings)

    report = {
        "generated_at": datetime.now().isoformat(),
        "metrics": {"ctr": ctr, "wtr": wtr, "avg_rating": avg_rating},
        "data_summary": {
            "users": len(recommendations),
            "recommendation_events": sum(
                len(recs) for recs in recommendations.values()
            ),
            "watches": sum(len(movies) for user, movies in watches.items()),
            "ratings": sum(len(movies) for user, movies in ratings.items()),
        },
    }

    # Save report
    with open("evaluation_results/online_evaluation_report.json", "w") as f:
        json.dump(report, f, indent=2)

    # Print summary
    print("\nEvaluation Results:")
    print(f"Click-Through Rate: {ctr:.2f}%")
    print(f"Watch-Through Rate: {wtr:.2f}%")
    print(f"Average Rating: {avg_rating:.2f}/5")

    return report


def generate_visualizations(recommendations, watches, ratings):
    """Generate visualizations for the report"""

    # Calculate metrics per day for the last 14 days
    days = 14
    end_date = datetime.now()

    daily_ctr = []
    daily_wtr = []
    daily_ratings = []
    dates = []

    for day in range(days):
        start_time = end_date - timedelta(days=day + 1)
        end_time = end_date - timedelta(days=day)
        dates.append(start_time.strftime("%m-%d"))

        # Filter recommendations for this day
        day_recommendations = {}
        for user, recs in recommendations.items():
            day_recommendations[user] = []
            for rec in recs:
                rec_time = datetime.fromisoformat(rec["timestamp"])
                if start_time <= rec_time < end_time:
                    day_recommendations[user].append(rec)

        # Compute metrics for this day
        if sum(len(recs) for recs in day_recommendations.values()) > 0:
            ctr = compute_ctr(day_recommendations, watches)
            wtr = compute_wtr(day_recommendations, watches)
            avg_rating = compute_avg_rating(day_recommendations, ratings)
        else:
            ctr = 0
            wtr = 0
            avg_rating = 0

        daily_ctr.append(ctr)
        daily_wtr.append(wtr)
        daily_ratings.append(avg_rating)

    # Reverse lists to show chronological order
    dates.reverse()
    daily_ctr.reverse()
    daily_wtr.reverse()
    daily_ratings.reverse()

    # Create visualizations
    plt.figure(figsize=(15, 10))

    # CTR subplot
    plt.subplot(2, 2, 1)
    plt.plot(dates, daily_ctr, marker="o", linestyle="-", color="blue")
    plt.title("Click-Through Rate")
    plt.xlabel("Date")
    plt.ylabel("CTR (%)")
    plt.xticks(rotation=45)
    plt.grid(True, linestyle="--", alpha=0.7)

    # WTR subplot
    plt.subplot(2, 2, 2)
    plt.plot(dates, daily_wtr, marker="o", linestyle="-", color="green")
    plt.title("Watch-Through Rate")
    plt.xlabel("Date")
    plt.ylabel("WTR (%)")
    plt.xticks(rotation=45)
    plt.grid(True, linestyle="--", alpha=0.7)

    # Ratings subplot
    plt.subplot(2, 2, 3)
    plt.plot(dates, daily_ratings, marker="o", linestyle="-", color="red")
    plt.title("Average Rating")
    plt.xlabel("Date")
    plt.ylabel("Rating (0-5)")
    plt.ylim(0, 5)
    plt.xticks(rotation=45)
    plt.grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()

    # Save the visualization
    plt.savefig("evaluation_results/metrics_over_time.png")
    print("Visualization saved to evaluation_results/metrics_over_time.png")

    return "evaluation_results/metrics_over_time.png"


if __name__ == "__main__":
    print("Generating sample telemetry data...")
    recommendations, watches, ratings = generate_sample_data()

    print("Computing evaluation metrics...")
    report = generate_report(recommendations, watches, ratings)

    print("Generating visualizations...")
    viz_path = generate_visualizations(recommendations, watches, ratings)

    print("\nOnline evaluation complete!")
    print(f"Report saved to: evaluation_results/online_evaluation_report.json")
    print(f"Visualization saved to: {viz_path}")

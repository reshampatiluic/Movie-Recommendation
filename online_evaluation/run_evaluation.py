import argparse
from datetime import datetime
from online_evaluation.metrics import MetricsComputer

def main():
    parser = argparse.ArgumentParser(description="Run online evaluation of the recommendation system")
    parser.add_argument("--visualize", action="store_true", help="Generate visualizations")
    parser.add_argument("--telemetry_dir", default="telemetry", help="Path to telemetry data directory")
    parser.add_argument("--output_dir", default="evaluation_results", help="Directory to save evaluation results")
    args = parser.parse_args()

    print("="*60)
    print(f"📊 Starting Online Evaluation at {datetime.now().isoformat()}")
    print("="*60)

    # Initialize metrics computer
    metrics = MetricsComputer(telemetry_dir=args.telemetry_dir, output_dir=args.output_dir)

    # Generate report
    report = metrics.generate_report()

    # Print Summary
    data_summary = report.get('data_summary', {})
    print("\n🧾 Evaluation Summary")
    print("-"*40)
    print(f"Users with recommendations: {data_summary.get('users_with_recommendations', 0)}")
    print(f"Users with watches:         {data_summary.get('users_with_watches', 0)}")
    print(f"Users with ratings:         {data_summary.get('users_with_ratings', 0)}")

    # Print All-Time Metrics
    print("\n📈 Metrics (All Time)")
    print("-"*40)
    all_time = report.get('metrics', {}).get('all_time', {})
    print(f"Click-Through Rate:   {all_time.get('ctr', 0):.2f}%")
    print(f"Watch-Through Rate:   {all_time.get('wtr', 0):.2f}%")
    print(f"Average Rating:       {all_time.get('avg_rating', 0):.2f} / 5")

    # Visualizations
    if args.visualize:
        print("\n🖼️  Generating visualizations...")
        plots = metrics.generate_visualizations()
        for plot in plots:
            print(f"  ✔ {plot}")
        print(f"\n✅ {len(plots)} visualization files generated")

    print(f"\n✅ Evaluation Completed at {datetime.now().isoformat()}")
    print("="*60)

if __name__ == "__main__":
    main()
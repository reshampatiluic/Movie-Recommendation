from flask import Flask, jsonify
import pickle
import pandas as pd
from prometheus_client import make_wsgi_app
from werkzeug.middleware.dispatcher import DispatcherMiddleware


app = Flask(__name__)
# Attach /metrics endpoint to Flask app
app.wsgi_app = DispatcherMiddleware(app.wsgi_app, {
    '/metrics': make_wsgi_app()
})


# Load your model
model = pickle.load(open("trained_models/trained_model.pkl", "rb"))

# Load your movie data directly (don't use `data` module)
movies_df = pd.read_csv("data/final_processed_data.csv")

# Helper function to recommend movies
def recommend_movies(user_id, top_k=20):
    all_movies = movies_df["Movie_Name"].unique()
    predictions = []
    for movie in all_movies:
        try:
            pred = model.predict(user_id, movie).est
            predictions.append((movie, pred))
        except Exception as e:
            continue
    predictions.sort(key=lambda x: x[1], reverse=True)
    top_movies = [m for m, _ in predictions[:top_k]]
    return top_movies

@app.route("/recommendations/<int:user_id>", methods=["GET"])
def recommendations(user_id):
    try:
        recs = recommend_movies(user_id)
        return jsonify({"user_id": user_id, "recommendations": recs})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run server
if __name__ == "__main__":
    import sys
    port = 5050  # Default port
    if len(sys.argv) > 2 and sys.argv[1] == "--port":
        port = int(sys.argv[2])
    app.run(host='0.0.0.0', port=8000, debug=True)

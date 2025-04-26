from fastapi import FastAPI, Path
from app.models import SVDRecommender
from app.logger import logger
from app.schema import RecommendationsResponse
<<<<<<< HEAD
from datetime import datetime
from app.provenance import log_provenance 
=======
>>>>>>> 0b1a69eb1906812fea80cf35c163e016fd45e467

logger.info("Initializing SVD model")
svd = SVDRecommender()
logger.info("Finished initializing SVD model")

app = FastAPI()


@app.get(
    "/recommend/{user_id}",
    response_model=RecommendationsResponse,
    responses={200: {"description": "Success"}},
)
async def get_recommendations(
    user_id: int = Path(..., description="User ID to get recommendations for"),
):
    """
<<<<<<< HEAD
    Generates recommendations for a given user and logs provenance information
    """
    # Generate recommendations
    recommendations, _ = svd.recommend(user_id)

    # Convert recommendations to strings for the schema
    recommendations = [str(rec) for rec in recommendations]  # 🔥 This is essential

    # Provenance Logging
    provenance_info = {
        "timestamp": datetime.utcnow().isoformat(),
        "user_id": user_id,
        "model_version": str(svd.saved_model_path),
        "data_version": str(svd.dataset_path),
        "num_recommendations": len(recommendations),
    }
    logger.info(f"Provenance Log: {provenance_info}")

    # Also log to provenance file
    from app.provenance import log_provenance  # Make sure this import is present
    log_provenance(
        user_id=user_id,
        model_version=str(svd.saved_model_path),
        data_version=str(svd.dataset_path),
        recommendations=recommendations
    )

    return RecommendationsResponse(recommendations=recommendations)
=======
    Generates recommendations for a given user
    """
    recommendations, _ = svd.recommend(user_id)
    return RecommendationsResponse(recommendations=recommendations)
>>>>>>> 0b1a69eb1906812fea80cf35c163e016fd45e467

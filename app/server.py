from fastapi import FastAPI, Path
from app.models import SVDRecommender
from app.logger import logger
from app.schema import RecommendationsResponse

logger.info("Initializing SVD model")
svd = SVDRecommender()
logger.info("Finished initializing SVD model")

app = FastAPI()


@app.get("/recommend/{user_id}", response_model=RecommendationsResponse, responses={200: {"description": "Success"}})

async def get_recommendations(
        user_id: int = Path(..., description="User ID to get recommendations for"),
):
    """
    Generates recommendations for a given user
    """
    recommendations, _ = svd.recommend(user_id)
    return RecommendationsResponse(recommendations=recommendations)
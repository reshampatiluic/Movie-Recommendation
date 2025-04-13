from pydantic import BaseModel


class RecommendationsResponse(BaseModel):
    recommendations: list[str]

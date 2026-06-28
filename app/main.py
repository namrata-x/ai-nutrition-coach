from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from app.models import MealRequest, MealResponse, IngredientBreakdown
from app.agent import analyze_meal_with_agent
from app.classifier import load_model, predict_calorie_range
from app.verification import verify_results
from dotenv import load_dotenv

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


app = FastAPI(
    title="AI Nutrition Coach",
    description="Estimates meal calories using Mistral Agents and PyTorch",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/analyze-meal", response_model=MealResponse)
async def analyze_meal(request: MealRequest):
    try:
        agent_result = await analyze_meal_with_agent(
            request.meal_description
        )

        classifier_result = predict_calorie_range(
            request.meal_description
        )

        verification = verify_results(
            agent_calories=agent_result["estimated_calories"],
            classifier_range=classifier_result["predicted_range"]
        )

        breakdown = [
            IngredientBreakdown(
                ingredient=item["ingredient"],
                amount=item["amount"],
                calories=item["calories"]
            )
            for item in agent_result.get("breakdown", [])
        ]

        return MealResponse(
            meal_description=request.meal_description,
            estimated_calories=agent_result["estimated_calories"],
            calorie_range=classifier_result["predicted_range"],
            breakdown=breakdown,
            confidence=verification["confidence"],
            verification_status=verification["status"]
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )
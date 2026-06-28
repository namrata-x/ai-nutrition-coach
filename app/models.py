from pydantic import BaseModel, validator
from typing import Optional

class MealRequest(BaseModel):
    meal_description: str

    @validator("meal_description")
    def must_not_be_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("Meal description cannot be empty")
        if len(v) > 500:
            raise ValueError("Meal description too long")
        return v.strip()


class IngredientBreakdown(BaseModel):
    ingredient: str
    amount: str
    calories: int


class MealResponse(BaseModel):
    meal_description: str
    estimated_calories: int
    calorie_range: str
    breakdown: list[IngredientBreakdown]
    confidence: str
    verification_status: str
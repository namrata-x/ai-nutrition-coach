import httpx
import os
import re

USDA_API_KEY = os.getenv("USDA_API_KEY", "DEMO_KEY")
USDA_BASE_URL = "https://api.nal.usda.gov/fdc/v1"


async def get_nutrition_data(ingredient: str, amount: str) -> dict:
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{USDA_BASE_URL}/foods/search",
            params={
                "query": ingredient,
                "api_key": USDA_API_KEY,
                "pageSize": 1,
                "dataType": "SR Legacy,Foundation"
            },
            timeout=10.0
        )

    if response.status_code != 200:
        return {
            "ingredient": ingredient,
            "amount": amount,
            "calories_per_100g": None,
            "found": False,
            "note": "USDA API error. Please estimate from general knowledge."
        }

    data = response.json()

    if not data.get("foods"):
        return {
            "ingredient": ingredient,
            "amount": amount,
            "calories_per_100g": None,
            "found": False,
            "note": "Not found. Please estimate from general knowledge."
        }

    food = data["foods"][0]

    calories_per_100g = next(
        (n["value"] for n in food.get("foodNutrients", [])
         if n.get("nutrientId") == 1008),
        None
    )

    grams = parse_amount_to_grams(amount)

    calories_for_portion = None
    if calories_per_100g and grams:
        calories_for_portion = round(calories_per_100g * grams / 100)

    return {
        "ingredient": ingredient,
        "amount": amount,
        "grams": grams,
        "calories_per_100g": calories_per_100g,
        "calories_for_portion": calories_for_portion,
        "food_name": food.get("description", ingredient),
        "found": True
    }


def parse_amount_to_grams(amount: str) -> float:
    amount_lower = amount.lower().strip()

    conversions = {
        "1 cup": 240,
        "half cup": 120,
        "0.5 cup": 120,
        "1 tbsp": 15,
        "1 tsp": 5,
        "1 slice": 30,
        "1 piece": 100,
        "1 medium": 150,
        "1 large": 200,
        "1 small": 80,
        "100g": 100,
        "150g": 150,
        "200g": 200,
        "250g": 250,
    }

    for key, grams in conversions.items():
        if key in amount_lower:
            return grams

    match = re.search(r"(\d+)\s*g", amount_lower)
    if match:
        return float(match.group(1))

    return 100
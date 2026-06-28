# AI Nutrition Coach

A backend API that estimates calories from a natural language meal description using a Mistral AI Agent and a PyTorch text classifier.

## What It Does

You type a meal description. The app returns an estimated calorie count with a per-ingredient breakdown and a confidence level.

**Input:**
```json
{
  "meal_description": "grilled chicken and brown rice"
}
```

**Output:**
```json
{
  "meal_description": "grilled chicken and brown rice",
  "estimated_calories": 1102,
  "calorie_range": "medium",
  "breakdown": [
    { "ingredient": "grilled chicken breast", "amount": "150g", "calories": 226 },
    { "ingredient": "brown rice", "amount": "1 cup", "calories": 876 }
  ],
  "confidence": "medium",
  "verification_status": "minor_disagreement"
}
```

## How It Works

1. A Mistral Agent parses the meal into ingredients and calls the USDA FoodData Central API for each one to get real calorie data.
2. A PyTorch text classifier independently predicts whether the meal is low, medium, or high calorie.
3. A verification layer compares both results and returns a confidence level.

## Tech Stack

- **FastAPI** — backend API
- **Pydantic** — request and response validation
- **Mistral Agents** — tool calling to query nutrition data
- **USDA FoodData Central API** — real nutrition database
- **PyTorch** — text classification model
- **scikit-learn** — TF-IDF vectorization
- **vLLM / SGLang** — local inference benchmarking

## Setup

```bash
git clone https://github.com/namrata-x/ai-nutrition-coach
cd ai-nutrition-coach
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install torch scikit-learn
cp .env.example .env
# Add your MISTRAL_API_KEY to .env
python training/train.py
uvicorn app.main:app --reload
```

## Test It

```bash
curl -X POST http://127.0.0.1:8000/analyze-meal \
  -H "Content-Type: application/json" \
  -d '{"meal_description": "grilled chicken and brown rice"}'
```

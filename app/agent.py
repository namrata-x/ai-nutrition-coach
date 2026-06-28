import json
import os
from mistralai.client import Mistral
from app.nutrition_api import get_nutrition_data

client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))

NUTRITION_TOOL = {
    "type": "function",
    "function": {
        "name": "get_nutrition_data",
        "description": "Look up calorie data for a food ingredient from USDA database.",
        "parameters": {
            "type": "object",
            "properties": {
                "ingredient": {
                    "type": "string",
                    "description": "Food ingredient name e.g. grilled chicken breast"
                },
                "amount": {
                    "type": "string",
                    "description": "Amount e.g. 150g, 1 cup, 1 medium"
                }
            },
            "required": ["ingredient", "amount"]
        }
    }
}

SYSTEM_PROMPT = """You are a nutrition assistant.
When given a meal description:
1. Identify each ingredient and estimate a realistic portion size
2. Call get_nutrition_data for EACH ingredient separately
3. Calculate total calories from all ingredients
4. Return ONLY valid JSON in this exact format with no extra text:
{
  "estimated_calories": <total integer>,
  "breakdown": [
    {"ingredient": "<name>", "amount": "<amount>", "calories": <integer>}
  ]
}"""


async def analyze_meal_with_agent(meal_description: str) -> dict:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Analyze this meal: {meal_description}"}
    ]

    max_tool_calls = 10
    tool_call_count = 0

    while True:
        response = client.chat.complete(
            model="mistral-small-latest",
            messages=messages,
            tools=[NUTRITION_TOOL],
            tool_choice="auto",
            temperature=0.1
        )

        message = response.choices[0].message

        if message.tool_calls:
            messages.append({"role": "assistant", "content": None, "tool_calls": message.tool_calls})

            for tool_call in message.tool_calls:
                tool_call_count += 1

                if tool_call_count > max_tool_calls:
                    break

                args = json.loads(tool_call.function.arguments)

                result = await get_nutrition_data(
                    ingredient=args["ingredient"],
                    amount=args["amount"]
                )

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result)
                })

        else:
            raw_text = message.content.strip()

            if raw_text.startswith("```"):
                lines = raw_text.split("\n")
                raw_text = "\n".join(lines[1:-1])

            return json.loads(raw_text)
import json
import os
from mistralai import Mistral
from app.nutrition_api import get_nutrition_data

client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))

NUTRITION_TOOL = {
    "type": "function",
    "function": {
        "name": "get_nutrition_data",
        "description": "Look up calorie data for a food ingredient or dish from USDA database.",
        "parameters": {
            "type": "object",
            "properties": {
                "ingredient": {
                    "type": "string",
                    "description": "Food ingredient or dish name e.g. paneer biryani"
                },
                "amount": {
                    "type": "string",
                    "description": "Amount in grams e.g. 150g"
                }
            },
            "required": ["ingredient", "amount"]
        }
    }
}

SYSTEM_PROMPT = """You are a nutrition assistant.
Estimate calories for meals by looking up each ingredient.

STRICT RULE: Call get_nutrition_data only ONE time per response.
Never make more than one tool call at a time.
Wait for the result before making the next call.

Use these portion sizes:
- Protein (meat, fish, eggs, paneer, tofu): 150g
- Grain or starch (rice, pasta, bread, naan, roti): 150g
- Vegetable: 80g
- Sauce, gravy, dressing: 30g
- Dairy (cheese, yogurt, cream): 50g
- Fruit: 100g
- Legume (lentils, dal, beans): 150g
- Ghee or oil: 30g

After looking up ALL ingredients one by one return ONLY this JSON:
{
  "estimated_calories": <total integer>,
  "breakdown": [
    {"ingredient": "<name>", "amount": "<Xg>", "calories": <integer>}
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
            temperature=0.1,
            parallel_tool_calls=False
        )

        message = response.choices[0].message

        if message.tool_calls:

            # Only process the FIRST tool call to enforce one at a time
            tool_call = message.tool_calls[0]

            # Append assistant message with only the first tool call
            messages.append({
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments
                        }
                    }
                ]
            })

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

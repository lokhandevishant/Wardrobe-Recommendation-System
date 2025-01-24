import os
from vertexai.preview.generative_models import GenerativeModel
import json

def generate_outfit_combinations(items):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "ambient-skyline-430315-k3-c0bde7455bdd.json"
    model = GenerativeModel("gemini-1.0-pro")

    items_str = json.dumps(items, indent=2)
    prompt = f"""Given the following clothing items:

{items_str}

Generate 7 outfit combinations for a week, ensuring color coordination and style matching. 
Each outfit should consist of one top and one bottom.
Consider the following rules:
1. Match colors that complement according to the user and avoid using both dark and bold colors for tops and bottoms.
2. Pair similar styles (e.g., casual with casual, formal with formal).
3. Avoid repeating the same item in the same week for shirts.
4. Try to create a variety of looks throughout the week.

Return only the results as a JSON-formatted list of dictionaries. Each dictionary should represent a day and contain 'day', 'top', and 'bottom' keys with the corresponding item IDs. The output should be valid JSON that can be parsed by Python's json.loads() function. Do not include any additional text or comments.

Example output format:
[
  {{"day": 1, "top": 1, "bottom": 4}},
  {{"day": 2, "top": 2, "bottom": 5}},
  ...
]
"""

    response = model.generate_content(prompt)
    
    try:
        # Extract and clean the JSON part from the response text
        response_text = response.text.strip()
        print(f"Raw response: {response_text}")  # Log raw response for debugging
        
        json_start = response_text.find('[')
        json_end = response_text.rfind(']')
        if json_start == -1 or json_end == -1:
            raise ValueError("No valid JSON array found in the response.")
        
        json_str = response_text[json_start:json_end + 1]
        outfit_combinations = json.loads(json_str)
        
        # Validate the structure of the parsed data``
        if not isinstance(outfit_combinations, list):
            raise ValueError("Response is not a list")
        
        for outfit in outfit_combinations:
            if not isinstance(outfit, dict) or 'day' not in outfit or 'top' not in outfit or 'bottom' not in outfit:
                raise ValueError("Invalid outfit structure")
        
        return outfit_combinations
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        print(f"Raw response: {response.text}")
        return []
    except ValueError as e:
        print(f"Error validating response structure: {e}")
        return []
    except Exception as e:
        print(f"Unexpected error: {e}")
        return []

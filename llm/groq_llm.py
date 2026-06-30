import os
import json

from dotenv import load_dotenv
from groq import Groq

load_dotenv()


def get_structured_completion(prompt: str, response_model):
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    schema = response_model.model_json_schema()

    final_prompt = f"""
{prompt}

Return only valid JSON matching this schema:
{json.dumps(schema, indent=2)}
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "user",
                "content": final_prompt
            }
        ],
        temperature=0
    )

    content = response.choices[0].message.content

    data = json.loads(content)

    return response_model.model_validate(data)

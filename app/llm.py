import json
import os

import ollama
from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = os.getenv("OLLAMA_MODEL", "llama3:latest")
OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://localhost:11434",
)

client = ollama.Client(host=OLLAMA_BASE_URL)


def classify_text(text: str, categories: list[str]) -> str:
    if not categories:
        raise ValueError("At least one category is required.")

    response = client.chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "Classify the given text into exactly one of the "
                    "provided categories. Return only the category name."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Categories: {json.dumps(categories)}\n"
                    f"Text: {text}"
                ),
            },
        ],
    )

    result = response["message"]["content"].strip()

    for category in categories:
        if result.casefold() == category.casefold():
            return category

    raise ValueError(f"Invalid category returned by LLM: {result}")


def generate_workflow(description: str) -> dict:
    response = client.chat(
        model=MODEL_NAME,
        format="json",
        messages=[
            {
                "role": "system",
                "content": """
Create a workflow definition as a JSON object.

The required structure is:
{
  "name": "Workflow Name",
  "steps": [
    {
      "name": "Step Name",
      "type": "TRANSFORM_TEXT",
      "config": {
        "operation": "uppercase",
        "input_key": "customer_name",
        "output_key": "normalized_name"
      }
    }
  ]
}

Supported types and configurations:

TRANSFORM_TEXT:
operation must be uppercase, lowercase, or strip.
Required keys: operation, input_key, output_key.

CALCULATE:
operation must be add, subtract, multiply, or divide.
Required keys: operation, left_key, right_key, output_key.

CLASSIFY_TEXT:
Required keys: input_key, output_key, categories.
categories must be a list of strings.

Generate one or more steps in the correct sequence.
Only use the supported types and operations.
Return valid JSON only. Do not include Markdown.
Do not invent input values.
""",
            },
            {
                "role": "user",
                "content": description,
            },
        ],
    )

    workflow = json.loads(response["message"]["content"])

    if not isinstance(workflow, dict):
        raise ValueError("Generated workflow must be a JSON object.")

    return workflow
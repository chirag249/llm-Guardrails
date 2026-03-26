
import os
import google.genai as genai
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

models_to_test = [
    "gemini-flash-latest",
    "gemini-1.5-flash",
    "gemini-2.0-flash",
    "gemini-2.5-flash",
    "gemini-3-flash-preview"
]

for model in models_to_test:
    print(f"Testing model: {model}")
    try:
        response = client.models.generate_content(
            model=model,
            contents="test"
        )
        print(f"   Success")
    except Exception as e:
        print(f"   Failed: {e}")

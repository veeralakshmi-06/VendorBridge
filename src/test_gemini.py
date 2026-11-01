
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise SystemExit("ERROR: GEMINI_API_KEY was not found in .env")

client = genai.Client(
    api_key=api_key,
    http_options=types.HttpOptions(timeout=30000)
)

try:
    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents="Reply with exactly: Gemini API connection successful."
    )
    print("SUCCESS")
    print(response.text)

except Exception as error:
    print("Gemini API request failed.")
    print(type(error).__name__)
    print(error)
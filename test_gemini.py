from dotenv import dotenv_values
from google import genai

config = dotenv_values(".env")
model = config.get("GEMINI_MODEL")
api_key = config.get("GEMINI_API_KEY")

try:
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model,
        contents="Reply with OK."
    )
    print("SUCCESS:", response.text)

except Exception as e:
    print("ERROR TYPE:", type(e).__name__)
    print("HTTP CODE:", getattr(e, "code", "not exposed"))
    print("STATUS:", getattr(e, "status", "not exposed"))
    detail = str(getattr(e, "message", "") or e)
    if api_key:
        detail = detail.replace(api_key, "[REDACTED]")
    print("DETAIL:", detail[:500])
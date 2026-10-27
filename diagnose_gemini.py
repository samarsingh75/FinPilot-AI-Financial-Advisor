from dotenv import load_dotenv, dotenv_values
from google import genai

load_dotenv(".env", override=True)
config = dotenv_values(".env")
key = config.get("GEMINI_API_KEY", "").strip()
model = config.get("GEMINI_MODEL", "gemini-3.1-flash-lite")

try:
    client = genai.Client(api_key=key)
    response = client.models.generate_content(
        model=model,
        contents="Reply with exactly: OK"
    )
    print("SUCCESS:", response.text)

except Exception as exc:
    print("ERROR TYPE:", type(exc).__name__)
    print("HTTP CODE:", getattr(exc, "code", "not exposed"))
    print("STATUS:", getattr(exc, "status", "not exposed"))
    detail = str(getattr(exc, "message", "") or exc)
    if key:
        detail = detail.replace(key, "[REDACTED]")
    print("DETAIL:", detail[:500])
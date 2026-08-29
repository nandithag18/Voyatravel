import json
import os
import uuid
from datetime import date

from google import genai
from dotenv import load_dotenv
from pydantic import ValidationError

from schemas import PassengerInfo, LocationEntity, ResolutionStatus, TripIntent

load_dotenv()
_CLIENT = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

_SYSTEM_PROMPT = """You extract structured trip-booking details from a spoken transcript.

Return ONLY valid JSON (no markdown fences, no prose) with exactly this shape:
{{
  "origin_text": "<place name as spoken>",
  "destination_text": "<place name as spoken>",
  "mode": "TRAIN" | "BUS" | "FLIGHT" | "ANY",
  "travel_date": "YYYY-MM-DD",
  "return_date": "YYYY-MM-DD" or null,
  "budget_max_inr": <number> or null,
  "passengers": {{"adults": <int>, "children": <int>, "infants": <int>}}
}}

Today's date is {today}. Resolve relative dates ("tomorrow", "next Friday") to
actual calendar dates. If a field truly cannot be determined, use null (for
optional fields) or your best reasonable default (1 adult passenger if
unspecified).

Transcript: "{transcript}"
"""


def _call_llm(transcript: str) -> dict:
    prompt = _SYSTEM_PROMPT.format(today=date.today().isoformat(), transcript=transcript)
    response = _CLIENT.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt,
    )
    text = response.text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.split("\n", 1)[1] if "\n" in text else text
        text = text.rsplit("```", 1)[0]
    return json.loads(text)


def parse_intent(transcript: str, session_id: uuid.UUID | None = None) -> TripIntent:
    session_id = session_id or uuid.uuid4()
    last_error = None

    for attempt in range(2):  # one retry
        try:
            raw = _call_llm(transcript)
            intent = TripIntent(
                session_id=session_id,
                origin=LocationEntity(raw_text=raw["origin_text"], status=ResolutionStatus.UNRESOLVED),
                destination=LocationEntity(raw_text=raw["destination_text"], status=ResolutionStatus.UNRESOLVED),
                mode=raw.get("mode", "ANY"),
                travel_date=raw["travel_date"],
                return_date=raw.get("return_date"),
                budget_max_inr=raw.get("budget_max_inr"),
                passengers=PassengerInfo(**raw.get("passengers", {"adults": 1})),
                raw_transcript=transcript,
            )
            return intent
        except (json.JSONDecodeError, KeyError, ValidationError) as e:
            last_error = e
            continue

    raise ValueError(f"Failed to parse intent after retry: {last_error}")


if __name__ == "__main__":
    test_transcript = "I want a train from Chennai to Bangalore tomorrow morning for 2 adults"
    result = parse_intent(test_transcript)
    print(result.model_dump_json(indent=2))
"""
Entity Resolution Engine — FR-5, FR-6 (SRS Sec 3.3)

Takes the LocationEntity placeholders Person 1's parse_intent() hands off
(status=UNRESOLVED, resolved=None, candidates=[]) and fills them in for real.

Contract with Person 1 (confirmed):
    intent = parse_intent(transcript)              # Person 1, sync
    resolved_intent = resolve_entities(intent)      # Person 2, sync

raw_text is the LLM's extracted place name — cleaned-up as Gemini understood
it (e.g. "Bangalore", "Alleppey"), but NOT yet matched to any code. That's
this file's job.
"""

from __future__ import annotations

import difflib
import json
from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict

GAZETTEER_PATH = Path(__file__).parent / "gazetteer.json"
FUZZY_MATCH_THRESHOLD = 0.8  # difflib similarity ratio; tune against real transcripts


# ---------------------------------------------------------------------------
# Schemas — mirror SRS Sec 4.2. Swap these for imports from the real
# schemas.py once it exists; keep field names identical so nothing else
# in the pipeline needs to change.
# ---------------------------------------------------------------------------

class ResolutionStatus(str, Enum):
    RESOLVED = "RESOLVED"
    AMBIGUOUS = "AMBIGUOUS"
    UNRESOLVED = "UNRESOLVED"


class LocationCandidate(BaseModel):
    code: str
    type: str          # "rail" | "airport" | "bus"
    name: str
    city: str


class LocationEntity(BaseModel):
    raw_text: str
    status: ResolutionStatus
    resolved: Optional[LocationCandidate] = None
    candidates: list[LocationCandidate] = []


# TripIntent is Person 1's model. This is a minimal stand-in so this module
# is runnable standalone — replace with `from schemas import TripIntent`
# once you're wiring this into the real pipeline.
class TripIntent(BaseModel):
    model_config = ConfigDict(extra="allow")

    origin: LocationEntity
    destination: LocationEntity


# ---------------------------------------------------------------------------
# Gazetteer loading
# ---------------------------------------------------------------------------

def _load_gazetteer(path: Path = GAZETTEER_PATH) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["entries"]


def _build_alias_index(entries: list[dict]) -> dict[str, dict]:
    """alias (lowercased) -> gazetteer entry"""
    index: dict[str, dict] = {}
    for entry in entries:
        for alias in entry["aliases"]:
            index[alias.lower().strip()] = entry
    return index


_ENTRIES = _load_gazetteer()
_ALIAS_INDEX = _build_alias_index(_ENTRIES)
_ALL_ALIASES = list(_ALIAS_INDEX.keys())


# ---------------------------------------------------------------------------
# Core resolution logic
# ---------------------------------------------------------------------------

def _normalize(raw_text: str) -> str:
    return raw_text.strip().lower()


def _entry_to_candidates(entry: dict) -> list[LocationCandidate]:
    return [
        LocationCandidate(code=hub["code"], type=hub["type"], name=hub["name"], city=entry["city"])
        for hub in entry["hubs"]
    ]


def _fuzzy_lookup(normalized: str) -> Optional[dict]:
    """Fallback for spelling drift the alias list didn't anticipate.
    Returns the matched gazetteer entry, or None if nothing clears the bar."""
    matches = difflib.get_close_matches(normalized, _ALL_ALIASES, n=1, cutoff=FUZZY_MATCH_THRESHOLD)
    if not matches:
        return None
    return _ALIAS_INDEX[matches[0]]


def resolve_location(raw_text: str) -> LocationEntity:
    """Resolve a single spoken place name to RESOLVED / AMBIGUOUS / UNRESOLVED."""
    normalized = _normalize(raw_text)

    entry = _ALIAS_INDEX.get(normalized) or _fuzzy_lookup(normalized)

    if entry is None:
        return LocationEntity(raw_text=raw_text, status=ResolutionStatus.UNRESOLVED)

    candidates = _entry_to_candidates(entry)

    if len(candidates) == 1:
        return LocationEntity(
            raw_text=raw_text,
            status=ResolutionStatus.RESOLVED,
            resolved=candidates[0],
            candidates=[],
        )

    # City matched but has more than one hub (FR-6) — never guess.
    return LocationEntity(
        raw_text=raw_text,
        status=ResolutionStatus.AMBIGUOUS,
        resolved=None,
        candidates=candidates,
    )


def resolve_entities(intent: TripIntent) -> TripIntent:
    """Entry point matching the agreed contract: takes the TripIntent with
    placeholder origin/destination LocationEntity objects, returns a copy
    with both fully resolved."""
    resolved_origin = resolve_location(intent.origin.raw_text)
    resolved_destination = resolve_location(intent.destination.raw_text)

    return intent.model_copy(update={
        "origin": resolved_origin,
        "destination": resolved_destination,
    })


# ---------------------------------------------------------------------------
# Quick manual smoke test — run directly: python resolve_entities.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_cases = [
        "Alleppey",       # RESOLVED — single hub
        "Bangalore",      # AMBIGUOUS — rail + airport + bus
        "Bengaluru",      # AMBIGUOUS — same city, different alias
        "Kochi",          # AMBIGUOUS — airport + 2 rail stations
        "Bengalooru",     # UNRESOLVED-or-fuzzy — misspelling, tests fuzzy fallback
        "Atlantis",       # UNRESOLVED — no match at all
    ]

    for raw in test_cases:
        result = resolve_location(raw)
        print(f"\nraw_text = {raw!r}")
        print(f"  status = {result.status.value}")
        if result.resolved:
            print(f"  resolved = {result.resolved.code} ({result.resolved.name})")
        if result.candidates:
            codes = ", ".join(c.code for c in result.candidates)
            print(f"  candidates = [{codes}]")

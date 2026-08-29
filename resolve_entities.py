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
from pathlib import Path
from typing import Optional

from schemas import LocationCandidate, LocationEntity, ResolutionStatus, TransportMode, TripIntent

GAZETTEER_PATH = Path(__file__).parent / "gazetteer.json"
FUZZY_MATCH_THRESHOLD = 0.8  # difflib similarity ratio; tune against real transcripts

# TransportMode -> the LocationCandidate.type it corresponds to. Used to
# auto-narrow candidates when the user already told us which mode they want.
MODE_TO_HUB_TYPE = {
    TransportMode.TRAIN: "rail",
    TransportMode.BUS: "bus",
    TransportMode.FLIGHT: "airport",
}


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


def resolve_location(raw_text: str, mode: TransportMode = TransportMode.ANY) -> LocationEntity:
    """Resolve a single spoken place name to RESOLVED / AMBIGUOUS / UNRESOLVED.

    If `mode` is known (the user already said "by train" / "by flight" /
    "by bus"), narrow candidates to that hub type first. This can turn what
    would otherwise be AMBIGUOUS into RESOLVED without asking the user
    anything — e.g. Bangalore + mode=TRAIN resolves straight to SBC,
    because a mode was already specified and only one rail hub exists there.
    """
    normalized = _normalize(raw_text)

    entry = _ALIAS_INDEX.get(normalized) or _fuzzy_lookup(normalized)

    if entry is None:
        return LocationEntity(raw_text=raw_text, status=ResolutionStatus.UNRESOLVED)

    candidates = _entry_to_candidates(entry)

    if mode != TransportMode.ANY:
        wanted_type = MODE_TO_HUB_TYPE.get(mode)
        narrowed = [c for c in candidates if c.type == wanted_type]
        # Only apply the narrowing if it actually found something. If the
        # requested mode has no hub in this city at all, fall back to the
        # full list rather than silently discarding every option — that's
        # still worth surfacing to the user as a genuine ambiguity.
        if narrowed:
            candidates = narrowed

    if len(candidates) == 1:
        return LocationEntity(
            raw_text=raw_text,
            status=ResolutionStatus.RESOLVED,
            resolved=candidates[0],
            candidates=[],
        )

    # More than one candidate remains even after applying mode, if known
    # (FR-6) — never guess.
    return LocationEntity(
        raw_text=raw_text,
        status=ResolutionStatus.AMBIGUOUS,
        resolved=None,
        candidates=candidates,
    )


def resolve_entities(intent: TripIntent) -> TripIntent:
    """Entry point matching the agreed contract: takes the TripIntent with
    placeholder origin/destination LocationEntity objects, returns a copy
    with both fully resolved. Passes the intent's mode through so a known
    mode can auto-narrow candidates instead of triggering an unnecessary
    clarifying question."""
    resolved_origin = resolve_location(intent.origin.raw_text, mode=intent.mode)
    resolved_destination = resolve_location(intent.destination.raw_text, mode=intent.mode)

    return intent.model_copy(update={
        "origin": resolved_origin,
        "destination": resolved_destination,
    })


# ---------------------------------------------------------------------------
# Quick manual smoke test — run directly: python resolve_entities.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_cases = [
        ("Alleppey", TransportMode.ANY),       # RESOLVED — single hub
        ("Bangalore", TransportMode.ANY),      # AMBIGUOUS — rail + airport + bus
        ("Bengaluru", TransportMode.ANY),      # AMBIGUOUS — same city, different alias
        ("Kochi", TransportMode.ANY),          # AMBIGUOUS — airport + 2 rail stations
        ("Bengalooru", TransportMode.ANY),     # UNRESOLVED-or-fuzzy — misspelling, tests fuzzy fallback
        ("Atlantis", TransportMode.ANY),       # UNRESOLVED — no match at all
        ("Bangalore", TransportMode.TRAIN),    # mode-aware: auto-resolves to SBC, no ambiguity
        ("Kochi", TransportMode.TRAIN),        # mode-aware: narrows to 2 rail options, still AMBIGUOUS
    ]

    for raw, mode in test_cases:
        result = resolve_location(raw, mode=mode)
        print(f"\nraw_text = {raw!r}, mode = {mode.value}")
        print(f"  status = {result.status.value}")
        if result.resolved:
            print(f"  resolved = {result.resolved.code} ({result.resolved.name})")
        if result.candidates:
            codes = ", ".join(c.code for c in result.candidates)
            print(f"  candidates = [{codes}]")
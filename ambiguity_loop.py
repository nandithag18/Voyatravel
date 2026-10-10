"""
Conversational Ambiguity Loop — FR-7, FR-8 (SRS Sec 3.4)

When resolve_location() returns AMBIGUOUS or UNRESOLVED, this module takes
over: generate a clarifying question, get the user's reply, re-attempt
resolution. Bounded to a max number of rounds (FR-8) so it never spins
forever — after that it asks the user to rephrase or flags for escalation.

This is pipeline-agnostic about *how* a reply is obtained — you pass in a
get_reply(question: str) -> str callable. In the real voice agent that
callable wraps STT + whatever prompt-the-user mechanism Person 1's stack
uses; in tests/demos it can just be a scripted function or input().
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from typing import Callable

from resolve_entities import resolve_location
from schemas import LocationCandidate, LocationEntity, ResolutionStatus

MAX_CLARIFICATION_ATTEMPTS = 3  # FR-8 — bounded loop

# Words a user might use to point at a candidate by transit type, rather
# than by code or ordinal position.
TYPE_KEYWORDS = {
    "rail": {"train", "rail", "railway", "station"},
    "airport": {"flight", "plane", "air", "airport", "fly"},
    "bus": {"bus", "coach"},
}


@dataclass
class AmbiguityLoopResult:
    """What the loop hands back to the orchestrator."""
    entity: LocationEntity
    questions_asked: list[str] = field(default_factory=list)
    resolved: bool = False
    escalated: bool = False   # True if bounded attempts were exhausted


# ---------------------------------------------------------------------------
# Question generation
# ---------------------------------------------------------------------------

def _format_candidate_line(i: int, c: LocationCandidate) -> str:
    return f"{i}. {c.name} ({c.type}, code {c.code})"


def generate_clarifying_question(entity: LocationEntity) -> str:
    """FR-7 — natural-language question + options list for AMBIGUOUS,
    or a re-prompt for UNRESOLVED."""
    if entity.status == ResolutionStatus.AMBIGUOUS:
        lines = [_format_candidate_line(i, c) for i, c in enumerate(entity.candidates, start=1)]
        options = "\n".join(lines)
        return (
            f"I found more than one place matching \"{entity.raw_text}\". "
            f"Which one did you mean?\n{options}\n"
            f"You can reply with a number, the transit type (train/flight/bus), or the station name."
        )

    if entity.status == ResolutionStatus.UNRESOLVED:
        return (
            f"I couldn't find a station, airport, or bus hub matching \"{entity.raw_text}\". "
            f"Could you say the city or station name again, or try a nearby major city?"
        )

    # Shouldn't be called on a RESOLVED entity, but fail safely if it is.
    return f"Can you confirm the location you meant by \"{entity.raw_text}\"?"


def generate_escalation_message(entity: LocationEntity) -> str:
    """FR-8 — what to say once the bounded loop is exhausted without resolving."""
    return (
        f"I'm still not able to pin down \"{entity.raw_text}\" after a few tries. "
        f"Let's move on for now — you can try rephrasing it later, or enter the station manually."
    )


# ---------------------------------------------------------------------------
# Reply interpretation
# ---------------------------------------------------------------------------

def _match_by_ordinal(reply: str, candidates: list[LocationCandidate]) -> LocationCandidate | None:
    stripped = reply.strip()
    if stripped.isdigit():
        idx = int(stripped) - 1
        if 0 <= idx < len(candidates):
            return candidates[idx]
    return None


def _match_by_code(reply: str, candidates: list[LocationCandidate]) -> LocationCandidate | None:
    normalized = reply.strip().upper()
    for c in candidates:
        if c.code.upper() == normalized:
            return c
    return None


def _match_by_type_keyword(reply: str, candidates: list[LocationCandidate]) -> LocationCandidate | None:
    words = set(reply.lower().split())
    for transit_type, keywords in TYPE_KEYWORDS.items():
        if words & keywords:
            matches = [c for c in candidates if c.type == transit_type]
            if len(matches) == 1:
                return matches[0]
            # If the type itself is still ambiguous (e.g. two rail options),
            # don't guess — fall through and let the loop re-ask.
    return None


def _match_by_name(reply: str, candidates: list[LocationCandidate]) -> LocationCandidate | None:
    names = {c.name.lower(): c for c in candidates}
    close = difflib.get_close_matches(reply.lower().strip(), names.keys(), n=1, cutoff=0.6)
    return names[close[0]] if close else None


def interpret_reply(entity: LocationEntity, reply: str) -> LocationEntity:
    """Attempt to resolve an AMBIGUOUS or UNRESOLVED entity from the user's
    clarifying reply. Returns a new LocationEntity — RESOLVED if the reply
    pinned it down, otherwise still AMBIGUOUS/UNRESOLVED for another round."""

    if entity.status == ResolutionStatus.AMBIGUOUS:
        candidate = (
            _match_by_ordinal(reply, entity.candidates)
            or _match_by_code(reply, entity.candidates)
            or _match_by_type_keyword(reply, entity.candidates)
            or _match_by_name(reply, entity.candidates)
        )
        if candidate:
            return LocationEntity(
                raw_text=entity.raw_text,
                status=ResolutionStatus.RESOLVED,
                resolved=candidate,
                candidates=[],
            )
        # No match — stay AMBIGUOUS, same candidate list, loop will re-ask.
        return entity

    if entity.status == ResolutionStatus.UNRESOLVED:
        # Treat the reply as a fresh attempt at naming the place and re-run
        # full resolution (gazetteer + fuzzy match) on it.
        return resolve_location(reply)

    return entity


# ---------------------------------------------------------------------------
# The bounded loop itself
# ---------------------------------------------------------------------------

def run_ambiguity_loop(
    entity: LocationEntity,
    get_reply: Callable[[str], str],
    max_attempts: int = MAX_CLARIFICATION_ATTEMPTS,
) -> AmbiguityLoopResult:
    """FR-7 + FR-8. Repeatedly asks a clarifying question and re-attempts
    resolution until RESOLVED or max_attempts is exhausted."""

    questions_asked: list[str] = []
    current = entity

    if current.status == ResolutionStatus.RESOLVED:
        return AmbiguityLoopResult(entity=current, resolved=True)

    for _ in range(max_attempts):
        question = generate_clarifying_question(current)
        questions_asked.append(question)

        reply = get_reply(question)
        current = interpret_reply(current, reply)

        if current.status == ResolutionStatus.RESOLVED:
            return AmbiguityLoopResult(
                entity=current,
                questions_asked=questions_asked,
                resolved=True,
            )

    # Bounded loop exhausted without resolving (FR-8) — escalate gracefully.
    questions_asked.append(generate_escalation_message(current))
    return AmbiguityLoopResult(
        entity=current,
        questions_asked=questions_asked,
        resolved=False,
        escalated=True,
    )


# ---------------------------------------------------------------------------
# Smoke test with scripted replies — run directly: python ambiguity_loop.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from resolve_entities import resolve_location

    def scripted_replies(*replies):
        it = iter(replies)
        def _get_reply(question: str) -> str:
            print(f"\n[AGENT] {question}")
            reply = next(it)
            print(f"[USER]  {reply}")
            return reply
        return _get_reply

    print("=== Case 1: AMBIGUOUS resolved by picking a number ===")
    entity = resolve_location("Bangalore")
    result = run_ambiguity_loop(entity, scripted_replies("2"))
    print(f"\n-> resolved={result.resolved}, final code={result.entity.resolved.code if result.entity.resolved else None}")

    print("\n\n=== Case 2: AMBIGUOUS resolved by transit type ===")
    entity = resolve_location("Kochi")
    result = run_ambiguity_loop(entity, scripted_replies("the airport one"))
    print(f"\n-> resolved={result.resolved}, final code={result.entity.resolved.code if result.entity.resolved else None}")

    print("\n\n=== Case 3: UNRESOLVED recovered by rephrasing ===")
    entity = resolve_location("Poonjar")  # not in the gazetteer at all
    result = run_ambiguity_loop(entity, scripted_replies("Alleppey"))  # single-hub city -> resolves cleanly
    print(f"\n-> resolved={result.resolved}, final status={result.entity.status}")

    print("\n\n=== Case 4: Never resolves, loop exhausts and escalates ===")
    entity = resolve_location("Atlantis")
    result = run_ambiguity_loop(entity, scripted_replies("Narnia", "Neverland", "Wakanda"))
    print(f"\n-> resolved={result.resolved}, escalated={result.escalated}")
    print(f"   final message: {result.questions_asked[-1]}")

"""
schemas.py — Data Contracts (SRS Sec 4)

Every inter-stage payload in the pipeline is one of these Pydantic models.
Per NFR-7, no raw dict ever crosses a stage boundary.

Ownership (per the Person 1 / Person 2 split):
  Person 1 (Voice & Understanding)      -> TripIntent
  Person 2 (Resolution & Observability) -> LocationEntity, AgentTraceEvent
  Not yet owned (Milestone 2 / Final Delivery, unassigned for now):
                                         -> TripOption, BookingHandoff

Shared supporting types (enums + small nested models) are defined once here
so both halves of the pipeline import the same definitions.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Shared enums
# ---------------------------------------------------------------------------

class TransportMode(str, Enum):
    ANY = "ANY"
    TRAIN = "TRAIN"
    BUS = "BUS"
    FLIGHT = "FLIGHT"


class ResolutionStatus(str, Enum):
    """Owned by Person 2 — set for real in resolve_entities.py."""
    RESOLVED = "RESOLVED"
    AMBIGUOUS = "AMBIGUOUS"
    UNRESOLVED = "UNRESOLVED"


class TraceStatus(str, Enum):
    """Owned by Person 2 — used by the trace sink."""
    OK = "OK"
    ERROR = "ERROR"
    RETRY = "RETRY"


# ---------------------------------------------------------------------------
# Shared small nested models
# ---------------------------------------------------------------------------

class PassengerInfo(BaseModel):
    """Minimal passenger info needed to build TripIntent / the checkout form.
    Fields not fully specified in the SRS excerpt — confirm with Person 1
    before Milestone 1 sign-off if you need more than this."""
    model_config = ConfigDict(extra="forbid")

    adults: int = Field(1, ge=1)
    children: int = Field(0, ge=0)
    infants: int = Field(0, ge=0)


class LocationCandidate(BaseModel):
    """One resolved transit hub. Owned by Person 2 (Entity Resolution)."""
    model_config = ConfigDict(extra="forbid")

    code: str          # e.g. "ALLP", "COK", "SBC"
    type: str           # "rail" | "airport" | "bus"
    name: str            # human-readable hub name
    city: str


# ---------------------------------------------------------------------------
# 4.2 — LocationEntity (Person 2: Entity Resolution State)
# ---------------------------------------------------------------------------

class LocationEntity(BaseModel):
    """Placeholder on handoff from Person 1 (status=UNRESOLVED, resolved=None,
    candidates=[]). Person 2's resolve_entities() fills these in for real."""
    model_config = ConfigDict(extra="forbid")

    raw_text: str
    status: ResolutionStatus
    resolved: Optional[LocationCandidate] = None
    candidates: list[LocationCandidate] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# 4.1 — TripIntent (Person 1: LLM Parser Output)
# ---------------------------------------------------------------------------

class TripIntent(BaseModel):
    """Output of parse_intent(transcript). Strict mode per FR-3 — unexpected
    fields from the LLM are rejected, not silently dropped."""
    model_config = ConfigDict(extra="forbid")

    session_id: UUID
    origin: LocationEntity
    destination: LocationEntity
    mode: TransportMode = TransportMode.ANY
    travel_date: date
    return_date: Optional[date] = None
    budget_max_inr: Optional[float] = Field(None, ge=0)
    passengers: PassengerInfo
    raw_transcript: str


# ---------------------------------------------------------------------------
# 4.3 — TripOption (Provider Engine result — Milestone 2, not yet owned)
# ---------------------------------------------------------------------------

class TripOption(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider_name: str
    mode: TransportMode
    origin_code: str
    destination_code: str
    departure_dt: datetime
    arrival_dt: datetime
    price_inr: float = Field(..., ge=0)
    seats_available: Optional[int] = None
    booking_deeplink: Optional[str] = None


# ---------------------------------------------------------------------------
# 4.4 — BookingHandoff (Checkout Form Generation — Final Delivery, not yet owned)
# ---------------------------------------------------------------------------

class BookingHandoff(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: UUID
    selected_option: TripOption
    mmt_search_url: str
    booking_form_text: str
    requires_human_payment: bool = True   # always True — safety gate, FR-15
    handed_off_at: datetime


# ---------------------------------------------------------------------------
# 4.5 — AgentTraceEvent (Person 2: Observability)
# ---------------------------------------------------------------------------

class AgentTraceEvent(BaseModel):
    """Emitted at every pipeline stage, keyed by session_id (NFR-5, NFR-6)."""
    model_config = ConfigDict(extra="forbid")

    session_id: UUID
    step: str                              # e.g. "entity_resolve", "provider:train"
    status: TraceStatus
    started_at: datetime
    latency_ms: float
    input_summary: Optional[str] = None
    output_summary: Optional[str] = None
    error: Optional[str] = None

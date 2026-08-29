from typing import Optional
from enum import Enum
from pydantic import BaseModel, Field
from datetime import date
from uuid import UUID

class TransportMode(str, Enum):
    TRAIN = "train"
    BUS = "bus"
    FLIGHT = "flight"
    ANY = "any"


class ResolutionStatus(str, Enum):
    RESOLVED = "resolved"
    AMBIGUOUS = "ambiguous"
    UNRESOLVED = "unresolved"


class TraceStatus(str, Enum):
    OK = "ok"
    ERROR = "error"
    RETRY = "retry"


class LocationCandidate(BaseModel):

    code: str                      # e.g. 'SBC', 'BLR'
    name: str                      # e.g. 'Bengaluru City Junction'
    kind: TransportMode            # which mode this code belongs to


class LocationEntity(BaseModel):
    raw_text: str                                  # e.g. 'Alleppey'
    status: ResolutionStatus
    resolved: Optional[LocationCandidate] = None
    candidates: list[LocationCandidate] = Field(default_factory=list)  # populated when AMBIGUOUS

class PassengerInfo(BaseModel):
    adults: int = Field(1, ge=1)
    children: int = Field(0, ge=0)
    seniors: int = Field(0, ge=0)


class TripIntent(BaseModel):
    model_config = {"extra": "forbid"}  # reject hallucinated fields

    session_id: UUID
    origin: LocationEntity
    destination: LocationEntity
    mode: TransportMode = TransportMode.ANY
    travel_date: date
    return_date: Optional[date] = None
    budget_max_inr: Optional[float] = Field(None, ge=0)
    passengers: PassengerInfo
    raw_transcript: str
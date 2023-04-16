#using dataclass
from dataclasses import dataclass
from typing import List, Optional

#enum for restriction types
from enum import Enum

class RestrictionType(Enum):
    RATE_LIMIT = 1
    ABSOLUTE = 2

class RestrictionLevel(Enum):
    FORBIDDEN = 1
    REQUEST = 2
    ALLOWED = 3

@dataclass
class AllowFlag:
    name: str
    level: RestrictionLevel

@dataclass
class Restriction:
    name: str
    #for rate limited restrictions
    rate: int
    period_hours: int
    spending_limit: int
    #restrictions have time periods that max out when these actions can occur
    type: RestrictionType
    #allowed plugins and methods that an agent can use
    allowed_plugins: List[AllowFlag]
    allowed_methods: List[AllowFlag]








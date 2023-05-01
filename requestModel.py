from dataclasses import dataclass
from typing import List

@dataclass
class RequestModel:
    query: str
    links: List[str]
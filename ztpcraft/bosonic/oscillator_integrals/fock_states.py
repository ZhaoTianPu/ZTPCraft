from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class OscillatorFockState:
    oscillator: Any
    occupations: tuple[int, ...]


__all__ = ["OscillatorFockState"]

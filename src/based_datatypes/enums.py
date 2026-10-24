"""Documentation-friendly enum classes."""

from enum import StrEnum as StdStrEnum
from typing import Self


class StrEnum(StdStrEnum):
    """StrEnum which renders only its value in a string context."""

    @classmethod
    def parse(cls, value: str) -> Self:
        """Parse a string into a StrEnum of proper type."""
        return cls(value)

    def __repr__(self) -> str:  # noqa: D105
        return f"'{self.value}'"

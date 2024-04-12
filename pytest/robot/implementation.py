from typing import (
    Any,
)

from interface import (
    BoardIF,
)


def validate_positive_integer(
        value: Any,
        msg: str,
        allow_zero: bool = False
):
    """Validate if the value is integer.
    Args:
        value: The value to validate
        msg: The error message to raise with ValueError if the value is not integer.
        allow_zero: True if the value accepts zero.

    Raises:
        ValueError: When the value is not integer, or is not positive with allow_zero==False,
            or is negative.
    """
    if not isinstance(value, int) or value < 0 or (not allow_zero and value == 0):
        raise ValueError(msg)


class Board(BoardIF):
    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    def __init__(self, width: int, height: int):
        super().__init__(width=width, height=height)

        for _value in [width, height]:
            if not isinstance(_value, int) or _value < 1:
                msg: str = (
                    f"expected the board sizes are positive, got width:[{width}] height:[{height}]."
                )
                raise ValueError(msg)

        self._width: int = int(width)
        self._height: int = int(height)

    def is_on_board(self, x: int, y: int) -> bool:
        if not isinstance(x, int) or not isinstance(y, int):
            msg: str = f"expected the coordinate (x,y) as integer, got (x:[{x}] y:[{y}])."
            raise ValueError(msg)

        return (0 <= x < self.width) and (0 <= y < self.height)

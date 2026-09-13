from dataclasses import dataclass
from typing import Any, Protocol


class V41Engram(Protocol):
    """External Engram implementation contract.

    The DSV4.1 model owns the boundary metadata and delegates the operation.
    """

    def __call__(self, hidden: Any, *, input_ids: Any, image_mask: Any) -> Any: ...


class V41DSpark(Protocol):
    """External DSpark/draft integration contract; disabled by default."""

    def loss(self, hidden: Any, *, labels: Any, **kwargs: Any) -> Any: ...


@dataclass(frozen=True, slots=True)
class V41OptionalModules:
    engram: V41Engram | None = None
    dspark: V41DSpark | None = None

"""Built-in objectives.

Importing this package registers every objective, so `get_objective(name)` works.
"""

from . import concrete, synthetic  # noqa: F401

__all__ = ["concrete", "synthetic"]

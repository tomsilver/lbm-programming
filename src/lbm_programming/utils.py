"""Utilities."""

from typing import Set

from lbm_programming.structs import Dog


def get_good_dogs_of_breed(dogs: Set[Dog], breed: str) -> Set:
    """Get all good dogs of the specified breed."""
    assert False
    return {d for d in dogs if d.is_good() and d.breed == breed}

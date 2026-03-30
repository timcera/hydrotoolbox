"""Define hydrotoolbox package."""

# Import main functions from hydrotoolbox module
from hydrotoolbox.hydrotoolbox import (
    about,
    exceedance_time,
    flow_duration,
    indices,
    recession,
    storm_events,
)

# Define __all__ list
__all__ = [
    # Main functions
    "about",
    "exceedance_time",
    "flow_duration",
    "indices",
    "recession",
    "storm_events",
]

"""Cross-layer schema contracts that do not depend on reconstruction or simulation."""

from .obstacles import OBSTACLE_PARAMETER_DIMENSIONS, OBSTACLE_MODEL_TYPES, validate_obstacle_model_contract

__all__ = [
    "OBSTACLE_PARAMETER_DIMENSIONS",
    "OBSTACLE_MODEL_TYPES",
    "validate_obstacle_model_contract",
]

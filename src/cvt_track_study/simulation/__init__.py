"""Nominal vehicle and ideal-CVT simulation."""

from .integrator import SimulationTrace, run_simulation
from .models import SimulationInputError
from . import service as _service
from .obstacle_severity import install_obstacle_severity_patch

# Install before studies import resolve_simulation_cases from the service module.
install_obstacle_severity_patch(_service)

SimulationError = _service.SimulationError
resolve_simulation_cases = _service.resolve_simulation_cases

from .router_v10 import run_baseline_project

__all__ = [
    "SimulationError",
    "SimulationInputError",
    "SimulationTrace",
    "resolve_simulation_cases",
    "run_baseline_project",
    "run_simulation",
]

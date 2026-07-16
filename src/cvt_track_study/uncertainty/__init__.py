"""Uncertainty sampling and statistical decision support."""

from .distributions import choice_from_uniform, quantity_from_uniform, quantity_quantile_si
from .model import GateSampleIdentity, ScenarioDraw
from .registry import InputRegistry, RegisteredInput, build_input_registry
from .sampling import (
    CorrelationGroup,
    SamplingError,
    SamplingPlan,
    ScenarioSampler,
    correlation_groups_from_study,
)
from .statistics import (
    SummaryInterval,
    convergence_diagnostics,
    paired_design_statistics,
    summarize_samples,
)

__all__ = [
    "CorrelationGroup",
    "GateSampleIdentity",
    "InputRegistry",
    "RegisteredInput",
    "SamplingError",
    "SamplingPlan",
    "ScenarioDraw",
    "ScenarioSampler",
    "SummaryInterval",
    "build_input_registry",
    "choice_from_uniform",
    "convergence_diagnostics",
    "correlation_groups_from_study",
    "paired_design_statistics",
    "quantity_from_uniform",
    "quantity_quantile_si",
    "summarize_samples",
]

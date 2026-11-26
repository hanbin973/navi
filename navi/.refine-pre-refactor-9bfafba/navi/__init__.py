"""Navi's modular Bayesian update and prediction API."""

from .bayes_update import (
    AbstractIntegrator,
    AbstractPooling,
    AbstractProposal,
    AxialAttentionProposal,
    BayesCarry,
    BayesUpdate,
    MeanPooling,
    TanhRunningMeanIntegrator,
    advance_chunk,
)
from .distribution import FullSupportBarDistribution
from .fixed_point import ContextSnapshot
from .predictor import (
    AbstractContextEncoder,
    AbstractPredictor,
    AbstractQueryDecoder,
    AlternatingContextEncoder,
    AlternatingQueryDecoder,
    BarPredictor,
    LastContextQueryDecoder,
    NoContextEncoder,
)
from .training import rollout_loss

__all__ = [
    "AbstractContextEncoder",
    "AbstractIntegrator",
    "AbstractPooling",
    "AbstractPredictor",
    "AbstractProposal",
    "AbstractQueryDecoder",
    "AlternatingContextEncoder",
    "AlternatingQueryDecoder",
    "AxialAttentionProposal",
    "BarPredictor",
    "BayesCarry",
    "BayesUpdate",
    "ContextSnapshot",
    "FullSupportBarDistribution",
    "LastContextQueryDecoder",
    "MeanPooling",
    "NoContextEncoder",
    "TanhRunningMeanIntegrator",
    "advance_chunk",
    "rollout_loss",
]

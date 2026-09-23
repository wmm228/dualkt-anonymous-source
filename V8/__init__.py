"""DualKT model package."""

from .model import InductiveCausalDualGraphKT, ModularEvidenceKT

DualKT = ModularEvidenceKT

__all__ = ['DualKT', 'ModularEvidenceKT', 'InductiveCausalDualGraphKT']

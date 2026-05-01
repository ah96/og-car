"""Lightweight affordance ontology for OG-CAR."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional, Set

from .data_types import GridObject, Intervention


@dataclass
class AffordanceOntology:
    """Structured affordance knowledge base.

    The ontology intentionally avoids heavyweight dependencies so that the
    reasoning behavior is transparent and reproducible.
    """

    object_affordances: Dict[str, Set[str]] = field(default_factory=dict)
    burdens: Dict[str, float] = field(default_factory=dict)

    @classmethod
    def default(cls) -> "AffordanceOntology":
        """Return the default robot-librarian affordance ontology."""
        return cls(
            object_affordances={
                "chair": {"movable"},
                "door": {"openable"},
                "cabinet": {"openable", "movable"},
                "cart": {"movable"},
                "person": {"step_aside"},
                "table": {"movable"},
                "shelf": set(),
                "wall": set(),
            },
            burdens={
                "movable": 1.0,
                "openable": 0.5,
                "step_aside": 1.5,
            },
        )

    def affordances_for(self, object_type: str) -> Set[str]:
        """Return admissible affordances for an object type."""
        return set(self.object_affordances.get(object_type, set()))

    def is_semantically_valid(self, obj: GridObject, affordance: str) -> bool:
        """Return whether an affordance is valid for the object's type."""
        return affordance in self.affordances_for(obj.object_type)

    def burden(self, affordance: str) -> float:
        """Return intervention burden for an affordance."""
        return float(self.burdens.get(affordance, 1.0))

    def candidate_interventions(self, obj: GridObject) -> Iterable[Intervention]:
        """Yield unresolved semantically valid interventions for an object."""
        for affordance in obj.affordances:
            if not self.is_semantically_valid(obj, affordance):
                continue
            if obj.states.get(affordance, 1) == 0:
                yield Intervention(object_id=obj.object_id, affordance=affordance)

    def degrade_missing(self, missing_affordances: Set[tuple[str, str]]) -> "AffordanceOntology":
        """Return a copy with selected object-type affordances removed.

        Args:
            missing_affordances: Set of `(object_type, affordance)` pairs.
        """
        new_mapping = {k: set(v) for k, v in self.object_affordances.items()}
        for object_type, affordance in missing_affordances:
            new_mapping.setdefault(object_type, set()).discard(affordance)
        return AffordanceOntology(new_mapping, dict(self.burdens))

    def degrade_noisy(self, noisy_affordances: Set[tuple[str, str]]) -> "AffordanceOntology":
        """Return a copy with selected incorrect affordances added."""
        new_mapping = {k: set(v) for k, v in self.object_affordances.items()}
        for object_type, affordance in noisy_affordances:
            new_mapping.setdefault(object_type, set()).add(affordance)
        return AffordanceOntology(new_mapping, dict(self.burdens))

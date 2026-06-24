"""
mace.semantic — Ontology-guided semantic navigation sub-package for MACE.

MACE: Minimal-change Affordance Counterfactual Explanations
Paper codebase.

    from mace.semantic import (
        OntofactNavigator,
        build_navigation_ontology,
        AffordanceType, DoorState, SurfaceType, MobilityType,
        Ontology, OntologyIndividual,
        NavigationGraph, NavNode, NavEdge, NavPath,
        Counterfactual, PropertyChange,
    )
"""

from .ontology       import Ontology, OntologyClass, OntologyIndividual
from .domain         import (
    AffordanceType,
    DoorState,
    SurfaceType,
    MobilityType,
    build_navigation_ontology,
)
from .affordance     import AffordanceReasoner, AffordanceResult
from .navigation     import NavigationGraph, NavNode, NavEdge, NavPath, AStarPlanner
from .counterfactual import CounterfactualEngine, Counterfactual, PropertyChange
from .explanation    import ExplanationGenerator, NavigationExplanation
from .orchestrator   import OntofactNavigator
from .visualization  import draw_navigation_graph

__all__ = [
    "draw_navigation_graph",
    "OntofactNavigator",
    "build_navigation_ontology",
    "Ontology",
    "OntologyClass",
    "OntologyIndividual",
    "AffordanceType",
    "DoorState",
    "SurfaceType",
    "MobilityType",
    "AffordanceReasoner",
    "AffordanceResult",
    "NavigationGraph",
    "NavNode",
    "NavEdge",
    "NavPath",
    "AStarPlanner",
    "CounterfactualEngine",
    "Counterfactual",
    "PropertyChange",
    "ExplanationGenerator",
    "NavigationExplanation",
]

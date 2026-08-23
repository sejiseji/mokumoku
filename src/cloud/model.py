from __future__ import annotations

import math
from dataclasses import dataclass, field

from src import config
from src.enums import EdgeKind
from src.math3d import Vec3


@dataclass
class OriginEvidence:
    sea: float = 0.0
    town: float = 0.0
    forest: float = 0.0
    mountain: float = 0.0


@dataclass
class CloudNode:
    id: int
    lineage_id: int
    cluster_id: int
    position: Vec3
    previous_position: Vec3
    velocity: Vec3
    mass: float
    moisture: float
    temperature: float
    density: float
    charge: float
    updraft: float
    activation: float
    noise: float
    incubation: float
    untouched_time: float
    age: float
    fade: float
    is_pruning: bool
    parent_node_id: int | None
    generation: int
    origin_evidence: OriginEvidence
    sprite_seed: int
    sprite_family: str
    size_class: str
    excitation: float = 0.0
    refractory_until_frame: int = 0
    polarity: Vec3 = field(default_factory=lambda: Vec3(0.0, 1.0, 0.0))
    trait_seed: int = 0

    @property
    def radius(self) -> float:
        raw = config.RADIUS_SCALE * math.sqrt(self.mass / max(self.density, 0.001))
        return max(config.MIN_NODE_RADIUS, min(config.MAX_NODE_RADIUS, raw))


@dataclass
class CloudEdge:
    id: int
    lineage_id: int
    cluster_id: int
    node_a: int
    node_b: int
    kind: EdgeKind
    rest_length: float
    stiffness: float
    strength: float
    moisture_conductivity: float
    temperature_conductivity: float
    charge_conductivity: float
    age: float
    strain: float


@dataclass
class CloudCluster:
    id: int
    lineage_id: int
    node_ids: list[int]
    edge_ids: list[int]
    centroid: Vec3
    total_mass: float
    age: float = 0.0
    history_tags: set[str] = field(default_factory=set)


@dataclass
class CloudLineage:
    id: int
    active_cluster_ids: set[int]
    started_at: float
    ended_at: float | None
    extinct: bool
    history_tags: set[str] = field(default_factory=set)
    observed_signatures: set[str] = field(default_factory=set)
    scored_event_ids: set[str] = field(default_factory=set)
    merged_lineage_ids: set[int] = field(default_factory=set)
    total_origin_evidence: OriginEvidence = field(default_factory=OriginEvidence)


@dataclass
class CloudState:
    nodes: dict[int, CloudNode] = field(default_factory=dict)
    edges: dict[int, CloudEdge] = field(default_factory=dict)
    clusters: dict[int, CloudCluster] = field(default_factory=dict)
    lineages: dict[int, CloudLineage] = field(default_factory=dict)
    next_node_id: int = 1
    next_edge_id: int = 1
    next_cluster_id: int = 1
    next_lineage_id: int = 1

    def active_lineage(self) -> CloudLineage | None:
        for lineage in self.lineages.values():
            if not lineage.extinct:
                return lineage
        return None

    def live_nodes(self) -> list[CloudNode]:
        return [node for node in self.nodes.values() if node.fade > 0.0]

    def live_edges(self) -> list[CloudEdge]:
        return [
            edge
            for edge in self.edges.values()
            if edge.node_a in self.nodes and edge.node_b in self.nodes
        ]

    def can_add_node(self) -> bool:
        return len(self.nodes) < config.MAX_NODES_TOTAL

    def can_add_edge(self) -> bool:
        return len(self.edges) < config.MAX_EDGES_TOTAL

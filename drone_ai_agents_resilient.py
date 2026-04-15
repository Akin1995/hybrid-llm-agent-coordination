from __future__ import annotations

import json
import math
import os
import random
from collections import deque
from dataclasses import dataclass, field
from typing import Any, List, Tuple, Optional, Dict

try:
    import openai  # type: ignore
except Exception:
    openai = None  # type: ignore

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


Coordinate = Tuple[int, int]


@dataclass
class DynamicObstacle:
    """Repräsentiert ein bewegliches Hindernis."""
    position: Coordinate


@dataclass
class Message:
    """Einfache Kommunikationsnachricht zwischen Drohnen."""
    sender: int
    type: str
    position: Optional[Coordinate] = None
    step: int = 0
    confidence: float = 1.0
    payload: Dict[str, Any] = field(default_factory=dict)


class GridWorld:
    """Rasterwelt mit statischen und dynamischen Hindernissen."""

    def __init__(self, width: int, height: int, static_obstacles: List[Coordinate], dynamic_obstacles: List[Coordinate]):
        self.width = width
        self.height = height
        self.static_obstacles = set(static_obstacles)
        self.dynamic_obstacles: List[DynamicObstacle] = [DynamicObstacle(pos) for pos in dynamic_obstacles]

    @property
    def obstacles(self) -> set[Coordinate]:
        return self.static_obstacles | {obs.position for obs in self.dynamic_obstacles}

    def in_bounds(self, pos: Coordinate) -> bool:
        x, y = pos
        return 0 <= x < self.width and 0 <= y < self.height

    def passable(self, pos: Coordinate) -> bool:
        return pos not in self.obstacles

    def neighbors(self, pos: Coordinate, include_wait: bool = False) -> List[Coordinate]:
        x, y = pos
        candidates = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
        result = [p for p in candidates if self.in_bounds(p) and self.passable(p)]
        if include_wait and self.passable(pos):
            result = [pos] + result
        return result

    def all_free_cells(self) -> List[Coordinate]:
        return [(x, y) for y in range(self.height) for x in range(self.width) if self.passable((x, y))]

    def update_dynamic_obstacles(self, agent_positions: set[Coordinate], move_fraction: float = 0.3) -> None:
        num_to_move = max(0, int(len(self.dynamic_obstacles) * move_fraction))
        indices = list(range(len(self.dynamic_obstacles)))
        random.shuffle(indices)
        to_move = indices[:num_to_move]

        for idx in to_move:
            obs = self.dynamic_obstacles[idx]
            neighbors = [
                (obs.position[0] + 1, obs.position[1]),
                (obs.position[0] - 1, obs.position[1]),
                (obs.position[0], obs.position[1] + 1),
                (obs.position[0], obs.position[1] - 1),
            ]
            random.shuffle(neighbors)

            for new_pos in neighbors:
                if not self.in_bounds(new_pos):
                    continue
                if new_pos in self.static_obstacles:
                    continue
                if any(o.position == new_pos for o in self.dynamic_obstacles):
                    continue
                if new_pos in agent_positions:
                    continue
                obs.position = new_pos
                break


def manhattan(a: Coordinate, b: Coordinate) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def normalize_distribution(prob_map: Dict[Coordinate, float]) -> Dict[Coordinate, float]:
    total = sum(max(v, 0.0) for v in prob_map.values())
    if total <= 0:
        return prob_map
    return {k: max(v, 0.0) / total for k, v in prob_map.items()}


def diffuse_probability_map(world: GridWorld, prob_map: Dict[Coordinate, float]) -> Dict[Coordinate, float]:
    new_map: Dict[Coordinate, float] = {cell: 0.0 for cell in world.all_free_cells()}
    for cell, prob in prob_map.items():
        if prob <= 0 or not world.passable(cell):
            continue
        reachable = world.neighbors(cell, include_wait=True)
        if not reachable:
            reachable = [cell]
        share = prob / len(reachable)
        for nxt in reachable:
            if world.passable(nxt):
                new_map[nxt] = new_map.get(nxt, 0.0) + share
    return normalize_distribution(new_map)


def make_uniform_probability_map(world: GridWorld) -> Dict[Coordinate, float]:
    cells = world.all_free_cells()
    if not cells:
        return {}
    p = 1.0 / len(cells)
    return {cell: p for cell in cells}


def visible_cells(center: Coordinate, radius: int, world: GridWorld) -> List[Coordinate]:
    cells: List[Coordinate] = []
    for y in range(world.height):
        for x in range(world.width):
            cell = (x, y)
            if world.passable(cell) and manhattan(center, cell) <= radius:
                cells.append(cell)
    return cells


def generate_sector_search_points(world: GridWorld, num_drones: int, drone_id: int, spacing: int = 4) -> List[Coordinate]:
    """Erzeugt Suchpunkte nur im Sektor der Drohne."""
    points: List[Coordinate] = []
    sector_width = max(1, math.ceil(world.width / max(1, num_drones)))
    x_min = drone_id * sector_width
    x_max = min(world.width, (drone_id + 1) * sector_width)
    offset_x = x_min + max(0, spacing // 2)
    offset_y = max(0, spacing // 2)

    for y in range(offset_y, world.height, spacing):
        row_points: List[Coordinate] = []
        for x in range(offset_x, x_max, spacing):
            p = (x, y)
            if world.in_bounds(p) and world.passable(p):
                row_points.append(p)
        if ((y - offset_y) // spacing) % 2 == 1:
            row_points.reverse()
        points.extend(row_points)
    return points


def next_patrol_point(drone: "Drone", points: List[Coordinate], reserved: set[Coordinate]) -> Optional[Coordinate]:
    if not points:
        return None
    n = len(points)
    for k in range(n):
        idx = (drone.search_index + k) % n
        p = points[idx]
        if p not in reserved:
            drone.search_index = (idx + 1) % n
            return p
    return None


class Drone:
    """Drohne mit begrenzter Sicht, BDI-Zustand und probabilistischer Suchlogik."""

    def __init__(self, id_: int, position: Coordinate, sight_radius: int):
        self.id = id_
        self.position = position
        self.target: Optional[Coordinate] = None
        self.sight_radius = sight_radius
        self.last_known_thief_pos: Optional[Coordinate] = None
        self.prev_position: Coordinate = position
        self.search_index: int = 0
        self.search_target: Optional[Coordinate] = None
        self.secondary_target: Optional[Coordinate] = None
        self.role: str = "search"
        self.stuck_steps: int = 0

        self.beliefs: Dict[str, Any] = {
            "last_known_thief_pos": None,
            "last_known_thief_step": None,
            "estimated_thief_pos": None,
            "thief_probability_map": {},
            "visible_obstacles": [],
            "visible_free_cells": [],
            "teammate_reports": [],
            "world_step": 0,
            "role_history": [],
        }
        self.desires: List[str] = ["search_thief"]
        self.intentions: List[str] = []
        self.message_log: List[Message] = []

    def initialize_probability_map(self, world: GridWorld) -> None:
        if not self.beliefs.get("thief_probability_map"):
            self.beliefs["thief_probability_map"] = make_uniform_probability_map(world)

    def update_beliefs(self, world: GridWorld, thief: "Thief", step: int) -> List[Message]:
        messages: List[Message] = []
        self.beliefs["world_step"] = step
        self.initialize_probability_map(world)

        visible_obs = [obs for obs in world.obstacles if manhattan(obs, self.position) <= self.sight_radius]
        visible_free = visible_cells(self.position, self.sight_radius, world)
        self.beliefs["visible_obstacles"] = sorted(visible_obs)
        self.beliefs["visible_free_cells"] = sorted(visible_free)

        prob_map = diffuse_probability_map(world, dict(self.beliefs.get("thief_probability_map", {})))

        # Zellen im Sichtbereich ohne Dieb werden ausgeschlossen
        if manhattan(self.position, thief.position) > self.sight_radius:
            for cell in visible_free:
                prob_map[cell] = 0.0

        if manhattan(self.position, thief.position) <= self.sight_radius:
            self.last_known_thief_pos = thief.position
            self.beliefs["last_known_thief_pos"] = thief.position
            self.beliefs["last_known_thief_step"] = step
            prob_map = {cell: 0.0 for cell in prob_map}
            prob_map[thief.position] = 1.0
            predicted = thief.position
            msg = Message(
                sender=self.id,
                type="THIEF_SPOTTED",
                position=thief.position,
                step=step,
                confidence=1.0,
                payload={"observer_pos": self.position},
            )
            messages.append(msg)
            self.message_log.append(msg)
        else:
            predicted = max(prob_map.items(), key=lambda kv: kv[1])[0] if prob_map else None

        self.beliefs["estimated_thief_pos"] = predicted
        self.beliefs["thief_probability_map"] = normalize_distribution(prob_map)
        return messages

    def observe(self, world: GridWorld, thief: "Thief", step: int) -> List[Message]:
        return self.update_beliefs(world, thief, step)

    def receive_messages(self, messages: List[Message], world: Optional[GridWorld] = None) -> None:
        if not messages:
            return

        reports = list(self.beliefs.get("teammate_reports", []))
        prob_map = dict(self.beliefs.get("thief_probability_map", {}))
        for msg in messages:
            if msg.sender == self.id:
                continue
            self.message_log.append(msg)
            reports.append({
                "sender": msg.sender,
                "type": msg.type,
                "position": msg.position,
                "step": msg.step,
                "confidence": msg.confidence,
            })
            if msg.type == "THIEF_SPOTTED" and msg.position is not None:
                self.last_known_thief_pos = msg.position
                self.beliefs["last_known_thief_pos"] = msg.position
                self.beliefs["last_known_thief_step"] = msg.step
                if world is not None:
                    if not prob_map:
                        prob_map = make_uniform_probability_map(world)
                    prob_map = {cell: 0.0 for cell in prob_map}
                    prob_map[msg.position] = max(msg.confidence, 0.5)

        if prob_map:
            self.beliefs["thief_probability_map"] = normalize_distribution(prob_map)
            self.beliefs["estimated_thief_pos"] = max(prob_map.items(), key=lambda kv: kv[1])[0]
        self.beliefs["teammate_reports"] = reports[-20:]

    def decay_last_known_thief_position(self, step: int, forget_after: int = 6) -> None:
        """Vergisst alte Sichtungen nach einigen Schritten.

        Ohne diese Alterung bleibt last_known_thief_pos zu lange gesetzt und die
        Drohne verhält sich weiterhin so, als sei die Information noch frisch.
        """
        last_step = self.beliefs.get("last_known_thief_step")
        if last_step is None:
            return
        if step - int(last_step) > forget_after:
            self.last_known_thief_pos = None
            self.beliefs["last_known_thief_pos"] = None

    def update_desires(self) -> None:
        if self.beliefs.get("last_known_thief_pos") is not None:
            self.desires = ["capture_thief", "contain_escape_routes"]
        elif self.beliefs.get("estimated_thief_pos") is not None:
            self.desires = ["investigate_hotspot", "search_thief"]
        else:
            self.desires = ["search_thief"]

    def deliberate_role(self) -> str:
        if self.beliefs.get("last_known_thief_pos") is not None:
            return "intercept" if self.id == 0 else "contain"
        if self.stuck_steps >= 3:
            return "reposition"
        if self.beliefs.get("estimated_thief_pos") is not None:
            return "investigate"
        return "search"

    def set_intention(
        self,
        role: str,
        target: Optional[Coordinate],
        secondary_target: Optional[Coordinate] = None,
    ) -> None:
        self.role = role
        self.target = target
        self.secondary_target = secondary_target
        self.intentions = [f"role:{role}"]
        if target is not None:
            self.intentions.append(f"move_to:{target}")
        if secondary_target is not None:
            self.intentions.append(f"secondary:{secondary_target}")
        history = list(self.beliefs.get("role_history", []))
        history.append(role)
        self.beliefs["role_history"] = history[-25:]
        if target is not None and role in {"search", "investigate", "reposition"}:
            self.search_target = target
        elif role not in {"search", "investigate", "reposition"}:
            self.search_target = None

    def compute_path_with_vision(self, world: GridWorld, target: Coordinate, blocked: set[Coordinate]) -> List[Coordinate]:
        """Globale BFS-Pfadsuche.

        Der Name bleibt aus Kompatibilitätsgründen bestehen, aber die Suche ist
        nicht mehr auf den aktuellen Sicht-Radius beschränkt. Sonst behalten
        Drohnen nach einer Sichtung oft nur lokales Verhalten bei, statt den
        Dieb über mehrere Schritte konsequent zu verfolgen.
        """
        if target is None:
            return []

        start = self.position
        queue = deque([(start, [])])
        visited = {start}

        while queue:
            current, path = queue.popleft()
            if current == target:
                return path
            for neighbor in world.neighbors(current):
                if neighbor in visited or neighbor in blocked:
                    continue
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
        return []

    def propose_move(self, world: GridWorld, occupied_now: set[Coordinate], reserved_next: set[Coordinate]) -> Coordinate:
        if self.target is not None and self.position == self.target and self.secondary_target is not None:
            if world.passable(self.secondary_target):
                self.target, self.secondary_target = self.secondary_target, None

        if self.target is None:
            return self.position

        path = self.compute_path_with_vision(world, self.target, (occupied_now - {self.position}) | reserved_next)
        if path:
            return path[0]

        candidates = world.neighbors(self.position, include_wait=True)
        candidates = [p for p in candidates if p == self.position or (p not in occupied_now and p not in reserved_next)]
        if not candidates:
            return self.position

        prob_map = self.beliefs.get("thief_probability_map", {})
        def score(p: Coordinate) -> Tuple[float, int, int, int]:
            prob = float(prob_map.get(p, 0.0))
            dist = manhattan(p, self.target)
            return (prob, -dist, -manhattan(p, self.position), -self.id)

        candidates.sort(key=score, reverse=True)
        return candidates[0]

    def commit_position(self, new_position: Coordinate) -> None:
        self.prev_position = self.position
        self.position = new_position
        if self.position == self.prev_position:
            self.stuck_steps += 1
        else:
            self.stuck_steps = 0


class Thief:
    """Dieb mit begrenzter Sicht."""

    def __init__(self, position: Coordinate, sight_radius: int = 3, stamina_max: int = 25, stamina_recovery: int = 2):
        self.position = position
        self.prev_position = position
        self.sight_radius = sight_radius
        self.stamina_max = stamina_max
        self.stamina = stamina_max
        self.stamina_recovery = stamina_recovery

    def choose_action(self, world: GridWorld, drone_positions: List[Coordinate]) -> Coordinate:
        self.prev_position = self.position
        if self.stamina <= 0:
            self.recover()
            return self.position

        visible_drones = [d for d in drone_positions if manhattan(d, self.position) <= self.sight_radius]
        neighbors = world.neighbors(self.position)

        if not visible_drones:
            if neighbors:
                self.position = random.choice(neighbors)
                self.stamina -= 1
            else:
                self.recover()
            return self.position

        candidates = [self.position] + neighbors

        def min_dist_to_visible(pos: Coordinate) -> int:
            return min(manhattan(pos, d) for d in visible_drones)

        best_pos = max(candidates, key=min_dist_to_visible)
        if best_pos == self.position:
            self.recover()
        else:
            self.stamina -= 1
            self.position = best_pos
        return self.position

    def recover(self) -> None:
        self.stamina = min(self.stamina + self.stamina_recovery, self.stamina_max)


def collect_team_messages(world: GridWorld, drones: List[Drone], thief: Thief, step: int) -> List[Message]:
    messages: List[Message] = []
    for drone in drones:
        messages.extend(drone.observe(world, thief, step))
    return messages


def distribute_messages(world: GridWorld, drones: List[Drone], messages: List[Message]) -> None:
    for drone in drones:
        drone.receive_messages(messages, world)




def aggregate_team_probability_map(world: GridWorld, drones: List[Drone]) -> Dict[Coordinate, float]:
    team_map: Dict[Coordinate, float] = {cell: 0.0 for cell in world.all_free_cells()}
    for drone in drones:
        prob_map = drone.beliefs.get("thief_probability_map", {})
        for cell, prob in prob_map.items():
            if world.passable(cell):
                team_map[cell] = max(team_map.get(cell, 0.0), float(prob))
    return normalize_distribution(team_map)


def probability_entropy(prob_map: Dict[Coordinate, float]) -> float:
    eps = 1e-12
    return -sum(p * math.log(p + eps, 2) for p in prob_map.values() if p > 0)


def top_probability_hotspots(
    prob_map: Dict[Coordinate, float],
    top_k: int = 8,
    min_prob: float = 0.0,
) -> List[Dict[str, Any]]:
    items = [(cell, p) for cell, p in prob_map.items() if p > min_prob]
    items.sort(key=lambda kv: kv[1], reverse=True)
    result: List[Dict[str, Any]] = []
    for cell, p in items[:top_k]:
        result.append({
            "cell": [cell[0], cell[1]],
            "probability": round(float(p), 6),
        })
    return result


def coverage_overlap_score(world: GridWorld, drones: List[Drone]) -> float:
    seen_counts: Dict[Coordinate, int] = {}
    for drone in drones:
        for cell in visible_cells(drone.position, drone.sight_radius, world):
            seen_counts[cell] = seen_counts.get(cell, 0) + 1

    if not seen_counts:
        return 0.0

    overlap_cells = sum(1 for c in seen_counts.values() if c >= 2)
    total_seen = len(seen_counts)
    return overlap_cells / max(1, total_seen)


def approximate_chokepoints(world: GridWorld, top_k: int = 20) -> List[List[int]]:
    scored: List[Tuple[int, Coordinate]] = []
    for cell in world.all_free_cells():
        degree = len(world.neighbors(cell, include_wait=False))
        if degree <= 2:
            scored.append((degree, cell))
    scored.sort(key=lambda x: (x[0], x[1][1], x[1][0]))
    return [[c[0], c[1]] for _, c in scored[:top_k]]


def candidate_targets_for_drone(
    world: GridWorld,
    drone: "Drone",
    drones: List["Drone"],
    team_prob_map: Dict[Coordinate, float],
    max_candidates: int = 5,
) -> List[Dict[str, Any]]:
    candidates: List[Tuple[str, Coordinate, float]] = []

    last_known = drone.beliefs.get("last_known_thief_pos")
    estimated = drone.beliefs.get("estimated_thief_pos")

    if isinstance(last_known, tuple) and world.passable(last_known):
        candidates.append(("last_known", last_known, 1.0))
        ring = [
            (last_known[0] + 1, last_known[1]),
            (last_known[0] - 1, last_known[1]),
            (last_known[0], last_known[1] + 1),
            (last_known[0], last_known[1] - 1),
        ]
        for p in ring:
            if world.in_bounds(p) and world.passable(p):
                candidates.append(("contain_ring", p, 0.9))

    if isinstance(estimated, tuple) and world.passable(estimated):
        candidates.append(("estimated", estimated, 0.8))

    hotspot_cells = sorted(
        [cell for cell, p in team_prob_map.items() if p > 0],
        key=lambda c: (team_prob_map[c], -manhattan(drone.position, c)),
        reverse=True,
    )
    for cell in hotspot_cells[:8]:
        candidates.append(("hotspot", cell, float(team_prob_map[cell])))

    spacing = max(3, world.width // max(4, len(drones)))
    sector_points = generate_sector_search_points(world, len(drones), drone.id, spacing=spacing)
    if sector_points:
        patrol = sector_points[drone.search_index % len(sector_points)]
        candidates.append(("sector_patrol", patrol, 0.35))

    other_positions = [d.position for d in drones if d.id != drone.id]
    free_cells = world.all_free_cells()

    def reposition_score(cell: Coordinate) -> Tuple[int, float]:
        min_team_dist = min((manhattan(cell, p) for p in other_positions), default=0)
        prob = team_prob_map.get(cell, 0.0)
        return (min_team_dist, prob)

    if free_cells:
        reposition = max(free_cells, key=reposition_score)
        candidates.append(("reposition", reposition, 0.25))

    dedup: Dict[Coordinate, Tuple[str, float]] = {}
    for label, cell, score in candidates:
        if cell not in dedup or score > dedup[cell][1]:
            dedup[cell] = (label, score)

    result: List[Dict[str, Any]] = []
    for cell, (label, score) in dedup.items():
        result.append({
            "target": [cell[0], cell[1]],
            "type": label,
            "score": round(float(score), 6),
            "distance": manhattan(drone.position, cell),
        })

    result.sort(key=lambda x: (x["score"], -x["distance"]), reverse=True)
    return result[:max_candidates]


def build_strategic_planner_state(world: GridWorld, drones: List["Drone"], thief: "Thief") -> Dict[str, Any]:
    team_prob_map = aggregate_team_probability_map(world, drones)
    entropy = probability_entropy(team_prob_map)
    hotspots = top_probability_hotspots(team_prob_map, top_k=10, min_prob=0.0)
    overlap = coverage_overlap_score(world, drones)
    chokepoints = approximate_chokepoints(world, top_k=16)

    visible_now: List[int] = []
    sightings: List[Dict[str, Any]] = []
    stuck_drones: List[int] = []
    recent_reports: List[Dict[str, Any]] = []

    for drone in drones:
        if manhattan(drone.position, thief.position) <= drone.sight_radius:
            visible_now.append(drone.id)

        lk = drone.beliefs.get("last_known_thief_pos")
        lk_step = drone.beliefs.get("last_known_thief_step")
        if isinstance(lk, tuple):
            sightings.append({
                "position": [lk[0], lk[1]],
                "step": lk_step,
                "age": None if lk_step is None else max(0, drone.beliefs.get("world_step", 0) - int(lk_step)),
            })

        if drone.stuck_steps >= 2:
            stuck_drones.append(drone.id)

        for report in drone.beliefs.get("teammate_reports", [])[-3:]:
            recent_reports.append(report)

    recent_reports.sort(key=lambda r: (r.get("step", -1), r.get("confidence", 0.0)), reverse=True)
    recent_reports = recent_reports[:10]

    freshest_sighting_age = None
    if sightings:
        valid_ages = [s["age"] for s in sightings if s["age"] is not None]
        if valid_ages:
            freshest_sighting_age = min(valid_ages)

    if visible_now:
        suggested_mode = "containment_ring"
    elif freshest_sighting_age is not None and freshest_sighting_age <= 5:
        suggested_mode = "converge_last_seen"
    elif entropy < 6.0 and hotspots:
        suggested_mode = "probability_sweep"
    else:
        suggested_mode = "sector_search"

    drone_entries: List[Dict[str, Any]] = []
    for drone in drones:
        last_known = drone.beliefs.get("last_known_thief_pos")
        estimated = drone.beliefs.get("estimated_thief_pos")
        drone_entries.append({
            "id": drone.id,
            "position": [drone.position[0], drone.position[1]],
            "role": drone.role,
            "stuck_steps": drone.stuck_steps,
            "last_known_thief_pos": [last_known[0], last_known[1]] if isinstance(last_known, tuple) else None,
            "last_known_thief_step": drone.beliefs.get("last_known_thief_step"),
            "estimated_thief_pos": [estimated[0], estimated[1]] if isinstance(estimated, tuple) else None,
            "candidate_targets": candidate_targets_for_drone(
                world=world,
                drone=drone,
                drones=drones,
                team_prob_map=team_prob_map,
                max_candidates=5,
            ),
        })

    return {
        "grid_size": [world.width, world.height],
        "num_drones": len(drones),
        "step_context": {
            "world_step": max((d.beliefs.get("world_step", 0) for d in drones), default=0),
            "thief_visible_to": visible_now,
            "freshest_sighting_age": freshest_sighting_age,
            "sightings": sightings[-8:],
            "recent_reports": recent_reports,
            "suggested_mode": suggested_mode,
        },
        "team_metrics": {
            "probability_entropy": round(entropy, 6),
            "coverage_overlap_score": round(overlap, 6),
            "stuck_drones": stuck_drones,
            "hotspots": hotspots,
            "chokepoints": chokepoints,
        },
        "drones": drone_entries,
    }


def should_replan(
    step: int,
    drones: List["Drone"],
    planner_memory: Dict[str, Any],
    horizon: int = 5,
) -> bool:
    if step == 0:
        return True

    last_plan_step = int(planner_memory.get("last_plan_step", -999))
    if step - last_plan_step >= horizon:
        return True

    prev_visible = bool(planner_memory.get("thief_visible", False))
    now_visible = any(d.beliefs.get("last_known_thief_pos") is not None for d in drones)
    if now_visible != prev_visible:
        return True

    stuck_now = sum(1 for d in drones if d.stuck_steps >= 2)
    if stuck_now >= 2:
        return True

    prev_targets = planner_memory.get("targets", {})
    if not prev_targets:
        return True

    current_targets = {
        d.id: tuple(d.target) if isinstance(d.target, tuple) else None
        for d in drones
    }
    lost_targets = sum(1 for d in drones if current_targets.get(d.id) is None)
    if lost_targets >= max(1, len(drones) // 3):
        return True

    return False


ALLOWED_ROLES = {"intercept", "contain", "investigate", "search", "reposition"}


@dataclass
class PlannerDecision:
    source: str
    success: bool
    mode: str
    reason: str
    degraded: bool = False
    reused_cached_plan: bool = False
    assignments_applied: int = 0


class SafetyManager:
    """Kleine Safety-/Sanity-Layer über dem Missionsplaner.

    In echten Systemen würde hier auch Akku, Geofence und Health-Checks liegen.
    Für diese Simulation validieren wir primär Rollen/Ziele und erzwingen einen
    konservativen Degraded Mode.
    """

    @staticmethod
    def validate_target(world: GridWorld, target: Optional[Coordinate]) -> bool:
        if target is None:
            return True
        return world.in_bounds(target) and world.passable(target)

    @staticmethod
    def enforce_degraded_mode(world: GridWorld, drones: List[Drone], planner_memory: Dict[str, Any]) -> None:
        assign_targets_to_drones_with_last_known(world, drones, Thief((-1, -1)))
        planner_memory["degraded_mode"] = True
        planner_memory["mode"] = "degraded_fallback"


def snapshot_current_plan(drones: List[Drone]) -> Dict[int, Dict[str, Any]]:
    return {
        drone.id: {
            "role": drone.role,
            "target": list(drone.target) if isinstance(drone.target, tuple) else None,
            "secondary_target": list(drone.secondary_target) if isinstance(drone.secondary_target, tuple) else None,
        }
        for drone in drones
    }


def apply_plan_snapshot(world: GridWorld, drones: List[Drone], plan_snapshot: Dict[int, Dict[str, Any]]) -> int:
    applied = 0
    used_targets: set[Coordinate] = set()
    for drone in drones:
        entry = plan_snapshot.get(drone.id)
        if not entry:
            continue
        role = str(entry.get("role", "search"))
        target_raw = entry.get("target")
        secondary_raw = entry.get("secondary_target")
        target = tuple(target_raw) if isinstance(target_raw, list) and len(target_raw) == 2 else None
        secondary_target = tuple(secondary_raw) if isinstance(secondary_raw, list) and len(secondary_raw) == 2 else None
        if role not in ALLOWED_ROLES:
            role = "search"
        if target is not None:
            target = (int(target[0]), int(target[1]))
            if not SafetyManager.validate_target(world, target) or target in used_targets:
                continue
            used_targets.add(target)
        if secondary_target is not None:
            secondary_target = (int(secondary_target[0]), int(secondary_target[1]))
            if not SafetyManager.validate_target(world, secondary_target) or secondary_target == target:
                secondary_target = None
        drone.set_intention(role, target, secondary_target=secondary_target)
        applied += 1
    return applied


def remember_plan(planner_memory: Dict[str, Any], drones: List[Drone], mode: str, step: int) -> None:
    planner_memory["last_plan_step"] = step
    planner_memory["thief_visible"] = any(d.beliefs.get("last_known_thief_pos") is not None for d in drones)
    planner_memory["targets"] = {d.id: d.target for d in drones}
    planner_memory["cached_plan"] = snapshot_current_plan(drones)
    planner_memory["mode"] = mode
    planner_memory["degraded_mode"] = mode.startswith("degraded")


def validate_and_finalize_assignments(
    world: GridWorld,
    drones: List[Drone],
    proposed: Dict[int, Tuple[str, Optional[Coordinate], Optional[Coordinate]]],
    planner_memory: Dict[str, Any],
    mode: str,
) -> int:
    used_targets: set[Coordinate] = set()
    applied = 0
    assigned_ids: set[int] = set()

    for drone in drones:
        if drone.id not in proposed:
            continue
        role, target, secondary_target = proposed[drone.id]
        if role not in ALLOWED_ROLES:
            role = "search"
        if target is not None:
            if not SafetyManager.validate_target(world, target):
                continue
            if target in used_targets:
                continue
            used_targets.add(target)
        if secondary_target is not None:
            if not SafetyManager.validate_target(world, secondary_target):
                secondary_target = None
            elif secondary_target == target:
                secondary_target = None
        drone.set_intention(role, target, secondary_target=secondary_target)
        assigned_ids.add(drone.id)
        applied += 1

    team_prob_map = aggregate_team_probability_map(world, drones)
    reserved_targets = set(used_targets)
    for drone in drones:
        if drone.id in assigned_ids:
            continue
        candidates = candidate_targets_for_drone(world, drone, drones, team_prob_map, max_candidates=5)
        chosen = None
        chosen_role = "search"
        for cand in candidates:
            cell = tuple(cand["target"])
            if cell in reserved_targets or not world.passable(cell):
                continue
            chosen = (int(cell[0]), int(cell[1]))
            if cand["type"] in {"last_known", "estimated", "hotspot", "contain_ring"}:
                chosen_role = "investigate"
            elif cand["type"] == "reposition":
                chosen_role = "reposition"
            break
        drone.set_intention(chosen_role, chosen, secondary_target=None)
        if chosen is not None:
            reserved_targets.add(chosen)
        applied += 1

    remember_plan(planner_memory, drones, mode=mode, step=int(planner_memory.get("current_step", 0)))
    return applied


def fallback_or_hold_plan(world: GridWorld, drones: List[Drone], planner_memory: Dict[str, Any], reason: str) -> PlannerDecision:
    cached_plan = planner_memory.get("cached_plan")
    freeze_steps = int(planner_memory.get("freeze_steps_remaining", 0))
    if cached_plan and freeze_steps > 0:
        applied = apply_plan_snapshot(world, drones, cached_plan)
        planner_memory["freeze_steps_remaining"] = max(0, freeze_steps - 1)
        planner_memory["degraded_mode"] = True
        planner_memory["mode"] = "hold_cached_plan"
        return PlannerDecision(
            source="cached_plan",
            success=applied > 0,
            mode="hold_cached_plan",
            reason=reason,
            degraded=True,
            reused_cached_plan=True,
            assignments_applied=applied,
        )

    assign_targets_to_drones_with_last_known(world, drones, thief=None)
    remember_plan(planner_memory, drones, mode="degraded_fallback", step=int(planner_memory.get("current_step", 0)))
    planner_memory["freeze_steps_remaining"] = 0
    return PlannerDecision(
        source="fallback",
        success=True,
        mode="degraded_fallback",
        reason=reason,
        degraded=True,
        assignments_applied=len(drones),
    )


def choose_team_plan(
    world: GridWorld,
    drones: List[Drone],
    thief: Thief,
    planner_memory: Dict[str, Any],
    use_llm: bool,
    llm_api_key: Optional[str],
    llm_model: str,
    llm_structured: bool,
) -> PlannerDecision:
    horizon = int(planner_memory.get("horizon", 5))
    if not should_replan(int(planner_memory.get("current_step", 0)), drones, planner_memory, horizon=horizon):
        if planner_memory.get("cached_plan"):
            applied = apply_plan_snapshot(world, drones, planner_memory["cached_plan"])
            return PlannerDecision(
                source="cached_plan",
                success=applied > 0,
                mode=str(planner_memory.get("mode", "hold_cached_plan")),
                reason="plan still valid",
                reused_cached_plan=True,
                assignments_applied=applied,
            )
        return fallback_or_hold_plan(world, drones, planner_memory, "missing cached plan")

    if use_llm:
        planned = assign_targets_with_llm_hybrid(
            world=world,
            drones=drones,
            thief=thief,
            planner_memory=planner_memory,
            api_key=llm_api_key,
            model=llm_model,
            use_structured=llm_structured,
        )
        if planned:
            planner_memory["consecutive_llm_failures"] = 0
            planner_memory["freeze_steps_remaining"] = max(1, int(planner_memory.get("horizon", 5)) - 1)
            planner_memory["last_successful_source"] = "llm"
            planner_memory["degraded_mode"] = False
            return PlannerDecision(
                source="llm",
                success=True,
                mode=str(planner_memory.get("mode", "llm_hybrid")),
                reason="llm plan accepted",
                assignments_applied=len(drones),
            )

        planner_memory["consecutive_llm_failures"] = int(planner_memory.get("consecutive_llm_failures", 0)) + 1
        if int(planner_memory.get("consecutive_llm_failures", 0)) >= int(planner_memory.get("max_consecutive_llm_failures", 2)):
            planner_memory["llm_disabled_until_step"] = int(planner_memory.get("current_step", 0)) + int(planner_memory.get("llm_cooldown_steps", 8))
        return fallback_or_hold_plan(world, drones, planner_memory, "llm unavailable or invalid")

    return fallback_or_hold_plan(world, drones, planner_memory, "llm disabled")


HYBRID_ASSIGNMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "mode": {"type": "string"},
        "horizon": {"type": "integer"},
        "assignments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "role": {"type": "string"},
                    "primary_target": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "minItems": 2,
                        "maxItems": 2,
                    },
                    "secondary_target": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "minItems": 2,
                        "maxItems": 2,
                    },
                    "rationale": {"type": "string"},
                },
                "required": ["id", "role", "primary_target"],
                "additionalProperties": False,
            },
        },
        "replan_if": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": ["mode", "horizon", "assignments"],
    "additionalProperties": False,
}


HYBRID_SYSTEM_PROMPT = """
Du bist ein strategischer Team-Planer für ein kooperatives Multi-Agenten-System aus Drohnen in einer Grid-Welt.

Du steuerst NICHT einzelne Mikrobewegungen, sondern vergibst:
- einen Team-Modus,
- eine Rolle pro Drohne,
- ein Primärziel pro Drohne,
- optional ein Sekundärziel.

## Ziel
Maximiere die Wahrscheinlichkeit, den Dieb schnell zu finden und zu fangen.

## Grundprinzip
Nutze die Heuristik-Kandidaten der Drohnen.
Bevorzuge Kandidatenziele aus "candidate_targets".
Erfinde nur dann ein anderes Ziel, wenn es zwingend strategisch besser ist.

## Prioritäten
1. Wenn der Dieb aktuell sichtbar oder sehr frisch gesichtet wurde:
   - containment_ring oder converge_last_seen
   - eine Drohne intercept / investigate
   - andere contain an Ring- oder Fluchtzellen
2. Wenn nur Wahrscheinlichkeitsverteilung vorliegt:
   - probability_sweep
   - verteile Drohnen über mehrere Hotspots
   - vermeide, dass alle zum selben Hotspot gehen
3. Wenn Unsicherheit hoch ist:
   - sector_search
   - maximiere Coverage und minimiere Overlap
4. Wenn mehrere Drohnen feststecken:
   - gib mindestens einer Drohne reposition

## Harte Regeln
- Jede Drohne genau einmal zuweisen
- Keine zwei Drohnen mit identischem primary_target
- Targets müssen im Grid und frei sein
- Minimiere Ziel-Redundanz
- Bevorzuge stabile Pläne für mehrere Schritte

## Erlaubte Rollen
- intercept
- contain
- investigate
- search
- reposition

## Gute Strategie
- Höhere Probability ist wichtig, aber nicht alle Drohnen auf ein Ziel
- Berücksichtige Distanz
- Nutze chokepoints und Ringpositionen
- Nutze Coverage statt Clusterbildung
- Nutze secondary_target als robusten Backup-Plan bei blockiertem Primärziel
- Wenn es eine frische letzte Sichtung gibt: Fluchtwege blockieren
- Wenn die Lage unklar ist: Informationsgewinn priorisieren

## Ausgabe
Gib ausschließlich JSON im vorgegebenen Schema zurück.
Keine Erklärungen außerhalb des JSON.
"""


def build_llm_prompt_state(world: GridWorld, drones: List[Drone], thief: Thief) -> Dict[str, Any]:
    visible_obstacles: set[Coordinate] = set()
    thief_visible = False

    for drone in drones:
        if manhattan(drone.position, thief.position) <= drone.sight_radius:
            thief_visible = True
        for obs in world.obstacles:
            if manhattan(obs, drone.position) <= drone.sight_radius:
                visible_obstacles.add(obs)

    thief_pos = thief.position if thief_visible else None
    drone_info = []
    for d in drones:
        prob_map = d.beliefs.get("thief_probability_map", {})
        hotspot = max(prob_map.items(), key=lambda kv: kv[1])[0] if prob_map else None
        drone_info.append({
            "id": d.id,
            "position": d.position,
            "role": d.role,
            "beliefs": {
                "last_known_thief_pos": d.beliefs.get("last_known_thief_pos"),
                "estimated_thief_pos": d.beliefs.get("estimated_thief_pos"),
                "hotspot": hotspot,
                "teammate_reports": d.beliefs.get("teammate_reports", [])[-5:],
                "visible_obstacles": d.beliefs.get("visible_obstacles", []),
            },
            "desires": d.desires,
            "intentions": d.intentions,
        })

    return {
        "thief_position": thief_pos,
        "drones": drone_info,
        "visible_obstacles": sorted(list(visible_obstacles)),
        "grid_size": [world.width, world.height],
    }


def best_probability_target(drone: Drone, reserved_targets: set[Coordinate], world: GridWorld) -> Optional[Coordinate]:
    prob_map: Dict[Coordinate, float] = drone.beliefs.get("thief_probability_map", {})
    if not prob_map:
        return None
    candidates = [cell for cell, prob in prob_map.items() if prob > 0 and cell not in reserved_targets and world.passable(cell)]
    if not candidates:
        candidates = [cell for cell, prob in prob_map.items() if prob > 0 and world.passable(cell)]
    if not candidates:
        return None
    candidates.sort(key=lambda c: (prob_map.get(c, 0.0), -manhattan(drone.position, c)), reverse=True)
    return candidates[0]


def assign_targets_to_drones_with_last_known(world: GridWorld, drones: List[Drone], thief: Optional[Thief]) -> None:
    """Fallback-Heuristik mit formalisierter BDI-Rolle, Sektorensuche und Wahrscheinlichkeitskarte."""
    reserved_targets: set[Coordinate] = set()
    known_positions = [d.beliefs.get("last_known_thief_pos") for d in drones if d.beliefs.get("last_known_thief_pos") is not None]

    for drone in drones:
        drone.update_desires()

    if known_positions:
        target_base = known_positions[-1]
        surround_targets = [
            target_base,
            (target_base[0] + 1, target_base[1]),
            (target_base[0] - 1, target_base[1]),
            (target_base[0], target_base[1] + 1),
            (target_base[0], target_base[1] - 1),
            (target_base[0] + 1, target_base[1] + 1),
            (target_base[0] - 1, target_base[1] - 1),
            (target_base[0] + 1, target_base[1] - 1),
            (target_base[0] - 1, target_base[1] + 1),
        ]
        valid = [p for p in surround_targets if world.in_bounds(p) and world.passable(p)]
        for idx, drone in enumerate(drones):
            available = [p for p in valid if p not in reserved_targets] or valid[:]
            if not available:
                drone.set_intention("intercept", target_base)
                continue
            available.sort(key=lambda p: manhattan(drone.position, p))
            chosen = available[0]
            role = "intercept" if idx == 0 else "contain"
            drone.set_intention(role, chosen)
            reserved_targets.add(chosen)
        return

    # probabilistische Hotspots zuerst untersuchen
    for drone in drones:
        role = drone.deliberate_role()
        hotspot = best_probability_target(drone, reserved_targets, world)
        if role in {"investigate", "reposition"} and hotspot is not None:
            drone.set_intention(role, hotspot)
            reserved_targets.add(hotspot)
            continue

        points = generate_sector_search_points(world, len(drones), drone.id, spacing=max(3, world.width // max(4, len(drones))))
        if len(points) < 2:
            points = generate_sector_search_points(world, len(drones), drone.id, spacing=3)
        candidate = next_patrol_point(drone, points, reserved_targets)
        if candidate is None:
            candidate = hotspot
        drone.set_intention("search", candidate)
        if candidate is not None:
            reserved_targets.add(candidate)


def assign_targets_with_openai_partial(
    world: GridWorld,
    drones: List[Drone],
    thief: Thief,
    api_key: Optional[str] = None,
    model: str = "gpt-5.4-mini",
    use_structured: bool = True,
) -> bool:
    """LLM-basierte Zielzuweisung mit Nachrichten-/BDI-Kontext."""
    if openai is None:
        return False

    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        return False

    user_prompt = build_llm_prompt_state(world, drones, thief)
    system_prompt = """
Du bist ein zentraler Koordinationsagent für ein kooperatives Multi-Agenten-System aus Drohnen in einer Grid-Welt.

Jede Drohne besitzt:
- Beliefs (inkl. Wahrscheinlichkeitsverteilung über die Position des Diebs)
- Desires
- Intentions
- eingeschränkte Sicht
- empfangene Team-Nachrichten

Deine Aufgabe ist es, für JEDE Drohne genau eine Rolle und ein Ziel zu bestimmen, um den Dieb effizient zu finden und zu fangen.

# GLOBALZIEL
Maximiere die Wahrscheinlichkeit der Gefangennahme in minimaler Zeit.

# ENTSCHEIDUNGSSTRATEGIE (verpflichtend)

## 1. Situationsbewertung (hierarchisch)
Treffe Entscheidungen strikt nach Priorität:
1. Dieb aktuell sichtbar: sofortige Einkreisung (Containment + Interception)
2. Letzte bekannte Position vorhanden: konvergieren + Fluchtwege blockieren
3. Nur Wahrscheinlichkeitsverteilung vorhanden: verteile Drohnen proportional zu Hochwahrscheinlichkeitsgebieten
4. Keine Information: systematische, nicht überlappende Sektorsuche

## 2. Nutzung von Unsicherheit (KRITISCH)
- Nutze "thief_probability_map", "estimated_thief_pos" und "hotspot"
- Höhere Wahrscheinlichkeit = höhere Priorität
- Kombiniere Wahrscheinlichkeit mit Distanz (Trade-off)

## 3. Rollenvergabe (nur diese Rollen erlaubt)
- "intercept" -> direkte Verfolgung
- "contain" -> blockiert Fluchtwege
- "investigate" -> bewegt sich zu Wahrscheinlichkeits-Hotspots
- "search" -> systematische Exploration
- "reposition" -> löst lokale Stagnation

## 4. Team-Koordination (HARTE REGELN)
- Jede Drohne MUSS ein einzigartiges Ziel haben
- Maximiere räumliche Abdeckung (Coverage)
- Minimiere Redundanz
- Vermeide Clusterbildung ohne Grund
- Koordiniere Rollen komplementär (kein homogenes Verhalten)

## 5. Bewegungseffizienz
- Bevorzuge Ziele mit hoher Wahrscheinlichkeit und geringer Distanz
- Vermeide unnötige Richtungswechsel
- Stabilität ist wichtiger als kurzfristige Optimierung

## 6. Anti-Stuck Verhalten
Wenn eine Drohne feststeckt oder ineffektiv ist:
- Rolle = "reposition"
- Ziel = freier Bereich mit niedriger Drohnendichte

## 7. Kommunikation nutzen
- Berücksichtige "teammate_reports"
- Neuere Informationen sind wichtiger als ältere
- Ignoriere veraltete oder inkonsistente Daten

## 8. Strategische Muster
### Einkreisung (wenn Dieb lokalisiert)
- Eine Drohne: "intercept"
- Andere: "contain" auf angrenzenden Feldern
- Ziel: alle Fluchtoptionen schließen

### Unsicherheit
- Verteile Drohnen entlang der Wahrscheinlichkeitsverteilung
- Nicht alle zum gleichen Hotspot schicken

# CONSTRAINTS (streng)
- Targets müssen innerhalb des Grids liegen
- Targets dürfen keine Hindernisse sein
- Jede Drohne genau einmal zuweisen
- Keine zwei Drohnen mit identischem Ziel

# AUSGABEFORMAT (STRICT JSON)
Gib ausschließlich folgendes Format zurück:
{
  "assignments": [
    {"id": 0, "role": "intercept", "target": [x, y]},
    {"id": 1, "role": "contain", "target": [x, y]}
  ]
}

# VERBOTEN
- Kein Text außerhalb von JSON
- Keine Erklärungen
- Keine zusätzlichen Felder
- Keine ungültigen Koordinaten

# DENKWEISE
Handle wie ein strategischer Planer, nicht wie ein einzelner Agent:
Optimiere das Verhalten des gesamten Teams, nicht einzelner Drohnen.
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(user_prompt)},
    ]

    schema = {
        "type": "object",
        "properties": {
            "assignments": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer"},
                        "role": {"type": "string"},
                        "target": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "minItems": 2,
                            "maxItems": 2,
                        },
                    },
                    "required": ["id", "role", "target"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["assignments"],
        "additionalProperties": False,
    }

    try:
        client = openai.OpenAI(api_key=key)
        if use_structured:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": "drone_assignments", "schema": schema},
                },
            )
        else:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                response_format={"type": "json_object"},
            )

        content = response.choices[0].message.content
        if not content:
            return False
        data = json.loads(content)
    except Exception:
        return False

    assignments = data.get("assignments", [])
    assigned_map: Dict[int, Tuple[str, Coordinate]] = {}

    for assignment in assignments:
        try:
            drone_id = int(assignment["id"])
            role = str(assignment.get("role", "search"))
            target_raw = assignment["target"]
            target = (int(target_raw[0]), int(target_raw[1]))
            if not world.in_bounds(target) or not world.passable(target):
                continue
            assigned_map[drone_id] = (role, target)
        except Exception:
            continue

    for drone in drones:
        if drone.id in assigned_map:
            role, target = assigned_map[drone.id]
            drone.set_intention(role, target)
        else:
            fallback_target = drone.beliefs.get("estimated_thief_pos") or drone.beliefs.get("last_known_thief_pos")
            fallback_target = fallback_target if isinstance(fallback_target, tuple) else None
            drone.set_intention("investigate" if fallback_target else "search", fallback_target)
    return True




def assign_targets_with_llm_hybrid(
    world: GridWorld,
    drones: List[Drone],
    thief: Thief,
    planner_memory: Dict[str, Any],
    api_key: Optional[str] = None,
    model: str = "gpt-5.4-mini",
    use_structured: bool = True,
) -> bool:
    if openai is None:
        return False

    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        return False

    planner_state = build_strategic_planner_state(world, drones, thief)

    messages = [
        {"role": "system", "content": HYBRID_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(planner_state)},
    ]

    try:
        client = openai.OpenAI(api_key=key)
        if use_structured:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "hybrid_drone_plan",
                        "schema": HYBRID_ASSIGNMENT_SCHEMA,
                    },
                },
            )
        else:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                response_format={"type": "json_object"},
            )

        content = response.choices[0].message.content
        if not content:
            return False

        data = json.loads(content)
    except Exception:
        return False

    assignments = data.get("assignments", [])
    proposed: Dict[int, Tuple[str, Optional[Coordinate], Optional[Coordinate]]] = {}

    for item in assignments:
        try:
            drone_id = int(item["id"])
            role = str(item.get("role", "search"))
            primary = item["primary_target"]
            secondary = item.get("secondary_target")
            target = (int(primary[0]), int(primary[1]))
            secondary_target = None
            if isinstance(secondary, list) and len(secondary) == 2:
                secondary_target = (int(secondary[0]), int(secondary[1]))
            proposed[drone_id] = (role, target, secondary_target)
        except Exception:
            continue

    planner_memory["horizon"] = max(2, min(12, int(data.get("horizon", 5))))
    applied = validate_and_finalize_assignments(
        world=world,
        drones=drones,
        proposed=proposed,
        planner_memory=planner_memory,
        mode=str(data.get("mode", "llm_hybrid")),
    )
    return applied > 0


def draw_state(world: GridWorld, drones: List[Drone], thief: Thief, step: int, path: str, show_vision: bool = True) -> None:
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(0, world.width)
    ax.set_ylim(0, world.height)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.invert_yaxis()

    for (x, y) in world.static_obstacles:
        rect = plt.Rectangle((x, y), 1, 1, color="black")
        ax.add_patch(rect)

    for obs in world.dynamic_obstacles:
        rect = plt.Rectangle((obs.position[0], obs.position[1]), 1, 1, color="dimgray")
        ax.add_patch(rect)

    if show_vision:
        for drone in drones:
            vision = Circle((drone.position[0] + 0.5, drone.position[1] + 0.5), drone.sight_radius, color="blue", alpha=0.08)
            ax.add_patch(vision)

    # Wahrscheinlichkeits-Hotspot jedes Teams visualisieren
    team_prob: Dict[Coordinate, float] = {}
    for drone in drones:
        for cell, prob in drone.beliefs.get("thief_probability_map", {}).items():
            team_prob[cell] = max(team_prob.get(cell, 0.0), prob)
    if team_prob:
        top_cells = sorted(team_prob.items(), key=lambda kv: kv[1], reverse=True)[:20]
        for (x, y), prob in top_cells:
            alpha = min(0.25, 0.05 + prob)
            rect = plt.Rectangle((x, y), 1, 1, color="gold", alpha=alpha)
            ax.add_patch(rect)

    colors = ["blue", "green", "orange", "purple", "brown", "pink"]
    for i, drone in enumerate(drones):
        col = colors[i % len(colors)]
        circ = Circle((drone.position[0] + 0.5, drone.position[1] + 0.5), 0.35, color=col)
        ax.add_patch(circ)
        ax.text(drone.position[0] + 0.05, drone.position[1] + 0.2, f"{drone.id}:{drone.role[:3]}", fontsize=8)

    x = thief.position[0] + 0.5
    y = thief.position[1] + 0.5
    ax.plot(x, y, marker="x", color="red", markersize=16, markeredgewidth=4)
    filename = f"{path}/frame_{step:03d}.png"
    fig.savefig(filename, bbox_inches="tight")
    plt.close(fig)


def check_capture(drones: List[Drone], thief: Thief, world: GridWorld) -> bool:
    """Formale Capture-Regel.

    Capture liegt vor, wenn mindestens eine der Bedingungen gilt:
    1) Eine Drohne steht auf der Diebszelle.
    2) Eine Kantenkreuzung tritt auf (Drohne und Dieb tauschen Zellen im selben Takt).
    3) Die aktuelle Diebsposition und alle legalen Folgezellen sind durch Drohnen belegt.
    """
    drone_positions = {d.position for d in drones}
    if thief.position in drone_positions:
        return True

    for drone in drones:
        if drone.prev_position == thief.position and drone.position == thief.prev_position:
            return True

    legal_escape_cells = set(world.neighbors(thief.position, include_wait=True))
    if legal_escape_cells and legal_escape_cells.issubset(drone_positions):
        return True
    return False


def resolve_drone_moves(world: GridWorld, drones: List[Drone], step: int) -> None:
    occupied_now = {d.position for d in drones}
    order = sorted(drones, key=lambda d: (d.id - step) % len(drones))  # rotierende Fairness
    reserved_next: set[Coordinate] = set()
    proposals: Dict[int, Coordinate] = {}

    for drone in order:
        proposals[drone.id] = drone.propose_move(world, occupied_now, reserved_next)
        reserved_next.add(proposals[drone.id])

    for drone in order:
        drone.commit_position(proposals[drone.id])


def run_dynamic_simulation_llm(
    width: int = 40,
    height: int = 40,
    static_obstacle_ratio: float = 0.05,
    dynamic_obstacle_ratio: float = 0.05,
    num_drones: int = 6,
    sight_radius_drone: int = 5,
    sight_radius_thief: int = 4,
    max_steps: int = 250,
    output_dir: str = "frames",
    save_gif: str | None = "dynamic_simulation_llm.gif",
    use_llm: bool = True,
    llm_api_key: Optional[str] = None,
    llm_model: str = "gpt-5.4-mini",
    llm_structured: bool = True,
    seed: Optional[int] = None,
) -> None:
    """Simulation mit resilienter Planer-Architektur.

    Neu gegenüber der Ausgangsversion:
    - LLM ist optional und nie Single Point of Failure
    - letzter gültiger Plan kann kurz weiterverwendet werden
    - nach wiederholten LLM-Fehlern folgt Cooldown/Degraded Mode
    - Fallback-Planer übernimmt deterministisch
    """
    import shutil
    from PIL import Image

    if seed is not None:
        random.seed(seed)
    else:
        random.seed()

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    total_cells = width * height
    num_static = int(total_cells * static_obstacle_ratio)
    num_dynamic = int(total_cells * dynamic_obstacle_ratio)

    static_obstacles: set[Coordinate] = set()
    dynamic_obstacles: set[Coordinate] = set()

    while len(static_obstacles) < num_static:
        static_obstacles.add((random.randint(0, width - 1), random.randint(0, height - 1)))

    while len(dynamic_obstacles) < num_dynamic:
        pos = (random.randint(0, width - 1), random.randint(0, height - 1))
        if pos not in static_obstacles:
            dynamic_obstacles.add(pos)

    world = GridWorld(width, height, list(static_obstacles), list(dynamic_obstacles))

    def random_free_position(exclude: set[Coordinate]) -> Coordinate:
        while True:
            pos = (random.randint(0, width - 1), random.randint(0, height - 1))
            if pos not in world.obstacles and pos not in exclude:
                return pos

    thief = Thief(random_free_position(set()), sight_radius=sight_radius_thief)

    drones: List[Drone] = []
    occupied: set[Coordinate] = {thief.position}
    for i in range(num_drones):
        pos = random_free_position(occupied)
        drone = Drone(i, pos, sight_radius=sight_radius_drone)
        drone.initialize_probability_map(world)
        drones.append(drone)
        occupied.add(pos)

    planner_stats: Dict[str, int] = {
        "llm_success_count": 0,
        "fallback_count": 0,
        "cached_plan_reuse_count": 0,
        "degraded_mode_steps": 0,
    }
    total_messages_sent = 0
    first_sighting_step: Optional[int] = None
    planner_memory: Dict[str, Any] = {
        "last_plan_step": -999,
        "thief_visible": False,
        "targets": {},
        "mode": None,
        "horizon": 5,
        "current_step": 0,
        "cached_plan": {},
        "freeze_steps_remaining": 0,
        "consecutive_llm_failures": 0,
        "max_consecutive_llm_failures": 2,
        "llm_cooldown_steps": 8,
        "llm_disabled_until_step": -1,
        "last_successful_source": None,
        "degraded_mode": False,
    }

    for step in range(max_steps):
        agent_positions = {thief.position} | {d.position for d in drones}
        world.update_dynamic_obstacles(agent_positions)

        messages = collect_team_messages(world, drones, thief, step)
        total_messages_sent += len(messages)
        distribute_messages(world, drones, messages)

        for drone in drones:
            drone.decay_last_known_thief_position(step)

        if first_sighting_step is None and any(msg.type == "THIEF_SPOTTED" for msg in messages):
            first_sighting_step = step + 1

        planner_memory["current_step"] = step
        llm_allowed_now = bool(use_llm and step >= int(planner_memory.get("llm_disabled_until_step", -1)))
        decision = choose_team_plan(
            world=world,
            drones=drones,
            thief=thief,
            planner_memory=planner_memory,
            use_llm=llm_allowed_now,
            llm_api_key=llm_api_key,
            llm_model=llm_model,
            llm_structured=llm_structured,
        )

        if decision.source == "llm" and decision.success:
            planner_stats["llm_success_count"] += 1
        elif decision.reused_cached_plan:
            planner_stats["cached_plan_reuse_count"] += 1
        else:
            planner_stats["fallback_count"] += 1

        if decision.degraded:
            planner_stats["degraded_mode_steps"] += 1

        resolve_drone_moves(world, drones, step)
        thief.choose_action(world, [d.position for d in drones])
        draw_state(world, drones, thief, step, output_dir, show_vision=True)

        if check_capture(drones, thief, world):
            print(f"Dieb gefangen in Schritt {step + 1}.")
            break

    if save_gif:
        frames = []
        frame_files = sorted([f for f in os.listdir(output_dir) if f.endswith(".png")])
        for f in frame_files:
            img = Image.open(os.path.join(output_dir, f))
            frames.append(img)
        if frames:
            frames[0].save(
                save_gif,
                format="GIF",
                append_images=frames[1:],
                save_all=True,
                duration=300,
                loop=0,
            )

    print(
        f"Simulation beendet nach {step + 1} Schritten. "
        f"LLM genutzt: {planner_stats['llm_success_count']}×, "
        f"Cached-Plan genutzt: {planner_stats['cached_plan_reuse_count']}×, "
        f"Fallback: {planner_stats['fallback_count']}×, "
        f"Degraded-Mode-Schritte: {planner_stats['degraded_mode_steps']}, "
        f"Nachrichten gesendet: {total_messages_sent}."
    )

    if first_sighting_step is not None:
        print(f"Dieb erstmals gesichtet in Schritt {first_sighting_step}.")
    else:
        print("Der Dieb wurde nie gesichtet.")


if __name__ == "__main__":
    
    for i in range(30,60):
        print(f"========== RUN {i} ==========")
        run_dynamic_simulation_llm(
            num_drones=3,
            sight_radius_drone=5,
            sight_radius_thief=4,
            max_steps=200,
            use_llm=True,
            seed=i,
            output_dir=f"frames_run_{i}",
            save_gif=f"dynamic_simulation_resilient_{i}.gif",
        )

#if __name__ == "__main__":
 #   run_dynamic_simulation_llm(
 #       num_drones=6,
  #      sight_radius_drone=5,
   #     sight_radius_thief=4,
    #    max_steps=200,
     #   use_llm=True,
    #)

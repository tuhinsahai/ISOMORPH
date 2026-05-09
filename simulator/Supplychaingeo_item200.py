"""
Supply Chain Simulation — Day-level, 52560-step.
200-item.

1 step = 1 day | 365 steps = 1 year | 52560 steps = 144 years

logic:
  - (s,S) inventory policy at warehouses
  - Dijkstra routing (weight = travel_time / daily_capacity)
  - Greedy first-fit bin packing
  - Proactive shipping via pipeline_multiplier
  - Streaming CSV for large runs


Multi-echelon changes:
  - Only source nodes (SF, StLouis, Orlando) retain magic replenishment
  - Intermediate nodes pull inventory from upstream via network edges
  - Per-tier (s,S) parameters calibrated to demand flow
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Callable
import math
import heapq
import random
import os
import csv
import time as _time_module
from datetime import datetime, timedelta
import io
import base64

try:
    import matplotlib.pyplot as plt
    from branca.element import Element
except ImportError:
    pass

import numpy as np
import pandas as pd

try:
    import folium
    from folium.plugins import TimestampedGeoJson
    HAS_FOLIUM = True
except ImportError:
    HAS_FOLIUM = False


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class Item:
    item_id: str
    volume: float


@dataclass
class Edge:
    u: str
    v: str
    travel_time_days: float
    container_volume: float
    num_containers_per_day: int
    daily_containers: List[float] = field(default_factory=list)

    def reset_daily(self, capacity_factor: float = 1.0) -> None:
        effective_vol = self.container_volume * capacity_factor
        self.daily_containers = [effective_vol] * self.num_containers_per_day

    def find_container_slot(self, item_volume: float) -> Optional[int]:
        for idx, rem in enumerate(self.daily_containers):
            if rem >= item_volume:
                return idx
        return None

    def allocate_in_container(self, idx: int, item_volume: float) -> bool:
        if 0 <= idx < len(self.daily_containers) and \
                self.daily_containers[idx] >= item_volume:
            self.daily_containers[idx] -= item_volume
            return True
        return False

    @property
    def daily_total_capacity(self) -> float:
        return self.container_volume * max(self.num_containers_per_day, 0)


@dataclass
class Node:
    node_id: str
    lat: float
    lon: float
    is_destination: bool = False
    is_source: bool = False
    inventory: Dict[str, int] = field(default_factory=dict)
    s_levels: Dict[str, int] = field(default_factory=dict)
    S_levels: Dict[str, int] = field(default_factory=dict)
    lead_time_mean: Dict[str, float] = field(default_factory=dict)
    lead_time_std_frac: float = 0.2
    outstanding_orders: Dict[str, Optional[Tuple[int, int]]] = \
        field(default_factory=dict)
    backlog: Dict[str, int] = field(default_factory=dict)

    def receive_orders_today(self, day: int) -> None:
        to_clear = []
        for item_id, order in self.outstanding_orders.items():
            if order is None:
                continue
            arrival_day, qty = order
            if day >= arrival_day:
                self.inventory[item_id] = \
                    self.inventory.get(item_id, 0) + qty
                to_clear.append(item_id)
        for iid in to_clear:
            self.outstanding_orders[iid] = None

    def maybe_place_orders(self, day: int, rng: random.Random) -> None:
        if self.is_destination:
            return
        if not self.is_source:
            return
        for item_id, s in self.s_levels.items():
            on_hand = self.inventory.get(item_id, 0)
            if on_hand < s and \
                    self.outstanding_orders.get(item_id) is None:
                S = self.S_levels.get(item_id, on_hand)
                qty = max(S - on_hand, 0)
                if qty <= 0:
                    continue
                mean_lt = max(self.lead_time_mean.get(item_id, 1.0), 0.1)
                std = self.lead_time_std_frac * mean_lt
                sampled = rng.normalvariate(mean_lt, std)
                lt_days = max(1, int(math.ceil(sampled)))
                self.outstanding_orders[item_id] = (day + lt_days, qty)


# ============================================================================
# Network + Dijkstra
# ============================================================================

class Network:
    def __init__(self) -> None:
        self.nodes: Dict[str, Node] = {}
        self.edges: Dict[Tuple[str, str], Edge] = {}
        self.adj: Dict[str, List[str]] = {}
        self.weight_cache: Dict[Tuple[str, str], float] = {}
        self.paths_to_dest: Dict[
            str, Tuple[float, List[str], List[Tuple[str, str]]]] = {}

    def add_node(self, node: Node) -> None:
        self.nodes[node.node_id] = node
        self.adj.setdefault(node.node_id, [])

    def add_edge(self, edge: Edge) -> None:
        self.edges[(edge.u, edge.v)] = edge
        self.adj.setdefault(edge.u, []).append(edge.v)
        cap = max(edge.daily_total_capacity, 1e-9)
        self.weight_cache[(edge.u, edge.v)] = edge.travel_time_days / cap

    def reset_daily_edges(self, capacity_factor: float = 1.0) -> None:
        for e in self.edges.values():
            e.reset_daily(capacity_factor)

    def dijkstra(self, source: str, target: str) -> Tuple[float, List[str]]:
        if source == target:
            return 0.0, [source]
        dist: Dict[str, float] = {source: 0.0}
        prev: Dict[str, Optional[str]] = {source: None}
        pq = [(0.0, source)]
        visited: set = set()
        while pq:
            d, u = heapq.heappop(pq)
            if u in visited:
                continue
            visited.add(u)
            if u == target:
                break
            for v in self.adj.get(u, []):
                w = self.weight_cache[(u, v)]
                nd = d + w
                if nd < dist.get(v, float('inf')):
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(pq, (nd, v))
        if target not in dist:
            return float('inf'), []
        path: List[str] = []
        cur: Optional[str] = target
        while cur is not None:
            path.append(cur)
            cur = prev.get(cur)
        path.reverse()
        return dist[target], path

    def compute_paths_to_destination(self, dest_id: str) -> None:
        self.paths_to_dest.clear()
        for nid in self.nodes:
            if nid == dest_id:
                self.paths_to_dest[nid] = (0.0, [nid], [])
                continue
            d, pn = self.dijkstra(nid, dest_id)
            if not pn:
                self.paths_to_dest[nid] = (float('inf'), [], [])
            else:
                pe = [(pn[i], pn[i+1]) for i in range(len(pn)-1)]
                self.paths_to_dest[nid] = (d, pn, pe)


# ============================================================================
# Greedy First-Fit Bin Packing
# ============================================================================

def allocate_units_along_path_greedy(
    item: Item,
    max_units: int,
    path_edges: List[Tuple[str, str]],
    network_edges: Dict[Tuple[str, str], Edge],
) -> int:
    if max_units <= 0 or not path_edges:
        return 0
    edges_objs = [network_edges[eid] for eid in path_edges]
    placed = 0
    ivol = item.volume
    for _ in range(max_units):
        slots: List[Tuple[Edge, int]] = []
        ok = True
        for e in edges_objs:
            si = e.find_container_slot(ivol)
            if si is None:
                ok = False
                break
            slots.append((e, si))
        if not ok:
            break
        good = True
        for e, si in slots:
            if not e.allocate_in_container(si, ivol):
                good = False
                break
        if not good:
            break
        placed += 1
    return placed


# ============================================================================
# Simulation Engine
# ============================================================================

class SupplyChainSimulation:

    def __init__(
        self,
        network: Network,
        items: Dict[str, Item],
        destination_id: str,
        demand_fn: Callable[[int], Dict[str, int]],
        horizon_days: int,
        seed: int = 42,
        pipeline_multiplier: float = 0.0,
        streaming_out_dir: Optional[str] = None,
        packing: str = "greedy",
    ) -> None:
        assert destination_id in network.nodes and \
            network.nodes[destination_id].is_destination
        self.network = network
        self.items = items
        self.item_order: List[str] = sorted(self.items.keys())
        self.round_robin_items: bool = True
        self.per_item_daily_cap_units: Optional[int] = None

        self.destination_id = destination_id
        self.demand_fn = demand_fn
        self.horizon_days = horizon_days
        self.rng = random.Random(seed)
        self.packing = packing
        self.pipeline_multiplier = pipeline_multiplier

        # EMA warm-start at approximate per-day mean demand
        self.demand_ema: Dict[str, float] = {iid: 165.0 for iid in items}
        self.ema_alpha = 0.05
        self.item_intransit: Dict[str, int] = {iid: 0 for iid in items}

        self.streaming_out_dir = streaming_out_dir
        self._csv_files: Dict = {}
        self._csv_writers: Dict = {}
        self._csv_buffers: Dict[str, list] = {}
        self.dest_in_transit: Dict[int, Dict[str, int]] = {}

        self.daily_records: list = []
        self.shipments_log: list = []
        self.inventory_history: list = []
        self.backlog_history: list = []
        self.intransit_history: list = []

        self.svc_demand: Dict[str, int] = {iid: 0 for iid in items}
        self.svc_served: Dict[str, int] = {iid: 0 for iid in items}
        self.svc_backlog: Dict[str, int] = {iid: 0 for iid in items}

        network.compute_paths_to_destination(destination_id)
        self.sorted_warehouses = sorted(
            [(nid, d) for nid, (d, _, _) in network.paths_to_dest.items()
             if nid != destination_id and math.isfinite(d)],
            key=lambda x: x[1])

        # precompute supplier relationships for intermediate nodes
        self.node_suppliers: Dict[
            str,
            List[Tuple[str, float, List[str], List[Tuple[str, str]]]]
        ] = {}
        for nid, node in network.nodes.items():
            if node.is_destination or node.is_source:
                continue
            suppliers = []
            for (u, v) in network.edges:
                if v == nid:
                    d, path = network.dijkstra(u, nid)
                    if path and math.isfinite(d):
                        pe = [(path[i], path[i + 1])
                              for i in range(len(path) - 1)]
                        tt = sum(network.edges[e].travel_time_days
                                 for e in pe)
                        suppliers.append((u, tt, path, pe))
            suppliers.sort(key=lambda x: x[1])
            self.node_suppliers[nid] = suppliers

        # intermediate nodes ordered upstream-first for replenishment
        self.replenish_order: List[str] = [
            nid for nid, _ in reversed(self.sorted_warehouses)
            if nid in self.node_suppliers]

        self.avg_travel_time = 6.0
        if self.sorted_warehouses:
            _, _, pe = network.paths_to_dest[self.sorted_warehouses[0][0]]
            if pe:
                self.avg_travel_time = sum(
                    network.edges[e].travel_time_days for e in pe)

    def _total_tt(self, pe):
        return sum(self.network.edges[e].travel_time_days for e in pe)

    def _replenish_warehouses(self, day: int) -> None:
        """Inter-warehouse replenishment: intermediate nodes pull
        from upstream suppliers using (s,S) trigger + edge capacity."""
        net = self.network

        if self.round_robin_items and self.item_order:
            k = day % len(self.item_order)
            ids_today = self.item_order[k:] + self.item_order[:k]
        else:
            ids_today = self.item_order[:]

        for nid in self.replenish_order:
            node = net.nodes[nid]
            for iid in ids_today:
                on_hand = node.inventory.get(iid, 0)
                s = node.s_levels.get(iid, 0)
                if on_hand >= s:
                    continue
                if node.outstanding_orders.get(iid) is not None:
                    continue
                S = node.S_levels.get(iid, on_hand)
                qty_needed = max(S - on_hand, 0)
                if qty_needed <= 0:
                    continue

                for sup_id, tt, path, pe in \
                        self.node_suppliers.get(nid, []):
                    sup_node = net.nodes[sup_id]
                    avail = sup_node.inventory.get(iid, 0)
                    if avail <= 0:
                        continue
                    attempt = min(avail, qty_needed)
                    placed = allocate_units_along_path_greedy(
                        self.items[iid], attempt, pe, net.edges)
                    if placed <= 0:
                        continue

                    sup_node.inventory[iid] -= placed
                    arr = day + max(1, int(math.ceil(tt)))
                    node.outstanding_orders[iid] = (arr, placed)

                    r = [day, arr, sup_id, nid, iid, placed,
                         str(path),
                         str([net.edges[e].travel_time_days
                              for e in pe])]
                    if self.streaming_out_dir:
                        self._csv_buffers.setdefault(
                            'ship', []).append(r)
                    else:
                        self.shipments_log.append({
                            "day": day, "arrival_day": arr,
                            "from": sup_id, "to": nid,
                            "item": iid, "units": placed,
                            "path_nodes": path,
                            "edge_times": [
                                net.edges[e].travel_time_days
                                for e in pe]})
                    break

    def step(self, day: int) -> None:
        net = self.network
        dest = net.nodes[self.destination_id]

        # 1) Receive (s,S) replenishment at warehouses
        for node in net.nodes.values():
            node.receive_orders_today(day)

        # 2) Arrivals at destination
        arrivals = self.dest_in_transit.pop(day, {})
        for iid, qty in arrivals.items():
            self.item_intransit[iid] = max(
                0, self.item_intransit.get(iid, 0) - qty)
            bl = dest.backlog.get(iid, 0)
            if bl > 0:
                use = min(qty, bl)
                dest.backlog[iid] = bl - use
                qty -= use
            if qty > 0:
                dest.inventory[iid] = dest.inventory.get(iid, 0) + qty

        # 3) Reset edge containers
        net.reset_daily_edges(1.0)

        # 4) Demand at destination
        td = self.demand_fn(day)
        for iid in self.items:
            dq = int(td.get(iid, 0))
            self.demand_ema[iid] = (
                self.ema_alpha * dq +
                (1 - self.ema_alpha) * self.demand_ema[iid])
            oh = dest.inventory.get(iid, 0)
            if oh >= dq:
                served, unfilled = dq, 0
                dest.inventory[iid] = oh - dq
            else:
                served, unfilled = oh, dq - oh
                dest.inventory[iid] = 0
                dest.backlog[iid] = dest.backlog.get(iid, 0) + unfilled
            self.svc_demand[iid] += dq
            self.svc_served[iid] += served
            self.svc_backlog[iid] += unfilled
            rec = [day, iid, dq, served, unfilled,
                   dest.inventory.get(iid, 0), dest.backlog.get(iid, 0)]
            if self.streaming_out_dir:
                self._csv_buffers.setdefault('daily', []).append(rec)
            else:
                self.daily_records.append({
                    "day": day, "item": iid, "demand": dq,
                    "served_from_stock": served,
                    "new_backlog_today": unfilled,
                    "dest_on_hand_end_before_ship":
                        dest.inventory.get(iid, 0),
                    "dest_backlog_end_before_ship":
                        dest.backlog.get(iid, 0)})

        # 5) Ship
        if self.round_robin_items and self.item_order:
            k = day % len(self.item_order)
            ids_today = self.item_order[k:] + self.item_order[:k]
        else:
            ids_today = self.item_order[:]

        for iid in ids_today:
            item = self.items[iid]
            cb = dest.backlog.get(iid, 0)
            it = self.item_intransit.get(iid, 0)
            oh = dest.inventory.get(iid, 0)

            if self.pipeline_multiplier > 0:
                pt = self.demand_ema[iid] * self.pipeline_multiplier
                ship_target = max(0, int(math.ceil(cb + pt - it - oh)))
            else:
                S_dest = max(1, int(self.demand_ema[iid] * 3))
                ship_target = max(0, cb + S_dest - oh - it)

            if ship_target <= 0:
                continue

            remaining = ship_target
            shipped = 0
            for wid, _ in self.sorted_warehouses:
                if remaining <= 0:
                    break
                if self.per_item_daily_cap_units is not None and \
                        shipped >= self.per_item_daily_cap_units:
                    break
                wn = net.nodes[wid]
                avail = wn.inventory.get(iid, 0)
                if avail <= 0:
                    continue
                _, pn, pe = net.paths_to_dest[wid]
                if not pe:
                    continue
                attempt = min(avail, remaining)
                if self.per_item_daily_cap_units is not None:
                    attempt = min(attempt,
                                  self.per_item_daily_cap_units - shipped)
                placed = allocate_units_along_path_greedy(
                    item, attempt, pe, net.edges)
                if placed <= 0:
                    continue
                wn.inventory[iid] -= placed
                remaining -= placed
                shipped += placed
                arr = day + max(1, int(math.ceil(self._total_tt(pe))))
                self.dest_in_transit.setdefault(arr, {})
                self.dest_in_transit[arr][iid] = \
                    self.dest_in_transit[arr].get(iid, 0) + placed
                self.item_intransit[iid] = \
                    self.item_intransit.get(iid, 0) + placed
                r = [day, arr, wid, self.destination_id, iid, placed,
                     str(pn),
                     str([net.edges[e].travel_time_days for e in pe])]
                if self.streaming_out_dir:
                    self._csv_buffers.setdefault('ship', []).append(r)
                else:
                    self.shipments_log.append({
                        "day": day, "arrival_day": arr,
                        "from": wid, "to": self.destination_id,
                        "item": iid, "units": placed,
                        "path_nodes": pn,
                        "edge_times": [net.edges[e].travel_time_days
                                       for e in pe]})

        # 5b) Inter-warehouse replenishment
        self._replenish_warehouses(day)

        # 6) (s,S) orders at source warehouses
        for node in net.nodes.values():
            node.maybe_place_orders(day, self.rng)

        # 7) Snapshots
        did = self.destination_id
        itc: Dict[str, int] = {}
        for ad, im in self.dest_in_transit.items():
            if ad > day:
                for iid, q in im.items():
                    itc[iid] = itc.get(iid, 0) + int(q)

        if self.streaming_out_dir:
            for node in net.nodes.values():
                for iid in self.items:
                    self._csv_buffers.setdefault('inv', []).append(
                        [day, node.node_id, iid,
                         int(node.inventory.get(iid, 0))])
                    self._csv_buffers.setdefault('bl', []).append(
                        [day, node.node_id, iid,
                         int(node.backlog.get(iid, 0))])
            for iid in self.items:
                self._csv_buffers.setdefault('it', []).append(
                    [day, did, iid, itc.get(iid, 0)])
        else:
            for node in net.nodes.values():
                for iid in self.items:
                    self.inventory_history.append({
                        "day": day, "node": node.node_id, "item": iid,
                        "on_hand": int(node.inventory.get(iid, 0))})
                    self.backlog_history.append({
                        "day": day, "node": node.node_id, "item": iid,
                        "backlog": int(node.backlog.get(iid, 0))})
            for iid in self.items:
                self.intransit_history.append({
                    "day": day, "node": did, "item": iid,
                    "in_transit": itc.get(iid, 0)})

    # --- CSV streaming ---
    def _open_csv_files(self):
        os.makedirs(self.streaming_out_dir, exist_ok=True)
        hdrs = {
            'daily': ["day", "item", "demand", "served_from_stock",
                      "new_backlog_today", "dest_on_hand_end_before_ship",
                      "dest_backlog_end_before_ship"],
            'ship':  ["day", "arrival_day", "from", "to", "item",
                      "units", "path_nodes", "edge_times"],
            'inv':   ["day", "node", "item", "on_hand"],
            'bl':    ["day", "node", "item", "backlog"],
            'it':    ["day", "node", "item", "in_transit"],
        }
        fn = {'daily': 'daily_records', 'ship': 'shipments',
              'inv': 'inventory_history', 'bl': 'backlog_history',
              'it': 'intransit_history'}
        for k in hdrs:
            f = open(os.path.join(self.streaming_out_dir,
                                  f"{fn[k]}.csv"),
                     "w", newline="", buffering=65536)
            w = csv.writer(f)
            w.writerow(hdrs[k])
            self._csv_files[k] = f
            self._csv_writers[k] = w
            self._csv_buffers[k] = []

    def _flush_csv(self):
        for k in self._csv_buffers:
            if self._csv_buffers[k] and k in self._csv_writers:
                self._csv_writers[k].writerows(self._csv_buffers[k])
                self._csv_buffers[k] = []
        for f in self._csv_files.values():
            f.flush()

    def _close_csv(self):
        self._flush_csv()
        for f in self._csv_files.values():
            f.close()

    def run(self):
        if self.streaming_out_dir:
            self._open_csv_files()
        t0 = _time_module.time()
        ri = max(1, self.horizon_days // 100)
        for day in range(self.horizon_days):
            self.step(day)
            if self.streaming_out_dir and day % 500 == 0:
                self._flush_csv()
            if day % 5000 == 0:
                for k in [k for k in self.dest_in_transit if k < day]:
                    del self.dest_in_transit[k]
            if day % ri == 0 or day == self.horizon_days - 1:
                el = _time_module.time() - t0
                pct = (day + 1) / self.horizon_days * 100
                rate = (day + 1) / max(el, 0.001)
                eta = (self.horizon_days - day - 1) / max(rate, 0.001)
                print(f"\r  Day {day+1:>6}/{self.horizon_days} "
                      f"({pct:5.1f}%) {rate:6.1f} days/s "
                      f"ETA {eta:6.0f}s", end="", flush=True)
        print(f"\n  Simulation complete in "
              f"{_time_module.time()-t0:.1f}s")

        if self.streaming_out_dir:
            self._close_csv()
            rows = []
            for iid in sorted(self.items):
                td = self.svc_demand[iid]
                sv = self.svc_served[iid]
                bl = self.svc_backlog[iid]
                rows.append({
                    "item": iid, "total_demand": td,
                    "served_from_stock": sv,
                    "new_backlog_added": bl,
                    "fill_rate_stock_only":
                        round(sv / td, 6) if td > 0 else 0.0})
            svc = pd.DataFrame(rows)
            svc.to_csv(os.path.join(self.streaming_out_dir,
                                    "service_summary.csv"), index=False)
            return (pd.DataFrame(), pd.DataFrame(), svc,
                    pd.DataFrame(), pd.DataFrame(), pd.DataFrame())

        dd = pd.DataFrame(self.daily_records)
        ds = pd.DataFrame(self.shipments_log) if self.shipments_log \
            else pd.DataFrame(columns=[
                "day", "arrival_day", "from", "to",
                "item", "units", "path_nodes", "edge_times"])
        svc = dd.groupby("item", as_index=False).agg(
            total_demand=("demand", "sum"),
            served_from_stock=("served_from_stock", "sum"),
            new_backlog_added=("new_backlog_today", "sum"))
        svc["fill_rate_stock_only"] = (
            svc["served_from_stock"] / svc["total_demand"]).fillna(0)
        di = pd.DataFrame(self.inventory_history,
                          columns=["day", "node", "item", "on_hand"])
        db = pd.DataFrame(self.backlog_history,
                          columns=["day", "node", "item", "backlog"])
        dt = pd.DataFrame(self.intransit_history,
                          columns=["day", "node", "item", "in_transit"])
        return dd, ds, svc, di, db, dt


# ============================================================================
# Build network from adjacency
# ============================================================================

def build_network_from_adjacency(nodes_meta, adjacency):
    n = len(nodes_meta)
    assert all(len(r) == n for r in adjacency)
    net = Network()
    for meta in nodes_meta:
        net.add_node(Node(
            node_id=meta["id"],
            lat=float(meta["lat"]), lon=float(meta["lon"]),
            is_destination=bool(meta.get("is_destination", False)),
            is_source=bool(meta.get("is_source", False)),
            inventory=dict(meta.get("inventory", {})),
            s_levels=dict(meta.get("s_levels", {})),
            S_levels=dict(meta.get("S_levels", {})),
            lead_time_mean=dict(meta.get("lead_time_mean", {})),
            backlog=dict(meta.get("backlog", {}))))
    ids = [m["id"] for m in nodes_meta]
    for i in range(n):
        for j in range(n):
            s = adjacency[i][j]
            if s is None:
                continue
            tt, cv, nc = s
            net.add_edge(Edge(
                u=ids[i], v=ids[j],
                travel_time_days=float(tt),
                container_volume=float(cv),
                num_containers_per_day=int(nc)))
    return net


# ============================================================================
# Demand Generator  — day-level
# ============================================================================

def build_demand_fn(
    item_ids, n_steps, seed=42,
    base_lambda_range=(80, 250),
):
    """

    Structure:
      - Yearly cycle   T=365 days  (moderate amplitude, two harmonics)
      - Weekly cycle   T=7 days    (small texture)
      - AR(1) drift                (dominant low-frequency, decade-scale)
      - Per-item spikes            (sustained 1-6 months, clear amplitude)
      - Global macro events        (rare, wide, correlated across items)
      - Poisson sampling
    """
    rng = np.random.default_rng(seed)
    n_items = len(item_ids)

    spy = 365
    spw = 7

    t = np.arange(n_steps, dtype=np.float64)
    yearly_phase = 2 * np.pi * t / spy
    weekly_phase = 2 * np.pi * (t % spw) / spw

    lam = np.zeros((n_steps, n_items), dtype=np.float64)

    # Global macro events
    gs = np.zeros(n_steps)
    n_global = int(rng.integers(5, 12))
    for _ in range(n_global):
        si  = int(rng.integers(0, n_steps))
        dur = int(rng.integers(180, 1100))
        end = min(si + dur, n_steps)
        h   = rng.uniform(0.20, 0.60)
        for k in range(si, end):
            p = (k - si) / max(dur, 1)
            if p < 0.15:
                gs[k] += h * (p / 0.15)
            elif p < 0.75:
                gs[k] += h
            else:
                gs[k] += h * (1.0 - (p - 0.75) / 0.25)

    for j, iid in enumerate(item_ids):
        base = rng.uniform(*base_lambda_range)

        ya1 = rng.uniform(0.12, 0.28)
        ya2 = rng.uniform(0.04, 0.10)
        yo  = rng.uniform(0, 2 * np.pi)
        yr  = (ya1 * np.sin(yearly_phase + yo) +
               ya2 * np.sin(2 * yearly_phase + yo * 0.7))

        wa = rng.uniform(0.04, 0.10)
        wo = rng.uniform(0, 2 * np.pi)
        wy = wa * np.sin(weekly_phase + wo)

        ac     = rng.uniform(0.9990, 0.9996)
        ar_std = rng.uniform(0.008, 0.018)
        dr = np.zeros(n_steps)
        dr[0] = rng.normal(0, 0.10)
        for i in range(1, n_steps):
            dr[i] = ac * dr[i-1] + rng.normal(0, ar_std)
        dr = np.clip(dr, -0.60, 0.60)

        sr = rng.uniform(0.0002, 0.001)
        sm = rng.random(n_steps) < sr
        sp = np.zeros(n_steps)
        for si in np.where(sm)[0]:
            dur = int(rng.integers(30, 180))
            end = min(si + dur, n_steps)
            h   = rng.uniform(0.20, 0.70)
            for k in range(si, end):
                p = (k - si) / max(dur, 1)
                if p < 0.15:
                    sp[k] += h * (p / 0.15)
                elif p < 0.75:
                    sp[k] += h
                else:
                    sp[k] += h * (1.0 - (p - 0.75) / 0.25)

        gsens = rng.uniform(0.4, 1.2)
        fac = 1.0 + yr + wy + dr + sp + gsens * gs
        fac = np.clip(fac, 0.08, None)
        lam[:, j] = base * fac

    def demand_fn(day):
        idx = day % n_steps
        return {iid: int(rng.poisson(lam=max(0.01, lam[idx, j])))
                for j, iid in enumerate(item_ids)}

    return demand_fn, lam


# ============================================================================
# Build example simulation  (200-item variant)
# ============================================================================

def build_example_simulation_from_adjacency(
    seed=123,
    horizon_days=52560,
    pipeline_multiplier=3.0,
    streaming_out_dir=None,
    packing="greedy",
):
    """
    Day-level supply chain: 1 step = 1 day, 52560 days = 144 years.
    Multi-echelon design:
      Sources: magic replenishment (factory)
      Intermediate nodes: pull from upstream via network edges
      Per-tier (s,S) calibrated to flow rate and upstream lead time
    """
    random.seed(2025)
    item_ids = [f"I{i:03d}" for i in range(1, 201)]              # 200 items
    items = {iid: Item(iid, round(random.uniform(1.0, 4.0), 2))
             for iid in item_ids}

    def make_pol(inv_base=4000, inv_var=500,
                 s_base=600,   s_var=100,
                 S_base=6000,  S_var=500,
                 lt_mean=5,    lt_var=1):
        inv, s, S, lt = {}, {}, {}, {}
        for iid in item_ids:
            si = max(0, int(round(s_base + random.uniform(-s_var, s_var))))
            Si = max(si + 1, int(round(
                S_base + random.uniform(-S_var, S_var))))
            ii = int(round(inv_base + random.uniform(-inv_var, inv_var)))
            ii = max(si, min(Si, max(0, ii)))
            li = max(1, int(round(
                lt_mean + random.uniform(-lt_var, lt_var))))
            s[iid], S[iid], inv[iid], lt[iid] = si, Si, ii, li
        return inv, s, S, lt

    nodes_meta = [
        {"id": "NewYork",      "lat": 40.7128, "lon": -74.0060,
         "is_destination": True,
         "inventory": {iid: 600 for iid in item_ids},
         "backlog":   {iid: 0   for iid in item_ids}},

        {"id": "SanFrancisco", "lat": 37.7749, "lon": -122.4194,
         "is_source": True},
        {"id": "StLouis",      "lat": 38.6270, "lon":  -90.1994,
         "is_source": True},
        {"id": "Orlando",      "lat": 28.5383, "lon":  -81.3792,
         "is_source": True},
        {"id": "Nashville",    "lat": 36.1627, "lon":  -86.7816},
        {"id": "Atlanta",      "lat": 33.7490, "lon":  -84.3880},
        {"id": "Chicago",      "lat": 41.8781, "lon":  -87.6298},
        {"id": "Charlotte",    "lat": 35.2271, "lon":  -80.8431},
        {"id": "Columbus",     "lat": 39.9612, "lon":  -82.9988},
        {"id": "Richmond",     "lat": 37.5407, "lon":  -77.4360},
        {"id": "Philadelphia", "lat": 39.9526, "lon":  -75.1652},
        {"id": "Baltimore",    "lat": 39.2904, "lon":  -76.6122},
        {"id": "Memphis",      "lat": 35.1495, "lon":  -90.0490},
    ]

    tier_params = {
        "SanFrancisco": dict(inv_base=4000, inv_var=400,
                             s_base=400,  s_var=60,
                             S_base=4000, S_var=400,
                             lt_mean=3,   lt_var=1),
        "StLouis":      dict(inv_base=4000, inv_var=400,
                             s_base=400,  s_var=60,
                             S_base=4000, S_var=400,
                             lt_mean=3,   lt_var=1),
        "Orlando":      dict(inv_base=4000, inv_var=400,
                             s_base=400,  s_var=60,
                             S_base=4000, S_var=400,
                             lt_mean=3,   lt_var=1),
        "Nashville":    dict(inv_base=8000, inv_var=800,
                             s_base=1000, s_var=150,
                             S_base=8000, S_var=800,
                             lt_mean=3,   lt_var=1),
        "Atlanta":      dict(inv_base=6000, inv_var=600,
                             s_base=500,  s_var=80,
                             S_base=6000, S_var=600,
                             lt_mean=1,   lt_var=0),
        "Chicago":      dict(inv_base=5000, inv_var=500,
                             s_base=1000, s_var=150,
                             S_base=5000, S_var=500,
                             lt_mean=8,   lt_var=1),
        "Charlotte":    dict(inv_base=5000, inv_var=500,
                             s_base=1000, s_var=150,
                             S_base=5000, S_var=500,
                             lt_mean=7,   lt_var=1),
        "Memphis":      dict(inv_base=3000, inv_var=300,
                             s_base=500,  s_var=80,
                             S_base=3000, S_var=300,
                             lt_mean=7,   lt_var=1),
        "Columbus":     dict(inv_base=4000, inv_var=400,
                             s_base=500,  s_var=80,
                             S_base=4000, S_var=400,
                             lt_mean=2,   lt_var=0),
        "Richmond":     dict(inv_base=4000, inv_var=400,
                             s_base=500,  s_var=80,
                             S_base=4000, S_var=400,
                             lt_mean=2,   lt_var=0),
        "Philadelphia": dict(inv_base=3000, inv_var=300,
                             s_base=500,  s_var=80,
                             S_base=3000, S_var=300,
                             lt_mean=1,   lt_var=0),
        "Baltimore":    dict(inv_base=3000, inv_var=300,
                             s_base=500,  s_var=80,
                             S_base=3000, S_var=300,
                             lt_mean=2,   lt_var=0),
    }

    for m in nodes_meta:
        nid = m["id"]
        if m.get("is_destination", False):
            continue
        inv, s, S, lt = make_pol(**tier_params[nid])
        m["inventory"]      = inv
        m["s_levels"]       = s
        m["S_levels"]       = S
        m["lead_time_mean"] = lt

    n = len(nodes_meta)
    adj = [[None]*n for _ in range(n)]
    idx = {m["id"]: i for i, m in enumerate(nodes_meta)}

    def se(u, v, tt, cv, nc):
        adj[idx[u]][idx[v]] = (tt, cv, nc)

    # Travel times in days; upstream capacity generous (not bottleneck)
    se("SanFrancisco", "Nashville",    4, 20000.0, 3)
    se("StLouis",      "Nashville",    2, 20000.0, 3)
    se("Orlando",      "Nashville",    2, 20000.0, 3)
    se("Nashville",    "Atlanta",      1, 60000.0, 3)
    se("Atlanta",      "Chicago",      8, 16000.0, 3)
    se("Atlanta",      "Charlotte",    7, 16000.0, 3)
    se("Atlanta",      "Memphis",      7, 16000.0, 3)
    se("Chicago",      "Columbus",     2, 16000.0, 3)
    se("Charlotte",    "Richmond",     2, 16000.0, 3)
    se("Columbus",     "Philadelphia", 2, 16000.0, 3)
    se("Richmond",     "Philadelphia", 1, 16000.0, 3)
    se("Richmond",     "Baltimore",    3, 12000.0, 3)
    se("Columbus",     "Baltimore",    3, 12000.0, 3)
    se("Memphis",      "Baltimore",    2, 12000.0, 3)
    # Last-mile: placeholder, overwritten dynamically below
    se("Philadelphia", "NewYork",      1,  4000.0, 3)
    se("Baltimore",    "NewYork",      2,  4000.0, 3)

    net = build_network_from_adjacency(nodes_meta, adj)

    print("Building day-level demand signals (200 items)...")
    demand_fn, demand_signals = build_demand_fn(
        item_ids, horizon_days, seed,
        base_lambda_range=(80, 250))
    print(f"  Shape: {demand_signals.shape}  "
          f"({horizon_days} days ≈ {horizon_days/365:.0f} years)")
    print(f"  Lambda: mean={demand_signals.mean():.1f}  "
          f"min={demand_signals.min():.1f}  "
          f"max={demand_signals.max():.1f}")

    actual_mean_lam = float(demand_signals.mean())
    avg_vol = 2.5
    n_items = len(item_ids)
    total_demand_vol = n_items * actual_mean_lam * avg_vol
    target_ratio = 1.20
    packing_eff   = 0.93
    raw_needed    = total_demand_vol * target_ratio / packing_eff
    philadelphia_cv = round(raw_needed * 0.55 / 3 / 100) * 100
    baltimore_cv    = round(raw_needed * 0.45 / 3 / 100) * 100
    net.edges[("Philadelphia", "NewYork")].container_volume = float(philadelphia_cv)
    net.edges[("Baltimore",    "NewYork")].container_volume = float(baltimore_cv)
    for eid in [("Philadelphia", "NewYork"), ("Baltimore", "NewYork")]:
        e = net.edges[eid]
        net.weight_cache[eid] = e.travel_time_days / max(
            e.daily_total_capacity, 1e-9)
    last_mile_cap = (philadelphia_cv + baltimore_cv) * 3
    print(f"  Demand vol: {total_demand_vol:.0f}/day  "
          f"Last-mile cap: {last_mile_cap:.0f}/day  "
          f"Ratio: {last_mile_cap/total_demand_vol:.1%}  "
          f"(Philadelphia={philadelphia_cv:.0f} Baltimore={baltimore_cv:.0f})")

    sim = SupplyChainSimulation(
        network=net,
        items=items,
        destination_id="NewYork",
        demand_fn=demand_fn,
        horizon_days=horizon_days,
        seed=seed,
        pipeline_multiplier=pipeline_multiplier,
        streaming_out_dir=streaming_out_dir,
        packing=packing)

    return sim, net, items, demand_signals


# ============================================================================
# Folium helpers
# ============================================================================

def plot_network_folium(network, tiles="cartodbpositron", zoom_start=5):
    lats = [n.lat for n in network.nodes.values()]
    lons = [n.lon for n in network.nodes.values()]
    c = (sum(lats)/len(lats), sum(lons)/len(lons))
    m = folium.Map(location=c, zoom_start=zoom_start, tiles=tiles)
    for (u, v), e in network.edges.items():
        nu, nv = network.nodes[u], network.nodes[v]
        folium.PolyLine(
            [(nu.lat, nu.lon), (nv.lat, nv.lon)],
            weight=2, opacity=0.7,
            tooltip=f"{u}→{v} t={e.travel_time_days:.0f}d "
                    f"cap={e.daily_total_capacity:.0f}/day").add_to(m)
    for nid, n in network.nodes.items():
        folium.CircleMarker(
            (n.lat, n.lon),
            radius=6 if n.is_destination else 4, fill=True,
            tooltip=f"{nid} dest={n.is_destination}").add_to(m)
    return m


def export_map_with_animation(
    network, sdf, inv_df=None, bl_df=None, it_df=None,
    out_html="supply_chain_map.html", start_date="2025-01-01",
    zoom_start=5,
):
    m = plot_network_folium(network, zoom_start=zoom_start)
    m.save(out_html)
    return out_html


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(
        description="Supply Chain Simulation — day-level, 52560-step, "
                    "200-item variant")
    ap.add_argument("--days",          type=int,   default=52560)
    ap.add_argument("--seed",          type=int,   default=2025)
    ap.add_argument("--out_dir",       type=str,   default="test_output")
    ap.add_argument("--pipeline_mult", type=float, default=0.0,
                    help="Days of EMA demand to keep in pipeline. "
                         "0 = reactive mode (backlog + 3-day buffer).")
    ap.add_argument("--no_streaming",  action="store_true")
    args = ap.parse_args()

    streaming = not args.no_streaming and args.days > 500

    print("=== Supply Chain Simulation (Multi-Echelon, 200 items) ===")
    print(f"  Days: {args.days:,} ({args.days/365:.1f} years)  "
          f"Seed: {args.seed}")
    print(f"  1 step = 1 day | 365 = 1 year | 52560 = 144 years")
    print(f"  Items: 200  Pipeline: {args.pipeline_mult}  "
          f"Streaming: {streaming}")
    print()

    sim, net, items, dsig = build_example_simulation_from_adjacency(
        seed=args.seed,
        horizon_days=args.days,
        pipeline_multiplier=args.pipeline_mult,
        streaming_out_dir=args.out_dir if streaming else None,
        packing="greedy")

    dd, ds, svc, di, db, dt = sim.run()

    os.makedirs(args.out_dir, exist_ok=True)

    # Save demand signals
    print("Saving demand signals...")
    np.save(os.path.join(args.out_dir, "demand_signals.npy"),
            dsig[:args.days])
    with open(os.path.join(args.out_dir, "demand_signals_cols.txt"), "w") as f:
        f.write(",".join(sorted(items.keys())) + "\n")
    print(f"  Saved shape={dsig[:args.days].shape}")

    if not streaming:
        dd.to_csv(os.path.join(args.out_dir, "daily_records.csv"),
                  index=False)
        ds.to_csv(os.path.join(args.out_dir, "shipments.csv"), index=False)
        svc.to_csv(os.path.join(args.out_dir, "service_summary.csv"),
                   index=False)
        if args.days <= 500:
            di.to_csv(os.path.join(args.out_dir, "inventory_history.csv"),
                      index=False)
            db.to_csv(os.path.join(args.out_dir, "backlog_history.csv"),
                      index=False)
            dt.to_csv(os.path.join(args.out_dir, "intransit_history.csv"),
                      index=False)

    fr = svc['fill_rate_stock_only']
    print(f"\n=== Service Summary (200 items) ===")
    print(f"  Fill rate: mean={fr.mean():.3f}  "
          f"median={fr.median():.3f}  "
          f"min={fr.min():.3f}  max={fr.max():.3f}")
    print(f"  Total demand:  {svc['total_demand'].sum():,}")
    print(f"  Total served:  {svc['served_from_stock'].sum():,}")
    print(f"  Total backlog: {svc['new_backlog_added'].sum():,}")
    print(f"\nOutputs → {args.out_dir}/")
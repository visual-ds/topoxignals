"""FastAPI server for the temporal TDA viewer and its bundled datasets."""

from __future__ import annotations

import json
import os
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

import networkx as nx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.11+ provides tomllib.
    import tomli as tomllib  # type: ignore[no-redef]


_REPO_ROOT = Path(__file__).resolve().parent.parent
_WEB_DIR = _REPO_ROOT / "web"


@dataclass
class Dataset:
    """In-memory representation of one selectable viewer dataset."""

    identifier: str
    label: str
    graphs: dict[int, nx.Graph]
    scores: dict[int, Any] = field(default_factory=dict)
    raw_distances: Optional[Mapping[str, Any]] = None
    explanation_distances: Optional[Mapping[str, Any]] = None
    raw_series: Optional[list[float]] = None
    explanation_series: Optional[list[float]] = None
    analysis_by_key: dict[tuple[str, str], Mapping[str, list[float]]] = field(default_factory=dict)
    fixed_positions: dict[int, Mapping[Any, Any]] = field(default_factory=dict)
    score_threshold: float = 0.8
    sample_size: Optional[int] = None
    sample_selector: str = "id"
    random_state: int = 0
    explain_all_edges: bool = False


class DataStore:
    """Load the toy, email, and HSNet datasets bundled under ``data/``."""

    def __init__(self) -> None:
        config = self._load_server_config()
        paths_cfg = self._section(config, "paths")
        score_cfg = self._section(config, "edge_scores")
        viewer_cfg = self._section(config, "viewer")

        root_value = Path(str(paths_cfg.get("dataset_root", "data"))).expanduser()
        self._data_root = root_value if root_value.is_absolute() else _REPO_ROOT / root_value
        self._positions: dict[tuple[str, int], dict[Any, Any]] = {}
        self._datasets = {
            "toy": self._load_toy(),
            "email": self._load_email(paths_cfg, score_cfg),
            "hsnet": self._load_hsnet(),
        }
        self.default_dataset = str(viewer_cfg.get("default_dataset", "hsnet"))
        if self.default_dataset not in self._datasets:
            raise ValueError("viewer.default_dataset must be toy, email, or hsnet.")

    @staticmethod
    def _section(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
        section = config.get(name, {})
        if not isinstance(section, Mapping):
            raise TypeError(f"[{name}] must be a TOML table.")
        return section

    @staticmethod
    def _optional_int(value: Any) -> Optional[int]:
        return None if value is None else int(value)

    @staticmethod
    def _required_path(section: Mapping[str, Any], key: str) -> Path:
        value = section.get(key)
        if not value:
            raise KeyError(f"Missing required configuration value: [{key}].")
        return Path(str(value))

    @staticmethod
    def _load_pickle(path: Path, description: str) -> Any:
        if not path.is_file():
            raise FileNotFoundError(f"Missing {description}: {path}")
        with path.open("rb") as file:
            return pickle.load(file)

    @staticmethod
    def _load_json(path: Path, description: str) -> Mapping[str, Any]:
        if not path.is_file():
            raise FileNotFoundError(f"Missing {description}: {path}")
        with path.open(encoding="utf-8") as file:
            data = json.load(file)
        if not isinstance(data, Mapping):
            raise TypeError(f"{path} must contain a JSON object.")
        return data

    @staticmethod
    def _load_server_config() -> Mapping[str, Any]:
        configured_path = os.environ.get("TEMPORAL_TDA_CONFIG")
        path = Path(configured_path).expanduser().resolve() if configured_path else _REPO_ROOT / "config.toml"
        if not path.is_file():
            raise FileNotFoundError(
                f"Server configuration not found: {path}. "
                "Copy config.toml.example to config.toml and set the dataset paths."
            )
        with path.open("rb") as file:
            return tomllib.load(file)

    def _load_toy(self) -> Dataset:
        data = self._load_pickle(self._data_root / "toy" / "syn_dataset.pkl", "toy dataset")
        if not isinstance(data, Mapping) or not isinstance(data.get("G"), nx.Graph):
            raise TypeError("data/toy/syn_dataset.pkl must contain a NetworkX graph in 'G'.")
        positions = data.get("pos", {})
        return Dataset(
            identifier="toy",
            label="Toy graph",
            graphs={0: data["G"].copy()},
            fixed_positions={0: positions} if isinstance(positions, Mapping) else {},
            raw_series=[0.0],
            explanation_series=[0.0],
            explain_all_edges=True,
        )

    def _load_email(self, paths_cfg: Mapping[str, Any], score_cfg: Mapping[str, Any]) -> Dataset:
        graphs_path = self._data_root / self._required_path(paths_cfg, "dataset_file")
        raw_path = self._data_root / self._required_path(paths_cfg, "output_file")
        scores_path = self._data_root / self._required_path(score_cfg, "scores_file")
        explanation_path = self._data_root / self._required_path(score_cfg, "output_file")
        snapshots = self._load_pickle(graphs_path, "email graph snapshots")
        scores = self._load_pickle(scores_path, "email edge scores")
        if not isinstance(snapshots, Mapping) or not isinstance(scores, Mapping):
            raise TypeError("The email dataset must contain graph_id mappings.")
        return Dataset(
            identifier="email",
            label="Email network",
            graphs={int(graph_id): nx.Graph(adjacency) for graph_id, adjacency in snapshots.items()},
            scores={int(graph_id): value for graph_id, value in scores.items()},
            raw_distances=self._load_json(raw_path, "email raw persistence distances"),
            explanation_distances=self._load_json(explanation_path, "email explanation persistence distances"),
            score_threshold=float(score_cfg.get("score_threshold", 0.8)),
            sample_size=self._optional_int(score_cfg.get("sample_size")),
            sample_selector=str(score_cfg.get("sample_selector", "id")),
            random_state=int(score_cfg.get("random_state", 0)),
        )

    def _load_hsnet(self) -> Dataset:
        latest_dir = self._data_root / "hsnet" / "latest"
        snapshots = self._load_pickle(latest_dir / "monday1_dfs.pkl", "latest HSNet snapshots")
        if not isinstance(snapshots, list):
            raise TypeError("HSNet snapshots must be a list.")

        graphs: dict[int, nx.Graph] = {}
        scores: dict[int, dict[tuple[int, int], float]] = {}
        for entry in snapshots:
            if not isinstance(entry, Mapping):
                raise TypeError("Each HSNet snapshot must be a mapping.")
            snapshot_id = int(entry["snapshot"])
            nodes, edges = entry["nodes"], entry["edges"]
            graph = nx.Graph()
            for _, node in nodes.iterrows():
                graph.add_node(int(node["student_id"]), Ci=int(node.get("label", 0)))
            score_map: dict[tuple[int, int], float] = {}
            for _, edge in edges.iterrows():
                source, target = int(edge["src_student"]), int(edge["dst_student"])
                score = float(edge.get("importance", 0.0))
                key = self._edge_key(source, target)
                graph.add_edge(source, target)
                score_map[key] = max(score_map.get(key, 0.0), score)
            graphs[snapshot_id] = graph
            scores[snapshot_id] = score_map

        analysis_by_key: dict[tuple[str, str], Mapping[str, list[float]]] = {}
        for metric_prefix, metric in (("b", "bottleneck"), ("w", "wasserstein")):
            for dimension in ("0", "1"):
                result = self._load_json(
                    latest_dir / "results" / f"{metric_prefix}{dimension}_timeseries_monday1.json",
                    f"HSNet {metric} H{dimension} time series",
                )
                tc, raw_change, explanation_change = (
                    result.get("dist_G_E"),
                    result.get("dist_G_seq"),
                    result.get("dist_E_seq"),
                )
                graph_jaccard, explanation_jaccard = result.get("jacc_G_seq"), result.get("jacc_E_seq")
                if not all(isinstance(values, list) for values in (tc, raw_change, explanation_change, graph_jaccard, explanation_jaccard)):
                    raise TypeError("HSNet time series must define TC, sequential distances, and Jaccard baselines.")
                analysis_by_key[(metric, dimension)] = {
                    "tc": [float(value) for value in tc],
                    "ts": [float(graph) - float(explanation) for graph, explanation in zip(raw_change, explanation_change)],
                    "jaccard_graph": [float(value) for value in graph_jaccard],
                    "jaccard_explanation": [float(value) for value in explanation_jaccard],
                }

        return Dataset(
            identifier="hsnet",
            label="HSNet contact network (latest)",
            graphs=graphs,
            scores=scores,
            analysis_by_key=analysis_by_key,
            score_threshold=0.5,
        )

    def list_datasets(self) -> dict[str, Any]:
        return {
            "default": self.default_dataset,
            "datasets": [{"id": data.identifier, "label": data.label} for data in self._datasets.values()],
        }

    def _dataset(self, identifier: Optional[str]) -> Dataset:
        selected = identifier or self.default_dataset
        dataset = self._datasets.get(selected)
        if dataset is None:
            raise ValueError("dataset must be toy, email, or hsnet.")
        return dataset

    @staticmethod
    def _edge_key(source: Any, target: Any) -> tuple[int, int]:
        a, b = int(source), int(target)
        return (a, b) if a <= b else (b, a)

    @staticmethod
    def _sampled_nodes(graph: nx.Graph, dataset: Dataset) -> list[Any]:
        nodes = sorted(graph.nodes(), key=lambda node: int(node))
        if dataset.sample_size is None or len(nodes) <= dataset.sample_size:
            return nodes
        if dataset.sample_selector == "id":
            return nodes[: dataset.sample_size]
        if dataset.sample_selector == "degree":
            return [node for node, _ in sorted(graph.degree(), key=lambda item: item[1], reverse=True)[: dataset.sample_size]]
        if dataset.sample_selector == "random":
            import numpy as np

            indices = np.random.default_rng(dataset.random_state).choice(
                len(nodes), size=dataset.sample_size, replace=False
            )
            return [nodes[index] for index in indices]
        raise ValueError("edge_scores.sample_selector must be 'id', 'degree', or 'random'.")

    def _score_map(self, graph_id: int, graph: nx.Graph, dataset: Dataset) -> dict[tuple[int, int], float]:
        if dataset.explain_all_edges:
            return {self._edge_key(source, target): 1.0 for source, target in graph.edges()}
        score_values = dataset.scores.get(graph_id)
        if score_values is None:
            return {}
        sampled = graph.subgraph(self._sampled_nodes(graph, dataset))
        edges = sorted(self._edge_key(source, target) for source, target in sampled.edges())
        if isinstance(score_values, Mapping):
            return {
                self._edge_key(edge[0], edge[1]): float(score)
                for edge, score in score_values.items()
                if isinstance(edge, (tuple, list)) and len(edge) == 2
            }
        if hasattr(score_values, "tolist"):
            score_values = score_values.tolist()
        try:
            return {edge: float(score) for edge, score in zip(edges, score_values)}
        except TypeError as error:
            raise TypeError(f"Unsupported edge-score format for {dataset.identifier} graph {graph_id}.") from error

    @staticmethod
    def _distance_index(payload: Mapping[str, Any], metric: str, dimension: str) -> dict[int, float]:
        metric_key = f"{metric}_by_dim"
        result: dict[int, float] = {}
        for entry in payload.get("results", []):
            if not isinstance(entry, Mapping):
                continue
            pair, values = entry.get("pair"), entry.get(metric_key)
            if isinstance(pair, list) and len(pair) == 2 and isinstance(values, Mapping) and str(dimension) in values:
                result[int(pair[0])] = float(values[str(dimension)])
        return result

    @staticmethod
    def _timeline(values: list[float]) -> list[dict[str, float]]:
        return [{"timestep": time, "value": value} for time, value in enumerate(values)]

    @staticmethod
    def _transition_timeline(values: list[float]) -> list[dict[str, float]]:
        """Place a t→t+1 quantity halfway between its two snapshots."""
        return [
            {"timestep": time + 0.5, "start_time": time, "end_time": time + 1, "value": value}
            for time, value in enumerate(values)
        ]

    def _jaccard_baseline(self, dataset: Dataset) -> tuple[list[float], list[float]]:
        """Sequential Jaccard similarity for raw and explanation edge sets."""
        raw_values: list[float] = []
        explanation_values: list[float] = []
        graph_ids = sorted(dataset.graphs)
        for current_id, next_id in zip(graph_ids, graph_ids[1:]):
            if next_id != current_id + 1:
                continue
            current_graph, next_graph = dataset.graphs[current_id], dataset.graphs[next_id]
            current_edges = {self._edge_key(source, target) for source, target in current_graph.edges()}
            next_edges = {self._edge_key(source, target) for source, target in next_graph.edges()}
            current_scores = self._score_map(current_id, current_graph, dataset)
            next_scores = self._score_map(next_id, next_graph, dataset)
            current_explanation = {edge for edge, score in current_scores.items() if score >= dataset.score_threshold}
            next_explanation = {edge for edge, score in next_scores.items() if score >= dataset.score_threshold}

            def similarity(left: set[tuple[int, int]], right: set[tuple[int, int]]) -> float:
                union = left | right
                return len(left & right) / len(union) if union else 1.0

            raw_values.append(similarity(current_edges, next_edges))
            explanation_values.append(similarity(current_explanation, next_explanation))
        return raw_values, explanation_values

    def get_timelines(
        self, dimension: str, metric: str, identifier: Optional[str], view: str = "topology"
    ) -> dict[str, Any]:
        if dimension not in {"0", "1"}:
            raise ValueError("dimension must be 0 or 1.")
        if metric not in {"bottleneck", "wasserstein"}:
            raise ValueError("metric must be bottleneck or wasserstein.")
        if view not in {"topology", "jaccard"}:
            raise ValueError("view must be topology or jaccard.")
        dataset = self._dataset(identifier)
        initial_timestep = min(dataset.graphs) if dataset.graphs else None
        analysis = dataset.analysis_by_key.get((metric, dimension))
        if analysis is not None:
            if view == "topology":
                return {
                    "raw": self._timeline(analysis["tc"]),
                    "exp": self._transition_timeline(analysis["ts"]),
                    "labels": {"raw": "Topological Consistency — TC(t)", "exp": "Topological Stability — TS(t)"},
                    "initial_timestep": initial_timestep,
                    "domain": [0, max(len(analysis["tc"]) - 1, len(analysis["ts"]))],
                    "zero_reference": {"raw": False, "exp": True},
                }
            jaccard_gap = [
                graph - explanation
                for graph, explanation in zip(analysis["jaccard_graph"], analysis["jaccard_explanation"])
            ]
            return {
                "raw": self._transition_timeline(analysis["jaccard_graph"]),
                "exp": self._transition_timeline(jaccard_gap),
                "labels": {"raw": "Graph Jaccard similarity", "exp": "Jaccard stability gap — J_G − J_E"},
                "initial_timestep": initial_timestep,
                "domain": [0, max(len(analysis["jaccard_graph"]), len(analysis["jaccard_explanation"]))],
                "zero_reference": {"raw": False, "exp": True},
            }
        if dataset.raw_distances is not None and dataset.explanation_distances is not None:
            raw = self._distance_index(dataset.raw_distances, metric, dimension)
            explanation = self._distance_index(dataset.explanation_distances, metric, dimension)
            timesteps = sorted(set(raw) | set(explanation))
            if view == "topology":
                ts = {time: raw.get(time, 0.0) - explanation.get(time, 0.0) for time in timesteps}
                return {
                    "raw": [{"timestep": time, "value": -1.0} for time in timesteps],
                    "exp": [
                        {"timestep": time + 0.5, "start_time": time, "end_time": time + 1, "value": ts[time]}
                        for time in timesteps
                    ],
                    "labels": {"raw": "TC(t) (not precomputed for Email)", "exp": "TS(t) = ΔG − ΔE"},
                    "initial_timestep": initial_timestep,
                    "domain": [min(timesteps), max(timesteps) + 1] if timesteps else [0, 1],
                    "zero_reference": {"raw": False, "exp": True},
                }
            graph_jaccard, explanation_jaccard = self._jaccard_baseline(dataset)
            jaccard_gap = [graph - explanation for graph, explanation in zip(graph_jaccard, explanation_jaccard)]
            return {
                "raw": self._transition_timeline(graph_jaccard),
                "exp": self._transition_timeline(jaccard_gap),
                "labels": {"raw": "Graph Jaccard similarity", "exp": "Jaccard stability gap — J_G − J_E"},
                "initial_timestep": initial_timestep,
                "domain": [0, max(len(graph_jaccard), len(explanation_jaccard))],
                "zero_reference": {"raw": False, "exp": True},
            }
        label = "Topological Consistency — TC(t)" if view == "topology" else "Graph Jaccard similarity"
        comparison_label = "Topological Stability — TS(t)" if view == "topology" else "Explanation Jaccard similarity"
        return {
            "raw": [],
            "exp": [],
            "labels": {"raw": f"{label} (not defined for static Toy graph)", "exp": f"{comparison_label} (not defined for static Toy graph)"},
            "initial_timestep": initial_timestep,
            "domain": [0, 0],
            "zero_reference": {"raw": False, "exp": False},
        }

    def _format_graph(self, graph_id: int, dataset: Dataset) -> Optional[dict[str, Any]]:
        graph = dataset.graphs.get(graph_id)
        if graph is None:
            return None
        cache_key = (dataset.identifier, graph_id)
        positions = dataset.fixed_positions.get(graph_id)
        if positions is None:
            if cache_key not in self._positions:
                self._positions[cache_key] = nx.spring_layout(graph, seed=42, method="energy")
            positions = self._positions[cache_key]
        scores = self._score_map(graph_id, graph, dataset)
        nodes = [
            {
                "id": int(node),
                "x": float(positions[node][0]),
                "y": float(positions[node][1]),
                "class": graph.nodes[node].get("Ci", 0),
            }
            for node in graph.nodes()
        ]
        edges = [
            {
                "source": int(source),
                "target": int(target),
                "importance": score if (score := scores.get(self._edge_key(source, target), 0.0)) >= dataset.score_threshold else 0.0,
            }
            for source, target in graph.edges()
        ]
        return {"num_nodes": graph.number_of_nodes(), "nodes": nodes, "edges": edges}

    def get_graphs(self, timestep: int, identifier: Optional[str]) -> dict[str, Optional[dict[str, Any]]]:
        dataset = self._dataset(identifier)
        return {"curr": self._format_graph(timestep, dataset), "next": self._format_graph(timestep + 1, dataset)}


app = FastAPI(title="Temporal TDA Viewer API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=False, allow_methods=["*"], allow_headers=["*"])
if _WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_WEB_DIR)), name="static")
store = DataStore()


@app.get("/")
def index() -> Any:
    html = _WEB_DIR / "index.html"
    return FileResponse(str(html)) if html.exists() else {"message": "API is running. See /docs for endpoints."}


@app.get("/api/datasets")
def api_datasets() -> dict[str, Any]:
    return store.list_datasets()


@app.get("/api/timelines/{dimension}/{metric}")
def api_timelines(dimension: str, metric: str, dataset: Optional[str] = None, view: str = "topology") -> dict[str, Any]:
    try:
        return store.get_timelines(dimension, metric, dataset, view)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.get("/api/graphs/{timestep}")
def api_graphs(timestep: int, dataset: Optional[str] = None) -> dict[str, Optional[dict[str, Any]]]:
    try:
        return store.get_graphs(timestep, dataset)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


def main() -> None:
    """Run the local viewer."""
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)

from __future__ import annotations

import math
import sys
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch_geometric.data import Data

import gnns
from graph_editor.metadata import (
    SUPPORTED_DATASETS,
    SUPPORTED_SPLITS,
    atom_type_options_for_dataset,
    default_atom_type_for_dataset,
    feature_labels_for_dataset,
    infer_feature_mode,
    infer_node_label,
    is_supported_dataset,
    is_supported_split,
    node_label_mode_for_dataset,
    normalize_dataset_name,
    resolve_model_path,
)
from utils.dataset import get_datasets


class GraphEditorError(Exception):
    def __init__(self, status_code: int, message: str, details: Optional[dict] = None):
        super().__init__(message)
        self.status_code = int(status_code)
        self.message = message
        self.details = details or {}


class GraphEditorRuntime:
    def __init__(
        self,
        project_root: Path,
        data_root: Path,
        param_root: Path,
        device: str = "cpu",
        default_dataset: str = "mutag",
    ):
        self.project_root = Path(project_root).expanduser().resolve()
        self.data_root = Path(data_root).expanduser().resolve()
        self.param_root = Path(param_root).expanduser().resolve()
        self.device = self._resolve_device(device)
        self.requested_device = str(device)
        self.default_dataset = normalize_dataset_name(default_dataset)
        self._dataset_cache: Dict[str, Dict[str, object]] = {}
        self._model_cache: Dict[Tuple[str, str, str], torch.nn.Module] = {}
        self._inject_pickled_gnn_symbols()

    def _resolve_device(self, requested: str) -> torch.device:
        requested = str(requested).strip() or "cpu"
        if requested.startswith("cuda") and not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device(requested)

    def _inject_pickled_gnn_symbols(self) -> None:
        main_module = sys.modules.get("__main__")
        if main_module is None:
            return

        for name in getattr(gnns, "__all__", []):
            if hasattr(gnns, name) and not hasattr(main_module, name):
                setattr(main_module, name, getattr(gnns, name))

    def list_datasets(self) -> dict:
        datasets = []
        for dataset_name in SUPPORTED_DATASETS:
            dataset_info = {
                "name": dataset_name,
                "display_name": dataset_name,
                "splits": {split: 0 for split in SUPPORTED_SPLITS},
                "data_available": False,
                "data_error": None,
                "model_available": False,
                "model_path": None,
                "default_feature_mode": "vector",
            }

            try:
                split_map = self._load_dataset(dataset_name)
                dataset_info["splits"] = {
                    split: len(split_map[split]) for split in SUPPORTED_SPLITS
                }
                dataset_info["data_available"] = True
                sample = self._first_available_graph(split_map)
                if sample is not None:
                    feature_mode, _ = self._infer_feature_spec(dataset_name, sample)
                    dataset_info["default_feature_mode"] = feature_mode
            except Exception as exc:
                dataset_info["data_error"] = str(exc)

            model_path = resolve_model_path(dataset_name, self.param_root)
            dataset_info["model_available"] = model_path is not None
            dataset_info["model_path"] = str(model_path) if model_path else None
            datasets.append(dataset_info)

        default_dataset = self.default_dataset
        if default_dataset not in SUPPORTED_DATASETS:
            default_dataset = SUPPORTED_DATASETS[0]

        return {
            "datasets": datasets,
            "supported_datasets": list(SUPPORTED_DATASETS),
            "default_dataset": default_dataset,
            "device": str(self.device),
            "requested_device": self.requested_device,
        }

    def get_graph(self, dataset: str, split: str, index: int) -> dict:
        dataset = self._validate_dataset(dataset)
        split = self._validate_split(split)
        graph = self._get_graph_object(dataset, split, index)
        feature_mode, feature_labels = self._infer_feature_spec(dataset, graph)
        edge_pairs = self._canonical_edges_from_edge_index(graph.edge_index)
        nodes = self._serialize_nodes(graph, dataset, feature_mode, feature_labels, edge_pairs)
        edges = self._serialize_edges(edge_pairs)
        ground_truth_motif = self._extract_ground_truth_motif(dataset, graph)

        original_prediction = None
        model_error = None
        model_path = resolve_model_path(dataset, self.param_root)
        if model_path is None:
            model_error = f"No compatible checkpoint found under '{self.param_root}'."
        else:
            try:
                original_prediction = self._predict_data(dataset, graph)
            except GraphEditorError as exc:
                model_error = exc.message

        y_value = getattr(graph, "y", None)
        if y_value is not None:
            y_value = int(torch.as_tensor(y_value).view(-1)[0].item())

        payload = {
            "dataset": dataset,
            "split": split,
            "source_index": int(index),
            "graph_meta": {
                "dataset": dataset,
                "split": split,
                "split_size": len(self._load_dataset(dataset)[split]),
                "name": getattr(graph, "name", f"{dataset}_{index}"),
                "idx": int(getattr(graph, "idx", index)),
                "y": y_value,
                "num_nodes": int(graph.num_nodes),
                "num_edges": len(edges),
                "x_dim": int(self._ensure_2d_features(graph.x).size(1)),
                "num_classes": len(original_prediction["probabilities"]) if original_prediction else 2,
                "feature_mode": feature_mode,
                "feature_labels": feature_labels,
                "node_label_mode": node_label_mode_for_dataset(
                    dataset,
                    feature_mode,
                    int(self._ensure_2d_features(graph.x).size(1)),
                ),
                "atom_type_options": atom_type_options_for_dataset(dataset),
                "default_atom_type": default_atom_type_for_dataset(dataset),
                "prediction_available": original_prediction is not None,
                "model_available": model_path is not None,
                "model_path": str(model_path) if model_path else None,
                "model_error": model_error,
                "ground_truth_motif_available": bool(ground_truth_motif.get("available")),
            },
            "nodes": nodes,
            "edges": edges,
            "original_prediction": original_prediction,
            "ground_truth_motif": ground_truth_motif,
        }
        return payload

    def predict_from_payload(self, payload: dict) -> dict:
        dataset = self._validate_dataset(payload.get("dataset"))
        split = self._validate_split(payload.get("split"))
        source_index = self._parse_int(payload.get("source_index"), "source_index")
        source_graph = self._get_graph_object(dataset, split, source_index)
        feature_mode, feature_labels = self._infer_feature_spec(dataset, source_graph)

        normalized_nodes, normalized_edges, edited_graph = self._payload_to_data(
            dataset=dataset,
            feature_mode=feature_mode,
            feature_labels=feature_labels,
            x_dim=int(self._ensure_2d_features(source_graph.x).size(1)),
            nodes=payload.get("nodes"),
            edges=payload.get("edges"),
        )

        current_prediction = self._predict_data(dataset, edited_graph)
        original_prediction = self._predict_data(dataset, source_graph)
        delta = [
            round(curr - orig, 6)
            for curr, orig in zip(
                current_prediction["probabilities"], original_prediction["probabilities"]
            )
        ]

        return {
            "current_prediction": current_prediction,
            "delta_vs_original": delta,
            "graph_stats": {
                "dataset": dataset,
                "split": split,
                "source_index": int(source_index),
                "num_nodes": len(normalized_nodes),
                "num_edges": len(normalized_edges),
                "x_dim": int(edited_graph.x.size(1)),
                "feature_mode": feature_mode,
            },
            "normalized_graph": {
                "nodes": normalized_nodes,
                "edges": normalized_edges,
            },
        }

    def _validate_dataset(self, dataset: str) -> str:
        dataset = normalize_dataset_name(dataset)
        if not is_supported_dataset(dataset):
            raise GraphEditorError(
                404,
                f"Unsupported dataset '{dataset}'.",
                {"supported_datasets": list(SUPPORTED_DATASETS)},
            )
        return dataset

    def _validate_split(self, split: str) -> str:
        split = str(split).strip().lower()
        if not is_supported_split(split):
            raise GraphEditorError(
                404,
                f"Unsupported split '{split}'.",
                {"supported_splits": list(SUPPORTED_SPLITS)},
            )
        return split

    def _parse_int(self, value, field_name: str) -> int:
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise GraphEditorError(422, f"Field '{field_name}' must be an integer.") from exc

    def _load_dataset(self, dataset: str) -> Dict[str, object]:
        dataset = normalize_dataset_name(dataset)
        if dataset in self._dataset_cache:
            return self._dataset_cache[dataset]

        try:
            train_dataset, evaluation_dataset, test_dataset = get_datasets(
                name=dataset,
                root=str(self.data_root),
            )
        except Exception as exc:
            raise GraphEditorError(500, f"Failed to load dataset '{dataset}': {exc}") from exc

        split_map = {
            "training": train_dataset,
            "evaluation": evaluation_dataset,
            "testing": test_dataset,
        }
        self._dataset_cache[dataset] = split_map
        return split_map

    def _first_available_graph(self, split_map: Dict[str, object]):
        for split in SUPPORTED_SPLITS:
            if len(split_map[split]) > 0:
                return split_map[split][0]
        return None

    def _get_graph_object(self, dataset: str, split: str, index: int):
        split_map = self._load_dataset(dataset)
        target_split = split_map[split]
        if index < 0 or index >= len(target_split):
            raise GraphEditorError(
                404,
                f"Graph index {index} is out of range for {dataset}/{split}.",
                {"split_size": len(target_split)},
            )
        return target_split[index]

    def _infer_feature_spec(self, dataset: str, graph) -> Tuple[str, List[str]]:
        features = self._ensure_2d_features(graph.x).detach().cpu().tolist()
        feature_mode = infer_feature_mode(features)
        labels = feature_labels_for_dataset(dataset, len(features[0]) if features else 0, feature_mode)
        return feature_mode, labels

    def _ensure_2d_features(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.as_tensor(x)
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        return x.float()

    def _canonical_edges_from_edge_index(self, edge_index: torch.Tensor) -> List[Tuple[int, int]]:
        if edge_index is None or edge_index.numel() == 0:
            return []

        seen = set()
        edges = []
        for src, dst in edge_index.t().tolist():
            src = int(src)
            dst = int(dst)
            if src == dst:
                continue
            edge = (src, dst) if src < dst else (dst, src)
            if edge in seen:
                continue
            seen.add(edge)
            edges.append(edge)

        edges.sort()
        return edges

    def _serialize_nodes(
        self,
        graph,
        dataset: str,
        feature_mode: str,
        feature_labels: Sequence[str],
        edge_pairs: Sequence[Tuple[int, int]],
    ) -> List[dict]:
        x = self._ensure_2d_features(graph.x).detach().cpu()
        raw_features = x.tolist()
        labels = [
            infer_node_label(
                feature,
                feature_mode,
                feature_labels,
                node_id,
                dataset=dataset,
            )
            for node_id, feature in enumerate(raw_features)
        ]
        positions = self._layout_positions(labels, edge_pairs)
        nodes = []
        for node_id in range(x.size(0)):
            feature = [float(value) for value in raw_features[node_id]]
            nodes.append(
                {
                    "id": int(node_id),
                    "label": labels[node_id],
                    "feature": feature,
                    "pos": positions[node_id],
                }
            )
        return nodes

    def _serialize_edges(self, edge_pairs: Sequence[Tuple[int, int]]) -> List[dict]:
        edges = []
        for src, dst in edge_pairs:
            edges.append({"source": int(src), "target": int(dst)})
        return edges

    def _extract_ground_truth_motif(self, dataset: str, graph) -> dict:
        if dataset not in {"alkane_carbonyl", "fluoride_carbonyl"}:
            return {
                "available": False,
                "positive_class_only": True,
                "is_positive_sample": False,
                "description": None,
                "reason": "Ground-truth positive motif overlay is only configured for alkane_carbonyl and fluoride_carbonyl.",
                "node_ids": [],
                "edges": [],
            }

        y_value = getattr(graph, "y", None)
        is_positive_sample = False
        if y_value is not None:
            is_positive_sample = int(torch.as_tensor(y_value).view(-1)[0].item()) == 1

        descriptions = {
            "alkane_carbonyl": "Dataset GT motif: union of valid explanations containing an unbranched alkane together with a carbonyl (C=O).",
            "fluoride_carbonyl": "Dataset GT motif: union of valid explanations containing fluoride atom(s) together with a carbonyl (C=O).",
        }

        node_mask = getattr(graph, "node_mask", None)
        edge_mask = getattr(graph, "edge_mask", None)
        if node_mask is None or edge_mask is None:
            return {
                "available": False,
                "positive_class_only": True,
                "is_positive_sample": is_positive_sample,
                "description": descriptions.get(dataset),
                "reason": "This graph does not expose node_mask/edge_mask ground-truth annotations.",
                "node_ids": [],
                "edges": [],
            }

        node_mask = torch.as_tensor(node_mask).view(-1)
        edge_mask = torch.as_tensor(edge_mask).view(-1)
        edge_index = torch.as_tensor(graph.edge_index, dtype=torch.long)

        motif_node_ids = [
            int(node_id)
            for node_id, mask_value in enumerate(node_mask.tolist())
            if float(mask_value) > 0.0
        ]
        motif_edges = self._canonical_edges_from_masked_edge_index(edge_index, edge_mask)

        available = is_positive_sample and (len(motif_node_ids) > 0 or len(motif_edges) > 0)
        reason = None
        if not is_positive_sample:
            reason = "This sample is labeled 0, so there is no positive-class motif to overlay."
        elif not available:
            reason = "Ground-truth masks were present, but no positive motif nodes or edges were marked."

        return {
            "available": available,
            "positive_class_only": True,
            "is_positive_sample": is_positive_sample,
            "description": descriptions.get(dataset),
            "reason": reason,
            "node_ids": motif_node_ids,
            "edges": [{"source": int(src), "target": int(dst)} for src, dst in motif_edges],
        }

    def _canonical_edges_from_masked_edge_index(
        self,
        edge_index: torch.Tensor,
        edge_mask: torch.Tensor,
    ) -> List[Tuple[int, int]]:
        if edge_index is None or edge_index.numel() == 0 or edge_mask is None or edge_mask.numel() == 0:
            return []

        seen = set()
        edges = []
        num_edges = min(edge_index.size(1), edge_mask.numel())
        for edge_idx in range(num_edges):
            if float(edge_mask[edge_idx].item()) <= 0.0:
                continue
            src = int(edge_index[0, edge_idx].item())
            dst = int(edge_index[1, edge_idx].item())
            if src == dst:
                continue
            edge = (src, dst) if src < dst else (dst, src)
            if edge in seen:
                continue
            seen.add(edge)
            edges.append(edge)

        edges.sort()
        return edges

    def _layout_positions(
        self,
        labels: Sequence[str],
        edge_pairs: Sequence[Tuple[int, int]],
    ) -> List[dict]:
        num_nodes = len(labels)
        if num_nodes <= 0:
            return []
        if num_nodes == 1:
            return [{"x": 450.0, "y": 300.0}]

        molecule_positions = self._molecule_like_positions(labels, edge_pairs)
        if molecule_positions is not None:
            return molecule_positions

        return self._force_layout_positions(num_nodes, edge_pairs)

    def _molecule_like_positions(
        self,
        labels: Sequence[str],
        edge_pairs: Sequence[Tuple[int, int]],
    ) -> Optional[List[dict]]:
        hydrogen_nodes = [idx for idx, label in enumerate(labels) if label == "H"]
        heavy_nodes = [idx for idx, label in enumerate(labels) if label != "H"]
        if not hydrogen_nodes or not heavy_nodes:
            return None

        heavy_edges = [
            (src, dst)
            for src, dst in edge_pairs
            if src in heavy_nodes and dst in heavy_nodes
        ]
        if not heavy_edges and len(heavy_nodes) > 1:
            return None

        heavy_position_map = self._force_layout_map(
            nodes=heavy_nodes,
            edge_pairs=heavy_edges,
            margin=92.0,
        )
        if not heavy_position_map:
            return None

        neighbors = {idx: [] for idx in range(len(labels))}
        for src, dst in edge_pairs:
            neighbors[src].append(dst)
            neighbors[dst].append(src)

        positions = {idx: dict(pos) for idx, pos in heavy_position_map.items()}
        canvas_center = {"x": 450.0, "y": 310.0}

        for heavy_node in heavy_nodes:
            attached_hydrogens = [
                node for node in neighbors[heavy_node] if labels[node] == "H"
            ]
            if not attached_hydrogens:
                continue

            heavy_pos = positions[heavy_node]
            heavy_neighbors = [
                node
                for node in neighbors[heavy_node]
                if labels[node] != "H" and node in positions
            ]

            if heavy_neighbors:
                avg_x = sum(positions[node]["x"] for node in heavy_neighbors) / len(heavy_neighbors)
                avg_y = sum(positions[node]["y"] for node in heavy_neighbors) / len(heavy_neighbors)
                base_angle = math.atan2(heavy_pos["y"] - avg_y, heavy_pos["x"] - avg_x)
            else:
                base_angle = math.atan2(
                    heavy_pos["y"] - canvas_center["y"],
                    heavy_pos["x"] - canvas_center["x"],
                )

            spread = 0.74 if len(attached_hydrogens) > 1 else 0.0
            start_angle = base_angle - spread * (len(attached_hydrogens) - 1) / 2.0
            radius = 58.0
            for offset, hydrogen_node in enumerate(attached_hydrogens):
                angle = start_angle + offset * spread
                positions[hydrogen_node] = {
                    "x": round(max(24.0, min(876.0, heavy_pos["x"] + radius * math.cos(angle))), 3),
                    "y": round(max(24.0, min(596.0, heavy_pos["y"] + radius * math.sin(angle))), 3),
                }

        missing_nodes = [idx for idx in range(len(labels)) if idx not in positions]
        if missing_nodes:
            fallback = self._force_layout_map(
                nodes=missing_nodes,
                edge_pairs=[
                    (src, dst)
                    for src, dst in edge_pairs
                    if src in missing_nodes and dst in missing_nodes
                ],
                margin=120.0,
            )
            positions.update(fallback)

        if len(positions) != len(labels):
            return None

        return [positions[idx] for idx in range(len(labels))]

    def _force_layout_positions(
        self,
        num_nodes: int,
        edge_pairs: Sequence[Tuple[int, int]],
    ) -> List[dict]:
        position_map = self._force_layout_map(
            nodes=list(range(num_nodes)),
            edge_pairs=edge_pairs,
            margin=58.0,
        )
        if len(position_map) == num_nodes:
            return [position_map[idx] for idx in range(num_nodes)]
        return self._circle_positions(num_nodes)

    def _force_layout_map(
        self,
        nodes: Sequence[int],
        edge_pairs: Sequence[Tuple[int, int]],
        margin: float,
    ) -> Dict[int, dict]:
        node_list = list(nodes)
        if not node_list:
            return {}
        if len(node_list) == 1:
            return {node_list[0]: {"x": 450.0, "y": 310.0}}

        node_set = set(node_list)
        filtered_edges = [
            (src, dst) for src, dst in edge_pairs if src in node_set and dst in node_set
        ]

        try:
            import networkx as nx

            graph_nx = nx.Graph()
            graph_nx.add_nodes_from(node_list)
            graph_nx.add_edges_from(filtered_edges)

            # Kamada-Kawai keeps small molecular scaffolds readable; spring_layout
            # handles sparse/disconnected generic graphs more naturally.
            if filtered_edges and len(node_list) <= 80:
                raw_pos = nx.kamada_kawai_layout(graph_nx, scale=1.0)
            else:
                raw_pos = nx.spring_layout(
                    graph_nx,
                    seed=17,
                    iterations=180,
                    k=1.35 / math.sqrt(max(len(node_list), 1)),
                    scale=1.0,
                )
            raw = {
                int(node): (float(coords[0]), float(coords[1]))
                for node, coords in raw_pos.items()
            }
        except Exception:
            raw = self._deterministic_force_layout(node_list, filtered_edges)

        return self._scale_layout(raw, margin=margin)

    def _deterministic_force_layout(
        self,
        nodes: Sequence[int],
        edge_pairs: Sequence[Tuple[int, int]],
    ) -> Dict[int, Tuple[float, float]]:
        node_list = list(nodes)
        index_map = {node: idx for idx, node in enumerate(node_list)}
        n_nodes = len(node_list)
        positions = {}
        for idx, node in enumerate(node_list):
            angle = 2.399963229728653 * idx
            radius = 0.28 + 0.72 * math.sqrt((idx + 1) / n_nodes)
            positions[node] = [radius * math.cos(angle), radius * math.sin(angle)]

        edges = [(src, dst) for src, dst in edge_pairs if src in index_map and dst in index_map]
        ideal = 0.55
        area = 4.0
        repulsion = math.sqrt(area / max(n_nodes, 1))

        for _ in range(180):
            disp = {node: [0.0, 0.0] for node in node_list}

            for left_idx, left in enumerate(node_list):
                lx, ly = positions[left]
                for right in node_list[left_idx + 1:]:
                    rx, ry = positions[right]
                    dx = lx - rx
                    dy = ly - ry
                    dist = max(math.sqrt(dx * dx + dy * dy), 1e-4)
                    force = (repulsion * repulsion) / dist
                    fx = dx / dist * force
                    fy = dy / dist * force
                    disp[left][0] += fx
                    disp[left][1] += fy
                    disp[right][0] -= fx
                    disp[right][1] -= fy

            for src, dst in edges:
                sx, sy = positions[src]
                dx_pos, dy_pos = positions[dst]
                dx = sx - dx_pos
                dy = sy - dy_pos
                dist = max(math.sqrt(dx * dx + dy * dy), 1e-4)
                force = (dist * dist) / ideal
                fx = dx / dist * force
                fy = dy / dist * force
                disp[src][0] -= fx
                disp[src][1] -= fy
                disp[dst][0] += fx
                disp[dst][1] += fy

            step = 0.025
            for node in node_list:
                positions[node][0] += disp[node][0] * step
                positions[node][1] += disp[node][1] * step

        return {node: (xy[0], xy[1]) for node, xy in positions.items()}

    def _scale_layout(self, raw: Dict[int, Tuple[float, float]], margin: float) -> Dict[int, dict]:
        if not raw:
            return {}

        min_x = min(pos[0] for pos in raw.values())
        max_x = max(pos[0] for pos in raw.values())
        min_y = min(pos[1] for pos in raw.values())
        max_y = max(pos[1] for pos in raw.values())
        width = max(max_x - min_x, 1e-6)
        height = max(max_y - min_y, 1e-6)
        usable_w = 900.0 - 2.0 * margin
        usable_h = 620.0 - 2.0 * margin
        scale = min(usable_w / width, usable_h / height)

        layout_w = width * scale
        layout_h = height * scale
        offset_x = (900.0 - layout_w) / 2.0
        offset_y = (620.0 - layout_h) / 2.0

        return {
            node: {
                "x": round(offset_x + (pos[0] - min_x) * scale, 3),
                "y": round(offset_y + (pos[1] - min_y) * scale, 3),
            }
            for node, pos in raw.items()
        }

    def _circle_positions(self, num_nodes: int) -> List[dict]:
        radius = 220.0 if num_nodes <= 12 else min(280.0, 120.0 + num_nodes * 6.0)
        center_x = 450.0
        center_y = 310.0
        positions = []
        for idx in range(num_nodes):
            angle = (2.0 * math.pi * idx / num_nodes) - (math.pi / 2.0)
            positions.append(
                {
                    "x": round(center_x + radius * math.cos(angle), 3),
                    "y": round(center_y + radius * math.sin(angle), 3),
                }
            )
        return positions
        center_x = 450.0
        center_y = 300.0
        positions = []
        for idx in range(num_nodes):
            angle = (2.0 * math.pi * idx / num_nodes) - (math.pi / 2.0)
            positions.append(
                {
                    "x": round(center_x + radius * math.cos(angle), 3),
                    "y": round(center_y + radius * math.sin(angle), 3),
                }
            )
        return positions

    def _get_model(self, dataset: str) -> Tuple[torch.nn.Module, Path]:
        model_path = resolve_model_path(dataset, self.param_root)
        if model_path is None:
            raise GraphEditorError(
                409,
                f"No compatible checkpoint found for dataset '{dataset}' under '{self.param_root}'.",
            )

        cache_key = (dataset, str(model_path), str(self.device))
        if cache_key in self._model_cache:
            return self._model_cache[cache_key], model_path

        try:
            model = torch.load(str(model_path), map_location=self.device)
            model = model.to(self.device)
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)
        except Exception as exc:
            raise GraphEditorError(
                500,
                f"Failed to load checkpoint '{model_path.name}' for dataset '{dataset}': {exc}",
            ) from exc

        self._model_cache[cache_key] = model
        return model, model_path

    def _predict_data(self, dataset: str, graph) -> dict:
        model, model_path = self._get_model(dataset)
        x = self._ensure_2d_features(graph.x).to(self.device)
        edge_index = torch.as_tensor(graph.edge_index, dtype=torch.long, device=self.device)
        batch = torch.zeros(x.size(0), dtype=torch.long, device=self.device)

        with torch.no_grad():
            output = model(x, edge_index, batch)

        if isinstance(output, (tuple, list)):
            logits = output[-1]
        else:
            logits = output

        logits = torch.as_tensor(logits)
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)

        probabilities = torch.softmax(logits, dim=1)[0].detach().cpu().tolist()
        predicted_class = int(torch.argmax(logits, dim=1).item())

        return {
            "probabilities": [round(float(prob), 6) for prob in probabilities],
            "predicted_class": predicted_class,
            "logits": [round(float(value), 6) for value in logits[0].detach().cpu().tolist()],
            "model_path": str(model_path),
            "device": str(self.device),
        }

    def _payload_to_data(
        self,
        dataset: str,
        feature_mode: str,
        feature_labels: Sequence[str],
        x_dim: int,
        nodes,
        edges,
    ) -> Tuple[List[dict], List[dict], Data]:
        if not isinstance(nodes, list) or not nodes:
            raise GraphEditorError(422, "Payload field 'nodes' must be a non-empty list.")
        if not isinstance(edges, list):
            raise GraphEditorError(422, "Payload field 'edges' must be a list.")

        original_ids = []
        features = []
        normalized_nodes = []
        label_mode = node_label_mode_for_dataset(dataset, feature_mode, x_dim)

        for idx, node in enumerate(nodes):
            if not isinstance(node, dict):
                raise GraphEditorError(422, f"Node #{idx} must be an object.")
            if "id" not in node:
                raise GraphEditorError(422, f"Node #{idx} is missing field 'id'.")

            node_id = node["id"]
            if node_id in original_ids:
                raise GraphEditorError(422, f"Duplicate node id '{node_id}' is not allowed.")

            feature = node.get("feature")
            if not isinstance(feature, list):
                raise GraphEditorError(422, f"Node '{node_id}' must provide 'feature' as a list.")
            vector = self._validate_feature_vector(feature, x_dim, feature_mode, node_id)
            original_ids.append(node_id)
            features.append(vector)

        id_to_new = {node_id: new_id for new_id, node_id in enumerate(original_ids)}

        for old_id, vector, node in zip(original_ids, features, nodes):
            pos = self._validate_position(node.get("pos"))
            label = node.get("label")
            if label_mode != "node_id" or not isinstance(label, str) or not label.strip():
                label = infer_node_label(
                    vector,
                    feature_mode,
                    feature_labels,
                    id_to_new[old_id],
                    dataset=dataset,
                )

            normalized_nodes.append(
                {
                    "id": int(id_to_new[old_id]),
                    "label": label,
                    "feature": vector,
                    "pos": pos,
                }
            )

        canonical_edges = []
        seen_edges = set()
        for idx, edge in enumerate(edges):
            if not isinstance(edge, dict):
                raise GraphEditorError(422, f"Edge #{idx} must be an object.")
            if "source" not in edge or "target" not in edge:
                raise GraphEditorError(422, f"Edge #{idx} must include 'source' and 'target'.")

            source = edge["source"]
            target = edge["target"]
            if source not in id_to_new or target not in id_to_new:
                raise GraphEditorError(
                    422,
                    f"Edge #{idx} references unknown node ids ({source}, {target}).",
                )

            new_source = id_to_new[source]
            new_target = id_to_new[target]
            if new_source == new_target:
                raise GraphEditorError(422, "Self-loops are not allowed in the editor payload.")

            canonical = (
                (new_source, new_target)
                if new_source < new_target
                else (new_target, new_source)
            )
            if canonical in seen_edges:
                raise GraphEditorError(
                    422,
                    f"Duplicate undirected edge ({canonical[0]}, {canonical[1]}) is not allowed.",
                )
            seen_edges.add(canonical)
            canonical_edges.append(canonical)

        canonical_edges.sort()
        edge_index = self._edge_list_to_edge_index(canonical_edges)
        x_tensor = torch.tensor(features, dtype=torch.float)
        graph = Data(x=x_tensor, edge_index=edge_index, num_nodes=len(features))

        normalized_edges = [
            {"source": int(src), "target": int(dst)} for src, dst in canonical_edges
        ]
        return normalized_nodes, normalized_edges, graph

    def _validate_feature_vector(
        self,
        feature: Sequence[object],
        x_dim: int,
        feature_mode: str,
        node_id,
    ) -> List[float]:
        try:
            vector = [float(value) for value in feature]
        except (TypeError, ValueError) as exc:
            raise GraphEditorError(
                422,
                f"Node '{node_id}' contains a non-numeric feature value.",
            ) from exc

        if len(vector) != x_dim:
            raise GraphEditorError(
                422,
                f"Node '{node_id}' feature length {len(vector)} does not match x_dim={x_dim}.",
            )

        if feature_mode == "onehot":
            if any(value < -1e-4 or value > 1.0 + 1e-4 for value in vector):
                raise GraphEditorError(
                    422,
                    f"Node '{node_id}' one-hot feature values must stay inside [0, 1].",
                )
            if abs(sum(vector) - 1.0) > 1e-4:
                raise GraphEditorError(
                    422,
                    f"Node '{node_id}' one-hot feature values must sum to 1.",
                )

        return [round(value, 6) for value in vector]

    def _validate_position(self, pos) -> dict:
        if not isinstance(pos, dict):
            return {"x": 450.0, "y": 300.0}

        try:
            x = float(pos.get("x", 450.0))
            y = float(pos.get("y", 300.0))
        except (TypeError, ValueError) as exc:
            raise GraphEditorError(422, "Node positions must be numeric.") from exc

        return {"x": round(x, 3), "y": round(y, 3)}

    def _edge_list_to_edge_index(self, edges: Sequence[Tuple[int, int]]) -> torch.Tensor:
        if not edges:
            return torch.empty((2, 0), dtype=torch.long)

        directed_edges = []
        for src, dst in edges:
            directed_edges.append((src, dst))
            directed_edges.append((dst, src))
        return torch.tensor(directed_edges, dtype=torch.long).t().contiguous()

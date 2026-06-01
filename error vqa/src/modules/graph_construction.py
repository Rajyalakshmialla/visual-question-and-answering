"""
Graph Construction Module
Converts technical diagrams into graph structures representing nodes and edges
"""

import cv2
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class GraphNode:
    """Represents a node in the diagram graph"""

    node_id: str
    label: str
    position: Tuple[int, int]
    shape_type: str
    attributes: Dict


@dataclass
class GraphEdge:
    """Represents an edge in the diagram graph"""

    source: str
    target: str
    edge_type: str  # connection, arrow, etc.
    label: Optional[str] = None
    weight: float = 1.0


class GraphConstructor:
    """Constructs graph structures from technical diagrams"""

    def __init__(self):
        """Initialize the graph constructor"""
        self.graph = nx.DiGraph()
        self.nodes: List[GraphNode] = []
        self.edges: List[GraphEdge] = []

    def construct_graph(
        self, image: np.ndarray, vision_results: Dict, text_detections: List
    ) -> nx.DiGraph:
        """
        Construct a graph from diagram components.

        Args:
            image: Input diagram image
            vision_results: Results from vision processing
            text_detections: Detected text and labels

        Returns:
            NetworkX directed graph
        """
        self.graph = nx.DiGraph()
        self.nodes = []
        self.edges = []

        try:
            # Extract nodes from detected shapes
            self._extract_nodes(vision_results, text_detections)

            # Extract edges from connections
            self._extract_edges(image, vision_results)

            # Add nodes to graph
            for node in self.nodes:
                self.graph.add_node(node.node_id, **asdict(node))

            # Add edges to graph
            for edge in self.edges:
                self.graph.add_edge(edge.source, edge.target, edge_type=edge.edge_type, weight=edge.weight)

            logger.info(f"Constructed graph with {len(self.nodes)} nodes and {len(self.edges)} edges")
            return self.graph

        except Exception as e:
            logger.error(f"Error constructing graph: {str(e)}")
            return self.graph

    def _extract_nodes(self, vision_results: Dict, text_detections: List) -> None:
        """Extract nodes from detected shapes and text"""
        node_id_counter = 0

        # Create nodes from detected shapes
        for shape in vision_results.get("shapes", []):
            node_id = f"node_{node_id_counter}"
            node_id_counter += 1

            # Find associated text
            label = self._find_associated_text(shape, text_detections)

            node = GraphNode(
                node_id=node_id,
                label=label or shape.shape_type,
                position=shape.center,
                shape_type=shape.shape_type,
                attributes={"area": shape.area, "confidence": shape.confidence},
            )

            self.nodes.append(node)

    def _find_associated_text(self, shape, text_detections: List) -> Optional[str]:
        """Find text label associated with a shape"""
        shape_x1, shape_y1, shape_x2, shape_y2 = shape.location
        shape_center = ((shape_x1 + shape_x2) / 2, (shape_y1 + shape_y2) / 2)

        # Find text closest to shape center
        closest_text = None
        closest_distance = float("inf")

        for text in text_detections:
            text_x1, text_y1, text_x2, text_y2 = text.location
            text_center = ((text_x1 + text_x2) / 2, (text_y1 + text_y2) / 2)

            distance = np.sqrt(
                (shape_center[0] - text_center[0]) ** 2 + (shape_center[1] - text_center[1]) ** 2
            )

            if distance < closest_distance and distance < 50:  # Within 50 pixels
                closest_distance = distance
                closest_text = text.text

        return closest_text

    def _extract_edges(self, image: np.ndarray, vision_results: Dict) -> None:
        """Extract edges from connections between shapes"""
        edge_id_counter = 0

        if image.dtype == np.float32:
            img_uint8 = (image * 255).astype(np.uint8)
        else:
            img_uint8 = image

        # Convert to grayscale
        gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY) if len(img_uint8.shape) == 3 else img_uint8

        # Detect lines (connections)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=30, maxLineGap=10)

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]

                # Find which nodes this line connects
                source_node = self._find_closest_node((x1, y1))
                target_node = self._find_closest_node((x2, y2))

                if source_node and target_node and source_node != target_node:
                    edge_type = self._classify_connection_type((x1, y1, x2, y2), img_uint8)

                    edge = GraphEdge(
                        source=source_node.node_id,
                        target=target_node.node_id,
                        edge_type=edge_type,
                        weight=1.0,
                    )

                    # Avoid duplicate edges
                    if not self._edge_exists(edge):
                        self.edges.append(edge)
                        edge_id_counter += 1

    def _find_closest_node(self, point: Tuple[int, int]) -> Optional[GraphNode]:
        """Find the closest node to a given point"""
        closest_node = None
        closest_distance = float("inf")

        for node in self.nodes:
            node_x, node_y = node.position
            distance = np.sqrt((point[0] - node_x) ** 2 + (point[1] - node_y) ** 2)

            if distance < closest_distance and distance < 100:  # Within 100 pixels
                closest_distance = distance
                closest_node = node

        return closest_node

    def _classify_connection_type(self, line: Tuple, image: np.ndarray) -> str:
        """Classify the type of connection (arrow, simple line, dashed, etc.)"""
        x1, y1, x2, y2 = line

        # Check for arrow by looking at endpoint patterns
        endpoint_region = self._get_endpoint_region(image, (x2, y2))
        if self._is_arrow_endpoint(endpoint_region):
            return "arrow"
        else:
            return "line"

    def _get_endpoint_region(self, image: np.ndarray, point: Tuple[int, int], window_size: int = 20) -> np.ndarray:
        """Get region around an endpoint"""
        x, y = point
        x1 = max(0, x - window_size // 2)
        x2 = min(image.shape[1], x + window_size // 2)
        y1 = max(0, y - window_size // 2)
        y2 = min(image.shape[0], y + window_size // 2)

        return image[y1:y2, x1:x2]

    def _is_arrow_endpoint(self, region: np.ndarray) -> bool:
        """Check if region looks like an arrow endpoint"""
        if region.size == 0:
            return False

        # Simple heuristic: check pixel patterns
        white_pixels = np.sum(region > 127)
        return white_pixels > 10

    def _edge_exists(self, edge: GraphEdge) -> bool:
        """Check if edge already exists in the graph"""
        for existing_edge in self.edges:
            if existing_edge.source == edge.source and existing_edge.target == edge.target:
                return True
        return False

    def get_graph_statistics(self) -> Dict:
        """Get statistics about the constructed graph"""
        return {
            "num_nodes": self.graph.number_of_nodes(),
            "num_edges": self.graph.number_of_edges(),
            "density": nx.density(self.graph),
            "num_connected_components": nx.number_strongly_connected_components(self.graph),
            "has_cycles": not nx.is_directed_acyclic_graph(self.graph),
        }

    def get_node_degrees(self) -> Dict[str, int]:
        """Get in-degree and out-degree for all nodes"""
        degrees = {}
        for node in self.graph.nodes():
            degrees[node] = {
                "in_degree": self.graph.in_degree(node),
                "out_degree": self.graph.out_degree(node),
            }
        return degrees

    def find_paths(self, source: str, target: str) -> List:
        """Find all paths between two nodes"""
        try:
            paths = list(nx.all_simple_paths(self.graph, source, target))
            return paths
        except nx.NetworkXNoPath:
            return []

    def get_graph_visualization_data(self) -> Dict:
        """Get data for graph visualization"""
        return {
            "nodes": [asdict(node) for node in self.nodes],
            "edges": [
                {
                    "source": edge.source,
                    "target": edge.target,
                    "edge_type": edge.edge_type,
                    "label": edge.label,
                    "weight": edge.weight,
                }
                for edge in self.edges
            ],
            "statistics": self.get_graph_statistics(),
        }

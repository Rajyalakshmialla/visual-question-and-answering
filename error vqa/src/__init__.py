"""
Error-Aware Visual Question Answering System for Technical Diagrams
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "VQA Research Team"

from src.dataset_loader import load_manifest, scan_dataset

__all__ = [
    "VisionProcessor",
    "GraphConstructor",
    "DiagramQAEngine",
    "load_manifest",
    "scan_dataset",
]


def __getattr__(name):
    if name == "VisionProcessor":
        from src.modules.vision import VisionProcessor

        return VisionProcessor
    if name == "GraphConstructor":
        from src.modules.graph_construction import GraphConstructor

        return GraphConstructor
    if name == "DiagramQAEngine":
        from src.vqa_engine import DiagramQAEngine

        return DiagramQAEngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

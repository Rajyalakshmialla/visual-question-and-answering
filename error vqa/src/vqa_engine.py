import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image

from src.modules.graph_construction import GraphConstructor
from src.modules.vision import VisionProcessor


class DiagramQAEngine:
    """A lightweight diagram QA engine for image-based questions."""

    def __init__(self, use_easyocr: bool = True):
        self.vision_processor = VisionProcessor(use_easyocr=use_easyocr)
        self.graph_constructor = GraphConstructor()

    def analyze_image(self, image: np.ndarray) -> Dict:
        """Analyze the uploaded diagram image."""
        vision_results = self.vision_processor.process_diagram(image)
        graph = self.graph_constructor.construct_graph(image, vision_results, vision_results.get("text", []))
        return {
            "vision_results": vision_results,
            "graph": graph,
            "shape_counts": self._count_shapes(vision_results),
            "text_labels": self._extract_text_labels(vision_results),
            "arrow_count": self._count_arrows(graph),
            "node_count": graph.number_of_nodes(),
            "edge_count": graph.number_of_edges(),
            "diagram_type": self._detect_diagram_type(vision_results, graph),
        }

    def preprocess_image(self, image_path: str) -> Optional[np.ndarray]:
        try:
            image = Image.open(image_path).convert("RGB")
            return np.array(image)
        except Exception:
            return None

    def answer_question(self, question: str, analysis: Dict) -> Dict:
        question_text = question.strip()
        if not question_text:
            return {
                "answer": "Please enter a valid question.",
                "explanation": "The question is empty.",
                "confidence": 0.0,
            }

        q = question_text.lower()
        shape_counts = analysis["shape_counts"]
        text_labels = analysis["text_labels"]
        arrow_count = analysis["arrow_count"]
        diagram_type = analysis["diagram_type"]
        answer = ""
        explanation_lines = []

        if "summary" in q or "summarize" in q or "describe" in q or "predict" in q:
            summary = self.summarize_diagram(analysis)
            answer = summary["prediction"]
            explanation_lines = summary["summary_lines"]
        elif "how many" in q or "count" in q:
            answer, explanation_lines = self._answer_count_question(q, shape_counts, arrow_count, text_labels)
        elif "what" in q or "which" in q:
            answer, explanation_lines = self._answer_what_question(q, analysis)
        elif "is" in q or "are" in q or "does" in q:
            answer, explanation_lines = self._answer_yes_no_question(q, analysis)
        else:
            answer, explanation_lines = self._answer_what_question(q, analysis)

        if not answer:
            answer = f"This looks like a {diagram_type}. I detected {shape_counts.get('rectangle', 0)} boxes, {shape_counts.get('circle', 0)} circles, and {arrow_count} connections."
            explanation_lines.append("No direct answer rule matched, so the system returned a structure summary.")

        confidence = self._estimate_confidence(answer, explanation_lines)
        explanation = " \n".join(explanation_lines)

        return {
            "answer": answer,
            "explanation": explanation,
            "confidence": float(confidence),
        }

    def summarize_diagram(self, analysis: Dict) -> Dict:
        """Return a direct prediction and structured summary for the diagram."""
        shape_counts = analysis["shape_counts"]
        text_labels = analysis["text_labels"]
        diagram_type = analysis["diagram_type"]
        node_count = analysis["node_count"]
        edge_count = analysis["edge_count"]
        arrow_count = analysis["arrow_count"]

        prediction = f"Predicted diagram type: {diagram_type}."
        structure = (
            f"Detected structure: {shape_counts.get('rectangle', 0)} rectangles, "
            f"{shape_counts.get('circle', 0)} circles, {shape_counts.get('diamond', 0)} diamonds, "
            f"{node_count} graph nodes, and {edge_count} connections."
        )

        summary_lines = [prediction, structure]

        if text_labels:
            summary_lines.append(f"Detected text labels: {', '.join(text_labels[:8])}.")
        else:
            summary_lines.append("No reliable text labels were detected, so the prediction is based mainly on shapes and connections.")

        if arrow_count == 0 and node_count > 1:
            summary_lines.append("Possible issue: multiple elements were found, but no clear arrow/connection was detected.")
        elif arrow_count > 0:
            summary_lines.append(f"Relationship summary: {arrow_count} arrow or line connection(s) link the detected elements.")

        if diagram_type == "UML class diagram":
            summary_lines.append("Interpretation: the rectangles are treated as class/component-like elements.")
        elif diagram_type == "UML use case diagram":
            summary_lines.append("Interpretation: circular elements and actor/use-case labels suggest use-case structure.")
        else:
            summary_lines.append("Interpretation: the image is a technical diagram, but OCR/shape evidence is not strong enough for a more specific UML subtype.")

        return {
            "prediction": prediction,
            "summary": "\n".join(summary_lines),
            "summary_lines": summary_lines,
        }

    def _count_shapes(self, vision_results: Dict) -> Dict[str, int]:
        counts = {"rectangle": 0, "circle": 0, "triangle": 0, "diamond": 0, "polygon": 0, "text": 0}
        for shape in vision_results.get("shapes", []):
            if hasattr(shape, "shape_type"):
                shape_type = shape.shape_type
            else:
                shape_type = shape.get("shape_type", "")
            if shape_type in counts:
                counts[shape_type] += 1
            else:
                counts[shape_type] = counts.get(shape_type, 0) + 1
        counts["text"] = len(vision_results.get("text", []))
        return counts

    def _extract_text_labels(self, vision_results: Dict) -> List[str]:
        labels = []
        for detected in vision_results.get("text", []):
            text = getattr(detected, "text", None) if hasattr(detected, "text") else detected.get("text", "")
            if text and text.strip():
                labels.append(text.strip())
        return labels

    def _count_arrows(self, graph) -> int:
        count = 0
        for _, _, data in graph.edges(data=True):
            if data.get("edge_type") == "arrow":
                count += 1
        return count if count > 0 else graph.number_of_edges()

    def _detect_diagram_type(self, vision_results: Dict, graph) -> str:
        text = " ".join(self._extract_text_labels(vision_results)).lower()
        counts = self._count_shapes(vision_results)

        if re.search(r"\bclass\b|\binterface\b|\bentity\b|\bcontroller\b|\bservice\b", text):
            return "UML class diagram"
        if counts.get("rectangle", 0) >= 2 and graph.number_of_edges() >= 1 and counts.get("circle", 0) == 0:
            return "UML class diagram"
        if re.search(r"\bactor\b|\buse case\b|\busecase\b", text) or counts.get("circle", 0) >= 2:
            return "UML use case diagram"
        if re.search(r"\bsequence\b|\bmessage\b|\basync\b", text) or graph.number_of_edges() >= 6:
            return "UML sequence diagram"
        if re.search(r"\bcomponent\b|\bdeployment\b|\bnode\b", text):
            return "UML component or deployment diagram"
        return "technical diagram"

    def _answer_count_question(self, q: str, shape_counts: Dict[str, int], arrow_count: int, text_labels: List[str]) -> (str, List[str]):
        explanation = []
        if "class" in q or "classes" in q:
            count = shape_counts.get("rectangle", 0)
            answer = f"I counted {count} rectangle-shaped class elements in the diagram."
            explanation.append(f"Detected {count} rectangles as class-like elements.")
            return answer, explanation
        if "actor" in q:
            count = sum(1 for label in text_labels if re.search(r"\bactor\b", label.lower()))
            answer = f"I found {count} actors by text labels and diagram structure."
            explanation.append("Actor detection uses OCR label keywords.")
            return answer, explanation
        if "box" in q or "rectangle" in q:
            count = shape_counts.get("rectangle", 0)
            answer = f"The diagram contains {count} rectangle boxes."
            explanation.append(f"Detected {count} rectangle shapes.")
            return answer, explanation
        if "circle" in q:
            count = shape_counts.get("circle", 0)
            answer = f"The diagram contains {count} circle shapes."
            explanation.append(f"Detected {count} circle shapes.")
            return answer, explanation
        if "arrow" in q or "connection" in q or "link" in q:
            answer = f"The diagram contains {arrow_count} connections or arrows."
            explanation.append(f"Detected {arrow_count} edges in the diagram graph.")
            return answer, explanation
        if "text" in q or "label" in q:
            count = len(text_labels)
            answer = f"I detected {count} text labels in the diagram."
            explanation.append("Text is extracted using OCR.")
            return answer, explanation
        answer = f"I counted {sum(v for k, v in shape_counts.items() if k != 'text')} shape elements and {arrow_count} connections."
        explanation.append("The question asks for a count, so I returned the total shape and connection count.")
        return answer, explanation

    def _answer_what_question(self, q: str, analysis: Dict) -> (str, List[str]):
        explanation = []
        diagram_type = analysis["diagram_type"]
        shape_counts = analysis["shape_counts"]
        text_labels = analysis["text_labels"]

        if "type of diagram" in q or "diagram is" in q or "what is this" in q:
            answer = f"This appears to be a {diagram_type}."
            explanation.append(f"Detected diagram type based on OCR keywords and shape counts: {diagram_type}.")
            return answer, explanation
        if "main components" in q or "components" in q or "classes" in q:
            labels = [label for label in text_labels if len(label.split()) <= 4][:8]
            if labels:
                answer = f"Detected components: {', '.join(labels[:5])}."
                explanation.append("These labels were extracted from the diagram text.")
            else:
                answer = f"I detected {shape_counts.get('rectangle', 0)} major elements in the diagram."
                explanation.append("No reliable text labels were found, so I returned a shape-based summary.")
            return answer, explanation
        if "error" in q or "missing" in q or "broken" in q:
            answer = "I can check the diagram for broken connections and missing arrows."
            explanation.append("The current analysis includes graph structure and line connectivity.")
            return answer, explanation
        answer = f"This looks like a {diagram_type} with {shape_counts.get('rectangle', 0)} boxes and {analysis['edge_count']} connections."
        explanation.append("The answer is based on diagram shape counts and graph edges.")
        return answer, explanation

    def _answer_yes_no_question(self, q: str, analysis: Dict) -> (str, List[str]):
        explanation = []
        shape_counts = analysis["shape_counts"]
        diagram_type = analysis["diagram_type"]
        arrow_count = analysis["arrow_count"]

        if "missing arrow" in q or "missing arrows" in q:
            is_missing = arrow_count == 0 and shape_counts.get("rectangle", 0) > 1
            answer = "Yes, the diagram may be missing arrows." if is_missing else "No, I found connections between diagram elements."
            explanation.append(f"Arrow count: {arrow_count}. Shape count: {shape_counts.get('rectangle', 0)}.")
            return answer, explanation
        if "is this" in q and "class diagram" in q:
            is_class = diagram_type == "UML class diagram"
            answer = "Yes, this looks like a UML class diagram." if is_class else "No, this does not look like a UML class diagram."
            explanation.append(f"Detected diagram type: {diagram_type}.")
            return answer, explanation
        if "does" in q and ("contain" in q or "have" in q):
            if "arrow" in q or "connection" in q:
                answer = "Yes, it contains connections." if arrow_count > 0 else "No, I did not detect clear connections."
                explanation.append(f"Connections detected: {arrow_count}.")
                return answer, explanation
        answer = f"I am not fully certain, but the diagram appears to be a {diagram_type}."
        explanation.append("The question could not be matched to a specific yes/no rule.")
        return answer, explanation

    def _estimate_confidence(self, answer: str, explanation_lines: List[str]) -> float:
        score = min(1.0, 0.4 + len(explanation_lines) * 0.1)
        if "may be" in answer.lower() or "appears" in answer.lower():
            score *= 0.7
        return score

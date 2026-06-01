"""
Main VQA Pipeline
A simplified diagram QA entrypoint optimized for UML and technical diagrams.
"""

import sys
import logging
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.vqa_engine import DiagramQAEngine

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ErrorAwareVQASystem:
    """Simplified Error-Aware VQA System for technical diagrams."""

    def __init__(self):
        logger.info("Initializing Error-Aware VQA System...")
        self.engine = DiagramQAEngine(use_easyocr=True)
        logger.info("VQA engine initialized")

    def process_question(self, image_path: str, question: str) -> dict:
        """Process a diagram image and return an answer."""
        result = {
            "success": False,
            "image_path": image_path,
            "question": question,
            "answer": None,
            "explanation": None,
            "confidence": 0.0,
            "analysis": {},
            "error": None,
        }

        try:
            image = self.engine.preprocess_image(image_path)
            if image is None:
                raise FileNotFoundError(f"Unable to open image: {image_path}")

            analysis = self.engine.analyze_image(image)
            response = self.engine.answer_question(question, analysis)

            result.update(
                {
                    "success": True,
                    "answer": response["answer"],
                    "explanation": response["explanation"],
                    "confidence": response["confidence"],
                    "analysis": analysis,
                }
            )

        except Exception as exc:
            logger.error(f"Error processing question: {exc}", exc_info=True)
            result["error"] = str(exc)

        return result

    def get_system_info(self) -> dict:
        return {
            "name": "Error-Aware Diagram QA",
            "version": "1.0.0",
            "description": "A lightweight diagram question-answering system optimized for UML and technical diagrams.",
            "capabilities": [
                "Upload diagram image",
                "OCR text extraction",
                "Shape detection",
                "Graph-based analysis",
                "Question-driven answer generation",
                "Explainable results",
            ],
        }


def main():
    vqa_system = ErrorAwareVQASystem()
    info = vqa_system.get_system_info()

    print("Error-Aware Diagram QA System")
    print("=" * 40)
    print(f"Name: {info['name']}")
    print(f"Version: {info['version']}")
    print("Capabilities:")
    for cap in info["capabilities"]:
        print(f"  - {cap}")

    print("\nRun the web interface with:")
    print("python src/web_app.py")


if __name__ == "__main__":
    main()

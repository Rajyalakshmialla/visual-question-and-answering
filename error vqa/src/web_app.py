import json
import os
import socket
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import gradio as gr

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from src.dataset_loader import load_manifest
    from src.vqa_engine import DiagramQAEngine
except ModuleNotFoundError:
    # Fallback for `python src/web_app.py` when `src` is not on the import path
    ROOT_DIR_SRC = Path(__file__).resolve().parent
    if str(ROOT_DIR_SRC) not in sys.path:
        sys.path.insert(0, str(ROOT_DIR_SRC))
    from dataset_loader import load_manifest
    from vqa_engine import DiagramQAEngine


def find_free_port(start: int = 7860, end: int = 7880) -> int:
    for port in range(start, end + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("0.0.0.0", port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"No free port found in range {start}-{end}")


def choose_port() -> int:
    env_port = os.environ.get("GRADIO_SERVER_PORT")
    if env_port and env_port.isdigit():
        requested_port = int(env_port)
        try:
            return find_free_port(requested_port, requested_port)
        except RuntimeError:
            print(f"Port {requested_port} is already in use. Trying the next available port...")
            return find_free_port(requested_port + 1, requested_port + 20)

    return find_free_port(7860, 7880)

DATASET_MANIFEST_PATH = ROOT_DIR / "data" / "uml_dataset_manifest.json"

engine = DiagramQAEngine(use_easyocr=True)
dataset_manifest = load_manifest(DATASET_MANIFEST_PATH)
sample_choices = [item["path"] for item in dataset_manifest][:80]


def load_sample_image(sample_path: str):
    if not sample_path:
        return None
    return sample_path


def build_prediction_summary(analysis):
    return engine.summarize_diagram(analysis)["prediction"]


def predict_answer(image, question):
    if image is None:
        return "Please upload a diagram image.", "", ""

    question_text = question.strip() if question else "Predict and summarize this diagram."

    if isinstance(image, Image.Image):
        input_image = image.convert("RGB")
    else:
        input_image = Image.fromarray(np.array(image)).convert("RGB")

    try:
        image_array = np.array(input_image)
        analysis = engine.analyze_image(image_array)
        summary = engine.summarize_diagram(analysis)
        result = engine.answer_question(question_text, analysis)

        answer = result.get("answer") or summary["prediction"]
        if "summary" in question_text.lower() or "predict" in question_text.lower() or question_text == "Predict and summarize this diagram.":
            explanation = summary["summary"]
        else:
            explanation = result.get("explanation") or summary["summary"]

        debug_info = {
            "diagram_type": analysis.get("diagram_type"),
            "shape_counts": analysis.get("shape_counts"),
            "arrow_count": analysis.get("arrow_count"),
            "node_count": analysis.get("node_count"),
            "edge_count": analysis.get("edge_count"),
            "text_labels": analysis.get("text_labels", [])[:10],
            "confidence": result.get("confidence"),
        }

        return answer, explanation, json.dumps(debug_info, indent=2)
    except Exception as exc:
        return (
            f"Prediction failed: {exc}",
            "The app could not finish analyzing this image. Restart the server after code changes, then upload the image again.",
            json.dumps({"error": str(exc)}, indent=2),
        )


def launch_app(port: int | None = None):
    description = (
        "Upload a technical diagram or select a sample from your dataset. "
        "Ask a question about the diagram and get a clear answer with explanation."
    )

    with gr.Blocks(title="Error-Aware Diagram QA", theme=gr.themes.Default()) as demo:
        gr.Markdown("# Error-Aware Diagram QA")
        gr.Markdown(description)

        with gr.Row():
            with gr.Column(scale=1):
                sample_selector = gr.Dropdown(
                    choices=sample_choices,
                    label="Select a sample dataset image",
                    info="Choose one of the first 80 images from data/uml_dataset_manifest.json if available.",
                )
                load_button = gr.Button("Load Sample Image")
                image_input = gr.Image(type="pil", label="Upload Diagram Image", tool="editor")
                question_input = gr.Textbox(
                    label="Question",
                    placeholder="e.g. How many classes are shown?",
                    lines=2,
                )
                submit_button = gr.Button("Get Answer")

            with gr.Column(scale=1):
                answer_output = gr.Textbox(label="Predicted Answer", interactive=False)
                explanation_output = gr.Textbox(label="Detailed Explanation", interactive=False, lines=10)
                debug_output = gr.Code(label="Analysis Debug Info")

        load_button.click(fn=load_sample_image, inputs=[sample_selector], outputs=[image_input])
        submit_button.click(
            fn=predict_answer,
            inputs=[image_input, question_input],
            outputs=[answer_output, explanation_output, debug_output],
        )

    if port is None:
        port = choose_port()

    print(f"Launching web app on http://127.0.0.1:{port}")
    demo.launch(server_name="0.0.0.0", server_port=port, show_error=True)


if __name__ == "__main__":
    launch_app()

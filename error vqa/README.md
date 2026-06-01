# Error-Aware Visual Question Answering for Technical Diagrams

## 🎯 Vision

Imagine a system that not only understands a technical diagram but also **verifies whether it is correct**. Our project introduces an **error-aware framework** that detects structural issues, explains anomalies, and suggests corrections through self-healing mechanisms—transforming passive diagram understanding into intelligent validation.

## 📋 Overview

An intelligent Visual Question Answering (VQA) system for technical diagrams that:
- ✅ **Detects** structural errors and anomalies early
- 💡 **Explains** detected issues through intelligent reasoning
- 🔧 **Heals** diagrams by suggesting and implementing corrections

This system combines multiple advanced techniques to understand and analyze technical diagrams:

- **Image Preprocessing**: Resizing, normalization, noise removal, and edge enhancement
- **Early Error Detection**: Identifies anomalies like missing arrows, broken connections, invalid symbols, and disconnected nodes
- **Self-Healing**: Automatically repairs detected diagram errors
- **OCR & Vision**: Extracts text and identifies shapes using computer vision techniques
- **Graph Construction**: Converts diagrams into graph structures representing nodes and edges
- **Multimodal Fusion**: Combines visual, textual, graph, and question embeddings
- **Reasoning Engine**: Performs structural and logical reasoning on diagram relationships
- **Answer Generation**: Provides answers with step-by-step explanations

## 🔴 Problem Statement

Current Visual Question Answering systems for technical diagrams interpret and answer queries but **lack validation mechanisms** to verify diagram correctness. These systems assume diagrams are always structurally and logically accurate—an assumption often violated in real-world applications.

### Key Challenges:
- ❌ **Unvalidated Input**: No pre-processing validation before reasoning
- ❌ **Undetected Errors**: Missing arrows, invalid connections, inconsistent structures go unnoticed
- ❌ **Lack of Interpretability**: Detected errors are not explained to users
- ❌ **No Correction Capabilities**: Systems cannot suggest or implement fixes

**Impact**: Incorrect reasoning, unreliable outputs, and reduced trust in AI-powered systems—especially critical in safety-sensitive domains.

## 🎓 Research Focus

| Domain | Focus |
|--------|-------|
| **Primary** | Computer Vision & Visual Question Answering |
| **Sub-Area 1** | Document & Diagram Understanding (symbol/text extraction) |
| **Sub-Area 2** | Graph-Based Reasoning (structure analysis & logical flow) |
| **Sub-Area 3** | Intelligent Validation & Self-Healing Systems (error detection & correction) |

---

## 🟣 Three Core Objectives

### **OBJECTIVE 1: Early Error Detection in Diagrams**
Detect structural errors and anomalies before reasoning using rule-based validation and graph structure analysis.
- **Gap Addressed**: Existing models assume diagrams are correct
- **Metrics**: Error Detection Accuracy (>90%), Consistency Score (>0.85)

### **OBJECTIVE 2: Anomaly Reasoning**  
Explain detected errors through intelligent reasoning to enhance interpretability and user trust.
- **Gap Addressed**: Detected inconsistencies lack explanation
- **Metrics**: Explanation Accuracy (>85%), Reasoning Consistency (>0.90)

### **OBJECTIVE 3: Self-Healing Diagram System**
Suggest and implement automatic corrections for detected errors to reduce manual intervention.
- **Gap Addressed**: No mechanism to correct diagrams
- **Metrics**: Correction Accuracy (>85%), User Acceptance (>80%)

---

📖 **For detailed research methodology, algorithms, and metrics, see [RESEARCH_OVERVIEW.md](RESEARCH_OVERVIEW.md)**

---

## ✨ Features

**Core Capabilities:**
- Automatic error detection and correction
- Multi-modal information fusion
- Advanced reasoning and inference
- Detailed answer explanations
- Support for various diagram types

🎯 **Use Cases:**
- Technical education and training
- Engineering documentation analysis
- Process flow understanding
- Circuit diagram interpretation
- System architecture analysis

## Project Structure

```
error-aware-vqa/
├── src/
│   ├── modules/
│   │   ├── vision.py              # OCR and shape detection
│   │   └── graph_construction.py  # Graph-based connection analysis
│   ├── dataset_loader.py         # Import UML dataset images and save manifest
│   ├── vqa_engine.py             # Diagram QA engine and question answering
│   ├── main.py                   # Command-line system info and entrypoint
│   ├── web_app.py                # Gradio-based demo interface
│   └── __init__.py               # Package exports for the simplified demo
├── data/
│   └── import_uml_dataset.py     # Create a dataset manifest from a folder
├── requirements.txt              # Python dependencies
├── setup.py                      # Package setup
└── README.md                     # This file
```

## Installation

### Prerequisites
- Python 3.9+
- pip or conda
- (Optional) Tesseract-OCR for advanced text extraction

### Setup

1. **Clone the repository**
```bash
cd error-aware-vqa
```

2. **Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Optional: Install Tesseract OCR**
```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr

# macOS
brew install tesseract

# Windows
# Download from: https://github.com/UB-Mannheim/tesseract/wiki
```

## Quick Start

### Basic Usage

```python
from src.main import ErrorAwareVQASystem

# Initialize the VQA system
vqa_system = ErrorAwareVQASystem()

# Process a diagram with a question
result = vqa_system.process_question(
    image_path="path/to/diagram.png",
    question="What is shown in this diagram?"
)

# Access the answer
print("Answer:", result['answer'])

# View detailed explanation
print("\nExplanation:")
print(result['explanation'])

# Check confidence
print("\nConfidence:", f"{result['confidence']:.1%}")
```

### Using Your UML Dataset

Run this command to import your dataset folder and create a manifest:

```bash
python data/import_uml_dataset.py "c:\\Users\\srisb\\Downloads\\UML_Diagram_Dataset"
```

Then run the web interface:

```bash
python src/web_app.py
```

### Advanced Features

```python
# Batch processing multiple diagrams
images = ["diagram1.png", "diagram2.png", "diagram3.png"]
questions = [
    "What are the main components?",
    "How are these connected?",
    "What errors exist in this diagram?"
]

results = vqa_system.batch_process(images, questions)

# Access detailed pipeline information
for i, result in enumerate(results):
    print(f"\nDiagram {i+1}:")
    print(f"  Anomalies detected: {result['pipeline_info']['anomalies']}")
    print(f"  Shapes found: {result['pipeline_info']['shapes_detected']}")
    print(f"  Text extracted: {result['pipeline_info']['text_detected']}")
```

## System Architecture

### Pipeline Flow

```
Input Image
    ↓
[Preprocessing] → [Error Detection] → [Self-Healing]
    ↓
[Vision Processing] → [Graph Construction]
    ↓
[Multimodal Fusion] ← Question Encoding
    ↓
[Reasoning Engine]
    ↓
[Answer Generation]
    ↓
Output: Answer + Explanation
```

### Key Components

1. **ImagePreprocessor**: Enhances image quality through normalization and noise reduction
2. **ErrorDetector**: Identifies structural anomalies using image analysis
3. **SelfHealingModule**: Repairs detected errors using morphological operations
4. **VisionProcessor**: Extracts text (OCR) and detects shapes/symbols
5. **GraphConstructor**: Builds graph representation from diagram elements
6. **MultimodalFusion**: Combines multiple feature representations
7. **ReasoningEngine**: Performs logical inference on the diagram
8. **AnswerExplanation**: Generates human-readable answers with evidence

## Configuration

### Configuration Files

The current cleaned version of this project uses a simplified built-in diagram QA engine and does not require custom YAML configuration for the demo.

To import your UML dataset, use the provided `data/import_uml_dataset.py` script and a folder of diagram images.

## Testing

Run the test suite:

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_modules.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## API Documentation

### Three Core Objectives in Practice

#### **Objective 1: Early Error Detection**
```python
from src.modules.error_detection import ErrorDetector

detector = ErrorDetector(detection_threshold=0.5)
anomalies = detector.detect_errors(image)

# Results include:
# - Anomaly type (missing_arrow, broken_connection, etc.)
# - Location and confidence score
# - Severity level (low, medium, high)
```

#### **Objective 2: Anomaly Reasoning**
```python
from src.modules.reasoning_engine import ReasoningEngine

reasoning_engine = ReasoningEngine()
results = reasoning_engine.reason_about_diagram(
    graph, embedding, question, visual_context
)

# Results include:
# - Explanation of detected anomalies
# - Reasoning chain showing how conclusions were reached
# - Relevant diagram elements
# - Logical paths through the diagram
```

#### **Objective 3: Self-Healing System**
```python
from src.modules.self_healing import SelfHealingModule

healer = SelfHealingModule(healing_strength=0.8)
healed_image = healer.heal_diagram(image, anomalies)

# Features:
# - Automatic repair of broken connections
# - Addition of missing arrows
# - Fixing invalid symbols
# - Reconnecting disconnected nodes
```

### ErrorAwareVQASystem

Main class for the VQA system.

**Methods:**
- `process_question(image_path, question)`: Process a single image with a question
- `batch_process(image_paths, questions)`: Process multiple images
- `get_system_info()`: Get system capabilities and information

### Result Format

Each `process_question()` call returns:

```python
{
    "success": bool,              # Whether processing succeeded
    "question": str,              # The input question
    "answer": str,                # Generated answer
    "explanation": [              # Step-by-step explanation
        {
            "step": int,
            "title": str,
            "description": str,
            "evidence": [str],
            "confidence": float
        }
    ],
    "confidence": float,          # Overall confidence score
    "pipeline_info": {            # Detailed pipeline information
        "preprocessing": dict,
        "anomalies": dict,
        "shapes_detected": int,
        "text_detected": int,
        "graph_stats": dict
    },
    "error": str or None          # Error message if processing failed
}
```

## Performance Considerations

- **Image Size**: Default 768×768 balances quality and speed
- **OCR**: EasyOCR is faster than Tesseract but requires more memory
- **GPU Support**: PyTorch operations can use CUDA if available
- **Batch Processing**: Use for multiple images to reduce overhead

## Troubleshooting

### Issue: OCR not working
- Install Tesseract-OCR separately
- Or set `use_easyocr=False` in VisionProcessor

### Issue: Memory issues with large images
- Reduce `IMAGE_SIZE` in configuration
- Process images sequentially instead of batch

### Issue: Low confidence scores
- Check image quality and diagram clarity
- Ensure diagram follows standard conventions
- Adjust `DETECTION_THRESHOLD` in configuration

## Applications

1. **Technical Education**
   - Automatic diagram interpretation for students
   - Step-by-step learning explanations

2. **Engineering Documentation**
   - Extract information from circuit/system diagrams
   - Validate diagram correctness

3. **Process Analysis**
   - Understand workflow and process flows
   - Identify bottlenecks and issues

4. **Accessible Documentation**
   - Generate text descriptions of visual diagrams
   - Support for individuals with visual impairments

## Future Enhancements

### Research & Development Roadmap

**Enhanced Error Detection (Objective 1)**
- [ ] Learning-based error detection using neural networks
- [ ] Domain-specific rule libraries for different diagram types
- [ ] Real-time error detection in video streams
- [ ] Probabilistic error prediction

**Advanced Anomaly Reasoning (Objective 2)**
- [ ] Multi-modal explanation generation (visual + textual)
- [ ] Domain-specific reasoning for specialized diagrams
- [ ] Knowledge graph integration for semantic reasoning
- [ ] Interactive explanation refinement

**Intelligent Self-Healing (Objective 3)**
- [ ] Machine learning-based correction strategies
- [ ] User feedback loop for improvement
- [ ] Batch correction optimization
- [ ] Alternative solution generation

**System Enhancements**
- [ ] Support for color diagram analysis
- [ ] Multi-language question support
- [ ] Real-time video diagram processing
- [ ] Custom model fine-tuning interface
- [ ] Web-based user interface
- [ ] REST API deployment
- [ ] Mobile app development

---

## License

MIT License - See LICENSE file for details

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## Citation

If you use this system in your research, please cite:

```bibtex
@software{error_aware_vqa_2024,
  title={Error-Aware Visual Question Answering for Technical Diagrams},
  year={2024}
}
```

## Support

For issues, questions, or suggestions:
- Create an issue on GitHub
- Check existing documentation
- Review test cases for usage examples

## Acknowledgments

Built with:
- PyTorch and TensorFlow for deep learning
- OpenCV for computer vision
- EasyOCR/Tesseract for text recognition
- NetworkX for graph analysis
- HuggingFace Transformers for NLP

---

**Version**: 1.0.0  
**Last Updated**: 2024  
**Status**: Active Development

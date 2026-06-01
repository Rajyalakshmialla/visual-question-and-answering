from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="error-aware-vqa",
    version="1.0.0",
    author="VQA Research Team",
    description="Error-Aware Visual Question Answering for Technical Diagrams with Anomaly Reasoning and Self-Healing Mechanisms",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/error-aware-vqa",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.1.0",
        "torchvision>=0.16.0",
        "opencv-python>=4.6.0",
        "transformers>=4.35.0",
        "networkx>=3.2",
        "numpy>=1.26.0",
        "pillow>=10.0.0",
        "pytesseract>=0.3.10",
        "easyocr>=1.7.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.12.0",
            "flake8>=6.1.0",
            "mypy>=1.7.0",
        ],
    },
)

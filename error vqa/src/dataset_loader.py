import json
from pathlib import Path
from typing import Dict, List

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff", ".svg"}


def scan_dataset(root_path: Path) -> List[Dict[str, str]]:
    """Scan a folder recursively for diagram images."""
    root_path = Path(root_path)
    images = []

    if not root_path.exists() or not root_path.is_dir():
        return images

    for image_path in sorted(root_path.rglob("*")):
        if image_path.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(
                {
                    "path": str(image_path.resolve()),
                    "name": image_path.name,
                    "category": str(image_path.parent.relative_to(root_path).as_posix()) if image_path.parent != root_path else "root",
                }
            )

    return images


def save_manifest(images: List[Dict[str, str]], manifest_path: Path) -> None:
    """Save a dataset manifest to JSON."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(images, f, indent=2)


def load_manifest(manifest_path: Path) -> List[Dict[str, str]]:
    """Load a dataset manifest from JSON."""
    if not manifest_path.exists():
        return []

    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)

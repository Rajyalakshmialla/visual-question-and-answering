import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

def main():
    parser = argparse.ArgumentParser(description="Import a UML diagram dataset and save a JSON manifest.")
    parser.add_argument("source_path", nargs="?", help="Path to the UML dataset folder")
    parser.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parent / "uml_dataset_manifest.json"),
        help="Path to save the dataset manifest JSON",
    )
    args = parser.parse_args()

    if not args.source_path:
        parser.print_help()
        print('\nExample: python data/import_uml_dataset.py "C:\\Users\\srisb\\Downloads\\UML_Diagram_Dataset"')
        return

    source = Path(args.source_path)
    if not source.exists() or not source.is_dir():
        print(f"Dataset folder not found: {source}")
        return

    from src.dataset_loader import scan_dataset, save_manifest

    manifest_path = Path(args.output)
    images = scan_dataset(source)
    save_manifest(images, manifest_path)

    print(f"Found {len(images)} diagram images in {source}")
    print(f"Saved manifest to: {manifest_path}")


if __name__ == "__main__":
    main()

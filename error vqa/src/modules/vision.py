"""
Vision Processing Module
Extracts textual information through OCR and identifies symbols/shapes
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

try:
    import pytesseract
    from PIL import Image
    if not hasattr(Image, "ANTIALIAS"):
        Image.ANTIALIAS = Image.Resampling.LANCZOS
    PYTESSERACT_AVAILABLE = True
except ImportError:
    PYTESSERACT_AVAILABLE = False
    logger.warning("pytesseract not available, OCR functionality limited")

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    logger.warning("easyocr not available, EasyOCR functionality disabled")


@dataclass
class DetectedText:
    """Data class for detected text"""

    text: str
    location: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    language: str


@dataclass
class DetectedShape:
    """Data class for detected shapes"""

    shape_type: str  # circle, rectangle, triangle, diamond, etc.
    location: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    center: Tuple[int, int]
    area: float


class VisionProcessor:
    """Processes visual information including text and shape detection"""

    def __init__(self, use_easyocr: bool = True):
        """
        Initialize the vision processor.

        Args:
            use_easyocr: Whether to use EasyOCR instead of Tesseract
        """
        self.use_easyocr = use_easyocr and EASYOCR_AVAILABLE
        self.ocr_reader = None

        if self.use_easyocr:
            try:
                self.ocr_reader = easyocr.Reader(["en"])
                logger.info("EasyOCR reader initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize EasyOCR: {str(e)}, falling back to Tesseract")
                self.use_easyocr = False

    def process_diagram(self, image: np.ndarray) -> Dict:
        """
        Extract textual and visual information from diagram.

        Args:
            image: Input diagram image

        Returns:
            Dictionary containing detected text and shapes
        """
        results = {"text": [], "shapes": [], "features": []}

        try:
            # Extract text
            text_detections = self.extract_text(image)
            results["text"] = text_detections

            # Detect shapes
            shape_detections = self.detect_shapes(image)
            results["shapes"] = shape_detections

            # Extract visual features
            features = self.extract_visual_features(image)
            results["features"] = features

            logger.info(f"Processed diagram: {len(text_detections)} texts, {len(shape_detections)} shapes")
            return results

        except Exception as e:
            logger.error(f"Error processing diagram: {str(e)}")
            return results

    def extract_text(self, image: np.ndarray) -> List[DetectedText]:
        """Extract text from the diagram using OCR"""
        detections = []

        try:
            if image.dtype == np.float32:
                img_uint8 = (image * 255).astype(np.uint8)
            else:
                img_uint8 = image

            if self.use_easyocr and self.ocr_reader:
                detections = self._extract_text_easyocr(img_uint8)
            elif PYTESSERACT_AVAILABLE:
                detections = self._extract_text_tesseract(img_uint8)
            else:
                logger.warning("No OCR engine available")

        except Exception as e:
            logger.error(f"Error extracting text: {str(e)}")

        return detections

    def _extract_text_easyocr(self, image: np.ndarray) -> List[DetectedText]:
        """Extract text using EasyOCR"""
        detections = []

        try:
            results = self.ocr_reader.readtext(image, detail=1)

            for detection in results:
                bbox, text, confidence = detection
                # bbox is a list of 4 points
                x_coords = [p[0] for p in bbox]
                y_coords = [p[1] for p in bbox]

                x1, y1 = int(min(x_coords)), int(min(y_coords))
                x2, y2 = int(max(x_coords)), int(max(y_coords))

                detections.append(
                    DetectedText(
                        text=text,
                        location=(x1, y1, x2, y2),
                        confidence=float(confidence),
                        language="en",
                    )
                )

        except Exception as e:
            logger.error(f"EasyOCR extraction failed: {str(e)}")

        return detections

    def _extract_text_tesseract(self, image: np.ndarray) -> List[DetectedText]:
        """Extract text using Tesseract"""
        detections = []

        try:
            # Convert to PIL Image
            pil_image = Image.fromarray(image)

            # Get detailed results from Tesseract
            data = pytesseract.image_to_data(pil_image, output_type=pytesseract.Output.DICT)

            for i in range(len(data["text"])):
                if int(data["conf"][i]) < 0:
                    continue

                text = data["text"][i]
                if text.strip():
                    x1 = int(data["left"][i])
                    y1 = int(data["top"][i])
                    x2 = x1 + int(data["width"][i])
                    y2 = y1 + int(data["height"][i])

                    detections.append(
                        DetectedText(
                            text=text,
                            location=(x1, y1, x2, y2),
                            confidence=int(data["conf"][i]) / 100.0,
                            language="en",
                        )
                    )

        except Exception as e:
            logger.error(f"Tesseract extraction failed: {str(e)}")

        return detections

    def detect_shapes(self, image: np.ndarray) -> List[DetectedShape]:
        """Detect shapes and symbols in the diagram"""
        detections = []

        try:
            if image.dtype == np.float32:
                img_uint8 = (image * 255).astype(np.uint8)
            else:
                img_uint8 = image

            gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY) if len(img_uint8.shape) == 3 else img_uint8

            blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            binary = cv2.adaptiveThreshold(
                blurred,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                31,
                9,
            )
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

            detections.extend(self._detect_rectangles_from_lines(gray))

            contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            image_area = gray.shape[0] * gray.shape[1]

            for contour in contours:
                area = cv2.contourArea(contour)

                if area < 150 or area > image_area * 0.8:
                    continue

                x, y, w, h = cv2.boundingRect(contour)
                if w < 12 or h < 12:
                    continue

                if w > gray.shape[1] * 0.95 and h > gray.shape[0] * 0.95:
                    continue

                shape_type, confidence = self._classify_shape(contour)
                center = (x + w // 2, y + h // 2)

                if self._is_duplicate_shape(detections, (x, y, x + w, y + h)):
                    continue

                detections.append(
                    DetectedShape(
                        shape_type=shape_type,
                        location=(x, y, x + w, y + h),
                        confidence=confidence,
                        center=center,
                        area=float(area),
                    )
                )

        except Exception as e:
            logger.error(f"Error detecting shapes: {str(e)}")

        return detections

    def _classify_shape(self, contour: np.ndarray) -> Tuple[str, float]:
        """Classify shape type and return confidence"""
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        x, y, w, h = cv2.boundingRect(contour)

        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter ** 2)
        else:
            circularity = 0

        approx = cv2.approxPolyDP(contour, 0.03 * perimeter, True)
        num_vertices = len(approx)
        extent = area / float(w * h) if w * h else 0
        aspect_ratio = w / float(h) if h else 0

        if circularity > 0.8:
            return "circle", circularity
        elif num_vertices == 3:
            return "triangle", 0.85
        elif num_vertices == 4 and 0.6 <= extent <= 1.05:
            if 0.75 <= aspect_ratio <= 1.25:
                return "diamond", 0.75
            return "rectangle", 0.9
        elif num_vertices == 4:
            return "rectangle", 0.75
        elif num_vertices == 5 and 0.75 <= aspect_ratio <= 1.25:
            return "diamond", 0.8
        else:
            return "polygon", 0.7

    def _detect_rectangles_from_lines(self, gray: np.ndarray) -> List[DetectedShape]:
        """Detect UML-style boxes from straight line segments."""
        detections = []
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 45, minLineLength=25, maxLineGap=6)
        if lines is None:
            return detections

        horizontal = []
        vertical = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(y1 - y2) <= 4 and abs(x1 - x2) >= 25:
                horizontal.append((min(x1, x2), y1, max(x1, x2)))
            elif abs(x1 - x2) <= 4 and abs(y1 - y2) >= 25:
                vertical.append((x1, min(y1, y2), max(y1, y2)))

        for top_idx, (top_x1, top_y, top_x2) in enumerate(horizontal):
            for bottom_x1, bottom_y, bottom_x2 in horizontal[top_idx + 1 :]:
                height = abs(bottom_y - top_y)
                if height < 25 or height > gray.shape[0] * 0.8:
                    continue

                y1, y2 = sorted((top_y, bottom_y))
                x1 = max(top_x1, bottom_x1)
                x2 = min(top_x2, bottom_x2)
                if x2 - x1 < 25:
                    continue

                left_edges = [x for x, vy1, vy2 in vertical if abs(x - x1) <= 8 and vy1 <= y1 + 8 and vy2 >= y2 - 8]
                right_edges = [x for x, vy1, vy2 in vertical if abs(x - x2) <= 8 and vy1 <= y1 + 8 and vy2 >= y2 - 8]
                if not left_edges or not right_edges:
                    continue

                left = int(round(sum(left_edges) / len(left_edges)))
                right = int(round(sum(right_edges) / len(right_edges)))
                if right - left < 25:
                    continue

                location = (left, y1, right, y2)
                if self._is_duplicate_shape(detections, location):
                    continue

                detections.append(
                    DetectedShape(
                        shape_type="rectangle",
                        location=location,
                        confidence=0.9,
                        center=((left + right) // 2, (y1 + y2) // 2),
                        area=float((right - left) * (y2 - y1)),
                    )
                )

        return detections

    def _is_duplicate_shape(self, detections: List[DetectedShape], location: Tuple[int, int, int, int]) -> bool:
        """Avoid returning the same boxed element multiple times."""
        x1, y1, x2, y2 = location
        area = max(1, (x2 - x1) * (y2 - y1))

        for detected in detections:
            dx1, dy1, dx2, dy2 = detected.location
            ix1, iy1 = max(x1, dx1), max(y1, dy1)
            ix2, iy2 = min(x2, dx2), min(y2, dy2)
            if ix2 <= ix1 or iy2 <= iy1:
                continue

            intersection = (ix2 - ix1) * (iy2 - iy1)
            detected_area = max(1, (dx2 - dx1) * (dy2 - dy1))
            if intersection / min(area, detected_area) > 0.85:
                return True

        return False

    def extract_visual_features(self, image: np.ndarray) -> Dict:
        """Extract visual features from the diagram"""
        features = {}

        try:
            if image.dtype == np.float32:
                img_uint8 = (image * 255).astype(np.uint8)
            else:
                img_uint8 = image

            # Convert to grayscale
            gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY) if len(img_uint8.shape) == 3 else img_uint8

            # Edge features
            edges = cv2.Canny(gray, 50, 150)
            features["edge_density"] = float(np.sum(edges > 0) / edges.size)

            # Corner features
            corners = cv2.goodFeaturesToTrack(gray, 100, 0.01, 10)
            features["num_corners"] = int(len(corners)) if corners is not None else 0

            # Line features
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=30, maxLineGap=10)
            features["num_lines"] = int(len(lines)) if lines is not None else 0

            # Color features
            if len(img_uint8.shape) == 3:
                features["dominant_color"] = self._get_dominant_color(img_uint8)
                features["num_colors"] = self._count_unique_colors(img_uint8)

        except Exception as e:
            logger.error(f"Error extracting visual features: {str(e)}")

        return features

    def _get_dominant_color(self, image: np.ndarray) -> Tuple[int, int, int]:
        """Get the dominant color in the image"""
        pixels = image.reshape(-1, 3)
        unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)
        dominant_idx = np.argmax(counts)
        return tuple(unique_colors[dominant_idx].tolist())

    def _count_unique_colors(self, image: np.ndarray, bins: int = 8) -> int:
        """Count unique colors using binning"""
        quantized = image // (256 // bins)
        unique_colors = len(np.unique(quantized.reshape(-1, 3), axis=0))
        return unique_colors

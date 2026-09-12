"""
ocr.py
OCR extraction module wrapping pytesseract.

Designed so that the OCR backend (Tesseract) can be swapped for
PaddleOCR or EasyOCR by replacing this module.
"""

import logging
import os
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Allow users to override the Tesseract executable via an env variable.
# Example (Windows):  set TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
_TESSERACT_CMD = os.environ.get("TESSERACT_CMD")


def _configure_tesseract() -> None:
    """Point pytesseract at the correct Tesseract binary.

    Raises:
        EnvironmentError: If Tesseract cannot be found.
    """
    try:
        import pytesseract  # noqa: PLC0415

        if _TESSERACT_CMD:
            pytesseract.pytesseract.tesseract_cmd = _TESSERACT_CMD
    except ImportError as exc:
        raise EnvironmentError(
            "pytesseract is not installed. Run: pip install pytesseract"
        ) from exc


def extract_text(image: np.ndarray) -> str:
    """Run Tesseract OCR on a preprocessed image and return the extracted text.

    The image should already be preprocessed (grayscale, binarised) before
    being passed here.  pytesseract accepts NumPy arrays directly.

    Args:
        image: Preprocessed grayscale/binary image as a NumPy array.

    Returns:
        Extracted text string, stripped of leading/trailing whitespace.

    Raises:
        EnvironmentError: If Tesseract is not installed or cannot be located.
        RuntimeError: If OCR produces no output (empty result).
    """
    _configure_tesseract()

    try:
        import pytesseract  # noqa: PLC0415
    except ImportError as exc:
        raise EnvironmentError(
            "pytesseract is not installed. Run: pip install pytesseract"
        ) from exc

    try:
        # Page segmentation mode 6 (PSM 6): assume a single uniform block of text.
        # OEM 3: use both legacy Tesseract and LSTM engines.
        custom_config = r"--oem 3 --psm 6"
        raw = pytesseract.image_to_string(image, config=custom_config)
    except pytesseract.pytesseract.TesseractNotFoundError as exc:
        raise EnvironmentError(
            "Tesseract OCR is not installed or could not be located.\n"
            "Please install Tesseract and configure TESSERACT_CMD if required.\n"
            "See README.md for installation instructions."
        ) from exc
    except Exception as exc:
        raise RuntimeError(f"OCR failed unexpectedly: {exc}") from exc

    text = raw.strip()
    if not text:
        logger.warning("OCR returned an empty result. The image may have no readable text.")

    return text

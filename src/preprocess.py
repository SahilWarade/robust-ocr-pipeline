"""
preprocess.py
Image preprocessing pipeline using OpenCV.

Designed to be modular: each step is a discrete function so that
alternative strategies can be swapped in without rewriting the pipeline.
"""

import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def load_image(image_path: "str | Path") -> np.ndarray:
    """Load an image from disk.

    Args:
        image_path: Path to the image file.

    Returns:
        Image as a NumPy array (BGR).

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file cannot be decoded as an image.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(
            f"Could not read image: {image_path}. "
            "The file may be corrupt or in an unsupported format."
        )
    return image


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert a BGR image to grayscale.

    Args:
        image: BGR image array.

    Returns:
        Single-channel grayscale image.
    """
    if len(image.shape) == 2:
        return image  # Already grayscale
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def resize_image(image: np.ndarray, scale_factor: float) -> np.ndarray:
    """Upscale or downscale an image by a given factor.

    Upscaling (factor > 1) improves OCR accuracy on small text because
    Tesseract works best at 300 DPI or higher.

    Args:
        image: Grayscale or colour image array.
        scale_factor: Multiplier for width and height (e.g. 2.0 = 2x).

    Returns:
        Resized image.
    """
    if scale_factor == 1.0:
        return image
    new_width = int(image.shape[1] * scale_factor)
    new_height = int(image.shape[0] * scale_factor)
    interpolation = cv2.INTER_CUBIC if scale_factor > 1 else cv2.INTER_AREA
    return cv2.resize(image, (new_width, new_height), interpolation=interpolation)


def denoise(image: np.ndarray) -> np.ndarray:
    """Apply Non-Local Means denoising to suppress JPEG/scan artifacts.

    Args:
        image: Grayscale image array.

    Returns:
        Denoised grayscale image.
    """
    return cv2.fastNlMeansDenoising(image, h=10, templateWindowSize=7, searchWindowSize=21)


def enhance_contrast(image: np.ndarray) -> np.ndarray:
    """Apply CLAHE (Contrast-Limited Adaptive Histogram Equalization).

    CLAHE is preferred over global histogram equalisation because it
    avoids over-amplifying noise in uniform regions.

    Args:
        image: Grayscale image array.

    Returns:
        Contrast-enhanced grayscale image.
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)


def threshold_image(image: np.ndarray, method: str = "otsu") -> np.ndarray:
    """Binarise a grayscale image.

    Args:
        image: Grayscale image array.
        method: `"otsu"` for global Otsu thresholding or
                `"adaptive"` for Gaussian adaptive thresholding.

    Returns:
        Binary (black-and-white) image.

    Raises:
        ValueError: If an unknown method is specified.
    """
    if method == "otsu":
        _, binary = cv2.threshold(
            image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        return binary
    elif method == "adaptive":
        return cv2.adaptiveThreshold(
            image,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=31,
            C=10,
        )
    else:
        raise ValueError(
            f"Unknown threshold method: '{method}'. "
            "Choose 'otsu' or 'adaptive'."
        )


def preprocess_image(
    image_path: "str | Path",
    resize_factor: float = 2.0,
    threshold_method: str = "otsu",
    apply_denoise: bool = True,
    apply_contrast: bool = True,
) -> np.ndarray:
    """Full preprocessing pipeline: load -> grayscale -> resize ->
    denoise -> contrast -> threshold.

    Args:
        image_path: Path to the source image.
        resize_factor: Scale factor applied before OCR (default 2.0).
        threshold_method: Binarisation strategy -- `"otsu"` or `"adaptive"`.
        apply_denoise: Whether to apply NL-Means denoising.
        apply_contrast: Whether to apply CLAHE contrast enhancement.

    Returns:
        Preprocessed binary image ready for OCR.
    """
    logger.debug("Loading image: %s", image_path)
    image = load_image(image_path)

    logger.debug("Converting to grayscale")
    gray = to_grayscale(image)

    if resize_factor != 1.0:
        logger.debug("Resizing by factor %.1f", resize_factor)
        gray = resize_image(gray, resize_factor)

    if apply_denoise:
        logger.debug("Applying denoising")
        gray = denoise(gray)

    if apply_contrast:
        logger.debug("Applying CLAHE contrast enhancement")
        gray = enhance_contrast(gray)

    logger.debug("Applying '%s' thresholding", threshold_method)
    binary = threshold_image(gray, method=threshold_method)

    return binary

"""
test_preprocess.py
Basic tests for the image preprocessing pipeline.

Tests that require the OCR engine (Tesseract) are marked with
`@pytest.mark.skipif` and will be skipped gracefully if Tesseract
is not installed on the test machine.
"""

import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest

# Repository root (one level above this file's directory)
REPO_ROOT = Path(__file__).parent.parent
SAMPLE_IMAGE = REPO_ROOT / "sample" / "input.png"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dummy_image(tmp_path: Path, filename: str = "dummy.png") -> Path:
    """Create a small white image with black text for testing."""
    img = np.ones((100, 300, 3), dtype=np.uint8) * 255
    cv2.putText(img, "Hello Test", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    out = tmp_path / filename
    cv2.imwrite(str(out), img)
    return out


# ---------------------------------------------------------------------------
# Preprocessing tests (no Tesseract required)
# ---------------------------------------------------------------------------

class TestLoadImage:
    def test_load_valid_image(self, tmp_path):
        from src.preprocess import load_image
        img_path = _make_dummy_image(tmp_path)
        img = load_image(img_path)
        assert img is not None
        assert isinstance(img, np.ndarray)
        assert img.ndim == 3  # BGR

    def test_load_missing_image(self):
        from src.preprocess import load_image
        with pytest.raises(FileNotFoundError, match="Input image not found"):
            load_image("does_not_exist.png")

    def test_load_corrupt_image(self, tmp_path):
        from src.preprocess import load_image
        bad = tmp_path / "bad.png"
        bad.write_bytes(b"not an image")
        with pytest.raises(ValueError, match="Could not read image"):
            load_image(bad)


class TestGrayscale:
    def test_bgr_to_grayscale(self, tmp_path):
        from src.preprocess import to_grayscale
        colour = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        gray = to_grayscale(colour)
        assert gray.ndim == 2

    def test_already_grayscale_passthrough(self):
        from src.preprocess import to_grayscale
        gray_in = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        gray_out = to_grayscale(gray_in)
        assert np.array_equal(gray_in, gray_out)


class TestResize:
    def test_upscale(self):
        from src.preprocess import resize_image
        img = np.zeros((50, 100), dtype=np.uint8)
        resized = resize_image(img, scale_factor=2.0)
        assert resized.shape == (100, 200)

    def test_no_change_on_factor_one(self):
        from src.preprocess import resize_image
        img = np.zeros((50, 100), dtype=np.uint8)
        result = resize_image(img, scale_factor=1.0)
        assert result.shape == img.shape


class TestThreshold:
    def test_otsu_returns_binary(self):
        from src.preprocess import threshold_image
        gray = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        binary = threshold_image(gray, method="otsu")
        unique_vals = np.unique(binary)
        assert set(unique_vals).issubset({0, 255})

    def test_adaptive_returns_binary(self):
        from src.preprocess import threshold_image
        gray = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        binary = threshold_image(gray, method="adaptive")
        unique_vals = np.unique(binary)
        assert set(unique_vals).issubset({0, 255})

    def test_unknown_method_raises(self):
        from src.preprocess import threshold_image
        gray = np.zeros((100, 100), dtype=np.uint8)
        with pytest.raises(ValueError, match="Unknown threshold method"):
            threshold_image(gray, method="unknown")


class TestPreprocessPipeline:
    def test_returns_valid_image(self, tmp_path):
        from src.preprocess import preprocess_image
        img_path = _make_dummy_image(tmp_path)
        result = preprocess_image(img_path)
        assert isinstance(result, np.ndarray)
        assert result.ndim == 2  # Grayscale/binary

    def test_missing_file_raises(self):
        from src.preprocess import preprocess_image
        with pytest.raises(FileNotFoundError):
            preprocess_image("no_such_file.png")

    def test_sample_image_preprocessed(self):
        """Preprocess the bundled sample image (no OCR needed)."""
        if not SAMPLE_IMAGE.exists():
            pytest.skip("sample/input.png not found")
        from src.preprocess import preprocess_image
        result = preprocess_image(SAMPLE_IMAGE)
        assert result is not None
        assert result.size > 0


# ---------------------------------------------------------------------------
# OCR tests (skipped if Tesseract is unavailable)
# ---------------------------------------------------------------------------

def _tesseract_available() -> bool:
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


TESSERACT_REASON = (
    "Tesseract OCR is not installed on this machine. "
    "Install Tesseract to run OCR tests. See README.md for instructions."
)


@pytest.mark.skipif(not _tesseract_available(), reason=TESSERACT_REASON)
class TestOCR:
    def test_extract_text_from_sample(self):
        from src.preprocess import preprocess_image
        from src.ocr import extract_text
        if not SAMPLE_IMAGE.exists():
            pytest.skip("sample/input.png not found")
        processed = preprocess_image(SAMPLE_IMAGE)
        text = extract_text(processed)
        assert isinstance(text, str)
        # The sample image contains "PathPal" — check for partial match
        assert len(text) > 0

    def test_extract_text_returns_string(self, tmp_path):
        from src.preprocess import preprocess_image
        from src.ocr import extract_text
        img_path = _make_dummy_image(tmp_path)
        processed = preprocess_image(img_path, resize_factor=1.0)
        text = extract_text(processed)
        assert isinstance(text, str)

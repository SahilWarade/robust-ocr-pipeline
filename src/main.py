"""
main.py
Entry point for the OCR pipeline.

Run from the repository root:
    python -m src.main --image sample/input.png
    python -m src.main --image sample/input.png --output sample/output.txt
"""

import argparse
import logging
import sys
from pathlib import Path

from src.preprocess import preprocess_image
from src.ocr import extract_text

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Robust OCR Pipeline: extract text from an image using OpenCV + Tesseract."
    )
    parser.add_argument(
        "--image",
        required=True,
        metavar="PATH",
        help="Path to the input image.",
    )
    parser.add_argument(
        "--output",
        metavar="PATH",
        help="Optional path to save extracted text as a .txt file.",
    )
    parser.add_argument(
        "--resize-factor",
        type=float,
        default=2.0,
        metavar="FACTOR",
        help="Upscale factor applied before OCR (default: 2.0).",
    )
    parser.add_argument(
        "--threshold",
        choices=["otsu", "adaptive"],
        default="otsu",
        help="Thresholding method (default: otsu).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose debug logging.",
    )
    return parser.parse_args(argv)


def run_pipeline(
    image_path: str,
    resize_factor: float = 2.0,
    threshold_method: str = "otsu",
) -> str:
    """Preprocess the image and extract text.

    Args:
        image_path: Path to the input image.
        resize_factor: Upscale factor for the image.
        threshold_method: `"otsu"` or `"adaptive"`.

    Returns:
        Extracted text string.
    """
    logger.info("Preprocessing image: %s", image_path)
    processed = preprocess_image(
        image_path,
        resize_factor=resize_factor,
        threshold_method=threshold_method,
    )

    logger.info("Running OCR...")
    text = extract_text(processed)
    return text


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # --- Input validation ---
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: Input image not found: {args.image}", file=sys.stderr)
        return 1

    # --- Pipeline ---
    try:
        text = run_pipeline(
            str(image_path),
            resize_factor=args.resize_factor,
            threshold_method=args.threshold,
        )
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except EnvironmentError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    # --- Output ---
    print("\n--- Extracted Text ---")
    print(text if text else "(no text detected)")
    print("----------------------\n")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
        logger.info("Saved output to: %s", output_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())

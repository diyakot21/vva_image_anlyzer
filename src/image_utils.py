"""Simple image utilities for basic operations."""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional
from PIL import Image


def save_image_simple(image: np.ndarray, output_path: Path) -> bool:
    """
    Save image using basic OpenCV method.

    Args:
        image: Image array to save
        output_path: Where to save the image

    Returns:
        True if successful, False otherwise
    """
    try:
        # Simple OpenCV save
        success = cv2.imwrite(str(output_path), image)
        return bool(success)

    except Exception as e:
        print(f"Error saving image: {e}")
        return False


def get_image_properties(image_path: Path) -> dict:
    """
    Get basic properties of an image file.

    Args:
        image_path: Path to the image

    Returns:
        Dictionary with image properties
    """
    properties = {}

    try:
        # Get file size
        properties["file_size_mb"] = image_path.stat().st_size / (1024 * 1024)

        # Use PIL for basic properties
        with Image.open(image_path) as img:
            properties.update(
                {
                    "format": img.format,
                    "mode": img.mode,
                    "size": img.size,
                    "width": img.width,
                    "height": img.height,
                }
            )

        # Use OpenCV to check actual loaded data
        cv_image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if cv_image is not None:
            properties.update(
                {
                    "cv_shape": cv_image.shape,
                    "cv_dtype": str(cv_image.dtype),
                    "cv_channels": len(cv_image.shape),
                    "cv_depth": cv_image.dtype.itemsize * 8,  # bits per pixel
                }
            )

    except Exception as e:
        properties["error"] = str(e)

    return properties


def compare_image_properties(original_path: Path, processed_path: Path) -> dict:
    """
    Compare properties between original and processed images.

    Args:
        original_path: Path to original image
        processed_path: Path to processed image

    Returns:
        Dictionary with comparison results
    """
    original_props = get_image_properties(original_path)
    processed_props = get_image_properties(processed_path)

    comparison = {
        "original": original_props,
        "processed": processed_props,
        "size_reduction_percent": 0,
        "quality_preserved": True,
        "warnings": [],
    }

    # Calculate size reduction
    if "file_size_mb" in original_props and "file_size_mb" in processed_props:
        orig_size = original_props["file_size_mb"]
        proc_size = processed_props["file_size_mb"]
        reduction = ((orig_size - proc_size) / orig_size) * 100
        comparison["size_reduction_percent"] = reduction

        if reduction > 50:
            comparison["warnings"].append(
                f"Large file size reduction: {reduction:.1f}%"
            )
            comparison["quality_preserved"] = False

    # Check dimensions
    if original_props.get("size") != processed_props.get("size"):
        comparison["warnings"].append("Image dimensions changed")
        comparison["quality_preserved"] = False

    # Check bit depth
    if original_props.get("cv_dtype") != processed_props.get("cv_dtype"):
        comparison["warnings"].append("Image bit depth changed")
        comparison["quality_preserved"] = False

    return comparison

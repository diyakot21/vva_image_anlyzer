"""Image reader module for loading images from local directories."""

import os
from pathlib import Path
from typing import Generator, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image


class ImageReader:
    """
    A class to read and load images from local directories.

    Supports common image formats (jpg, jpeg, png, tiff, bmp).
    """

    SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp"}

    def __init__(self, directory_path: str):
        """
        Initialize the ImageReader with a directory path.

        Args:
            directory_path: Path to the directory containing images

        Raises:
            ValueError: If directory doesn't exist or is not a directory
        """
        self.directory_path = Path(directory_path)
        if not self.directory_path.exists():
            raise ValueError(f"Directory {directory_path} does not exist")
        if not self.directory_path.is_dir():
            raise ValueError(f"{directory_path} is not a directory")

    def get_image_paths(self) -> List[Path]:
        """
        Get all supported image file paths from the directory.

        Returns:
            List of Path objects for supported image files
        """
        image_paths = []
        for file_path in self.directory_path.iterdir():
            if (
                file_path.is_file()
                and file_path.suffix.lower() in self.SUPPORTED_FORMATS
            ):
                image_paths.append(file_path)
        return sorted(image_paths)

    def load_image(
        self, image_path: Path, color_mode: str = "color"
    ) -> Optional[np.ndarray]:
        """
        Load a single image from the given path.

        Args:
            image_path: Path to the image file
            color_mode: 'color', 'grayscale', or 'unchanged'

        Returns:
            Loaded image as numpy array, or None if loading failed
        """
        try:
            if color_mode == "color":
                image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            elif color_mode == "grayscale":
                image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            elif color_mode == "unchanged":
                image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
            else:
                raise ValueError(f"Unsupported color mode: {color_mode}")

            if image is None:
                print(f"Warning: Could not load image {image_path}")
                return None

            return image
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None

    def load_all_images(
        self, color_mode: str = "color"
    ) -> Generator[Tuple[Path, np.ndarray], None, None]:
        """
        Generator to load all images from the directory one by one.

        Args:
            color_mode: 'color', 'grayscale', or 'unchanged'

        Yields:
            Tuple of (image_path, image_array) for successfully loaded images
        """
        image_paths = self.get_image_paths()

        for image_path in image_paths:
            image = self.load_image(image_path, color_mode)
            if image is not None:
                yield image_path, image

    def get_image_info(self, image_path: Path) -> Optional[dict]:
        """
        Get basic information about an image file.

        Args:
            image_path: Path to the image file

        Returns:
            Dictionary with image information or None if file cannot be read
        """
        try:
            with Image.open(image_path) as img:
                return {
                    "filename": image_path.name,
                    "format": img.format,
                    "mode": img.mode,
                    "size": img.size,
                    "width": img.width,
                    "height": img.height,
                }
        except Exception as e:
            print(f"Error getting info for {image_path}: {e}")
            return None

    def count_images(self) -> int:
        """
        Count the number of supported image files in the directory.

        Returns:
            Number of supported image files
        """
        return len(self.get_image_paths())

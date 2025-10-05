"""Image analyzer module for brain image processing using OpenCV Hough Circle Transform."""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class CircleDetectionResult:
    """
    Data class to store circle detection results.

    Attributes:
        image_path: Path to the analyzed image
        circles: List of detected circles (x, y, radius)
        circle_count: Number of circles detected
        processing_params: Parameters used for detection
    """

    image_path: str
    circles: List[Tuple[int, int, int]]  # (x, y, radius)
    circle_count: int
    processing_params: dict


class BrainImageAnalyzer:
    """
    Analyzer for detecting circular structures in brain images using Hough Circle Transform.

    This class is designed to identify PV protein development in mice brain images
    by detecting circular patterns that may indicate protein clusters.
    """

    def __init__(
        self,
        dp: float = 1.0,
        min_dist: int = 50,
        param1: int = 50,
        param2: int = 30,
        min_radius: int = 10,
        max_radius: int = 100,
    ):
        """
        Initialize the brain image analyzer with Hough Circle Transform parameters.

        Args:
            dp: Inverse ratio of accumulator resolution to image resolution
            min_dist: Minimum distance between circle centers
            param1: Upper threshold for edge detection (Canny)
            param2: Accumulator threshold for center detection
            min_radius: Minimum circle radius to detect
            max_radius: Maximum circle radius to detect
        """
        self.dp = dp
        self.min_dist = min_dist
        self.param1 = param1
        self.param2 = param2
        self.min_radius = min_radius
        self.max_radius = max_radius

    def preprocess_image(
        self,
        image: np.ndarray,
        gaussian_blur_kernel: int = 9,
        gaussian_sigma: float = 2.0,
    ) -> np.ndarray:
        """
        Preprocess the image for better circle detection.

        Args:
            image: Input image (color or grayscale)
            gaussian_blur_kernel: Kernel size for Gaussian blur (must be odd)
            gaussian_sigma: Standard deviation for Gaussian blur

        Returns:
            Preprocessed grayscale image
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(
            gray, (gaussian_blur_kernel, gaussian_blur_kernel), gaussian_sigma
        )

        # Apply histogram equalization to improve contrast
        equalized = cv2.equalizeHist(blurred)

        return equalized

    def detect_circles(
        self, image: np.ndarray, preprocess: bool = True
    ) -> Optional[np.ndarray]:
        """
        Detect circles in the image using Hough Circle Transform.

        Args:
            image: Input image
            preprocess: Whether to apply preprocessing

        Returns:
            Array of detected circles (x, y, radius) or None if no circles found
        """
        # Preprocess image if requested
        if preprocess:
            processed_image = self.preprocess_image(image)
        else:
            processed_image = (
                image
                if len(image.shape) == 2
                else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            )

        # Apply Hough Circle Transform
        circles = cv2.HoughCircles(
            processed_image,
            cv2.HOUGH_GRADIENT,
            dp=self.dp,
            minDist=self.min_dist,
            param1=self.param1,
            param2=self.param2,
            minRadius=self.min_radius,
            maxRadius=self.max_radius,
        )

        return circles

    def analyze_image(
        self, image: np.ndarray, image_path: str = "unknown"
    ) -> CircleDetectionResult:
        """
        Perform complete analysis on a single image.

        Args:
            image: Input image to analyze
            image_path: Path or identifier for the image

        Returns:
            CircleDetectionResult with analysis results
        """
        circles = self.detect_circles(image)

        # Process detected circles
        circle_list = []
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for x, y, r in circles:
                circle_list.append((x, y, r))

        # Store processing parameters used
        processing_params = {
            "dp": self.dp,
            "min_dist": self.min_dist,
            "param1": self.param1,
            "param2": self.param2,
            "min_radius": self.min_radius,
            "max_radius": self.max_radius,
        }

        return CircleDetectionResult(
            image_path=image_path,
            circles=circle_list,
            circle_count=len(circle_list),
            processing_params=processing_params,
        )

    def draw_circles(
        self,
        image: np.ndarray,
        circles: List[Tuple[int, int, int]],
        circle_color: Tuple[int, int, int] = (0, 255, 0),
        center_color: Tuple[int, int, int] = (0, 0, 255),
        thickness: int = 2,
    ) -> np.ndarray:
        """
        Draw detected circles on the image for visualization.

        Args:
            image: Input image
            circles: List of circles (x, y, radius)
            circle_color: Color for circle outline (BGR)
            center_color: Color for circle center (BGR)
            thickness: Line thickness for drawing

        Returns:
            Image with circles drawn
        """
        output_image = image.copy()

        for x, y, r in circles:
            # Draw circle outline
            cv2.circle(output_image, (x, y), r, circle_color, thickness)
            # Draw center point
            cv2.circle(output_image, (x, y), 2, center_color, 3)

        return output_image

    def update_parameters(self, **kwargs):
        """
        Update detection parameters.

        Args:
            **kwargs: Parameter names and values to update
        """
        for param, value in kwargs.items():
            if hasattr(self, param):
                setattr(self, param, value)
            else:
                raise ValueError(f"Unknown parameter: {param}")

    def get_circle_statistics(self, results: List[CircleDetectionResult]) -> dict:
        """
        Calculate statistics across multiple analysis results.

        Args:
            results: List of CircleDetectionResult objects

        Returns:
            Dictionary with statistical information
        """
        if not results:
            return {}

        circle_counts = [result.circle_count for result in results]
        all_radii = []

        for result in results:
            for _, _, radius in result.circles:
                all_radii.append(radius)

        statistics = {
            "total_images": len(results),
            "total_circles": sum(circle_counts),
            "avg_circles_per_image": np.mean(circle_counts) if circle_counts else 0,
            "min_circles_per_image": min(circle_counts) if circle_counts else 0,
            "max_circles_per_image": max(circle_counts) if circle_counts else 0,
            "avg_radius": np.mean(all_radii) if all_radii else 0,
            "min_radius": min(all_radii) if all_radii else 0,
            "max_radius": max(all_radii) if all_radii else 0,
            "std_radius": np.std(all_radii) if all_radii else 0,
        }

        return statistics

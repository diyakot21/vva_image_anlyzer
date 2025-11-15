"""Test script to analyze the existing images in test_images folder."""

import pytest
from pathlib import Path
import cv2
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pnn_analyzer import PNNAnalyzer, PNNDetectionResult
from src.image_reader import ImageReader


class TestImageAnalysis:
    """Test class for image analysis functionality."""

    @classmethod
    def setup_class(cls):
        """Set up test fixtures."""
        cls.test_images_dir = Path(__file__).parent / "test_images"
        cls.analyzer = PNNAnalyzer(
            min_pnn_radius_mm=0.005,
            max_pnn_radius_mm=0.050,
            pixel_size_mm=0.001
        )
        cls.reader = ImageReader(str(cls.test_images_dir))

    def test_simple_circles_detection(self):
        """Test circle detection on simple_circles.png if it exists."""
        simple_circles_path = self.test_images_dir / "simple_circles.png"

        image = self.reader.load_image(simple_circles_path)
        assert image is not None, "Should be able to load simple_circles.png"

        result = self.analyzer.analyze_image(image, str(simple_circles_path))

        assert isinstance(result, PNNDetectionResult)
        assert result.pnn_count == 4, "Circle count should be 4"
        assert (
            len(result.pnn_circles) == result.pnn_count
        ), "Circle list length should match count"

    def test_brain_like_detection(self):
        """Test circle detection on brain_like.png if it exists."""
        brain_like_path = self.test_images_dir / "brain_like.png"

        if not brain_like_path.exists():
            pytest.skip("brain_like.png not found")

        image = self.reader.load_image(brain_like_path)
        assert image is not None, "Should be able to load brain_like.png"

        result = self.analyzer.analyze_image(image, str(brain_like_path))

        assert isinstance(result, PNNDetectionResult)
        assert result.pnn_count >= 0, "Circle count should be non-negative"

        print(f"brain_like.png: Detected {result.pnn_count} circles")
        if result.pnn_count <= 20:
            for i, (x, y, r) in enumerate(result.pnn_circles):
                print(f"  Circle {i+1}: ({x}, {y}, radius={r})")
        else:
            print(f"  (Too many circles to list individually)")

    def test_all_test_images(self):
        """Test circle detection on all available test images."""
        results = []

        for image_path, image in self.reader.load_all_images():
            # Skip annotated images
            if "annotated" in image_path.name:
                continue

            print(f"\nTesting: {image_path.name}")

            result = self.analyzer.analyze_image(image, str(image_path))
            results.append((image_path.name, result))

            # Basic assertions
            assert isinstance(result, PNNDetectionResult)
            assert result.pnn_count >= 0
            assert len(result.pnn_circles) == result.pnn_count
            assert result.image_path == str(image_path)

            print(f"  Detected: {result.pnn_count} circles")
            print(f"  Image size: {image.shape}")

            # Test circle coordinates are within reasonable bounds
            for x, y, r in result.pnn_circles:
                assert (
                    -r <= x <= image.shape[1] + r
                ), f"Circle x={x} too far outside image width {image.shape[1]}"
                assert (
                    -r <= y <= image.shape[0] + r
                ), f"Circle y={y} too far outside image height {image.shape[0]}"
                assert r > 0, f"Circle radius {r} should be positive"

        assert len(results) > 0, "Should have processed at least one image"

        # Summary statistics
        total_circles = sum(result.pnn_count for _, result in results)
        avg_circles = total_circles / len(results) if results else 0

        print(f"\n=== Test Summary ===")
        print(f"Images tested: {len(results)}")
        print(f"Total circles detected: {total_circles}")
        print(f"Average circles per image: {avg_circles:.2f}")

    def test_analyzer_parameters(self):
        """Test that analyzer parameters are properly set."""
        assert self.analyzer.pixel_size_mm == 0.001
        assert self.analyzer.min_pnn_radius_mm == 0.005
        assert self.analyzer.max_pnn_radius_mm == 0.050

    def test_draw_circles_functionality(self):
        """Test the circle drawing functionality."""
        test_image = np.ones((200, 300, 3), dtype=np.uint8) * 255
        test_circles = [(100, 100, 30), (200, 150, 20)]

        annotated = self.analyzer.draw_pnn_detections(test_image, test_circles)

        assert annotated.shape == test_image.shape
        assert annotated.dtype == test_image.dtype
        assert not np.array_equal(
            annotated, test_image
        ), "Image should be modified after drawing circles"

    def test_statistics_calculation(self):
        """Test basic statistics from results."""
        mock_results = [
            PNNDetectionResult("image1.png", [(10, 10, 5), (20, 20, 10)], [], 2, [0.8, 0.9], {}),
            PNNDetectionResult("image2.png", [(30, 30, 15)], [], 1, [0.7], {}),
            PNNDetectionResult("image3.png", [], [], 0, [], {}),
        ]

        total_pnns = sum(r.pnn_count for r in mock_results)
        avg_pnns = total_pnns / len(mock_results)

        assert total_pnns == 3
        assert avg_pnns == 1.0


def run_interactive_test():
    """Run an interactive test with detailed output."""
    print("🧪 Interactive Test of test_images Folder")
    print("=" * 50)

    test_images_dir = Path(__file__).parent / "test_images"

    if not test_images_dir.exists():
        print("❌ test_images directory not found!")
        return

    # Initialize components
    reader = ImageReader(str(test_images_dir))
    analyzer = PNNAnalyzer(
        min_pnn_radius_mm=0.005,
        max_pnn_radius_mm=0.050,
        pixel_size_mm=0.001
    )

    # Create output directory for annotated results
    output_dir = Path("test_results")
    output_dir.mkdir(exist_ok=True)

    image_paths = reader.get_image_paths()
    print(f"Found {len(image_paths)} images:")

    results = []

    for image_path in image_paths:
        # Skip already annotated images
        if "annotated" in image_path.name:
            print(f"  📷 {image_path.name} (skipping - already annotated)")
            continue

        print(f"\n📷 Testing: {image_path.name}")

        # Load and analyze image
        image = reader.load_image(image_path)
        if image is None:
            print("  ❌ Failed to load image")
            continue

        result = analyzer.analyze_image(image, str(image_path))
        results.append(result)

        print(f"  📊 Image size: {image.shape[1]}x{image.shape[0]} pixels")
        print(f"  🎯 Detected: {result.pnn_count} circles")

        # Show circle details
        if result.pnn_count > 0:
            if result.pnn_count <= 10:
                print("  📍 Circle positions (x, y, radius):")
                for i, (x, y, r) in enumerate(result.pnn_circles):
                    print(f"     {i+1}. ({x}, {y}, {r})")
            else:
                print("  📍 First 5 circles (x, y, radius):")
                for i, (x, y, r) in enumerate(result.pnn_circles[:5]):
                    print(f"     {i+1}. ({x}, {y}, {r})")
                print(f"     ... and {result.pnn_count - 5} more circles")

            # Create annotated image
            annotated = analyzer.draw_pnn_detections(image, result.pnn_circles)
            output_path = output_dir / f"test_result_{image_path.name}"
            cv2.imwrite(str(output_path), annotated)
            print(f"  💾 Saved annotated image: {output_path}")
        else:
            print("  ⚠️  No circles detected")

    # Overall statistics
    if results:
        total_pnns = sum(r.pnn_count for r in results)
        avg_pnns = total_pnns / len(results)
        pnn_counts = [r.pnn_count for r in results]

        print(f"\n📈 Overall Statistics:")
        print(f"   Images processed: {len(results)}")
        print(f"   Total circles found: {total_pnns}")
        print(f"   Average per image: {avg_pnns:.2f}")
        print(f"   Circle count range: {min(pnn_counts)} - {max(pnn_counts)}")

        print(f"\n📁 Annotated results saved in: {output_dir.absolute()}")
    else:
        print("\n❌ No images were successfully processed")


if __name__ == "__main__":
    # Run the interactive test
    run_interactive_test()

    print(f"\n🧪 To run formal tests, use:")
    print(f"   pytest test_existing_images.py -v")
    print(f"   or")
    print(f"   uv run pytest test_existing_images.py -v")

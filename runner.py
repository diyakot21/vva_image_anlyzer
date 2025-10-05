"""PNN detection runner (simplified)."""

import cv2
import numpy as np
from pathlib import Path

from src.pnn_analyzer import PNNAnalyzer as BasePNNAnalyzer
from src.image_reader import ImageReader


class PNNAnalyzer(BasePNNAnalyzer):
    """PNNAnalyzer with preferred thresholds (renamed from OptimalPNNAnalyzer)."""

    def __init__(self):
        super().__init__(
            min_pnn_radius=5,
            max_pnn_radius=65,
            contrast_threshold=1.05,
            uniformity_threshold=0.18,
            template_threshold=0.27,
            center_darkness_threshold=0.70,
            use_clahe=True,
            clahe_clip_limit=2.5,
            clahe_tile_grid=(8, 8),
            apply_background_subtraction=True,
            background_blur_radius=55,
        )


def main():
    reader = ImageReader("./images")
    output_dir = Path("./output")

    for image_path, image in reader.load_all_images(color_mode="color"):
        print(f"\n{'='*60}")
        print(f"PNN DETECTION: {image_path.name}")
        print(f"{'='*60}")

        analyzer = PNNAnalyzer()
        print(f"\n⚙️  PARAMETERS:")
        print(f"   Template threshold: {analyzer.template_threshold}")
        print(f"   Contrast threshold: {analyzer.contrast_threshold}")
        print(f"   Uniformity threshold: {analyzer.uniformity_threshold}")
        print(f"   Center darkness max ratio: {analyzer.center_darkness_threshold}")

        result = analyzer.analyze_image(image, str(image_path))
        print(f"\n📊 RESULTS:")
        print(f"   PNNs detected: {result.pnn_count}")
        if result.quality_scores:
            avg_quality = float(np.mean(result.quality_scores))
            min_quality = float(min(result.quality_scores))
            max_quality = float(max(result.quality_scores))
            print(
                f"   Quality scores: avg={avg_quality:.3f}, range={min_quality:.3f}-{max_quality:.3f}"
            )
            excellent = sum(1 for q in result.quality_scores if q > 0.8)
            good = sum(1 for q in result.quality_scores if 0.7 < q <= 0.8)
            moderate = sum(1 for q in result.quality_scores if 0.6 < q <= 0.7)
            poor = sum(1 for q in result.quality_scores if q <= 0.6)
            print(f"\n📈 QUALITY BREAKDOWN:")
            print(f"   Excellent (>0.8): {excellent}")
            print(f"   Good (0.7-0.8): {good}")
            print(f"   Moderate (0.6-0.7): {moderate}")
            print(f"   Poor (≤0.6): {poor}")

        annotated = analyzer.draw_pnn_detections(
            image, result.pnn_circles, result.quality_scores
        )
        for i, (x, y, r) in enumerate(result.pnn_circles):
            if i < len(result.quality_scores):
                q = result.quality_scores[i]
                if q > 0.8:
                    color = (0, 255, 0)
                elif q > 0.7:
                    color = (0, 255, 255)
                elif q > 0.6:
                    color = (255, 255, 0)
                else:
                    color = (255, 0, 0)
            else:
                color = (128, 128, 128)
            cv2.circle(annotated, (x, y), r, color, 2)
            cv2.circle(annotated, (x, y), 2, (0, 0, 255), -1)

        output_path = output_dir / f"pnn_{image_path.name}"
        cv2.imwrite(str(output_path), annotated)
        print(f"\n💾 PNN detection saved: {output_path}")
        print(f"\n🎨 Color Legend: 🟢 >0.8  🟡 0.7-0.8  🔵 0.6-0.7  🔴 ≤0.6")


if __name__ == "__main__":
    main()

"""Perineuronal net (PNN) detection using Hough circles and quality assessment."""

from dataclasses import dataclass
from typing import List, Tuple, Optional

import cv2
import numpy as np


@dataclass
class PNNDetectionResult:
    image_path: str
    pnn_circles: List[Tuple[int, int, int]]  # (x, y, r) in pixels
    pnn_circles_mm: List[Tuple[float, float, float]]  # (x, y, r) in millimeters
    pnn_count: int
    quality_scores: List[float]
    detection_params: dict


class PNNAnalyzer:
    def __init__(
        self,
        min_pnn_radius_mm: float = 0.008,  # Default ~8 pixels at 1 micron/pixel
        max_pnn_radius_mm: float = 0.035,  # Default ~35 pixels at 1 micron/pixel
        pixel_size_mm: float = 0.001,  # Default: 1 micron per pixel
        contrast_threshold: float = 1.3,
        uniformity_threshold: float = 0.2,
        template_threshold: float = 0.32,
        center_darkness_threshold: float = 0.75,
        use_clahe: bool = True,
        clahe_clip_limit: float = 2.0,
        clahe_tile_grid: Tuple[int, int] = (8, 8),
        apply_background_subtraction: bool = True,
        background_blur_radius: int = 45,
    ) -> None:
        """Initialize PNN analyzer with millimeter-based size parameters.
        
        Radii are specified in mm and converted to pixels using pixel_size_mm.
        Quality thresholds control detection sensitivity.
        """
        if pixel_size_mm <= 0:
            raise ValueError("pixel_size_mm must be > 0")
        
        self.pixel_size_mm = pixel_size_mm
        self.min_pnn_radius_mm = min_pnn_radius_mm
        self.max_pnn_radius_mm = max_pnn_radius_mm
        
        # Convert mm to pixels for internal processing
        self.min_pnn_radius = max(1, int(round(min_pnn_radius_mm / pixel_size_mm)))
        self.max_pnn_radius = max(1, int(round(max_pnn_radius_mm / pixel_size_mm)))
        self.contrast_threshold = contrast_threshold
        self.uniformity_threshold = uniformity_threshold
        self.template_threshold = template_threshold
        self.center_darkness_threshold = center_darkness_threshold
        self.use_clahe = use_clahe
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_grid = clahe_tile_grid
        self.apply_background_subtraction = apply_background_subtraction
        self.background_blur_radius = background_blur_radius

    def preprocess_for_pnn(self, image: np.ndarray) -> np.ndarray:
        """Convert to grayscale, denoise, subtract background, and apply CLAHE."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        if gray.dtype == np.uint16:
            max_val = float(gray.max()) if gray.max() > 0 else 1.0
            gray = cv2.convertScaleAbs(gray, alpha=255.0 / max_val)
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)

        if self.apply_background_subtraction:
            blur_base = max(5, self.background_blur_radius)
            k = blur_base if blur_base % 2 == 1 else blur_base + 1
            background = cv2.GaussianBlur(denoised, (k, k), 0)
            subtracted = cv2.subtract(denoised, background)
            norm = cv2.normalize(subtracted, None, 0, 255, cv2.NORM_MINMAX)
        else:
            norm = denoised

        if self.use_clahe:
            clahe = cv2.createCLAHE(
                clipLimit=self.clahe_clip_limit, tileGridSize=self.clahe_tile_grid
            )
            enhanced = clahe.apply(norm)
        else:
            enhanced = norm
        return enhanced

    def _create_ring_template(
        self, outer_radius: int = 15, inner_radius: int = 8
    ) -> np.ndarray:
        """Create a ring-shaped template for template matching."""
        size = outer_radius * 2 + 1
        tmpl = np.zeros((size, size), dtype=np.float32)
        cv2.circle(tmpl, (outer_radius, outer_radius), outer_radius, 1.0, -1)
        cv2.circle(tmpl, (outer_radius, outer_radius), inner_radius, 0.0, -1)
        tmpl = cv2.GaussianBlur(tmpl, (3, 3), 1)
        return cv2.normalize(tmpl, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    def detect_candidate_circles(self, image: np.ndarray) -> List[Tuple[int, int, int]]:
        """Find candidate circles using Hough transform and template matching."""
        candidates: List[Tuple[int, int, int]] = []

        # Hough circle detection
        circles = cv2.HoughCircles(
            image,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=16,
            param1=70,
            param2=22,
            minRadius=self.min_pnn_radius,
            maxRadius=self.max_pnn_radius,
        )
        if circles is not None:
            for x, y, r in np.round(circles[0, :]).astype(int):
                candidates.append((x, y, r))
        
        # Template matching
        base_tmpl = self._create_ring_template()
        base_resp = cv2.matchTemplate(image, base_tmpl, cv2.TM_CCOEFF_NORMED)
        ys, xs = np.where(base_resp >= self.template_threshold)
        br = base_tmpl.shape[0] // 2
        for x0, y0 in zip(xs, ys):
            candidates.append((x0 + br, y0 + br, br))

        # De-duplication
        uniq = {}
        for cx, cy, r in candidates:
            key = (int(cx / 4), int(cy / 4))
            if key not in uniq or r < uniq[key][2]:
                uniq[key] = (cx, cy, r)
        return list(uniq.values())

    def analyze_pnn_quality(
        self, image: np.ndarray, x: int, y: int, r: int
    ) -> Tuple[bool, float, dict]:
        """Score a candidate circle based on contrast, uniformity, and center darkness."""
        h, w = image.shape[:2]
        if x - r < 0 or x + r >= w or y - r < 0 or y + r >= h:
            return False, 0.0, {}

        outer_mask = np.zeros_like(image, dtype=np.uint8)
        inner_mask = np.zeros_like(image, dtype=np.uint8)
        cv2.circle(outer_mask, (x, y), int(r), 255, -1)
        inner_r = max(1, int(r * 0.4))
        cv2.circle(inner_mask, (x, y), inner_r, 255, -1)
        ring_mask = outer_mask - inner_mask
        ring_pixels = image[ring_mask > 0]
        inner_pixels = image[inner_mask > 0]
        if ring_pixels.size == 0 or inner_pixels.size == 0:
            return False, 0.0, {}

        ring_mean = float(ring_pixels.mean())
        inner_mean = float(inner_pixels.mean())
        ring_std = float(ring_pixels.std())
        background_mean = float(image.mean())
        contrast_ratio = ring_mean / (inner_mean + 1e-6)
        ring_uniformity = 1 - (ring_std / (ring_mean + 1e-6))
        signal_to_background = ring_mean / (background_mean + 1e-6)
        center_darkness_ratio = inner_mean / (ring_mean + 1e-6)
        size_score = 1.0 if self.min_pnn_radius <= r <= self.max_pnn_radius else 0.5

        stats = dict(
            ring_mean=ring_mean,
            inner_mean=inner_mean,
            contrast_ratio=contrast_ratio,
            ring_uniformity=ring_uniformity,
            signal_to_background=signal_to_background,
            center_darkness_ratio=center_darkness_ratio,
            size_score=size_score,
            radius=r,
        )

        is_valid = (
            contrast_ratio > self.contrast_threshold
            and ring_uniformity > self.uniformity_threshold
            and signal_to_background > 1.02
            and center_darkness_ratio < self.center_darkness_threshold
            and size_score > 0.5
        )

        quality = (
            min(contrast_ratio / 2.0, 1.0) * 0.4
            + ring_uniformity * 0.25
            + min(signal_to_background / 2.0, 1.0) * 0.15
            + (1 - center_darkness_ratio) * 0.15
            + size_score * 0.05
        )
        return is_valid, float(quality), stats

    def remove_overlapping_detections(
        self, candidates: List[Tuple[int, int, int]], scores: List[float]
    ) -> List[Tuple[int, int, int]]:
        """Remove overlapping detections, keeping higher-scoring circles."""
        if len(candidates) <= 1:
            return candidates
        order = np.argsort(scores)[::-1]
        kept: List[Tuple[int, int, int]] = []
        for idx in order:
            x1, y1, r1 = candidates[idx]
            keep = True
            for x2, y2, r2 in kept:
                d = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
                if d < min(r1, r2) * 0.6:
                    keep = False
                    break
            if keep:
                kept.append((x1, y1, r1))
        return kept

    def analyze_image(
        self, image: np.ndarray, image_path: str = "unknown"
    ) -> PNNDetectionResult:
        """Detect PNNs in an image and return results in both pixels and millimeters."""
        processed = self.preprocess_for_pnn(image)
        candidates = self.detect_candidate_circles(processed)
        if len(candidates) > 4000:
            candidates = candidates[:4000]

        valid: List[Tuple[int, int, int]] = []
        scores: List[float] = []
        for x, y, r in candidates:
            ok, q, _ = self.analyze_pnn_quality(processed, x, y, r)
            if ok:
                valid.append((x, y, r))
                scores.append(q)
        final = self.remove_overlapping_detections(valid, scores)
        final_scores: List[float] = []
        for x, y, r in final:
            _, q, _ = self.analyze_pnn_quality(processed, x, y, r)
            final_scores.append(q)

        # Convert pixel coordinates to mm
        pnn_circles_mm = [
            (float(x) * self.pixel_size_mm, 
             float(y) * self.pixel_size_mm, 
             float(r) * self.pixel_size_mm)
            for x, y, r in final
        ]
        
        params = dict(
            min_pnn_radius_mm=self.min_pnn_radius_mm,
            max_pnn_radius_mm=self.max_pnn_radius_mm,
            pixel_size_mm=self.pixel_size_mm,
            min_pnn_radius_px=self.min_pnn_radius,
            max_pnn_radius_px=self.max_pnn_radius,
            contrast_threshold=self.contrast_threshold,
            uniformity_threshold=self.uniformity_threshold,
            template_threshold=self.template_threshold,
            use_clahe=self.use_clahe,
            apply_background_subtraction=self.apply_background_subtraction,
            center_darkness_threshold=self.center_darkness_threshold,
        )
        return PNNDetectionResult(
            image_path=image_path,
            pnn_circles=final,
            pnn_circles_mm=pnn_circles_mm,
            pnn_count=len(final),
            quality_scores=final_scores,
            detection_params=params,
        )

    def draw_pnn_detections(
        self,
        image: np.ndarray,
        pnn_circles: List[Tuple[int, int, int]],
        quality_scores: Optional[List[float]] = None,
    ) -> np.ndarray:
        """Draw detected circles on the image with quality-based coloring."""
        out = image.copy()
        if len(out.shape) == 2:
            out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
        for i, (x, y, r) in enumerate(pnn_circles):
            color = (0, 255, 0)
            cv2.circle(out, (x, y), int(r), color, 2)
            cv2.circle(out, (x, y), 2, (0, 0, 255), -1)
        return out

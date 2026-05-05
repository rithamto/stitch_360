import cv2
import numpy as np
from typing import Tuple

class SphericalWarper:
    def __init__(self, focal_length: float):
        self.focal_length = focal_length
        self.warper = cv2.PyRotationWarper("spherical", focal_length)

    def warp_point(self, pt: Tuple[float, float], K: np.ndarray, R: np.ndarray) -> Tuple[float, float]:
        return self.warper.warpPoint(pt, K, R)

    def warp_points(self, pts: np.ndarray, K: np.ndarray, R: np.ndarray) -> np.ndarray:
        if len(pts) == 0:
            return np.empty((0, 2), dtype=np.float32)
            
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        
        # Normalize to camera coordinates
        x_norm = (pts[:, 0] - cx) / fx
        y_norm = (pts[:, 1] - cy) / fy
        
        # To 3D rays
        rays = np.stack([x_norm, y_norm, np.ones(len(pts))], axis=1)  # (N, 3)
        
        # Apply rotation
        rays_rot = (R @ rays.T).T  # (N, 3)
        
        # Spherical projection
        x3, y3, z3 = rays_rot[:, 0], rays_rot[:, 1], rays_rot[:, 2]
        
        theta = np.arctan2(x3, z3)
        # Avoid division by zero
        norms = np.linalg.norm(rays_rot, axis=1)
        norms[norms == 0] = 1e-6
        phi = np.arcsin(y3 / norms)
        
        u = self.focal_length * theta
        v = self.focal_length * (phi + np.pi / 2)
        
        return np.column_stack([u, v]).astype(np.float32)

    def warp_image(self, img: np.ndarray, K: np.ndarray, R: np.ndarray) -> Tuple[Tuple[int, int], np.ndarray]:
        return self.warper.warp(img, K, R, cv2.INTER_LINEAR, cv2.BORDER_REFLECT)

    def warp_mask(self, mask: np.ndarray, K: np.ndarray, R: np.ndarray) -> Tuple[Tuple[int, int], np.ndarray]:
        return self.warper.warp(mask, K, R, cv2.INTER_NEAREST, cv2.BORDER_CONSTANT)

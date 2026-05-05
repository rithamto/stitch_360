import cv2
import numpy as np
from typing import Tuple

class CylindricalWarper:
    def __init__(self, focal_length: float):
        self.focal_length = focal_length
        self.warper = cv2.PyRotationWarper("cylindrical", focal_length)

    def warp_point(self, pt: Tuple[float, float], K: np.ndarray, R: np.ndarray) -> Tuple[float, float]:
        """
        Warp a 2D point from the flat image to the global cylindrical coordinate system.
        """
        return self.warper.warpPoint(pt, K, R)

    def warp_points(self, pts: np.ndarray, K: np.ndarray, R: np.ndarray) -> np.ndarray:
        """
        Warp an N x 2 array of points from the flat image to the global cylindrical coordinate system.
        """
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
        
        # Cylindrical projection
        x3, y3, z3 = rays_rot[:, 0], rays_rot[:, 1], rays_rot[:, 2]
        
        theta = np.arctan2(x3, z3)
        
        norms = np.sqrt(x3**2 + z3**2)
        norms[norms == 0] = 1e-6
        h = y3 / norms
        
        u = self.focal_length * theta
        v = self.focal_length * h
        
        return np.column_stack([u, v]).astype(np.float32)

    def warp_image(self, img: np.ndarray, K: np.ndarray, R: np.ndarray) -> Tuple[Tuple[int, int], np.ndarray]:
        """
        Warp a flat image into the global cylindrical coordinate system.
        Returns the top-left corner (x, y) of the warped image and the warped image itself.
        """
        return self.warper.warp(img, K, R, cv2.INTER_LINEAR, cv2.BORDER_REFLECT)

    def warp_mask(self, mask: np.ndarray, K: np.ndarray, R: np.ndarray) -> Tuple[Tuple[int, int], np.ndarray]:
        """
        Warp a mask into the global cylindrical coordinate system.
        """
        return self.warper.warp(mask, K, R, cv2.INTER_NEAREST, cv2.BORDER_CONSTANT)

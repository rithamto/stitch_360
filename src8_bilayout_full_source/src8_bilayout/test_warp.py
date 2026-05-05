import cv2
import numpy as np

def test_warp():
    focal = 1000.0
    warper = cv2.PyRotationWarper("cylindrical", focal)
    K = np.array([[focal, 0, 500], [0, focal, 500], [0, 0, 1]], dtype=np.float32)
    R = np.eye(3, dtype=np.float32)
    pt = (800, 500)
    warped_pt = warper.warpPoint(pt, K, R)
    print(f"Original pt: {pt}, Warped pt: {warped_pt}")
    
    img = np.ones((1000, 1000, 3), dtype=np.uint8) * 255
    corner, warped_img = warper.warp(img, K, R, cv2.INTER_LINEAR, cv2.BORDER_REFLECT)
    print(f"Corner: {corner}, img shape: {warped_img.shape}")

if __name__ == "__main__":
    test_warp()

import os
import time
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import cv2
import numpy as np
import argparse
import logging
import re
try:
    import torch
    from kornia.feature import LoFTR
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
import csv
from pathlib import Path
from scipy.optimize import least_squares

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(message)s")

def get_rotation_matrix(y, p, r):
    y, p, r = np.radians(y), np.radians(p), np.radians(r)
    Ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    Rp = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])
    Rr = np.array([[np.cos(r), -np.sin(r), 0], [np.sin(r), np.cos(r), 0], [0, 0, 1]])
    return (Ry @ Rp @ Rr).astype(np.float32)

def warp_points(pts, K, R, focal_length):
    if len(pts) == 0: return np.empty((0, 2), dtype=np.float32)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    x_norm, y_norm = (pts[:, 0] - cx) / fx, (pts[:, 1] - cy) / fy
    rays = np.stack([x_norm, y_norm, np.ones(len(pts))], axis=1)
    rays_rot = (R @ rays.T).T
    theta = np.arctan2(rays_rot[:, 0], rays_rot[:, 2])
    phi = np.arcsin(np.clip(rays_rot[:, 1] / (np.linalg.norm(rays_rot, axis=1) + 1e-6), -1, 1))
    return np.column_stack([focal_length * theta, focal_length * (phi + np.pi/2)]).astype(np.float32)

def get_sift_matches(img1, img2, match_max_dim=1600):
    if len(img1.shape) == 3: img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    if len(img2.shape) == 3: img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    scale1 = min(1.0, float(match_max_dim) / max(h1, w1))
    s_img1 = cv2.resize(img1, (0, 0), fx=scale1, fy=scale1)
    
    scale2 = min(1.0, float(match_max_dim) / max(h2, w2))
    s_img2 = cv2.resize(img2, (0, 0), fx=scale2, fy=scale2)
    
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(s_img1, None)
    kp2, des2 = sift.detectAndCompute(s_img2, None)
    
    if des1 is None or des2 is None or len(kp1) < 2 or len(kp2) < 2:
        return np.empty((0, 2)), np.empty((0, 2)), np.empty(0)
        
    bf = cv2.BFMatcher()
    try:
        matches = bf.knnMatch(des1, des2, k=2)
    except Exception:
        return np.empty((0, 2)), np.empty((0, 2)), np.empty(0)
    
    good = []
    mkpts0 = []
    mkpts1 = []
    
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
            pt1 = list(kp1[m.queryIdx].pt)
            pt2 = list(kp2[m.trainIdx].pt)
            pt1[0] /= scale1
            pt1[1] /= scale1
            pt2[0] /= scale2
            pt2[1] /= scale2
            mkpts0.append(pt1)
            mkpts1.append(pt2)
            
    mkpts0 = np.array(mkpts0)
    mkpts1 = np.array(mkpts1)
    mconf = np.ones(len(mkpts0))
    
    return mkpts0, mkpts1, mconf

def get_matches(matcher, img1, img2, device, match_max_dim=1200):
    if not TORCH_AVAILABLE:
        return get_sift_matches(img1, img2, match_max_dim)
        
    if len(img1.shape) == 3: img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    if len(img2.shape) == 3: img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    scale1 = min(1.0, float(match_max_dim) / max(h1, w1))
    new_w1, new_h1 = int(max(8, round(w1 * scale1 / 8) * 8)), int(max(8, round(h1 * scale1 / 8) * 8))
    s_img1 = cv2.resize(img1, (new_w1, new_h1))
    
    scale2 = min(1.0, float(match_max_dim) / max(h2, w2))
    new_w2, new_h2 = int(max(8, round(w2 * scale2 / 8) * 8)), int(max(8, round(h2 * scale2 / 8) * 8))
    s_img2 = cv2.resize(img2, (new_w2, new_h2))
    
    t_img1 = torch.from_numpy(s_img1)[None, None].float().to(device) / 255.0
    t_img2 = torch.from_numpy(s_img2)[None, None].float().to(device) / 255.0
    
    if device.type == "cuda":
        t_img1 = t_img1.half()
        t_img2 = t_img2.half()
    
    with torch.no_grad():
        input_dict = {"image0": t_img1, "image1": t_img2}
        correspondences = matcher(input_dict)
        
    mkpts0 = correspondences["keypoints0"].cpu().numpy()
    mkpts1 = correspondences["keypoints1"].cpu().numpy()
    mconf = correspondences["confidence"].cpu().numpy()
    
    # Free GPU memory immediately
    del t_img1, t_img2, input_dict, correspondences
    if device is not None and device.type == "cuda":
        torch.cuda.empty_cache()
    
    if len(mkpts0) > 0:
        mkpts0[:, 0] *= (w1 / new_w1)
        mkpts0[:, 1] *= (h1 / new_h1)
        mkpts1[:, 0] *= (w2 / new_w2)
        mkpts1[:, 1] *= (h2 / new_h2)
    
    return mkpts0, mkpts1, mconf

def parse_metadata(filename):
    # Support both with and without _f focal length field
    match = re.search(r"frame_(\d+)_y([\d.\-]+)_p([\d.\-]+)_r([\d.\-]+)(?:_f([\d.\-]+))?_t(\d+)", filename)
    if match:
        return {
            "id": int(match.group(1)),
            "y": float(match.group(2)),
            "p": float(match.group(3)),
            "r": float(match.group(4)),
            "f": float(match.group(5)) if match.group(5) else 0.0,
            "t": int(match.group(6)),
            "filename": filename
        }
    return None

def load_metadata_csv(csv_path):
    metadata = {}
    if not csv_path.exists():
        return metadata
    try:
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                fname = row["filename"]
                # Use 'fx' if available as it is usually the pixel focal length needed for stitching
                metadata[fname] = {
                    "y": float(row["yaw"]),
                    "p": float(row["pitch"]),
                    "r": float(row["roll"]),
                    "f": float(row.get("fx", row.get("focal_length", 0))),
                    "cluster_id": int(row.get("cluster_id", -1))
                }
    except Exception as e:
        logging.warning(f"Failed to load metadata CSV: {e}")
    return metadata

def get_row_name(pitch):
    if pitch > 70: return "Zenith"
    if pitch > 17.5: return "Top"
    if pitch >= -17.5: return "Horizon"
    if pitch >= -70: return "Bottom"
    return "Nadir"

def solve_row_ring(frames, edges, K_opt):
    num_frames = len(frames)
    initial_y = np.array([f["y"] for f in frames])
    initial_p = np.array([f["p"] for f in frames])
    initial_r = np.array([f["r"] for f in frames])
    f = K_opt[0, 0]

    def residuals(v):
        ys = v[:num_frames]
        ps = v[num_frames:2*num_frames]
        rs = v[2*num_frames:3*num_frames]
        
        K = np.array([[f, 0, K_opt[0, 2]], [0, f, K_opt[1, 2]], [0, 0, 1]], dtype=np.float32)
        
        res = [
            (ys - initial_y) * f * 0.1,   # Allow some yaw adjustment
            (ps - initial_p) * f * 5.0,   # Strong prior to keep pitch near IMU
            (rs - initial_r) * f * 10.0,  # Very strong prior to keep roll near IMU
            np.array([(ys[0] - initial_y[0]) * f * 100.0]) # Anchor first frame's yaw
        ]
        
        for e in edges:
            ia, ib = e["ia"], e["ib"]
            pts_a = warp_points(e["pts_a"], K, get_rotation_matrix(ys[ia], ps[ia], rs[ia]), f)
            pts_b = warp_points(e["pts_b"], K, get_rotation_matrix(ys[ib], ps[ib], rs[ib]), f)
            
            diff_x = pts_a[:, 0] - pts_b[:, 0]
            circumference = 2 * np.pi * f
            diff_x = np.where(diff_x > circumference / 2, diff_x - circumference, diff_x)
            diff_x = np.where(diff_x < -circumference / 2, diff_x + circumference, diff_x)
            
            res.append(diff_x)
            res.append(pts_a[:, 1] - pts_b[:, 1])
            
        return np.concatenate(res)

    initial_guess = np.concatenate([initial_y, initial_p, initial_r])
    sol = least_squares(residuals, initial_guess, loss="soft_l1", f_scale=2.0, max_nfev=250)
    
    return sol.x[:num_frames], sol.x[num_frames:2*num_frames], sol.x[2*num_frames:3*num_frames]

def solve_global_bundle(frames, edges, K_opt):
    num_frames = len(frames)
    initial_y = np.array([f["y"] for f in frames])
    initial_p = np.array([f["p"] for f in frames])
    initial_r = np.array([f["r"] for f in frames])
    initial_f = K_opt[0, 0]
    cx, cy = K_opt[0, 2], K_opt[1, 2]

    def residuals(v):
        ys = v[:num_frames]
        ps = v[num_frames:2*num_frames]
        rs = v[2*num_frames:3*num_frames]
        f = v[3*num_frames]
        
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float32)
        circumference = 2 * np.pi * f
        
        res = [
            (ys - initial_y) * f * 0.1,   # Yaw prior
            (ps - initial_p) * f * 5.0,   # Pitch prior
            (rs - initial_r) * f * 10.0,  # Roll prior
            np.array([(f - initial_f) * 0.5]),
            np.array([(ys[0] - initial_y[0]) * f * 100.0]) # Anchor North
        ]
        
        for e in edges:
            ia, ib = e["ia"], e["ib"]
            pts_a = warp_points(e["pts_a"], K, get_rotation_matrix(ys[ia], ps[ia], rs[ia]), f)
            pts_b = warp_points(e["pts_b"], K, get_rotation_matrix(ys[ib], ps[ib], rs[ib]), f)
            
            diff_x = pts_a[:, 0] - pts_b[:, 0]
            diff_x = np.where(diff_x > circumference / 2, diff_x - circumference, diff_x)
            diff_x = np.where(diff_x < -circumference / 2, diff_x + circumference, diff_x)
            
            res.append(diff_x)
            res.append(pts_a[:, 1] - pts_b[:, 1])
            
        return np.concatenate(res)

    initial_guess = np.concatenate([initial_y, initial_p, initial_r, [initial_f]])
    sol = least_squares(residuals, initial_guess, loss="soft_l1", f_scale=1.0, max_nfev=600)
    
    logging.info(f"[INFO] Optimized Focal Length: {sol.x[3*num_frames]:.2f} (Initial: {initial_f:.2f})")
    return sol.x[:num_frames], sol.x[num_frames:2*num_frames], sol.x[2*num_frames:3*num_frames], sol.x[3*num_frames]

def solve_bridging_ring(frames, ring_edges, top_anchors, top_edges, bottom_anchors, bottom_edges, K_opt):
    num_frames = len(frames)
    initial_y = np.array([f["y"] for f in frames])
    initial_p = np.array([f["p"] for f in frames])
    initial_r = np.array([f["r"] for f in frames])
    f = K_opt[0, 0]
    circumference = 2 * np.pi * f

    def residuals(v):
        ys = v[:num_frames]
        ps = v[num_frames:2*num_frames]
        rs = v[2*num_frames:3*num_frames]
        K = np.array([[f, 0, K_opt[0, 2]], [0, f, K_opt[1, 2]], [0, 0, 1]], dtype=np.float32)
        
        res = [
            (ys - initial_y) * f * 0.1,
            (ps - initial_p) * f * 5.0,
            (rs - initial_r) * f * 10.0,
        ]
        
        # Internal ring matches (Horizon-Horizon)
        for e in ring_edges:
            ia, ib = e["ia"], e["ib"]
            pts_a = warp_points(e["pts_a"], K, get_rotation_matrix(ys[ia], ps[ia], rs[ia]), f)
            pts_b = warp_points(e["pts_b"], K, get_rotation_matrix(ys[ib], ps[ib], rs[ib]), f)
            diff_x = pts_a[:, 0] - pts_b[:, 0]
            diff_x = np.where(diff_x > circumference / 2, diff_x - circumference, diff_x)
            diff_x = np.where(diff_x < -circumference / 2, diff_x + circumference, diff_x)
            res.append(diff_x)
            res.append(pts_a[:, 1] - pts_b[:, 1])
            
        # Top anchor matches (Horizon-Top)
        for e in top_edges:
            ia = e["ia"]
            anchor = top_anchors[e["ib"]]
            pts_a = warp_points(e["pts_a"], K, get_rotation_matrix(ys[ia], ps[ia], rs[ia]), f)
            pts_b = warp_points(e["pts_b"], K, get_rotation_matrix(anchor["y"], anchor["p"], anchor["r"]), f)
            diff_x = pts_a[:, 0] - pts_b[:, 0]
            diff_x = np.where(diff_x > circumference / 2, diff_x - circumference, diff_x)
            diff_x = np.where(diff_x < -circumference / 2, diff_x + circumference, diff_x)
            res.append(diff_x)
            res.append(pts_a[:, 1] - pts_b[:, 1])
            
        # Bottom anchor matches (Horizon-Bottom)
        for e in bottom_edges:
            ia = e["ia"]
            anchor = bottom_anchors[e["ib"]]
            pts_a = warp_points(e["pts_a"], K, get_rotation_matrix(ys[ia], ps[ia], rs[ia]), f)
            pts_b = warp_points(e["pts_b"], K, get_rotation_matrix(anchor["y"], anchor["p"], anchor["r"]), f)
            diff_x = pts_a[:, 0] - pts_b[:, 0]
            diff_x = np.where(diff_x > circumference / 2, diff_x - circumference, diff_x)
            diff_x = np.where(diff_x < -circumference / 2, diff_x + circumference, diff_x)
            res.append(diff_x)
            res.append(pts_a[:, 1] - pts_b[:, 1])
            
        return np.concatenate(res)

    initial_guess = np.concatenate([initial_y, initial_p, initial_r])
    sol = least_squares(residuals, initial_guess, loss="soft_l1", f_scale=2.0, max_nfev=250)
    return sol.x[:num_frames], sol.x[num_frames:2*num_frames], sol.x[2*num_frames:3*num_frames]

def solve_anchored_frame(frame, anchors, edges, K_opt):
    initial_y = frame["y"]
    initial_p = frame["p"]
    initial_r = frame["r"]
    f = K_opt[0, 0]
    
    def residuals(v):
        y, p, r = v
        K = np.array([[f, 0, K_opt[0, 2]], [0, f, K_opt[1, 2]], [0, 0, 1]], dtype=np.float32)
        
        res = [
            np.array([(y - initial_y) * f * 0.2]),
            np.array([(p - initial_p) * f * 5.0]),
            np.array([(r - initial_r) * f * 10.0]),
        ]
        
        for e in edges:
            anchor = anchors[e["ib"]]
            pts_a = warp_points(e["pts_a"], K, get_rotation_matrix(y, p, r), f)
            pts_b = warp_points(e["pts_b"], K, get_rotation_matrix(anchor["y"], anchor["p"], anchor["r"]), f)
            
            diff_x = pts_a[:, 0] - pts_b[:, 0]
            circumference = 2 * np.pi * f
            diff_x = np.where(diff_x > circumference / 2, diff_x - circumference, diff_x)
            diff_x = np.where(diff_x < -circumference / 2, diff_x + circumference, diff_x)
            
            res.append(diff_x)
            res.append(pts_a[:, 1] - pts_b[:, 1])
            
        if len(res) == 3:
            return np.array(res).flatten()
        return np.concatenate(res)

    sol = least_squares(residuals, [initial_y, initial_p, initial_r], loss="soft_l1", f_scale=2.0)
    return sol.x

def apply_blocks_gain_compensation(images, masks, corners):
    try:
        scale = 0.1
        imgs_small = [cv2.resize(img, (0,0), fx=scale, fy=scale) for img in images]
        masks_small = [cv2.resize(m, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST) for m in masks]
        corners_small = [(int(c[0]*scale), int(c[1]*scale)) for c in corners]

        compensator = cv2.detail.ExposureCompensator_createDefault(cv2.detail.ExposureCompensator_GAIN_BLOCKS)
        compensator.feed(corners_small, imgs_small, masks_small)
        for idx in range(len(images)):
            compensator.apply(idx, corners[idx], images[idx], masks[idx])
    except Exception as e:
        logging.warning(f"Gain compensation failed: {e}")

def find_seams(images, masks, corners, scale=0.3, seam_type=cv2.detail.SeamFinder_DP_SEAM):
    if len(images) <= 1:
        return [m.copy() for m in masks]
    try:
        imgs_small = [cv2.resize(img, (0,0), fx=scale, fy=scale) for img in images]
        masks_small = [cv2.resize(m, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST) for m in masks]
        corners_small = [(int(c[0]*scale), int(c[1]*scale)) for c in corners]
        imgs_f32_small = [img.astype(np.float32) for img in imgs_small]

        seam_finder = cv2.detail.SeamFinder_createDefault(seam_type)
        res_masks_small = seam_finder.find(imgs_f32_small, corners_small, masks_small)
        
        res_masks = []
        for i in range(len(images)):
            h, w = images[i].shape[:2]
            m_small = res_masks_small[i].get() if hasattr(res_masks_small[i], "get") else res_masks_small[i]
            res_masks.append(cv2.resize(m_small, (w, h), interpolation=cv2.INTER_NEAREST))
        return res_masks
    except Exception as e:
        logging.warning(f"SeamFinder failed: {e}")
        return masks

def blend_images(images, masks, corners, out_w, out_h, blend_strength=2.5):
    # Lower blend strength to preserve more high-frequency details
    blend_width = np.sqrt(out_w * out_h) * blend_strength / 100.0
    num_bands = max(2, int(np.ceil(np.log(blend_width) / np.log(2.))))
    blender = cv2.detail_MultiBandBlender(0, num_bands)
    blender.prepare((0, 0, out_w, out_h))
    
    fed_count = 0
    for i in range(len(images)):
        img = images[i].get() if hasattr(images[i], "get") else images[i]
        m = masks[i].get() if hasattr(masks[i], "get") else masks[i]
        
        # Skip images whose mask is entirely zero
        if m is None or img is None or img.size == 0 or m.size == 0:
            continue
            
        # Dilate mask slightly to ensure overlap and remove black seams
        kernel = np.ones((3, 3), np.uint8)
        m = cv2.dilate(m, kernel, iterations=2)
        
        if np.count_nonzero(m) == 0:
            continue
            
        x_off, y_off = corners[i]
        
        for shift in [-out_w, 0, out_w]:
            dx = x_off + shift
            if dx + img.shape[1] <= 0 or dx >= out_w: continue
            
            x0, x1 = max(0, dx), min(out_w, dx + img.shape[1])
            y0, y1 = max(0, y_off), min(out_h, y_off + img.shape[0])
            
            if x1 > x0 and y1 > y0:
                sub_img = img[y0-y_off:y1-y_off, x0-dx:x1-dx]
                sub_mask = m[y0-y_off:y1-y_off, x0-dx:x1-dx]
                
                # Only feed if the sub_mask has non-zero pixels
                if sub_img.size == 0 or sub_mask.size == 0 or np.count_nonzero(sub_mask) == 0:
                    continue
                    
                blender.feed(sub_img.astype(np.int16), sub_mask, (x0, y0))
                fed_count += 1
    
    if fed_count == 0:
        logging.warning("[WARN] Blender received no valid data, returning empty canvas.")
        result = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        result_mask = np.zeros((out_h, out_w), dtype=np.uint8)
        return result, result_mask
                
    result, result_mask = blender.blend(None, None)
    return cv2.convertScaleAbs(result), result_mask

def mask_for_row(row_name, h, w):
    """
    Returns a full mask. Global trimming will be applied after warping
    to precisely control the vertical height of the Horizon row.
    """
    return np.full((h, w), 255, dtype=np.uint8)

def get_cropped_strip(strip_img, strip_msk):
    y_indices, x_indices = np.where(strip_msk > 0)
    if len(y_indices) == 0:
        return strip_img, strip_msk, (0, 0)
        
    x_min, x_max = int(x_indices.min()), int(x_indices.max())
    y_min, y_max = int(y_indices.min()), int(y_indices.max())
    
    cropped_img = strip_img[y_min:y_max+1, x_min:x_max+1]
    cropped_msk = strip_msk[y_min:y_max+1, x_min:x_max+1]
    
    return cropped_img.copy(), cropped_msk.copy(), (x_min, y_min)

def main():
    start_time = time.time()
    cv2.ocl.setUseOpenCL(False)
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path(r"c:\Project\drive-download-20260513T035900Z-3-001"))
    parser.add_argument("--width", type=int, default=10000) # Increased default width for more detail
    parser.add_argument("--scale", type=float, default=1.0)
    args = parser.parse_args()

    # 1. Load Frames & Metadata
    csv_meta = load_metadata_csv(args.input / "metadata.csv")
    frames_meta = []
    images_dict = {}
    
    for f in args.input.glob("*.jpg"):
        if f.name not in csv_meta:
            continue
        meta = parse_metadata(f.name)
        if not meta:
            # Fallback for simple filenames like 0.jpg
            meta_id_match = re.search(r"(\d+)", f.name)
            meta = {
                "id": int(meta_id_match.group(1)) if meta_id_match else 0,
                "filename": f.name
            }
        
        meta.update(csv_meta[f.name])
        meta["path"] = f
        meta["row"] = get_row_name(meta["p"])
        
        cid = csv_meta[f.name].get("cluster_id", -1)
        if cid == -1:
            # Fallback for 44-point layout without cluster_ids in csv
            fid = meta["id"]
            if fid < 42:
                cid = fid % 14
            elif fid == 42:
                cid = 100 # Zenith
            elif fid == 43:
                cid = 200 # Nadir
                
        meta["cluster_id"] = cid
        frames_meta.append(meta)
            
    if not frames_meta:
        logging.error("No valid frames found!")
        return

    frames_meta.sort(key=lambda x: x["id"])
    
    # Pre-load images
    for m in frames_meta:
        img = cv2.imread(str(m["path"]))
        if args.scale != 1.0:
            img = cv2.resize(img, (0, 0), fx=args.scale, fy=args.scale)
        images_dict[m["id"]] = img

    img0 = images_dict[frames_meta[0]["id"]]
    h0, w0 = img0.shape[:2]
    
    focal = frames_meta[0]["f"] * args.scale
    if focal <= 0: focal = 2411.0
    K_opt = np.array([[focal, 0, w0/2], [0, focal, h0/2], [0, 0, 1]], dtype=np.float32)
    
    if TORCH_AVAILABLE:
        logging.info("[INFO] Torch and Kornia found. Initializing LoFTR...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        loftr = LoFTR(pretrained="indoor").to(device)
        if device.type == "cuda":
            loftr = loftr.half()
        loftr = loftr.eval()
    else:
        logging.warning("[WARNING] Torch/Kornia not found. Falling back to fast OpenCV SIFT matcher!")
        device = None
        loftr = None

    # 2. Global Bundle Adjustment (GBA)
    all_edges = []
    
    def match_and_add(idx1, idx2):
        m1, m2 = frames_meta[idx1], frames_meta[idx2]
        # Skip if already matched or same frame
        if idx1 == idx2: return
        
        # Use very low threshold to capture links in featureless areas
        pts_a, pts_b, conf = get_matches(loftr, images_dict[m1["id"]], images_dict[m2["id"]], device)
        mask = conf > 0.15 # Lowered significantly to find more links
        if np.sum(mask) > 8:
            all_edges.append({"ia": idx1, "ib": idx2, "pts_a": pts_a[mask], "pts_b": pts_b[mask]})
            return True
        return False

    logging.info("[INFO] Performing Global Matching (this may take a few minutes)...")
    
    # Matching strategy: Cluster-based
    # 1. Intra-cluster matches (Vertical)
    clusters = {}
    for i, f in enumerate(frames_meta):
        cid = f["cluster_id"]
        if cid not in clusters: clusters[cid] = []
        clusters[cid].append(i)
    
    for cid, indices in clusters.items():
        if cid < 0: continue # Skip if no cluster
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                match_and_add(indices[i], indices[j])
                
    # 2. Inter-cluster matches (Horizontal)
    # Sort normal clusters by yaw to find neighbors
    normal_clusters = sorted([cid for cid in clusters.keys() if 0 <= cid < 100], 
                             key=lambda cid: np.mean([frames_meta[i]["y"] for i in clusters[cid]]))
    
    for i in range(len(normal_clusters)):
        for j in [1, 2]: # Match with next 2 clusters to increase connectivity
            cid1 = normal_clusters[i]
            cid2 = normal_clusters[(i + j) % len(normal_clusters)]
            
            # Match corresponding rows between clusters
            for idx1 in clusters[cid1]:
                f1 = frames_meta[idx1]
                # Find best match in next cluster by pitch
                for idx2 in clusters[cid2]:
                    f2 = frames_meta[idx2]
                    if f1["row"] == f2["row"]:
                        match_and_add(idx1, idx2)
                    # Also try cross-row matching for stability
                    elif abs(f1["p"] - f2["p"]) < 40:
                        match_and_add(idx1, idx2)
        
    # 3. Zenith/Nadir matches
    zen_indices = [i for i, f in enumerate(frames_meta) if f["row"] == "Zenith"]
    nad_indices = [i for i, f in enumerate(frames_meta) if f["row"] == "Nadir"]
    top_indices = [i for i, f in enumerate(frames_meta) if f["row"] == "Top"]
    bot_indices = [i for i, f in enumerate(frames_meta) if f["row"] == "Bottom"]
    
    for z_idx in zen_indices:
        for t_idx in top_indices:
            match_and_add(z_idx, t_idx)
    for n_idx in nad_indices:
        for b_idx in bot_indices:
            match_and_add(n_idx, b_idx)

    logging.info(f"[INFO] Global Bundle Adjustment with {len(all_edges)} match pairs...")
    opt_y, opt_p, opt_r, opt_f = solve_global_bundle(frames_meta, all_edges, K_opt)
    
    # Update frames and K_opt
    for i, m in enumerate(frames_meta):
        m["y"], m["p"], m["r"] = opt_y[i], opt_p[i], opt_r[i]
        m["f"] = opt_f # Update focal length for all frames
    
    # Update target focal length for warping based on optimized focal
    # Wait, target_focal is usually based on output width. 
    # But we should use the optimized focal to ensure the warp is geometrically correct.
    # Actually, we keep the output focal consistent for the pano size, but we use the optimized one for K.
    
    # 4. Warping
    out_w = args.width
    out_h = out_w // 2
    target_focal = out_w / (2 * np.pi)
    warper = cv2.PyRotationWarper("spherical", target_focal)
    
    logging.info("[INFO] Warping frames to Spherical Panorama...")
    
    # Store by cluster for hierarchical blending
    cluster_data = {} # {cluster_id: {"imgs": [], "msks": [], "corners": []}}
    all_warped_imgs = [] 
    all_warped_msks = []
    all_corners = []
    
    for m in frames_meta:
        img = images_dict[m["id"]]
        cid = m["cluster_id"]
        if cid not in cluster_data:
            cluster_data[cid] = {"imgs": [], "msks": [], "corners": []}
        
        work_focal = m["f"] * args.scale
        work_scale = target_focal / work_focal
        interp = cv2.INTER_AREA if work_scale < 1.0 else cv2.INTER_LANCZOS4
        img = cv2.resize(img, (0,0), fx=work_scale, fy=work_scale, interpolation=interp)
        
        K = np.array([[target_focal, 0, img.shape[1]/2], [0, target_focal, img.shape[0]/2], [0, 0, 1]], dtype=np.float32)
        R = get_rotation_matrix(m["y"], m["p"], m["r"])
        
        corner, warped = warper.warp(img, K, R, cv2.INTER_LANCZOS4, cv2.BORDER_REFLECT)
        
        base_mask = mask_for_row(m["row"], img.shape[0], img.shape[1])
        _, mask = warper.warp(base_mask, K, R, cv2.INTER_NEAREST, cv2.BORDER_CONSTANT)
        
        # Relaxed vertical trimming to avoid gaps
        if m["row"] in ["Zenith"]:
            global_max_y = int(out_h * 0.35) # Increased from 0.25
            local_max_y = global_max_y - corner[1]
            if 0 < local_max_y < mask.shape[0]: mask[local_max_y:, :] = 0
        elif m["row"] in ["Nadir"]:
            global_min_y = int(out_h * 0.65) # Decreased from 0.75
            local_min_y = global_min_y - corner[1]
            if 0 < local_min_y < mask.shape[0]: mask[:local_min_y, :] = 0
        
        cluster_data[cid]["imgs"].append(warped)
        cluster_data[cid]["msks"].append(mask)
        cluster_data[cid]["corners"].append(corner)
        
        all_warped_imgs.append(warped)
        all_warped_msks.append(mask)
        all_corners.append(corner)

    # 5. Global Exposure Compensation
    logging.info("[INFO] Applying Global Exposure Compensation...")
    apply_blocks_gain_compensation(all_warped_imgs, all_warped_msks, all_corners)
    
    # 6. Hierarchical Seam Finding & Smooth Proxy Generation
    logging.info("[INFO] Generating smooth cluster proxies and finding final seams...")
    sorted_cids = sorted(cluster_data.keys())
    
    cluster_proxies = []
    cluster_proxy_masks = []
    cluster_proxy_corners = []
    
    # We will save the original images, their intra-cluster masks, and corners
    final_imgs = []
    final_msks = []
    final_corners = []
    final_img_cluster_map = [] # To map which image belongs to which cluster
    
    # Step 6a: Intra-cluster seams and smooth proxy generation
    for cid_idx, cid in enumerate(sorted_cids):
        data = cluster_data[cid]
        logging.info(f"[INFO] Processing cluster: {cid} ({len(data['imgs'])} images)...")
        
        if len(data["imgs"]) > 1:
            cmsks = find_seams(data["imgs"], data["msks"], data["corners"], scale=0.3, seam_type=cv2.detail.SeamFinder_DP_SEAM)
        else:
            cmsks = data["msks"]
            
        for i in range(len(data["imgs"])):
            final_imgs.append(data["imgs"][i])
            final_msks.append(cmsks[i])
            final_corners.append(data["corners"][i])
            final_img_cluster_map.append(cid_idx)
            
        # Blend the cluster into a single smooth strip proxy
        strip_img, strip_msk = blend_images(data["imgs"], cmsks, data["corners"], out_w, out_h, blend_strength=1.5)
        
        # Crop to bounding box for the inter-cluster seam finder
        c_img, c_msk, c_corner = get_cropped_strip(strip_img, strip_msk)
        cluster_proxies.append(c_img)
        cluster_proxy_masks.append(c_msk)
        cluster_proxy_corners.append(c_corner)

    # Step 6b: Inter-cluster seams using smooth proxies
    logging.info("[INFO] Finding seams between smooth cluster proxies...")
    final_cluster_masks = find_seams(cluster_proxies, cluster_proxy_masks, cluster_proxy_corners, scale=0.2, seam_type=cv2.detail.SeamFinder_DP_SEAM)
    
    del cluster_proxies
    del cluster_proxy_masks
    
    # Step 6c: Project inter-cluster seams back to individual original images
    logging.info("[INFO] Projecting seams back for single-pass blending...")
    
    # Pre-build full canvas masks for each cluster
    global_cluster_masks = []
    for cid_idx in range(len(sorted_cids)):
        c_mask = final_cluster_masks[cid_idx].get() if hasattr(final_cluster_masks[cid_idx], "get") else final_cluster_masks[cid_idx]
        c_x, c_y = cluster_proxy_corners[cid_idx]
        
        global_c_mask = np.zeros((out_h, out_w), dtype=np.uint8)
        for shift in [-out_w, 0, out_w]:
            dx = c_x + shift
            if dx + c_mask.shape[1] <= 0 or dx >= out_w: continue
            x0, x1 = max(0, dx), min(out_w, dx + c_mask.shape[1])
            y0, y1 = max(0, c_y), min(out_h, c_y + c_mask.shape[0])
            if x1 > x0 and y1 > y0:
                global_c_mask[y0:y1, x0:x1] = c_mask[y0-c_y:y1-c_y, x0-dx:x1-dx]
        global_cluster_masks.append(global_c_mask)

    # Apply the global cluster masks to the individual image masks
    kernel = np.ones((5, 5), np.uint8)
    for idx in range(len(final_imgs)):
        cid_idx = final_img_cluster_map[idx]
        img = final_imgs[idx]
        m = final_msks[idx]
        x_off, y_off = final_corners[idx]
        global_c_mask = global_cluster_masks[cid_idx]
        
        # Dilate the cluster mask to ensure no gaps between clusters
        dilated_c_mask = cv2.dilate(global_c_mask, kernel, iterations=1)
        
        new_m = np.zeros_like(m)
        for shift in [-out_w, 0, out_w]:
            dx = x_off + shift
            if dx + img.shape[1] <= 0 or dx >= out_w: continue
            x0, x1 = max(0, dx), min(out_w, dx + img.shape[1])
            y0, y1 = max(0, y_off), min(out_h, y_off + img.shape[0])
            if x1 > x0 and y1 > y0:
                sub_m = m[y0-y_off:y1-y_off, x0-dx:x1-dx]
                sub_c = dilated_c_mask[y0:y1, x0:x1]
                valid = (sub_m > 0) & (sub_c > 0)
                new_m[y0-y_off:y1-y_off, x0-dx:x1-dx][valid] = 255
                
        final_msks[idx] = new_m

    # 7. Final Blending (Single Pass)
    logging.info("[INFO] Final single-pass blending of original pixels...")
    final_pano, _ = blend_images(final_imgs, final_msks, final_corners, out_w, out_h, blend_strength=2.0)
    
    out_path = f"panorama_360_v11_cluster_{args.input.name}.jpg"
    cv2.imwrite(out_path, final_pano, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    elapsed_time = time.time() - start_time
    logging.info(f"Done! Panorama saved to {out_path}")
    logging.info(f"Total execution time: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()

import cv2
import numpy as np
import argparse
import logging
import re
import torch
import csv
from pathlib import Path
from scipy.optimize import least_squares
from kornia.feature import LoFTR

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

def get_loftr_matches(matcher, img1, img2, device, match_max_dim=1024):
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
    
    with torch.no_grad():
        input_dict = {"image0": t_img1, "image1": t_img2}
        correspondences = matcher(input_dict)
        
    mkpts0 = correspondences["keypoints0"].cpu().numpy()
    mkpts1 = correspondences["keypoints1"].cpu().numpy()
    mconf = correspondences["confidence"].cpu().numpy()
    
    if len(mkpts0) > 0:
        mkpts0[:, 0] *= (w1 / new_w1)
        mkpts0[:, 1] *= (h1 / new_h1)
        mkpts1[:, 0] *= (w2 / new_w2)
        mkpts1[:, 1] *= (h2 / new_h2)
    
    return mkpts0, mkpts1, mconf

def parse_metadata(filename):
    match = re.search(r"frame_(\d+)_y([\d.\-]+)_p([\d.\-]+)_r([\d.\-]+)_f([\d.\-]+)_t(\d+)", filename)
    if match:
        return {
            "id": int(match.group(1)),
            "y": float(match.group(2)),
            "p": float(match.group(3)),
            "r": float(match.group(4)),
            "f": float(match.group(5)),
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
                metadata[fname] = {
                    "y": float(row["yaw"]),
                    "p": float(row["pitch"]),
                    "r": float(row["roll"]),
                    "f": float(row["focal_length"])
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

def solve_horizon_ring(frames, edges, K_opt):
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
            (ps - initial_p) * f * 5.0,   # Very strong prior to keep pitch near IMU
            (rs - initial_r) * f * 10.0,  # Extremely strong prior to keep roll near IMU
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

        compensator = cv2.detail.ExposureCompensator_createDefault(cv2.detail.ExposureCompensator_GAIN)
        compensator.feed(corners_small, imgs_small, masks_small)
        for idx in range(len(images)):
            compensator.apply(idx, corners[idx], images[idx], masks[idx])
    except Exception as e:
        logging.warning(f"Gain compensation failed: {e}")

def find_seams(images, masks, corners):
    if len(images) <= 1:
        return [m.copy() for m in masks]
    try:
        # Increase scale for much higher precision in seam finding
        scale = 0.3
        imgs_small = [cv2.resize(img, (0,0), fx=scale, fy=scale) for img in images]
        masks_small = [cv2.resize(m, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST) for m in masks]
        corners_small = [(int(c[0]*scale), int(c[1]*scale)) for c in corners]
        imgs_f32_small = [img.astype(np.float32) for img in imgs_small]

        seam_finder = cv2.detail.SeamFinder_createDefault(cv2.detail.SeamFinder_DP_SEAM)
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
    
    for i in range(len(images)):
        img = images[i].get() if hasattr(images[i], "get") else images[i]
        m = masks[i].get() if hasattr(masks[i], "get") else masks[i]
        x_off, y_off = corners[i]
        
        for shift in [-out_w, 0, out_w]:
            dx = x_off + shift
            if dx + img.shape[1] <= 0 or dx >= out_w: continue
            
            x0, x1 = max(0, dx), min(out_w, dx + img.shape[1])
            y0, y1 = max(0, y_off), min(out_h, y_off + img.shape[0])
            
            if x1 > x0 and y1 > y0:
                sub_img = img[y0-y_off:y1-y_off, x0-dx:x1-dx].astype(np.int16)
                sub_mask = m[y0-y_off:y1-y_off, x0-dx:x1-dx]
                blender.feed(sub_img, sub_mask, (x0, y0))
                
    result, result_mask = blender.blend(None, None)
    return cv2.convertScaleAbs(result), result_mask

def mask_for_row(row_name, h, w):
    """
    Creates a tailored mask prioritizing the Horizon row.
    Horizon gets a full mask (255).
    Top row gets a mask that fades or cuts off at the bottom so it doesn't overwrite the horizon.
    Bottom row gets a mask that cuts off at the top.
    """
    mask = np.full((h, w), 255, dtype=np.uint8)
    if row_name == "Horizon":
        # Keep full mask for Horizon, but we could subtly trim edges if needed
        # For now, let's keep it 255 and let SeamFinder do the work at higher resolution
        pass
    elif row_name == "Top":
        # Cut off the bottom 30% of the Top row to give more room to Horizon
        cut_h = int(h * 0.70)
        mask[cut_h:, :] = 0
    elif row_name == "Bottom":
        # Cut off the top 30% of the Bottom row
        cut_h = int(h * 0.30)
        mask[:cut_h, :] = 0
    return mask

def main():
    cv2.ocl.setUseOpenCL(False)
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path(r"c:\Project\18"))
    parser.add_argument("--width", type=int, default=8192)
    parser.add_argument("--scale", type=float, default=1.0)
    args = parser.parse_args()

    # 1. Load Frames & Metadata
    csv_meta = load_metadata_csv(args.input / "metadata.csv")
    frames_meta = []
    images_dict = {}
    
    for f in args.input.glob("*.jpg"):
        meta = parse_metadata(f.name)
        if meta:
            if f.name in csv_meta:
                meta.update(csv_meta[f.name])
            meta["path"] = f
            meta["row"] = get_row_name(meta["p"])
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
    
    logging.info("[INFO] Initializing LoFTR...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loftr = LoFTR(pretrained="indoor").to(device).eval()

    # 2. Extract Horizon Row
    horizon_frames = sorted([f for f in frames_meta if f["row"] == "Horizon"], key=lambda x: x["y"])
    horizon_edges = []
    
    logging.info(f"[INFO] Stitching Horizon Row first ({len(horizon_frames)} images)...")
    for i in range(len(horizon_frames)):
        for j in range(i + 1, min(i + 3, len(horizon_frames))):
            m1, m2 = horizon_frames[i], horizon_frames[j]
            dy = abs((m1["y"] - m2["y"] + 180) % 360 - 180)
            if dy < 40:
                pts_a, pts_b, conf = get_loftr_matches(loftr, images_dict[m1["id"]], images_dict[m2["id"]], device)
                mask = conf > 0.8
                if np.sum(mask) > 10:
                    horizon_edges.append({"ia": i, "ib": j, "pts_a": pts_a[mask], "pts_b": pts_b[mask]})
                    
    # Also link last to first for 360 loop closure
    if len(horizon_frames) > 2:
        m1, m2 = horizon_frames[-1], horizon_frames[0]
        pts_a, pts_b, conf = get_loftr_matches(loftr, images_dict[m1["id"]], images_dict[m2["id"]], device)
        mask = conf > 0.8
        if np.sum(mask) > 10:
            horizon_edges.append({"ia": len(horizon_frames)-1, "ib": 0, "pts_a": pts_a[mask], "pts_b": pts_b[mask]})

    opt_y, opt_p, opt_r = solve_horizon_ring(horizon_frames, horizon_edges, K_opt)
    for i, m in enumerate(horizon_frames):
        m["y"], m["p"], m["r"] = opt_y[i], opt_p[i], opt_r[i]
        
    logging.info("[INFO] Horizon Row anchored and optimized.")

    # 3. Anchor Top and Bottom rows to Horizon
    def anchor_row(row_name):
        row_frames = [f for f in frames_meta if f["row"] == row_name]
        if not row_frames: return
        logging.info(f"[INFO] Anchoring {row_name} Row to Horizon ({len(row_frames)} images)...")
        
        for f in row_frames:
            # Find closest 2 horizon frames by Yaw
            h_candidates = sorted(horizon_frames, key=lambda h: abs((f["y"] - h["y"] + 180) % 360 - 180))[:2]
            f_edges = []
            
            for h in h_candidates:
                pts_a, pts_b, conf = get_loftr_matches(loftr, images_dict[f["id"]], images_dict[h["id"]], device)
                mask = conf > 0.8
                if np.sum(mask) > 10:
                    # ib = index in h_candidates
                    f_edges.append({"ib": h_candidates.index(h), "pts_a": pts_a[mask], "pts_b": pts_b[mask]})
            
            if f_edges:
                opt = solve_anchored_frame(f, h_candidates, f_edges, K_opt)
                f["y"], f["p"], f["r"] = opt[0], opt[1], opt[2]
                logging.info(f"  Anchored frame {f['id']} using {len(f_edges)} matches.")
            else:
                logging.warning(f"  Failed to anchor frame {f['id']}! Using IMU fallback.")
                
    anchor_row("Top")
    anchor_row("Bottom")

    # 4. Warping
    out_w = args.width
    out_h = out_w // 2
    target_focal = out_w / (2 * np.pi)
    warper = cv2.PyRotationWarper("spherical", target_focal)
    
    logging.info("[INFO] Warping frames to Spherical Panorama...")
    warped_imgs = []
    warped_msks = []
    corners = []
    
    # Priority order for processing: Horizon first, so it dictates exposure
    # When blending, OpenCV typically blends them evenly, but our custom masks will protect Horizon pixels
    priority_order = ["Horizon", "Top", "Bottom", "Zenith", "Nadir"]
    ordered_frames = []
    for r_name in priority_order:
        ordered_frames.extend([f for f in frames_meta if f["row"] == r_name])
        
    for m in ordered_frames:
        img = images_dict[m["id"]]
        
        work_focal = m["f"] * args.scale
        work_scale = target_focal / work_focal
        img = cv2.resize(img, (0,0), fx=work_scale, fy=work_scale)
        
        K = np.array([[target_focal, 0, img.shape[1]/2], [0, target_focal, img.shape[0]/2], [0, 0, 1]], dtype=np.float32)
        R = get_rotation_matrix(m["y"], m["p"], m["r"])
        
        corner, warped = warper.warp(img, K, R, cv2.INTER_LANCZOS4, cv2.BORDER_REFLECT)
        
        # Apply intelligent mask
        base_mask = mask_for_row(m["row"], img.shape[0], img.shape[1])
        _, mask = warper.warp(base_mask, K, R, cv2.INTER_NEAREST, cv2.BORDER_CONSTANT)
        
        warped_imgs.append(warped)
        warped_msks.append(mask)
        corners.append(corner)

    # 5. Global Exposure, Seams & Blending
    logging.info("[INFO] Applying Exposure Compensation...")
    apply_blocks_gain_compensation(warped_imgs, warped_msks, corners)
    
    logging.info("[INFO] Finding GraphCut seams...")
    warped_msks = find_seams(warped_imgs, warped_msks, corners)
    
    logging.info("[INFO] Multi-band Blending...")
    final_pano, _ = blend_images(warped_imgs, warped_msks, corners, out_w, out_h, blend_strength=5.0)
    
    # Post-processing
    logging.info("[INFO] Applying final polish...")
    gaussian = cv2.GaussianBlur(final_pano, (0, 0), 3)
    final_pano = cv2.addWeighted(final_pano, 1.2, gaussian, -0.2, 0)
    
    out_path = "panorama_360_v6.jpg"
    cv2.imwrite(out_path, final_pano, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    logging.info(f"Done! Panorama saved to {out_path}")

if __name__ == "__main__":
    main()

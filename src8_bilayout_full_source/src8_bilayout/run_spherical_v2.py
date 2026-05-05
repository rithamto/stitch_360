from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

import cv2
import numpy as np

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from src2.random360_pipeline import prepare_images
from src6.config import PipelineConfig
from src6.loftr_matcher import get_loftr_matcher
from src8.spherical_warper import SphericalWarper

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Multi-ring Spherical panorama stitching.")
    parser.add_argument("--input", type=Path, required=True, help="Folder with HEIC/JPG/PNG images.")
    parser.add_argument("--workdir", type=Path, default=None, help="Output workdir.")
    parser.add_argument("--max-edge", type=int, default=1800, help="Prepared JPEG max side.")
    parser.add_argument("--quality", type=int, default=88, help="Prepared JPEG quality.")
    parser.add_argument("--limit", type=int, default=0, help="Use only first N images.")
    parser.add_argument("--force", action="store_true", help="Rebuild prepared JPEGs.")
    parser.add_argument("--match-max-dim", type=int, default=640, help="Longest edge fed to LoFTR.")
    parser.add_argument("--min-matches", type=int, default=10, help="Minimum raw LoFTR matches per pair.")
    parser.add_argument("--focal-ratio", type=float, default=0.8, help="Ratio of Image Max_Edge for focal length.")
    parser.add_argument("--compose-scale", type=float, default=0.5, help="Scale of final blend.")
    parser.add_argument("--loop", action="store_true", help="Match first and last frame of horizontal ring.")
    parser.add_argument("--log-level", default="INFO", help="Logging level.")
    return parser

def parse_metadata(filename: str) -> dict:
    match = re.search(r"frame_(\d+)_y([\d_\-]+)_p([\d_\-]+)_r([\d_\-]+)_f([\d_\-]+)_t(\d+)", filename)
    if match:
        return {
            "id": int(match.group(1)),
            "yaw": float(match.group(2).replace("_", ".")),
            "pitch": float(match.group(3).replace("_", ".")),
            "roll": float(match.group(4).replace("_", ".")),
            "focal": float(match.group(5).replace("_", ".")),
            "time": int(match.group(6))
        }
    return {"id": -1, "yaw": 0.0, "pitch": 0.0, "roll": 0.0, "focal": 0.0, "time": 0}

def default_workdir(input_dir: Path) -> Path:
    return Path("loftr_try_runs") / f"{input_dir.name}_sph"

def get_rotation_matrix(pitch_deg: float, roll_deg: float) -> np.ndarray:
    px = np.radians(pitch_deg)
    rz = np.radians(roll_deg)
    Rx = np.array([[1, 0, 0], [0, np.cos(px), -np.sin(px)], [0, np.sin(px), np.cos(px)]], dtype=np.float32)
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]], dtype=np.float32)
    return Rx @ Rz

def resized_for_loftr(gray: np.ndarray, match_max_dim: int) -> tuple[np.ndarray, float, float]:
    h, w = gray.shape[:2]
    scale = min(1.0, float(match_max_dim) / float(max(h, w)))
    new_w = max(8, (int(round(w * scale)) // 8) * 8)
    new_h = max(8, (int(round(h * scale)) // 8) * 8)
    resized = cv2.resize(gray, (new_w, new_h))
    scale_x = new_w / float(w)
    scale_y = new_h / float(h)
    return resized, scale_x, scale_y

def get_loftr_matches(
    matcher,
    img1_path: Path,
    img2_path: Path,
    match_max_dim: int,
    min_matches: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    img1 = cv2.imread(str(img1_path), cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(str(img2_path), cv2.IMREAD_GRAYSCALE)
    if img1 is None or img2 is None:
        return np.empty((0, 2)), np.empty((0, 2)), np.empty((0,))

    s_img1, scale1_x, scale1_y = resized_for_loftr(img1, match_max_dim)
    s_img2, scale2_x, scale2_y = resized_for_loftr(img2, match_max_dim)

    mkpts0, mkpts1, mconf = matcher.match(s_img1, s_img2)
    raw_count = len(mkpts0)
    if raw_count < min_matches:
        return np.empty((0, 2)), np.empty((0, 2)), np.empty((0,))

    mkpts0 = mkpts0.copy()
    mkpts1 = mkpts1.copy()
    mkpts0[:, 0] /= scale1_x
    mkpts0[:, 1] /= scale1_y
    mkpts1[:, 0] /= scale2_x
    mkpts1[:, 1] /= scale2_y
    return mkpts0, mkpts1, mconf

def estimate_spherical_translation(
    pts0: np.ndarray, 
    pts1: np.ndarray, 
    warper: SphericalWarper, 
    K: np.ndarray, 
    R0: np.ndarray,
    R1: np.ndarray,
    vertical_mode: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    pts0_sph = warper.warp_points(pts0, K, R0)
    pts1_sph = warper.warp_points(pts1, K, R1)
    
    # Foreground rejection for horizontal sweeps (ignore floor)
    if not vertical_mode:
        h = K[1, 2] * 2
        structural_mask = (pts0[:, 1] < h * 0.70) & (pts1[:, 1] < h * 0.70)
        if structural_mask.sum() < 10:
            structural_mask = np.ones(len(pts0), dtype=bool)
    else:
        # If matching ceiling to horizontal, we don't reject floor (ceiling has no floor).
        structural_mask = np.ones(len(pts0), dtype=bool)
        
    valid_pts0_sph = pts0_sph[structural_mask]
    valid_pts1_sph = pts1_sph[structural_mask]
    
    diffs = valid_pts0_sph - valid_pts1_sph
    tx = np.median(diffs[:, 0])
    ty = np.median(diffs[:, 1])
    
    dists = np.linalg.norm((pts0_sph - pts1_sph) - np.array([tx, ty]), axis=1)
    inliers = (dists < 5.0).astype(np.uint8).reshape(-1, 1)
    
    if inliers.sum() > 0:
        valid_diffs = (pts0_sph - pts1_sph)[inliers.flatten() == 1]
        tx = np.mean(valid_diffs[:, 0])
        ty = np.mean(valid_diffs[:, 1])

    M = np.array([[1.0, 0.0, tx], [0.0, 1.0, ty]], dtype=np.float32)
    return M, inliers

def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="[%(levelname)s] %(message)s",
    )
    print("STARTING MAIN", flush=True)

    workdir = args.workdir or default_workdir(args.input)
    workdir = Path(workdir)
    prepared_dir = prepare_images(
        input_dir=Path(args.input),
        workdir=workdir,
        max_edge=args.max_edge,
        quality=args.quality,
        limit=args.limit,
        force=args.force,
    )

    files = list(prepared_dir.glob("*.jpg"))
    if len(files) == 0:
        raise SystemExit("No prepared images found.")

    file_info = []
    for f in files:
        meta = parse_metadata(f.stem)
        file_info.append((meta, f))

    if args.limit > 0:
        file_info = file_info[:args.limit]

    # Partition Rings
    backbone_info = []
    ceiling_info = []
    floor_info = []
    
    for info in file_info:
        p = info[0]["pitch"]
        if p > 15.0:
            ceiling_info.append(info)
        elif p < -15.0:
            floor_info.append(info)
        else:
            backbone_info.append(info)
            
    logging.info(f"[rings] Backbone={len(backbone_info)}, Ceiling={len(ceiling_info)}, Floor={len(floor_info)}")

    backbone_info.sort(key=lambda x: x[0]["id"])
    
    if len(backbone_info) < 2:
        raise SystemExit("Need at least 2 horizontal images for the backbone.")

    # Stabilize Pitch and Roll for Backbone
    median_pitch = np.median([m[0]["pitch"] for m in backbone_info])
    median_roll = np.median([m[0]["roll"] for m in backbone_info])
    for m in backbone_info:
        m[0]["pitch"] = median_pitch
        m[0]["roll"] = median_roll
    logging.info(f"[run] Backbone locked to median pitch={median_pitch:.2f}, roll={median_roll:.2f}")

    config = PipelineConfig(input_dir=prepared_dir, workdir=workdir)
    print("Initializing LoFTR...", flush=True)
    matcher = get_loftr_matcher(config)
    print("LoFTR initialized.", flush=True)

    # Focal Length Calculation
    img0 = cv2.imread(str(backbone_info[0][1]))
    h, w = img0.shape[:2]
    
    # Remove prefix like "033_" created by prepare_images
    orig_name = backbone_info[0][1].name
    if re.match(r"\d{3}_", orig_name):
        orig_name = orig_name[4:]
        
    orig_path = Path(args.input) / orig_name
    orig_img = cv2.imread(str(orig_path))
    if orig_img is not None and backbone_info[0][0]["focal"] > 0:
        ho, wo = orig_img.shape[:2]
        true_focal = backbone_info[0][0]["focal"]
        focal_length = true_focal * (float(max(h, w)) / float(max(ho, wo)))
    else:
        focal_length = max(h, w) * args.focal_ratio

    K = np.array([[focal_length, 0, w / 2], [0, focal_length, h / 2], [0, 0, 1]], dtype=np.float32)
    R_eye = np.eye(3, dtype=np.float32)
    warper = SphericalWarper(focal_length)

    # 1. PROCESS BACKBONE
    print(f"Processing backbone with {len(backbone_info)} images...", flush=True)
    backbone_transforms = [np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)]
    
    for i in range(1, len(backbone_info)):
        prev_m, prev_f = backbone_info[i-1]
        curr_m, curr_f = backbone_info[i]
        
        mkpts_prev, mkpts_curr, _ = get_loftr_matches(matcher, prev_f, curr_f, args.match_max_dim, args.min_matches)
        
        if len(mkpts_prev) > 0:
            R_prev = get_rotation_matrix(prev_m["pitch"], prev_m["roll"])
            R_curr = get_rotation_matrix(curr_m["pitch"], curr_m["roll"])
            M_step, inliers = estimate_spherical_translation(mkpts_prev, mkpts_curr, warper, K, R_prev, R_curr)
            logging.info(f"[backbone] {i-1}->{i} inliers={inliers.sum()} ty={M_step[1,2]:.1f}")
            
            M_step_3x3 = np.vstack([M_step, [0.0, 0.0, 1.0]])
            T_prev_3x3 = np.vstack([backbone_transforms[-1], [0.0, 0.0, 1.0]])
            T_global_3x3 = T_prev_3x3 @ M_step_3x3
            backbone_transforms.append(T_global_3x3[:2, :])
        else:
            logging.error(f"[backbone] {i-1}->{i} MATCH FAILED!")
            backbone_transforms.append(backbone_transforms[-1].copy())

    tx_span = 2.0 * np.pi * focal_length
    
    if args.loop and len(backbone_info) > 2:
        last_m, last_f = backbone_info[-1]
        first_m, first_f = backbone_info[0]
        mkpts_last, mkpts_first, _ = get_loftr_matches(matcher, last_f, first_f, args.match_max_dim, args.min_matches)
        
        if len(mkpts_last) > 0:
            R_last = get_rotation_matrix(last_m["pitch"], last_m["roll"])
            R_first = get_rotation_matrix(first_m["pitch"], first_m["roll"])
            M_step, inliers = estimate_spherical_translation(mkpts_last, mkpts_first, warper, K, R_last, R_first)
            
            T_last_3x3 = np.vstack([backbone_transforms[-1], [0.0, 0.0, 1.0]])
            M_step_3x3 = np.vstack([M_step, [0.0, 0.0, 1.0]])
            T_loop_3x3 = T_last_3x3 @ M_step_3x3
            
            tx_span = abs(T_loop_3x3[0, 2])
            drift_y = T_loop_3x3[1, 2]
            logging.info(f"[loop] closure tx_span={tx_span:.2f}, drift_y={drift_y:.2f}. Correcting...")
            
            for i in range(1, len(backbone_info)):
                weight = i / float(len(backbone_info))
                backbone_transforms[i][1, 2] -= drift_y * weight

    # Base list of ALL things to render
    all_render_items = [] # list of (info_dict, original_file, T_matrix)
    
    for i in range(len(backbone_info)):
        all_render_items.append((backbone_info[i][0], backbone_info[i][1], backbone_transforms[i]))

    # 2. PROCESS CEILING / FLOOR ANCHORS
    def anchor_ring(ring_info, name="Ring"):
        # For each image in ring, find the closest backbone image by yaw
        for r_m, r_f in ring_info:
            r_yaw = r_m["yaw"]
            
            # Find closest backbone index by yaw difference
            best_idx = 0
            best_diff = 999
            for i, (b_m, _) in enumerate(backbone_info):
                diff = abs((b_m["yaw"] - r_yaw + 180) % 360 - 180)
                if diff < best_diff:
                    best_diff = diff
                    best_idx = i
                    
            b_m, b_f = backbone_info[best_idx]
            T_backbone = backbone_transforms[best_idx]
            
            logging.info(f"[{name}] matching frame {r_m['id']}(yaw={r_m['yaw']:.1f}) to Backbone {b_m['id']}(yaw={b_m['yaw']:.1f})")
            
            mkpts_b, mkpts_r, _ = get_loftr_matches(matcher, b_f, r_f, args.match_max_dim, args.min_matches)
            if len(mkpts_b) > 0:
                R_b = get_rotation_matrix(b_m["pitch"], b_m["roll"])
                R_r = get_rotation_matrix(r_m["pitch"], r_m["roll"]) # Keep raw IMU rotation for vertical tilt!
                
                # Match Backbone -> Ring. M maps Ring to Backbone coordinate space.
                M_step, inliers = estimate_spherical_translation(mkpts_b, mkpts_r, warper, K, R_b, R_r, vertical_mode=True)
                
                # T_ring = T_backbone * M_step
                M_step_3x3 = np.vstack([M_step, [0.0, 0.0, 1.0]])
                T_b_3x3 = np.vstack([T_backbone, [0.0, 0.0, 1.0]])
                T_ring_3x3 = T_b_3x3 @ M_step_3x3
                all_render_items.append((r_m, r_f, T_ring_3x3[:2, :]))
                logging.info(f"  -> Inliers: {inliers.sum()} offset ty={M_step[1,2]:.1f}")
            else:
                logging.error(f"  -> FAILED TO MATCH {r_m['id']} to {b_m['id']}")
                # Fallback: Just rely on raw IMU placement relative to backbone
                R_b = get_rotation_matrix(b_m["pitch"], b_m["roll"])
                R_r = get_rotation_matrix(r_m["pitch"], r_m["roll"])
                # Without tx,ty offset, just place it. M = Identity
                all_render_items.append((r_m, r_f, T_backbone.copy()))

    anchor_ring(ceiling_info, "Ceiling")
    anchor_ring(floor_info, "Floor")

    # 3. GLOBAL RENDER
    scale = args.compose_scale
    out_w = int(np.ceil(tx_span * scale))
    # Equirectangular constraint: Width = 360 deg, Height = 180 deg -> Height = Width / 2
    out_h = out_w // 2 
    
    logging.info(f"[render] Canvas Size: {out_w}x{out_h}")
    pano = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    max_alpha_map = -np.ones((out_h, out_w), dtype=np.float32)

    warped_items = []
    backbone_centers = []

    # Pre-warp all images and collect corners
    for idx, (m, f, T) in enumerate(all_render_items):
        img = cv2.imread(str(f))
        R_im = get_rotation_matrix(m["pitch"], m["roll"])
        
        corner, warped_img = warper.warp_image(img, K, R_im)
        mask = np.ones(img.shape[:2], dtype=np.uint8) * 255
        _, warped_mask = warper.warp_mask(mask, K, R_im)
        
        # Transform coords
        base_pt = np.array([corner[0], corner[1], 1.0]).reshape(3, 1)
        T_3x3 = np.vstack([T, [0.0, 0.0, 1.0]])
        global_pt = T_3x3 @ base_pt
        gx, gy = global_pt[0, 0], global_pt[1, 0]
        
        gx_scaled = gx * scale
        gy_scaled = gy * scale
        
        s_img = cv2.resize(warped_img, (0,0), fx=scale, fy=scale)
        s_mask = cv2.resize(warped_mask, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        
        dist = cv2.distanceTransform(s_mask, cv2.DIST_L2, 3)
        alpha = dist / (dist.max() + 1e-6)
        
        warped_items.append({
            "img": s_img,
            "alpha": alpha,
            "gx": gx_scaled,
            "gy": gy_scaled
        })
        
        # If this is a backbone image, track its center Y
        if abs(m["pitch"]) < 15.0:
            backbone_centers.append(gy_scaled + s_img.shape[0] / 2.0)

    # Shift Y so Backbone Equator perfectly lands on out_h / 2
    if len(backbone_centers) > 0:
        avg_bb_y = np.mean(backbone_centers)
        y_shift = (out_h / 2.0) - avg_bb_y
        logging.info(f"[render] Shifting Y by {y_shift:.1f} pixels to center the horizon.")
    else:
        y_shift = 0

    # Blending Phase
    for item in warped_items:
        gx = item["gx"]
        gy = item["gy"] + y_shift
        img = item["img"]
        alpha = item["alpha"]
        
        x_off = int(round(gx))
        y_off = int(round(gy))
        
        h_im, w_im = img.shape[:2]
        
        # Clamp Y
        y_start = max(0, y_off)
        y_end = min(out_h, y_off + h_im)
        if y_end <= y_start:
            continue
            
        src_y_start = y_start - y_off
        src_y_end = src_y_start + (y_end - y_start)
        dst_ys = slice(y_start, y_end)
        src_ys = slice(src_y_start, src_y_end)
        
        # Wrap X
        start_u = 0
        while start_u < w_im:
            dx_start = (x_off + start_u) % out_w
            chunk_len = min(w_im - start_u, out_w - dx_start)
            
            src_xs = slice(start_u, start_u + chunk_len)
            dst_xs = slice(dx_start, dx_start + chunk_len)
            
            arr_img = img[src_ys, src_xs, :]
            arr_alpha = alpha[src_ys, src_xs]
            
            # Voronoi / Max-Alpha selection for sharp, ghost-free seams
            mask_update = arr_alpha > max_alpha_map[dst_ys, dst_xs]
            
            for c in range(3):
                pano_c = pano[dst_ys, dst_xs, c]
                img_c = arr_img[:, :, c]
                pano[dst_ys, dst_xs, c] = np.where(mask_update, img_c, pano_c)
                
            alpha_view = max_alpha_map[dst_ys, dst_xs]
            max_alpha_map[dst_ys, dst_xs] = np.where(mask_update, arr_alpha, alpha_view)
            
            start_u += chunk_len

    output_path = workdir / "spherical_panorama.jpg"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), pano)
    logging.info(f"[done] saved to {output_path}")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())

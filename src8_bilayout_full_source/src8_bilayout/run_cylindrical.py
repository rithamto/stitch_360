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
from src8.cylindrical_warper import CylindricalWarper


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cylindrical no-metadata panorama stitching with LoFTR.")
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
    parser.add_argument("--horizontal-only", action="store_true", help="Only use images with pitch near 0.")
    parser.add_argument("--loop", action="store_true", help="Also match last frame back to first.")
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
    return Path("loftr_try_runs") / f"{input_dir.name}_cyl"


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
    files: list[Path],
    i: int,
    j: int,
    match_max_dim: int,
    min_matches: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    img1 = cv2.imread(str(files[i]), cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(str(files[j]), cv2.IMREAD_GRAYSCALE)
    if img1 is None or img2 is None:
        logging.warning(f"[pair] {i}->{j} unreadable image")
        return np.empty((0, 2)), np.empty((0, 2)), np.empty((0,))

    s_img1, scale1_x, scale1_y = resized_for_loftr(img1, match_max_dim)
    s_img2, scale2_x, scale2_y = resized_for_loftr(img2, match_max_dim)

    mkpts0, mkpts1, mconf = matcher.match(s_img1, s_img2)
    raw_count = len(mkpts0)
    logging.info(f"[pair] {i}->{j} raw_matches={raw_count}")
    if raw_count < min_matches:
        return np.empty((0, 2)), np.empty((0, 2)), np.empty((0,))

    mkpts0 = mkpts0.copy()
    mkpts1 = mkpts1.copy()
    mkpts0[:, 0] /= scale1_x
    mkpts0[:, 1] /= scale1_y
    mkpts1[:, 0] /= scale2_x
    mkpts1[:, 1] /= scale2_y

    return mkpts0, mkpts1, mconf


def estimate_cylindrical_translation(
    pts0: np.ndarray, 
    pts1: np.ndarray, 
    warper: CylindricalWarper, 
    K: np.ndarray, 
    R0: np.ndarray,
    R1: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    pts0 is matched to pts1. We want transformation T that aligns 1 to 0 (or 0 to 1).
    We warp points to cylindrical coords first.
    """
    pts0_cyl = warper.warp_points(pts0, K, R0)
    pts1_cyl = warper.warp_points(pts1, K, R1)
    
    # Foreground Rejection:
    # Ignore points in the bottom 30% of the image (floor, furniture) which exhibit massive parallax
    h = K[1, 2] * 2
    structural_mask = (pts0[:, 1] < h * 0.70) & (pts1[:, 1] < h * 0.70)
    
    if structural_mask.sum() < 10: # Fallback if we rejected too much
        structural_mask = np.ones(len(pts0), dtype=bool)
        
    valid_pts0_cyl = pts0_cyl[structural_mask]
    valid_pts1_cyl = pts1_cyl[structural_mask]
    
    # Estimate Pure Translation to completely block Scale/Rotation accumulation drift.
    diffs = valid_pts0_cyl - valid_pts1_cyl
    
    tx = np.median(diffs[:, 0])
    ty = np.median(diffs[:, 1])
    
    # RANSAC filtering with a distance threshold (e.g., 5.0 pixels)
    dists = np.linalg.norm((pts0_cyl - pts1_cyl) - np.array([tx, ty]), axis=1)
    inliers = (dists < 5.0).astype(np.uint8).reshape(-1, 1)
    
    # Use full inliers (even lower ones that fit) to refine mean
    if inliers.sum() > 0:
        valid_diffs = (pts0_cyl - pts1_cyl)[inliers.flatten() == 1]
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

    # Parse metadata and sort by frame ID to ensure sequence
    file_info = []
    for f in files:
        meta = parse_metadata(f.stem)
        file_info.append((meta, f))

    # Sort strictly by frame index
    file_info.sort(key=lambda x: x[0]["id"])
    
    # Filter horizontal ring if requested
    if args.horizontal_only:
        file_info = [x for x in file_info if abs(x[0]["pitch"]) < 15.0]
        logging.info(f"[run] horizontal-only mode: kept {len(file_info)} images")

    # Apply limit
    if args.limit > 0:
        file_info = file_info[:args.limit]

    files = [x[1] for x in file_info]
    metadata_list = [x[0] for x in file_info]

    # Stabilize Pitch and Roll per ring to remove IMU high-frequency shake/wobble
    if metadata_list:
        median_pitch = np.median([m["pitch"] for m in metadata_list])
        median_roll = np.median([m["roll"] for m in metadata_list])
        for m in metadata_list:
            m["pitch"] = median_pitch
            m["roll"] = median_roll
        logging.info(f"[run] IMU noise suppressed. Locked ring slightly to median pitch={median_pitch:.2f}, roll={median_roll:.2f}")

    if len(files) < 2:
        raise SystemExit("Need at least 2 prepared images after filtering.")

    config = PipelineConfig(input_dir=prepared_dir, workdir=workdir)
    matcher = get_loftr_matcher(config)
    if matcher is None:
        raise SystemExit("LoFTR is unavailable in this environment.")

    n_images = len(files)
    logging.info(f"[run] prepared_images={n_images}")

    # Calculate True Scaled Focal Length
    img0 = cv2.imread(str(files[0]))
    h, w = img0.shape[:2]
    
    orig_path = Path(args.input) / files[0].name
    orig_img = cv2.imread(str(orig_path))
    if orig_img is not None and metadata_list[0]["focal"] > 0:
        ho, wo = orig_img.shape[:2]
        true_focal = metadata_list[0]["focal"]
        scale_ratio = float(max(h, w)) / float(max(ho, wo))
        focal_length = true_focal * scale_ratio
        logging.info(f"[run] true focal scaled: {true_focal} * {scale_ratio:.4f} = {focal_length:.2f}")
    else:
        focal_length = max(h, w) * args.focal_ratio
        logging.info(f"[run] estimated focal_length={focal_length} (fallback)")

    K = np.array([[focal_length, 0, w / 2], [0, focal_length, h / 2], [0, 0, 1]], dtype=np.float32)
    R = np.eye(3, dtype=np.float32)
    warper = CylindricalWarper(focal_length)

    transforms = [np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)] # Identity for first image
    
    success_pairs = 0
    # Sequential Matching
    # Note: M maps img(i) to img(i-1)'s coordinate space
    for i in range(1, n_images):
        mkpts_prev, mkpts_curr, _ = get_loftr_matches(matcher, files, i - 1, i, args.match_max_dim, args.min_matches)
        
        # Use yaw difference from metadata as a baseline prior if available
        yaw_diff = metadata_list[i]["yaw"] - metadata_list[i-1]["yaw"]
        if yaw_diff < -180: yaw_diff += 360 # Handle wraparound visually
        if yaw_diff > 180: yaw_diff -= 360
        logging.info(f"[pair] {i-1}->{i} expected yaw_diff={yaw_diff:.1f}")

        if len(mkpts_prev) > 0:
            R_prev = get_rotation_matrix(metadata_list[i-1]["pitch"], metadata_list[i-1]["roll"])
            R_curr = get_rotation_matrix(metadata_list[i]["pitch"], metadata_list[i]["roll"])
            M_step, inliers = estimate_cylindrical_translation(mkpts_prev, mkpts_curr, warper, K, R_prev, R_curr)
            logging.info(f"[pair] {i-1}->{i} inliers={inliers.sum()}/{len(inliers)} T={M_step[:, 2]}")
            
            # Combine transforms: Current to Global
            # T_global(i) = T_global(i-1) * M_step
            # For affine M(2x3), we can do:
            M_step_3x3 = np.vstack([M_step, [0.0, 0.0, 1.0]])
            T_prev_3x3 = np.vstack([transforms[-1], [0.0, 0.0, 1.0]])
            T_global_3x3 = T_prev_3x3 @ M_step_3x3
            transforms.append(T_global_3x3[:2, :])
            success_pairs += 1
        else:
            logging.error(f"[pair] {i-1}->{i} MATCH FAILED!")
            transforms.append(transforms[-1].copy()) # Fallback: No movement
            
    # Loop Closure logic
    if args.loop and n_images > 2:
        logging.info("[run] Performing Loop Closure matching (N-1 -> 0)")
        mkpts_last, mkpts_first, _ = get_loftr_matches(matcher, files, n_images - 1, 0, args.match_max_dim, args.min_matches)
        
        if len(mkpts_last) > 0:
            R_last = get_rotation_matrix(metadata_list[n_images - 1]["pitch"], metadata_list[n_images - 1]["roll"])
            R_first = get_rotation_matrix(metadata_list[0]["pitch"], metadata_list[0]["roll"])
            M_step, inliers = estimate_cylindrical_translation(mkpts_last, mkpts_first, warper, K, R_last, R_first)
            logging.info(f"[loop] inliers={inliers.sum()}/{len(inliers)} T={M_step[:, 2]}")
            
            # Global pos of wrapped image 0
            T_last_3x3 = np.vstack([transforms[-1], [0.0, 0.0, 1.0]])
            M_step_3x3 = np.vstack([M_step, [0.0, 0.0, 1.0]])
            T_loop_3x3 = T_last_3x3 @ M_step_3x3
            
            tx_loop = T_loop_3x3[0, 2]
            ty_loop = T_loop_3x3[1, 2]
            
            # Store loop span (tx_loop) globally to wrap our 360-degree panorama dynamically 
            # instead of trusting inaccurate metadata focal lengths
            globals()['tx_loop_val'] = tx_loop
            
            drift_y = ty_loop - 0.0
            
            logging.info(f"[loop] closure tx_span={tx_loop:.2f}, drift_y={drift_y:.2f}. Correcting sequence...")
            
            for i in range(1, n_images):
                weight = i / float(n_images)
                transforms[i][1, 2] -= drift_y * weight
        else:
            globals()['tx_loop_val'] = None
            logging.warning("[loop] Failed to match last frame with first frame. Skipping loop closure.")

    logging.info(f"[run] matched_pairs={success_pairs}")
    
    # Warping all images to cylindrical space
    warped_images = []
    warped_masks = []
    corners = []
    
    for i in range(n_images):
        img = cv2.imread(str(files[i]))
        corner, warped_img = warper.warp_image(img, K, R)
        mask = np.ones(img.shape[:2], dtype=np.uint8) * 255
        _, warped_mask = warper.warp_mask(mask, K, R)
        
        corners.append(corner)
        warped_images.append(warped_img)
        warped_masks.append(warped_mask)
        
    # Scale composition if needed
    scale = args.compose_scale
    
    # Transform corners to global space
    global_corners = []
    scaled_images = []
    scaled_masks = []
    for corner, img, mask, T in zip(corners, warped_images, warped_masks, transforms):
        # Top-left corner in current cylindrical coord
        base_x, base_y = corner
        # We need to apply T to the base coordinate 
        # But wait, T transforms points. So corner transform is:
        base_pt = np.array([base_x, base_y, 1.0]).reshape(3, 1)
        T_3x3 = np.vstack([T, [0.0, 0.0, 1.0]])
        global_pt = T_3x3 @ base_pt
        gx, gy = global_pt[0, 0], global_pt[1, 0]
        
        gx_scaled = gx * scale
        gy_scaled = gy * scale
        global_corners.append((gx_scaled, gy_scaled))
        
        s_img = cv2.resize(img, (0,0), fx=scale, fy=scale)
        s_mask = cv2.resize(mask, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        scaled_images.append(s_img)
        scaled_masks.append(s_mask)

    # Compute bounding box
    # Dynamic 360 Width from LoFTR Loop
    tx_span = globals().get('tx_loop_val', None)
    if tx_span is not None and abs(tx_span) > 0:
        out_w = int(np.ceil(abs(tx_span) * scale))
        logging.info(f"[run] using dynamic tx_loop bounds out_w={out_w}")
    else:
        out_w = int(np.ceil(2.0 * np.pi * focal_length * scale))
        
    min_y = min([c[1] for c in global_corners])
    max_y = max([c[1] + img.shape[0] for c, img in zip(global_corners, scaled_images)])
    out_h = int(np.ceil(max_y - min_y))
    
    logging.info(f"[run] creating closed 360 panorama {out_w}x{out_h} with max-alpha seam clipping")
    pano = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    max_alpha_map = -np.ones((out_h, out_w), dtype=np.float32)
    
    for corner, img, mask in zip(global_corners, scaled_images, scaled_masks):
        # Distance transform for feathering/alpha blending
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
        alpha = dist / (dist.max() + 1e-6)
        
        gx, gy = corner
        x_off = int(round(gx))
        y_off = int(round(gy - min_y))
        
        h_im, w_im = img.shape[:2]
        
        y_start = max(0, y_off)
        y_end = min(out_h, y_off + h_im)
        if y_end <= y_start:
            continue
            
        src_y_start = y_start - y_off
        src_y_end = src_y_start + (y_end - y_start)
        dst_ys = slice(y_start, y_end)
        src_ys = slice(src_y_start, src_y_end)
        
        start_u = 0
        while start_u < w_im:
            dx_start = (x_off + start_u) % out_w
            # Contiguous chunk until image end OR canvas wrap boundary
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

    output = pano
    
    output_path = workdir / "cylindrical_panorama.jpg"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), output)
    logging.info(f"[done] saved to {output_path}")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import csv
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
from src8.solver_backbone_extracted import solve_backbone_least_squares

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Spherical multi-row panorama stitching with backbone anchoring.")
    parser.add_argument("--input", type=Path, required=True, help="Folder with images.")
    parser.add_argument("--workdir", type=Path, default=None, help="Output workdir.")
    parser.add_argument("--max-edge", type=int, default=1800, help="Prepared JPEG max side.")
    parser.add_argument("--quality", type=int, default=88, help="Prepared JPEG quality.")
    parser.add_argument("--limit", type=int, default=0, help="Use only first N images.")
    parser.add_argument("--k1", type=float, default=0.0, help="Radial distortion k1.")
    parser.add_argument("--k2", type=float, default=0.0, help="Radial distortion k2.")
    parser.add_argument("--k3", type=float, default=0.0, help="Radial distortion k3.")
    parser.add_argument("--p1", type=float, default=0.0, help="Tangential distortion p1.")
    parser.add_argument("--p2", type=float, default=0.0, help="Tangential distortion p2.")
    parser.add_argument("--force", action="store_true", help="Rebuild prepared JPEGs.")
    parser.add_argument("--match-max-dim", type=int, default=640, help="Longest edge fed to LoFTR.")
    parser.add_argument("--min-matches", type=int, default=10, help="Minimum raw LoFTR matches per pair.")
    parser.add_argument("--pair-min-inliers", type=int, default=12, help="Minimum spherical inliers for accepting a pair.")
    parser.add_argument("--pair-min-confidence", type=float, default=0.28, help="Minimum mean LoFTR confidence for accepting a pair.")
    parser.add_argument("--focal-ratio", type=float, default=0.8, help="Ratio of Image Max_Edge for focal length.")
    parser.add_argument("--use-source-focal", action="store_true", help="Use focal from source metadata scaled via prepared_manifest mapping.")
    parser.add_argument("--export-bi-layout", action="store_true", default=True, help="Force export exactly 1024x512 canvas for Bi_Layout (DEFAULT ON in this version).")
    parser.add_argument("--compose-scale", type=float, default=0.5, help="Scale of final blend.")
    parser.add_argument("--upper-anchor-candidates", type=int, default=3, help="Try up to K nearest backbone anchors per upper frame.")
    parser.add_argument("--upper-anchor-max-yaw-diff", type=float, default=18.0, help="Max yaw difference when considering backbone anchors.")
    parser.add_argument("--upper-pitch-bound", type=float, default=8.0, help="Max pitch correction from IMU for upper band optimization.")
    parser.add_argument("--upper-roll-bound", type=float, default=1.0, help="Max roll correction from IMU for upper band optimization.")
    parser.add_argument("--upper-pitch-prior-weight", type=float, default=0.10, help="Weight of pitch prior for upper band optimization.")
    parser.add_argument("--upper-roll-prior-weight", type=float, default=2.00, help="Weight of roll prior for upper band optimization.")
    parser.add_argument("--upper-x-smooth-weight", type=float, default=0.10, help="Weight of yaw-derived X smoothness prior for upper band optimization.")
    parser.add_argument("--upper-p-smooth-weight", type=float, default=0.50, help="Weight of pitch smoothness prior for upper band optimization.")
    parser.add_argument("--upper-r-smooth-weight", type=float, default=1.00, help="Weight of roll smoothness prior for upper band optimization.")
    parser.add_argument("--anchor-x-weight", type=float, default=2.00, help="Weight of horizontal anchor constraint to avoid yaw drift.")
    parser.add_argument("--anchor-y-weight", type=float, default=2.00, help="Weight of vertical anchor constraint to avoid pitch drift.")
    parser.add_argument("--upper1-alpha", type=float, default=0.85, help="Blend alpha cap for upper1 rendering.")
    parser.add_argument("--upper2-alpha", type=float, default=0.70, help="Blend alpha cap for upper2 rendering.")
    parser.add_argument("--lower1-alpha", type=float, default=0.80, help="Blend alpha cap for lower1 rendering.")
    parser.add_argument("--lower2-alpha", type=float, default=0.60, help="Blend alpha cap for lower2 rendering.")
    parser.add_argument("--ceiling-alpha", type=float, default=0.15, help="Blend alpha cap for ceiling rendering.")
    parser.add_argument("--floor-alpha", type=float, default=0.15, help="Blend alpha cap for floor rendering.")
    parser.add_argument("--skip-ceiling-floor", action="store_true", default=True, help="Skip ceiling and floor stitching.")
    parser.add_argument("--inlier-base-threshold", type=float, default=5.0, help="Base inlier threshold in pixels (pitch=0).")
    parser.add_argument("--inlier-pitch-scale", type=float, default=20.0, help="Pitch offset in degrees to double the threshold.")
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
    return Path("loftr_try_runs") / f"{input_dir.name}_sph_v3"


def strip_prepared_prefix(filename: str) -> str:
    if re.match(r"\d{3}_", filename):
        return filename[4:]
    return filename


def load_prepared_source_map(manifest_path: Path, input_dir: Path) -> dict[str, Path]:
    source_map: dict[str, Path] = {}
    if not manifest_path.exists():
        return source_map
    with manifest_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            prepared_name = row.get("prepared_name")
            source_name = row.get("source_name")
            if prepared_name and source_name:
                source_map[prepared_name] = Path(input_dir) / source_name
    return source_map


def normalize_yaw_delta(delta_deg: float) -> float:
    if delta_deg <= -180.0:
        delta_deg += 360.0
    if delta_deg > 180.0:
        delta_deg -= 360.0
    return delta_deg


def yaw_delta_to_tx(delta_deg: float, focal_length: float) -> float:
    return float(np.radians(normalize_yaw_delta(delta_deg)) * focal_length)

def get_rotation_matrix(pitch_deg: float, roll_deg: float) -> np.ndarray:
    px = np.radians(pitch_deg)
    rz = np.radians(roll_deg)
    Rx = np.array([[1, 0, 0], [0, np.cos(px), -np.sin(px)], [0, np.sin(px), np.cos(px)]], dtype=np.float32)
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]], dtype=np.float32)
    return Rx @ Rz

def load_and_undistort_image(path: Path | str, K: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        return img
    if args.k1 == 0.0 and args.k2 == 0.0 and args.k3 == 0.0 and args.p1 == 0.0 and args.p2 == 0.0:
        return img
    D = np.array([args.k1, args.k2, args.p1, args.p2, args.k3], dtype=np.float32)
    return cv2.undistort(img, K, D, None, K)


def compute_exposure_gain(ref_img: np.ndarray, tgt_img: np.ndarray) -> float:
    """Compute gain factor so tgt_img matches ref_img brightness."""
    ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY).astype(np.float64)
    tgt_gray = cv2.cvtColor(tgt_img, cv2.COLOR_BGR2GRAY).astype(np.float64)
    ref_mean = max(ref_gray.mean(), 1.0)
    tgt_mean = max(tgt_gray.mean(), 1.0)
    gain = ref_mean / tgt_mean
    return float(np.clip(gain, 0.5, 2.0))


def apply_exposure_gain(image: np.ndarray, gain: float) -> np.ndarray:
    """Apply exposure gain, clamp to [0, 255]."""
    if abs(gain - 1.0) < 0.01:
        return image
    return np.clip(image.astype(np.float64) * gain, 0, 255).astype(np.uint8)

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
    matcher, img1: np.ndarray, img2: np.ndarray, match_max_dim: int, min_matches: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(img1.shape) == 3: img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    if len(img2.shape) == 3: img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    s_img1, scale1_x, scale1_y = resized_for_loftr(img1, match_max_dim)
    s_img2, scale2_x, scale2_y = resized_for_loftr(img2, match_max_dim)

    mkpts0, mkpts1, mconf = matcher.match(s_img1, s_img2)
    if len(mkpts0) < min_matches:
        return np.empty((0, 2)), np.empty((0, 2)), np.empty((0,))

    mkpts0 = mkpts0.copy()
    mkpts1 = mkpts1.copy()
    mkpts0[:, 0] /= scale1_x
    mkpts0[:, 1] /= scale1_y
    mkpts1[:, 0] /= scale2_x
    mkpts1[:, 1] /= scale2_y
    return mkpts0, mkpts1, mconf

def _geometry_point_mask(pts0, pts1, h, pitch0_deg, pitch1_deg, is_horizontal):
    """Geometry-aware ROI mask — relaxed for lower rings to preserve features."""
    if is_horizontal:
        avg_pitch = (pitch0_deg + pitch1_deg) / 2.0
        if avg_pitch < -35:  # lower2: very relaxed
            mask = (pts0[:, 1] < h * 0.55) & (pts1[:, 1] < h * 0.55)
        elif avg_pitch < -10:  # lower1
            mask = (pts0[:, 1] < h * 0.55) & (pts1[:, 1] < h * 0.55)
        elif avg_pitch > 35:  # upper2
            mask = (pts0[:, 1] >= h * 0.25) & (pts1[:, 1] >= h * 0.25)
        elif avg_pitch > 10:  # upper1
            mask = (pts0[:, 1] > h * 0.10) & (pts1[:, 1] > h * 0.10)
        else:
            mask = (pts0[:, 1] >= h * 0.15) & (pts0[:, 1] <= h * 0.85)
            mask &= (pts1[:, 1] >= h * 0.15) & (pts1[:, 1] <= h * 0.85)
    else:  # vertical anchor matching
        if pitch1_deg < pitch0_deg:
            mask = (pts0[:, 1] > h * 0.20) & (pts1[:, 1] < h * 0.80)
        else:
            mask = (pts0[:, 1] < h * 0.80) & (pts1[:, 1] > h * 0.20)
    return mask

def estimate_translation(
    pts0: np.ndarray,
    pts1: np.ndarray,
    warper: SphericalWarper,
    K: np.ndarray,
    R0: np.ndarray,
    R1: np.ndarray,
    pitch0_deg: float,
    pitch1_deg: float,
    args: argparse.Namespace,
    is_horizontal: bool,
    confidences: np.ndarray | None = None,
    prior_tx: float | None = None,
) -> dict:
    pts0_sph = warper.warp_points(pts0, K, R0)
    pts1_sph = warper.warp_points(pts1, K, R1)

    h = K[1, 2] * 2
    mask = _geometry_point_mask(pts0, pts1, h, pitch0_deg, pitch1_deg, is_horizontal)

    diffs = pts0_sph[mask] - pts1_sph[mask]
    if diffs.size == 0:
        empty_mask = np.zeros(len(pts0), dtype=bool)
        return {
            "tx": 0.0,
            "ty": 0.0,
            "inlier_mask": empty_mask,
            "inliers": 0,
            "mean_confidence": 0.0,
            "median_error": float("inf"),
        }

    avg_pitch = (abs(pitch0_deg) + abs(pitch1_deg)) / 2.0
    scale_factor = 1.0 + (avg_pitch / args.inlier_pitch_scale)
    adaptive_threshold = args.inlier_base_threshold * scale_factor

    if prior_tx is not None:
        # Prior-guided mode-seeking clustering
        focal_length = K[0, 0]
        max_yaw_drift_px = 15.0 * (np.pi / 180.0) * focal_length
        plausible_mask = np.abs(diffs[:, 0] - prior_tx) < max_yaw_drift_px
        
        plausible_diffs = diffs[plausible_mask]
        if len(plausible_diffs) > 0:
            best_vote = -1
            best_tx = 0.0
            best_ty = 0.0
            
            for i in range(len(plausible_diffs)):
                T_i = plausible_diffs[i]
                dists_i = np.linalg.norm(plausible_diffs - T_i, axis=1)
                inliers_i = int(np.sum(dists_i < adaptive_threshold))
                
                if inliers_i > best_vote:
                    best_vote = inliers_i
                    best_tx, best_ty = T_i
                elif inliers_i == best_vote:
                    if abs(T_i[0] - prior_tx) < abs(best_tx - prior_tx):
                        best_tx, best_ty = T_i
            
            tx = float(best_tx)
            ty = float(best_ty)
        else:
            tx = prior_tx
            ty = 0.0
    else:
        tx = float(np.median(diffs[:, 0]))
        ty = float(np.median(diffs[:, 1]))

    avg_pitch = (abs(pitch0_deg) + abs(pitch1_deg)) / 2.0
    scale_factor = 1.0 + (avg_pitch / args.inlier_pitch_scale)
    adaptive_threshold = args.inlier_base_threshold * scale_factor

    dists = np.linalg.norm((pts0_sph - pts1_sph) - np.array([tx, ty]), axis=1)
    inliers = dists < adaptive_threshold

    if inliers.sum() > 0:
        valid_diffs = (pts0_sph - pts1_sph)[inliers]
        if confidences is not None and len(confidences) == len(pts0):
            weights = np.clip(confidences[inliers].astype(np.float64), 1e-3, None)
            tx = np.average(valid_diffs[:, 0], weights=weights)
            ty = np.average(valid_diffs[:, 1], weights=weights)
        else:
            tx = np.mean(valid_diffs[:, 0])
            ty = np.mean(valid_diffs[:, 1])

    if confidences is not None and len(confidences) == len(pts0):
        if inliers.sum() > 0:
            mean_confidence = float(np.mean(confidences[inliers]))
        else:
            mean_confidence = float(np.mean(confidences))
    else:
        mean_confidence = 0.0

    median_error = float(np.median(dists[inliers])) if inliers.sum() > 0 else float(np.median(dists))
    return {
        "tx": float(tx),
        "ty": float(ty),
        "inlier_mask": inliers,
        "inliers": int(inliers.sum()),
        "mean_confidence": mean_confidence,
        "median_error": median_error,
    }


def accept_pair(raw_matches: int, stats: dict, min_inliers: int, min_confidence: float) -> tuple[bool, str]:
    if raw_matches <= 0:
        return False, "no_matches"
    if stats["inliers"] < min_inliers:
        return False, "low_inliers"
    if stats["mean_confidence"] < min_confidence:
        return False, "low_confidence"
    return True, "accepted"


def append_diag(rows: list[dict], **kwargs) -> None:
    rows.append(dict(kwargs))


def write_diagnostics(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_anchor_states(
    group_name: str,
    infos: list[tuple[dict, Path]],
    images: list[np.ndarray],
    xs: np.ndarray | list[float],
    pitches: np.ndarray | list[float],
    rolls: np.ndarray | list[float],
    priority: int,
) -> list[dict]:
    states = []
    for i, ((meta, file_path), image) in enumerate(zip(infos, images)):
        states.append(
            {
                "group": group_name,
                "priority": priority,
                "meta": meta,
                "file": file_path,
                "image": image,
                "x": float(xs[i]),
                "pitch": float(pitches[i]),
                "roll": float(rolls[i]),
            }
        )
    return states


def optimize_upper_ring(
    ring_name: str,
    ring_info: list[tuple[dict, Path]],
    anchor_states: list[dict],
    matcher,
    warper: SphericalWarper,
    K: np.ndarray,
    focal_length: float,
    args,
    diagnostics_rows: list[dict],
) -> tuple[list[dict], list[np.ndarray]]:
    num_ring = len(ring_info)
    if num_ring == 0:
        return [], []

    ring_images = []
    # Use first anchor image as exposure reference for ring
    ref_img_for_gain = anchor_states[0]["image"] if anchor_states else None
    for _, file_path in ring_info:
        image = load_and_undistort_image(file_path, K, args)
        if image is None:
            raise SystemExit(f"Failed to read ring image: {file_path}")
        if ref_img_for_gain is not None:
            gain = compute_exposure_gain(ref_img_for_gain, image)
            image = apply_exposure_gain(image, gain)
        ring_images.append(image)

    initial_X = np.zeros(num_ring, dtype=np.float64)
    initial_P = np.array([meta["pitch"] for meta, _ in ring_info], dtype=np.float64)
    initial_R = np.array([meta["roll"] for meta, _ in ring_info], dtype=np.float64)
    ring_pitch_meta = initial_P.copy()
    ring_roll_meta = initial_R.copy()
    valid_vertical = []

    v_kpts_ring = []
    v_target_gx = []
    v_target_gy = []
    v_ring_indices = []

    logging.info("[run] Starting %s alignment...", ring_name)

    for i in range(num_ring):
        r_meta, r_file = ring_info[i]
        img_ring = ring_images[i]

        anchor_candidates = []
        for a_idx, anchor in enumerate(anchor_states):
            diff = abs((r_meta["yaw"] - anchor["meta"]["yaw"] + 180.0) % 360.0 - 180.0)
            anchor_candidates.append((diff, anchor["priority"], a_idx))
        anchor_candidates.sort(key=lambda item: (item[0], item[1]))
        nearby = [item for item in anchor_candidates if item[0] <= args.upper_anchor_max_yaw_diff]
        if not nearby:
            nearby = anchor_candidates[:1]
        else:
            nearby = nearby[: max(1, args.upper_anchor_candidates)]

        best_score = None
        best_anchor_count = 0
        best_anchor_idx = nearby[0][2]
        R_ring_guess = get_rotation_matrix(r_meta["pitch"], r_meta["roll"])

        for rank, (yaw_gap, _priority, anchor_idx) in enumerate(nearby):
            anchor = anchor_states[anchor_idx]
            avg_pitch = (abs(r_meta["pitch"]) + abs(anchor["pitch"])) / 2.0
            effective_match_dim = args.match_max_dim
            effective_min_inliers = max(4, int(args.pair_min_inliers * (1.0 - avg_pitch / 70.0)))

            kpts_anchor, kpts_ring, conf = get_loftr_matches(
                matcher,
                anchor["image"],
                img_ring,
                effective_match_dim,
                args.min_matches,
            )
            if len(kpts_ring) > 0:
                R_anchor = get_rotation_matrix(anchor["pitch"], anchor["roll"])
                expected_dx = yaw_delta_to_tx(r_meta["yaw"] - anchor["meta"]["yaw"], K[0, 0])
                stats = estimate_translation(
                    kpts_anchor, kpts_ring, warper, K, R_anchor, R_ring_guess, 
                    anchor["pitch"], r_meta["pitch"], args, is_horizontal=False, confidences=conf, prior_tx=expected_dx
                )
                accept, reason = accept_pair(len(kpts_ring), stats, effective_min_inliers, args.pair_min_confidence)
            else:
                stats = {"tx": 0.0, "ty": 0.0, "inlier_mask": np.zeros(0, dtype=bool), "inliers": 0, "mean_confidence": 0.0}
                accept, reason = False, "match_failed"

            append_diag(
                diagnostics_rows,
                stage=f"{ring_name}_anchor",
                ring_idx=i,
                file_ring=r_file.name,
                anchor_idx=anchor_idx,
                file_anchor=anchor["file"].name,
                anchor_group=anchor["group"],
                anchor_rank=rank,
                yaw_diff_deg=f"{yaw_gap:.3f}",
                raw_matches=len(kpts_ring),
                inliers=stats["inliers"],
                mean_confidence=f"{stats['mean_confidence']:.4f}",
                tx=f"{stats['tx']:.3f}",
                ty=f"{stats['ty']:.3f}",
                accepted=int(accept),
                reason=reason,
            )
            if not accept:
                continue

            inlier_mask = stats["inlier_mask"]
            kpts_anchor_in = kpts_anchor[inlier_mask]
            kpts_ring_in = kpts_ring[inlier_mask]
            if len(kpts_ring_in) == 0:
                continue

            R_anchor = get_rotation_matrix(anchor["pitch"], anchor["roll"])
            pts_anchor_sph = warper.warp_points(kpts_anchor_in, K, R_anchor)
            gx_anchor = pts_anchor_sph[:, 0] + anchor["x"]
            gy_anchor = pts_anchor_sph[:, 1]
            pts_ring_sph = warper.warp_points(kpts_ring_in, K, R_ring_guess)

            v_kpts_ring.append(kpts_ring_in)
            v_target_gx.append(gx_anchor)
            v_target_gy.append(gy_anchor)
            v_ring_indices.append(i)
            best_anchor_count += 1

            dx = np.median(pts_anchor_sph[:, 0] - pts_ring_sph[:, 0])
            dy = np.median(pts_anchor_sph[:, 1] - pts_ring_sph[:, 1])
            candidate_x = anchor["x"] + dx
            candidate_p = r_meta["pitch"] - (dy / focal_length) * (180.0 / np.pi)
            score = (stats["inliers"], stats["mean_confidence"], -yaw_gap, -anchor["priority"])
            if best_score is None or score > best_score:
                best_score = score
                best_anchor_idx = anchor_idx
                initial_X[i] = candidate_x
                initial_P[i] = candidate_p
                initial_R[i] = r_meta["roll"]
            
            if rank == 0 and stats["inliers"] >= 30 and stats["mean_confidence"] >= 0.40:
                logging.info(f"  [{ring_name}_v] {r_file.stem} early exit on strong anchor (inliers={stats['inliers']})")
                break

        if best_score is not None:
            valid_vertical.append(True)
            best_anchor = anchor_states[best_anchor_idx]
            logging.info(
                f"  [{ring_name}_v] {r_file.stem} accepted {best_anchor_count} anchors; "
                f"best={best_anchor['group']}:{best_anchor['file'].stem} P={initial_P[i]:.1f}"
            )
        else:
            fallback_anchor = anchor_states[best_anchor_idx]
            if i > 0:
                yaw_dx = yaw_delta_to_tx(r_meta["yaw"] - ring_info[i - 1][0]["yaw"], focal_length)
                initial_X[i] = initial_X[i - 1] + yaw_dx
            else:
                yaw_dx = yaw_delta_to_tx(r_meta["yaw"] - fallback_anchor["meta"]["yaw"], focal_length)
                initial_X[i] = fallback_anchor["x"] + yaw_dx
            initial_P[i] = r_meta["pitch"]
            initial_R[i] = r_meta["roll"]
            valid_vertical.append(False)
            logging.warning(f"  [{ring_name}_v] {r_file.name} MISSING strong anchor, falling back to yaw prior.")

    h_kpts_prev = []
    h_kpts_curr = []
    h_prev_indices = []
    h_curr_indices = []
    for i in range(1, num_ring):
        prev_meta, prev_file = ring_info[i - 1]
        curr_meta, curr_file = ring_info[i]
        img_prev = ring_images[i - 1]
        img_curr = ring_images[i]

        avg_pitch = (abs(curr_meta["pitch"]) + abs(prev_meta["pitch"])) / 2.0
        effective_match_dim = args.match_max_dim
        effective_min_inliers = max(4, int(args.pair_min_inliers * (1.0 - avg_pitch / 70.0)))

        kpts_prev, kpts_curr, conf = get_loftr_matches(matcher, img_prev, img_curr, effective_match_dim, args.min_matches)
        expected_dx = yaw_delta_to_tx(curr_meta["yaw"] - prev_meta["yaw"], focal_length)

        if len(kpts_curr) > 0:
            R_prev = get_rotation_matrix(initial_P[i - 1], initial_R[i - 1])
            R_curr_guess = get_rotation_matrix(curr_meta["pitch"], curr_meta["roll"])
            stats = estimate_translation(
                kpts_prev, kpts_curr, warper, K, R_prev, R_curr_guess,
                initial_P[i - 1], curr_meta["pitch"], args, is_horizontal=True, confidences=conf, prior_tx=expected_dx
            )
            accept, reason = accept_pair(len(kpts_curr), stats, effective_min_inliers, args.pair_min_confidence)
            append_diag(
                diagnostics_rows,
                stage=f"{ring_name}_chain",
                idx_a=i - 1,
                idx_b=i,
                file_a=prev_file.name,
                file_b=curr_file.name,
                yaw_diff_deg=f"{normalize_yaw_delta(curr_meta['yaw'] - prev_meta['yaw']):.3f}",
                raw_matches=len(kpts_curr),
                inliers=stats["inliers"],
                mean_confidence=f"{stats['mean_confidence']:.4f}",
                tx=f"{stats['tx']:.3f}",
                ty=f"{stats['ty']:.3f}",
                prior_tx=f"{expected_dx:.3f}",
                accepted=int(accept),
                reason=reason,
            )
            if accept:
                inlier_mask = stats["inlier_mask"]
                h_kpts_prev.append(kpts_prev[inlier_mask])
                h_kpts_curr.append(kpts_curr[inlier_mask])
                h_prev_indices.append(i - 1)
                h_curr_indices.append(i)
                logging.info(
                    f"  [{ring_name}_h] {prev_file.stem} -> {curr_file.stem} "
                    f"inliers={stats['inliers']} conf={stats['mean_confidence']:.2f}"
                )
                if not valid_vertical[i]:
                    initial_X[i] = initial_X[i - 1] + stats["tx"]
                    initial_P[i] = initial_P[i - 1] - (stats["ty"] / focal_length) * (180.0 / np.pi)
                    initial_R[i] = curr_meta["roll"]
            else:
                logging.warning(f"  [{ring_name}_h] {prev_file.stem} -> {curr_file.stem} weak chain ({reason}).")
                if not valid_vertical[i]:
                    initial_X[i] = initial_X[i - 1] + expected_dx
                    initial_P[i] = curr_meta["pitch"]
                    initial_R[i] = curr_meta["roll"]
        else:
            logging.warning(f"  [{ring_name}_h] {prev_file.stem} -> {curr_file.stem} MISSING h-chain!")
            append_diag(
                diagnostics_rows,
                stage=f"{ring_name}_chain",
                idx_a=i - 1,
                idx_b=i,
                file_a=prev_file.name,
                file_b=curr_file.name,
                yaw_diff_deg=f"{normalize_yaw_delta(curr_meta['yaw'] - prev_meta['yaw']):.3f}",
                raw_matches=0,
                inliers=0,
                mean_confidence="0.0000",
                tx="0.000",
                ty="0.000",
                prior_tx=f"{expected_dx:.3f}",
                accepted=0,
                reason="match_failed",
            )
            if not valid_vertical[i]:
                initial_X[i] = initial_X[i - 1] + expected_dx
                initial_P[i] = curr_meta["pitch"]
                initial_R[i] = curr_meta["roll"]

    from scipy.optimize import least_squares

    deg_to_px = focal_length * (np.pi / 180.0)

    def residuals(vars):
        X = vars[:num_ring]
        P = vars[num_ring:2*num_ring]
        R = vars[2*num_ring:3*num_ring]
        
        res_arrays = []

        for idx in range(len(v_ring_indices)):
            ring_i = v_ring_indices[idx]
            pts_ring_sph = warper.warp_points(v_kpts_ring[idx], K, get_rotation_matrix(P[ring_i], R[ring_i]))
            gx_ring = pts_ring_sph[:, 0] + X[ring_i]
            gy_ring = pts_ring_sph[:, 1]
            res_arrays.append((v_target_gx[idx] - gx_ring) * args.anchor_x_weight)
            res_arrays.append((v_target_gy[idx] - gy_ring) * args.anchor_y_weight)

        for idx in range(len(h_prev_indices)):
            prev_i = h_prev_indices[idx]
            curr_i = h_curr_indices[idx]
            pts_prev_sph = warper.warp_points(h_kpts_prev[idx], K, get_rotation_matrix(P[prev_i], R[prev_i]))
            gx_prev = pts_prev_sph[:, 0] + X[prev_i]
            gy_prev = pts_prev_sph[:, 1]
            pts_curr_sph = warper.warp_points(h_kpts_curr[idx], K, get_rotation_matrix(P[curr_i], R[curr_i]))
            gx_curr = pts_curr_sph[:, 0] + X[curr_i]
            gy_curr = pts_curr_sph[:, 1]
            res_arrays.append(gx_prev - gx_curr)
            res_arrays.append(gy_prev - gy_curr)

        for i in range(num_ring):
            res_arrays.append(np.array([args.upper_pitch_prior_weight * deg_to_px * (P[i] - ring_pitch_meta[i])], dtype=np.float64))
            res_arrays.append(np.array([args.upper_roll_prior_weight * deg_to_px * (R[i] - ring_roll_meta[i])], dtype=np.float64))

        for i in range(1, num_ring):
            expected_dx = yaw_delta_to_tx(ring_info[i][0]["yaw"] - ring_info[i - 1][0]["yaw"], focal_length)
            smooth_res = (X[i] - X[i - 1]) - expected_dx
            res_arrays.append(np.array([args.upper_x_smooth_weight * smooth_res], dtype=np.float64))

        if len(res_arrays) == 0:
            return np.array([0.0])
        return np.concatenate(res_arrays)

    initial_guess = np.concatenate([initial_X, initial_P, initial_R])

    P_lower = np.array([float(ring_pitch_meta[i]) - args.upper_pitch_bound for i in range(num_ring)], dtype=np.float64)
    P_upper = np.array([float(ring_pitch_meta[i]) + args.upper_pitch_bound for i in range(num_ring)], dtype=np.float64)
    R_lower = np.array([float(ring_roll_meta[i]) - args.upper_roll_bound for i in range(num_ring)], dtype=np.float64)
    R_upper = np.array([float(ring_roll_meta[i]) + args.upper_roll_bound for i in range(num_ring)], dtype=np.float64)

    lower_bounds = np.concatenate([np.full(num_ring, -np.inf, dtype=np.float64), P_lower, R_lower])
    upper_bounds = np.concatenate([np.full(num_ring, np.inf, dtype=np.float64), P_upper, R_upper])

    initial_guess = np.clip(initial_guess, lower_bounds, upper_bounds)

    logging.info("[run] Solving %s optimization (per-image P/R)...", ring_name)
    solver_res = least_squares(
        residuals,
        initial_guess,
        bounds=(lower_bounds, upper_bounds),
        loss="soft_l1",
        f_scale=3.0,
        method="trf",
        verbose=0,
        max_nfev=100 if all(valid_vertical) else 200,
        xtol=1e-6,
        ftol=1e-6,
    )

    opt_X = solver_res.x[:num_ring]
    opt_P = solver_res.x[num_ring:2*num_ring]
    opt_R = solver_res.x[2*num_ring:3*num_ring]

    logging.info("  [%s_opt] Solver finished. Cost: %.2f", ring_name, solver_res.cost)

    ring_states = build_anchor_states(
        group_name=ring_name,
        infos=ring_info,
        images=ring_images,
        xs=opt_X,
        pitches=opt_P,
        rolls=opt_R,
        priority=0,
    )
    return ring_states, ring_images


def apply_blocks_gain_compensation(images, masks, corners):
    """Block-based exposure compensation for smooth transitions."""
    try:
        compensator = cv2.detail_BlocksGainCompensator()
        compensator.setBlockSize(32, 32)
        compensator.setNrFeeds(1)
        compensator.feed(corners=list(corners), images=images, masks=list(masks))
        for idx, image in enumerate(list(images)):
            adjusted = compensator.apply(idx, corners[idx], image, masks[idx])
            if adjusted is not None:
                images[idx] = np.asarray(adjusted, dtype=np.uint8)
    except Exception:
        pass

def find_graphcut_seams(images, masks, corners, downscale_factor=0.1):
    """Find optimal seam lines using graph-cut for smoother transitions. Downscales for speed."""
    if len(images) <= 1:
        return [m.copy() for m in masks]
    try:
        seam_finder = cv2.detail_GraphCutSeamFinder("COST_COLOR")
        
        scaled_images = []
        scaled_masks = []
        scaled_corners = []
        
        for img, msk, cnr in zip(images, masks, corners):
            s_img = cv2.resize(img, (0, 0), fx=downscale_factor, fy=downscale_factor, interpolation=cv2.INTER_AREA)
            s_msk = cv2.resize(msk, (0, 0), fx=downscale_factor, fy=downscale_factor, interpolation=cv2.INTER_NEAREST)
            scaled_images.append(s_img.astype(np.float32))
            scaled_masks.append(s_msk.astype(np.uint8))
            scaled_corners.append((int(cnr[0] * downscale_factor), int(cnr[1] * downscale_factor)))
            
        result_masks = seam_finder.find(scaled_images, scaled_corners, scaled_masks)
        
        if result_masks is None:
            return [m.copy() for m in masks]
            
        final_masks = []
        for i, res_mask in enumerate(result_masks):
            orig_h, orig_w = masks[i].shape[:2]
            upscaled = cv2.resize(np.asarray(res_mask, dtype=np.uint8), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
            # Ensure the upscaled mask doesn't exceed the original mask
            final_masks.append(cv2.bitwise_and(upscaled, masks[i]))
            
        return final_masks
    except Exception as e:
        logging.warning("GraphCutSeamFinder failed: %s", e)
        return [m.copy() for m in masks]

def compute_adaptive_alpha(state, base_alpha):
    """Higher alpha for frames with strong anchor evidence."""
    if state.get("anchor_inliers", 0) > 30:
        return min(1.0, base_alpha * 1.15)
    elif state.get("anchor_inliers", 0) < 10:
        return base_alpha * 0.7
    return base_alpha


def render_states_to_pano(
    pano_low_acc: np.ndarray,
    weight_low_acc: np.ndarray,
    pano_high_acc: np.ndarray,
    weight_high_acc: np.ndarray,
    out_w: int,
    out_h: int,
    scale_x: float,
    scale_y: float,
    min_y: int,
    K: np.ndarray,
    warper_render: SphericalWarper,
    states: list[dict],
    alpha_multiplier: float,
    log_prefix: str,
) -> None:
    if not states:
        return

    warped_images = []
    warped_masks = []
    corners = []
    final_gxs = []
    stretch_x = 1.0
    stretch_y = scale_y / scale_x
    
    for state in states:
        image = state["image"]
        R = get_rotation_matrix(state["pitch"], state["roll"])
        corner_render, warped = warper_render.warp_image(image, K, R)
        _, mask = warper_render.warp_mask(np.ones(image.shape[:2], dtype=np.uint8) * 255, K, R)
        s_img = cv2.resize(warped, (0, 0), fx=stretch_x, fy=stretch_y)
        s_mask = cv2.resize(mask, (0, 0), fx=stretch_x, fy=stretch_y, interpolation=cv2.INTER_NEAREST)
        
        final_gx = (corner_render[0] + state["x"] * scale_x) * stretch_x
        final_gy = corner_render[1] * stretch_y
        
        warped_images.append(s_img)
        warped_masks.append(s_mask)
        # For OpenCV compensator/seam finder, corner needs to be integer tuple
        corners.append((int(round(final_gx)), int(round(final_gy))))
        final_gxs.append(final_gx)

    # Step 2: Exposure compensation
    apply_blocks_gain_compensation(warped_images, warped_masks, corners)
    
    # Step 3: Seam optimization
    seam_masks = find_graphcut_seams(warped_images, warped_masks, corners)
    
    # Step 4: Render with optimized seams and adaptive alpha
    for idx, state in enumerate(states):
        adaptive_alpha = compute_adaptive_alpha(state, alpha_multiplier)
        draw_to_pano_two_band(
            pano_low_acc, weight_low_acc, pano_high_acc, weight_high_acc,
            out_w, out_h, (final_gxs[idx], corners[idx][1]),
            warped_images[idx], seam_masks[idx], min_y, adaptive_alpha
        )
        logging.info(
            "  [%s_draw] %s at gx=%.1f, pit=%.1f, rol=%.2f, alpha=%.2f",
            log_prefix,
            state["file"].stem,
            final_gxs[idx],
            state["pitch"],
            state["roll"],
            adaptive_alpha
        )


def draw_to_pano_two_band(
    pano_low_acc: np.ndarray, weight_low_acc: np.ndarray,
    pano_high_acc: np.ndarray, weight_high_acc: np.ndarray,
    out_w: int, out_h: int,
    corner: tuple[int, int], img: np.ndarray,
    mask: np.ndarray, min_y: int,
    alpha_multiplier: float = 1.0,
):
    """Accumulate into two frequency bands for preserving sharp details while matching exposure."""
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    normalized_dist = dist / (dist.max() + 1e-6)
    
    # Low frequency weight: very smooth and wide (linear falloff)
    weight_low = normalized_dist * alpha_multiplier
    # High frequency weight: softer transition for better edge blending (reduce from 20 -> 8)
    weight_high = (normalized_dist ** 8) * alpha_multiplier

    gx, gy = corner
    x_off = int(round(gx))
    y_off = int(round(gy - min_y))

    h_im, w_im = img.shape[:2]

    y_start = max(0, y_off)
    y_end = min(out_h, y_off + h_im)
    if y_end <= y_start:
        return

    src_y_start = y_start - y_off
    src_y_end = src_y_start + (y_end - y_start)
    dst_ys = slice(y_start, y_end)
    src_ys = slice(src_y_start, src_y_end)

    # Frequency separation
    arr_img_f = img.astype(np.float64)
    # Adaptive kernel: smaller for small images to preserve edges
    ksize = max(3, min(51, int(min(img.shape[:2]) * 0.08) | 1))
    arr_low = cv2.GaussianBlur(img, (ksize, ksize), 0).astype(np.float64)
    arr_high = arr_img_f - arr_low

    start_u = 0
    while start_u < w_im:
        dx_start = (x_off + start_u) % out_w
        chunk_len = min(w_im - start_u, out_w - dx_start)

        src_xs = slice(start_u, start_u + chunk_len)
        dst_xs = slice(dx_start, dx_start + chunk_len)

        c_arr_low = arr_low[src_ys, src_xs, :]
        c_arr_high = arr_high[src_ys, src_xs, :]
        c_w_low = weight_low[src_ys, src_xs]
        c_w_high = weight_high[src_ys, src_xs]

        pano_low_acc[dst_ys, dst_xs] += c_arr_low * c_w_low[:, :, np.newaxis]
        pano_high_acc[dst_ys, dst_xs] += c_arr_high * c_w_high[:, :, np.newaxis]

        weight_low_acc[dst_ys, dst_xs] += c_w_low
        weight_high_acc[dst_ys, dst_xs] += c_w_high
        start_u += chunk_len


def finalize_pano(pano_low_acc: np.ndarray, weight_low_acc: np.ndarray,
                  pano_high_acc: np.ndarray, weight_high_acc: np.ndarray) -> np.ndarray:
    """Combine low and high frequency bands and normalize to uint8."""
    pano = np.zeros(pano_low_acc.shape, dtype=np.uint8)
    valid_low = weight_low_acc > 1e-10
    valid_high = weight_high_acc > 1e-10
    
    valid_low_3d = valid_low[:, :, np.newaxis]
    valid_high_3d = valid_high[:, :, np.newaxis]
    weight_low_3d = weight_low_acc[:, :, np.newaxis] + 1e-10
    weight_high_3d = weight_high_acc[:, :, np.newaxis] + 1e-10

    low_band = np.where(valid_low_3d, pano_low_acc / weight_low_3d, 0)
    high_band = np.where(valid_high_3d, pano_high_acc / weight_high_3d, 0)
    
    composite = low_band + high_band
    pano = np.clip(composite, 0, 255).astype(np.uint8)
        
    return pano

def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO), format="[%(levelname)s] %(message)s")
    workdir = Path(args.workdir or default_workdir(args.input))
    workdir.mkdir(parents=True, exist_ok=True)

    prepared_dir = prepare_images(
        input_dir=Path(args.input), workdir=workdir, max_edge=args.max_edge, quality=args.quality, limit=args.limit, force=args.force
    )

    files = list(prepared_dir.glob("*.jpg"))
    if not files:
        raise SystemExit("No prepared images found.")

    valid_metadata = []
    for f in files:
        valid_metadata.append((parse_metadata(f.stem), f))

    # SPLIT: Backbone vs Rest
    backbone_info = []
    rings_to_process = [
        {"name": "upper1", "min_p": 15.0, "max_p": 35.0, "alpha": args.upper1_alpha, "info": [], "target_p": 25.0},
        {"name": "upper2", "min_p": 35.0, "max_p": 65.0, "alpha": args.upper2_alpha, "info": [], "target_p": 50.0},
        {"name": "lower1", "min_p": -35.0, "max_p": -15.0, "alpha": args.lower1_alpha, "info": [], "target_p": -25.0},
        {"name": "lower2", "min_p": -65.0, "max_p": -35.0, "alpha": args.lower2_alpha, "info": [], "target_p": -50.0},
        {"name": "ceiling", "min_p": 65.0, "max_p": 95.0, "alpha": args.ceiling_alpha, "info": [], "target_p": 85.0},
        {"name": "floor", "min_p": -95.0, "max_p": -65.0, "alpha": args.floor_alpha, "info": [], "target_p": -85.0},
    ]

    for meta, file_path in valid_metadata:
        p = meta["pitch"]
        if abs(p) < 15.0:
            backbone_info.append((meta, file_path))
        else:
            for band in rings_to_process:
                if band["min_p"] <= p < band["max_p"]:
                    band["info"].append((meta, file_path))
                    break

    # Sort by yaw
    backbone_info.sort(key=lambda x: x[0]["yaw"])
    for band in rings_to_process:
        band["info"].sort(key=lambda x: x[0]["yaw"])

    log_msg = f"Filtered: {len(backbone_info)} Backbone frames"
    for band in rings_to_process:
        log_msg += f", {len(band['info'])} {band['name'].capitalize()} frames"
    logging.info(log_msg)
    if len(backbone_info) < 2: raise SystemExit("Need at least 2 horizontal images for backbone.")

    # Apply IMU stabilization to Backbone
    median_pitch = np.median([m["pitch"] for m, _ in backbone_info])
    median_roll = np.median([m["roll"] for m, _ in backbone_info])
    for m, _ in backbone_info:
        m["pitch"] = median_pitch
        m["roll"] = median_roll
    logging.info(f"[run] Backbone locked to Pitch={median_pitch:.2f}, Roll={median_roll:.2f}")

    matcher = get_loftr_matcher(PipelineConfig(input_dir=prepared_dir, workdir=workdir))
    if matcher is None:
        raise SystemExit("LoFTR matcher unavailable. Install torch + kornia first.")

    diagnostics_rows: list[dict] = []
    source_lookup = load_prepared_source_map(workdir / "prepared_manifest.csv", args.input)

    # Determine focal length
    img0 = cv2.imread(str(backbone_info[0][1]))
    if img0 is None:
        raise SystemExit(f"Failed to read prepared image: {backbone_info[0][1]}")
    h, w = img0.shape[:2]

    orig_path = source_lookup.get(backbone_info[0][1].name, Path(args.input) / strip_prepared_prefix(backbone_info[0][1].name))
    orig_img = cv2.imread(str(orig_path))
    fallback_focal = max(h, w) * args.focal_ratio
    focal_length = fallback_focal
    if args.use_source_focal and orig_img is not None and backbone_info[0][0]["focal"] > 0:
        scale_ratio = max(h, w) / max(orig_img.shape[0], orig_img.shape[1])
        focal_length = backbone_info[0][0]["focal"] * scale_ratio
    elif args.use_source_focal:
        logging.warning("[run] Source focal requested but original image lookup failed for %s; falling back.", orig_path)

    logging.info(f"[run] Estimated Scaled Focal Length: {focal_length:.2f}")

    K = np.array([[focal_length, 0, w / 2], [0, focal_length, h / 2], [0, 0, 1]], dtype=np.float32)
    warper = SphericalWarper(focal_length)

    # --- PHASE 1: ALIGN BACKBONE ---
    logging.info("[run] Starting Phase 1: Backbone Align")
    bb_transforms = [{"tx": 0.0, "ty": 0.0}]
    bb_edges = []
    bb_yaw_priors = [0.0]
    bb_ref_img = load_and_undistort_image(backbone_info[0][1], K, args)
    bb_images = [bb_ref_img]

    for i in range(1, len(backbone_info)):
        img_prev = bb_images[-1]
        img_curr = load_and_undistort_image(backbone_info[i][1], K, args)
        # Exposure compensation: match brightness to first backbone image
        gain = compute_exposure_gain(bb_ref_img, img_curr)
        img_curr = apply_exposure_gain(img_curr, gain)
        bb_images.append(img_curr)

        kpts_prev, kpts_curr, conf = get_loftr_matches(matcher, img_prev, img_curr, args.match_max_dim, args.min_matches)
        yaw_diff = normalize_yaw_delta(backbone_info[i][0]["yaw"] - backbone_info[i - 1][0]["yaw"])
        prior_tx = yaw_delta_to_tx(yaw_diff, focal_length)
        bb_yaw_priors.append(prior_tx)
        if len(kpts_prev) > 0:
            Rp = get_rotation_matrix(backbone_info[i-1][0]["pitch"], backbone_info[i-1][0]["roll"])
            Rc = get_rotation_matrix(backbone_info[i][0]["pitch"], backbone_info[i][0]["roll"])
            stats = estimate_translation(
                kpts_prev, kpts_curr, warper, K, Rp, Rc, 
                backbone_info[i-1][0]["pitch"], backbone_info[i][0]["pitch"],
                args, is_horizontal=True, confidences=conf
            )
            accept, reason = accept_pair(len(kpts_prev), stats, args.pair_min_inliers, args.pair_min_confidence)
            append_diag(
                diagnostics_rows,
                stage="backbone",
                idx_a=i - 1,
                idx_b=i,
                file_a=backbone_info[i - 1][1].name,
                file_b=backbone_info[i][1].name,
                yaw_diff_deg=f"{yaw_diff:.3f}",
                raw_matches=len(kpts_prev),
                inliers=stats["inliers"],
                mean_confidence=f"{stats['mean_confidence']:.4f}",
                tx=f"{stats['tx']:.3f}",
                ty=f"{stats['ty']:.3f}",
                prior_tx=f"{prior_tx:.3f}",
                accepted=int(accept),
                reason=reason,
            )
            if accept:
                step_tx = stats["tx"]
                log_msg = f"  [bb] {i-1}->{i} inliers={stats['inliers']} conf={stats['mean_confidence']:.2f} tx={step_tx:.1f}"
                bb_edges.append({
                    "idx_a": i - 1,
                    "idx_b": i,
                    "pts_a": kpts_prev[stats["inlier_mask"]],
                    "pts_b": kpts_curr[stats["inlier_mask"]],
                    "weight": stats["mean_confidence"],
                    "is_loop": False
                })
            else:
                step_tx = prior_tx
                log_msg = (
                    f"  [bb] {i-1}->{i} weak pair ({reason}); using yaw prior tx={step_tx:.1f} "
                    f"instead of match tx={stats['tx']:.1f}"
                )

            global_tx = bb_transforms[-1]["tx"] + step_tx
            global_ty = 0.0
            bb_transforms.append({"tx": global_tx, "ty": global_ty})
            logging.info(log_msg)
        else:
            global_tx = bb_transforms[-1]["tx"] + prior_tx
            bb_transforms.append({"tx": global_tx, "ty": 0.0})
            append_diag(
                diagnostics_rows,
                stage="backbone",
                idx_a=i - 1,
                idx_b=i,
                file_a=backbone_info[i - 1][1].name,
                file_b=backbone_info[i][1].name,
                yaw_diff_deg=f"{yaw_diff:.3f}",
                raw_matches=0,
                inliers=0,
                mean_confidence="0.0000",
                tx="0.000",
                ty="0.000",
                prior_tx=f"{prior_tx:.3f}",
                accepted=0,
                reason="match_failed",
            )
            logging.warning(f"  [bb] {i-1}->{i} MATCH FAILED, using yaw prior tx={prior_tx:.1f}")

    # Loop closure for backbone
    kpts_last, kpts_first, conf_loop = get_loftr_matches(matcher, bb_images[-1], bb_images[0], args.match_max_dim, args.min_matches)
    expected_loop_tx = yaw_delta_to_tx(backbone_info[0][0]["yaw"] - backbone_info[-1][0]["yaw"], focal_length)
    if len(kpts_last) > 0:
        loop_stats = estimate_translation(
            kpts_last,
            kpts_first,
            warper,
            K,
            get_rotation_matrix(backbone_info[-1][0]["pitch"], backbone_info[-1][0]["roll"]),
            get_rotation_matrix(backbone_info[0][0]["pitch"], backbone_info[0][0]["roll"]),
            backbone_info[-1][0]["pitch"],
            backbone_info[0][0]["pitch"],
            args,
            is_horizontal=True,
            confidences=conf_loop,
        )
        loop_accept, loop_reason = accept_pair(len(kpts_last), loop_stats, args.pair_min_inliers, args.pair_min_confidence)
        if loop_accept:
            bb_edges.append({
                "idx_a": len(backbone_info) - 1,
                "idx_b": 0,
                "pts_a": kpts_last[loop_stats["inlier_mask"]],
                "pts_b": kpts_first[loop_stats["inlier_mask"]],
                "weight": loop_stats["mean_confidence"],
                "is_loop": True
            })
    else:
        loop_stats = {"tx": expected_loop_tx, "ty": 0.0, "inliers": 0, "mean_confidence": 0.0}
        loop_accept, loop_reason = False, "match_failed"

    append_diag(
        diagnostics_rows,
        stage="backbone_loop",
        idx_a=len(backbone_info) - 1,
        idx_b=0,
        file_a=backbone_info[-1][1].name,
        file_b=backbone_info[0][1].name,
        yaw_diff_deg=f"{normalize_yaw_delta(backbone_info[0][0]['yaw'] - backbone_info[-1][0]['yaw']):.3f}",
        raw_matches=len(kpts_last),
        inliers=loop_stats["inliers"],
        mean_confidence=f"{loop_stats['mean_confidence']:.4f}",
        tx=f"{loop_stats['tx']:.3f}",
        ty=f"{loop_stats['ty']:.3f}",
        prior_tx=f"{expected_loop_tx:.3f}",
        accepted=int(loop_accept),
        reason=loop_reason,
    )

    # Run global optimization
    if len(bb_edges) > 0 and len(backbone_info) > 1:
        logging.info("  [bb] Running global least_squares bundle adjustment...")
        opt_xs, opt_pitches, opt_rolls = solve_backbone_least_squares(
            num_states=len(backbone_info),
            edges=bb_edges,
            base_pitch=np.array([meta["pitch"] for meta, _ in backbone_info], dtype=np.float64),
            base_roll=np.array([meta["roll"] for meta, _ in backbone_info], dtype=np.float64),
            initial_x=np.array([t["tx"] for t in bb_transforms], dtype=np.float64),
            warper=warper,
            K=K,
            focal_length=focal_length,
            yaw_priors=np.array(bb_yaw_priors, dtype=np.float64),
            x_smooth_weight=1.0,
            pitch_prior_weight=0.5,
            roll_prior_weight=0.5,
        )
        
        # Update bb_transforms and backbone_info metadata
        for i in range(len(backbone_info)):
            bb_transforms[i]["tx"] = float(opt_xs[i])
            backbone_info[i][0]["pitch"] = float(opt_pitches[i])
            backbone_info[i][0]["roll"] = float(opt_rolls[i])
            
        logging.info("  [bb] Global optimization complete.")
    else:
        logging.info("  [bb] Not enough accepted edges for global optimization.")

    loop_tx = loop_stats["tx"] if loop_accept else expected_loop_tx
    tx_span = abs(bb_transforms[-1]["tx"] + loop_tx)
    if tx_span < 1.0:
        tx_span = max(abs(bb_transforms[-1]["tx"]), 2.0 * np.pi * focal_length)
        logging.info(f"  [bb] Loop span degenerate, falling back to width={tx_span:.1f}")

    logging.info(f"  [bb] Loop Closed. width={tx_span:.1f}")

    # Scale Composition Setup
    if args.export_bi_layout:
        out_w = 1024
        out_h = 512
        scale_x = float(out_w) / tx_span
        scale_y = float(out_h) / (np.pi * focal_length)
        logging.info(f"  [bb] Forcing Bi_Layout export: 1024x512 (scale_x={scale_x:.3f}, scale_y={scale_y:.3f})")
    else:
        scale = args.compose_scale
        out_w = max(1, int(np.ceil(tx_span * scale)))
        out_h = max(1, int(np.ceil(np.pi * focal_length * scale))) # 180 degrees vertical FOV mapped to pixels
        scale_x = scale
        scale_y = scale
    min_y = 0 # OpenCV Spherical Warper uses 0 for North Pole
    
    # Initialize Render-optimized Warper to skip 4K processing
    warper_render = SphericalWarper(focal_length * scale_x)
    stretch_x = 1.0
    stretch_y = scale_y / scale_x
    
    pano_low_acc = np.zeros((out_h, out_w, 3), dtype=np.float64)
    weight_low_acc = np.zeros((out_h, out_w), dtype=np.float64)
    pano_high_acc = np.zeros((out_h, out_w, 3), dtype=np.float64)
    weight_high_acc = np.zeros((out_h, out_w), dtype=np.float64)

    # Render Backbone
    logging.info("[run] Rendering Backbone Canvas...")
    for i in range(len(backbone_info)):
        img = bb_images[i]
        R = get_rotation_matrix(backbone_info[i][0]["pitch"], backbone_info[i][0]["roll"])
        corner_render, warped_img_render = warper_render.warp_image(img, K, R)
        _, warped_mask_render = warper_render.warp_mask(np.ones(img.shape[:2], dtype=np.uint8)*255, K, R)

        s_img = cv2.resize(warped_img_render, (0,0), fx=stretch_x, fy=stretch_y)
        s_mask = cv2.resize(warped_mask_render, (0,0), fx=stretch_x, fy=stretch_y, interpolation=cv2.INTER_NEAREST)

        gx = (corner_render[0] + bb_transforms[i]["tx"] * scale_x) * stretch_x
        gy = (corner_render[1] + bb_transforms[i]["ty"] * scale_x) * stretch_y
        draw_to_pano_two_band(pano_low_acc, weight_low_acc, pano_high_acc, weight_high_acc, out_w, out_h, (gx, gy), s_img, s_mask, min_y, 1.0)

    cv2.imwrite(str(workdir / "backbone_only.jpg"), finalize_pano(pano_low_acc, weight_low_acc, pano_high_acc, weight_high_acc))
    logging.info("  [done] Backbone rendered!")
    backbone_states = build_anchor_states(
        group_name="backbone",
        infos=backbone_info,
        images=bb_images,
        xs=[transform["tx"] for transform in bb_transforms],
        pitches=[meta["pitch"] for meta, _ in backbone_info],
        rolls=[meta["roll"] for meta, _ in backbone_info],
        priority=1,
    )

    # Iterate over dynamically defined rings in sequential target sequence
    completed_rings = [{"name": "backbone", "states": backbone_states, "target_p": 0.0}]
    current_pano_name = "backbone"

    for band in rings_to_process:
        band_name = band["name"]
        band_info = band["info"]

        if not band_info:
            logging.info(f"[run] No {band_name.capitalize()} frames detected; keeping previous canvas.")
            continue

        if args.skip_ceiling_floor and band_name in ("ceiling", "floor"):
            logging.info(f"[run] Skipping {band_name} (--skip-ceiling-floor)")
            continue

        logging.info(f"[run] Starting alignment for {band_name.capitalize()}")
        
        # Build anchor states based on pitch distance priority
        sorted_past_rings = sorted(
            completed_rings,
            key=lambda r: abs(r["target_p"] - band["target_p"])
        )

        current_anchor_states = []
        for i, past_ring in enumerate(sorted_past_rings[:2]):
            current_anchor_states.extend(
                {
                    **state,
                    "priority": i,
                }
                for state in past_ring["states"]
            )

        new_states, _ = optimize_upper_ring(
            ring_name=band_name,
            ring_info=band_info,
            anchor_states=current_anchor_states,
            matcher=matcher,
            warper=warper,
            K=K,
            focal_length=focal_length,
            args=args,
            diagnostics_rows=diagnostics_rows,
        )

        logging.info(f"[run] Rendering {band_name.capitalize()}...")
        render_states_to_pano(
            pano_low_acc=pano_low_acc,
            weight_low_acc=weight_low_acc,
            pano_high_acc=pano_high_acc,
            weight_high_acc=weight_high_acc,
            out_w=out_w,
            out_h=out_h,
            scale_x=scale_x,
            scale_y=scale_y,
            min_y=min_y,
            K=K,
            warper_render=warper_render,
            states=new_states,
            alpha_multiplier=band["alpha"],
            log_prefix=band_name,
        )

        completed_rings.append({"name": band_name, "states": new_states, "target_p": band["target_p"]})
        current_pano_name += f"_{band_name}"
        pano = finalize_pano(pano_low_acc, weight_low_acc, pano_high_acc, weight_high_acc)
        cv2.imwrite(str(workdir / f"{current_pano_name}.jpg"), pano)
        logging.info(f"  [done] Saved snapshot at {workdir / f'{current_pano_name}.jpg'}")

    pano = finalize_pano(pano_low_acc, weight_low_acc, pano_high_acc, weight_high_acc)
    cv2.imwrite(str(workdir / "spherical_panorama.jpg"), pano)
    write_diagnostics(workdir / "alignment_diagnostics.csv", diagnostics_rows)
    logging.info(f"[done] saved final at {workdir / 'spherical_panorama.jpg'}")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())

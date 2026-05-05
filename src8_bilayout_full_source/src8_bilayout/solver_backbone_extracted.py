from __future__ import annotations
import numpy as np
from scipy.optimize import least_squares
from src8.spherical_warper import SphericalWarper

def get_rotation_matrix(pitch_deg: float, roll_deg: float) -> np.ndarray:
    px = np.radians(pitch_deg)
    rz = np.radians(roll_deg)
    Rx = np.array([[1, 0, 0], [0, np.cos(px), -np.sin(px)], [0, np.sin(px), np.cos(px)]], dtype=np.float32)
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]], dtype=np.float32)
    return Rx @ Rz

def interpolate_pitch_delta(control: np.ndarray, num_states: int) -> np.ndarray:
    if num_states <= 1:
        return np.array([float(control[0])], dtype=np.float64)
    positions = np.linspace(0.0, 1.0, num_states)
    return np.interp(positions, [0.0, 0.5, 1.0], control).astype(np.float64)

def solve_backbone_least_squares(
    num_states: int,
    edges: list[dict],
    base_pitch: np.ndarray,
    base_roll: np.ndarray,
    initial_x: np.ndarray,
    warper: SphericalWarper,
    K: np.ndarray,
    focal_length: float,
    yaw_priors: np.ndarray,
    x_smooth_weight: float = 1.0,
    pitch_prior_weight: float = 0.5,
    roll_prior_weight: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solves for the global optimal x translation, pitch, and roll of the backbone images.
    edges: list of dicts with keys: 'idx_a', 'idx_b', 'pts_a', 'pts_b', 'weight', 'is_loop'
    Returns: (optimized_x, optimized_pitch, optimized_roll)
    """
    deg_to_px = float(focal_length) * (np.pi / 180.0)

    def residuals(vars_vec: np.ndarray) -> np.ndarray:
        xs = vars_vec[:num_states]
        roll_delta = float(vars_vec[num_states])
        pitch_ctrl = vars_vec[num_states + 1 : num_states + 4]
        pitch_delta = interpolate_pitch_delta(pitch_ctrl, num_states)
        pitches = base_pitch + pitch_delta
        rolls = base_roll + roll_delta

        # Anchor first image x to 0
        res_parts: list[np.ndarray] = [np.array([xs[0] * 100.0], dtype=np.float64)]
        
        for edge in edges:
            idx_a = edge["idx_a"]
            idx_b = edge["idx_b"]
            pts_a_raw = edge["pts_a"]
            pts_b_raw = edge["pts_b"]
            weight = edge.get("weight", 1.0)
            is_loop = edge.get("is_loop", False)
            
            if len(pts_a_raw) == 0:
                continue
                
            pts_a = warper.warp_points(pts_a_raw, K, get_rotation_matrix(pitches[idx_a], rolls[idx_a]))
            pts_b = warper.warp_points(pts_b_raw, K, get_rotation_matrix(pitches[idx_b], rolls[idx_b]))
            
            if is_loop:
                # Wrap around cylinder for loop closure
                res_parts.append(((pts_a[:, 0] + xs[idx_a]) - (pts_b[:, 0] + xs[idx_b] + 360.0 * deg_to_px)) * weight)
                res_parts.append((pts_a[:, 1] - pts_b[:, 1]) * weight)
            else:
                res_parts.append(((pts_a[:, 0] + xs[idx_a]) - (pts_b[:, 0] + xs[idx_b])) * weight)
                res_parts.append((pts_a[:, 1] - pts_b[:, 1]) * weight)

        # Smoothness prior for step distances
        for idx in range(1, num_states):
            prior_tx = yaw_priors[idx]
            smooth = (xs[idx] - xs[idx - 1]) - prior_tx
            res_parts.append(np.array([x_smooth_weight * smooth], dtype=np.float64))

        # Regularization on pitch and roll changes
        for value in pitch_ctrl:
            res_parts.append(np.array([pitch_prior_weight * deg_to_px * value], dtype=np.float64))
        res_parts.append(np.array([roll_prior_weight * deg_to_px * roll_delta], dtype=np.float64))
        
        return np.concatenate(res_parts) if res_parts else np.array([0.0], dtype=np.float64)

    initial_guess = np.concatenate([initial_x, np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)])
    lower = np.concatenate([np.full(num_states, -np.inf, dtype=np.float64), np.array([-1.0, -1.5, -1.5, -1.5], dtype=np.float64)])
    upper = np.concatenate([np.full(num_states, np.inf, dtype=np.float64), np.array([1.0, 1.5, 1.5, 1.5], dtype=np.float64)])
    
    solved = least_squares(
        residuals, initial_guess, bounds=(lower, upper), loss="soft_l1", f_scale=3.0, 
        method="trf", max_nfev=150, verbose=0, xtol=1e-6, ftol=1e-6
    )

    opt_xs = solved.x[:num_states]
    opt_roll_delta = float(solved.x[num_states])
    opt_pitch_ctrl = solved.x[num_states + 1 : num_states + 4]
    opt_pitch_delta = interpolate_pitch_delta(opt_pitch_ctrl, num_states)
    
    opt_pitches = base_pitch + opt_pitch_delta
    opt_rolls = base_roll + opt_roll_delta
    
    return opt_xs, opt_pitches, opt_rolls

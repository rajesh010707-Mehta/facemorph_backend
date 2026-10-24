"""
matcher.py
Compute facial structure similarity between two sets of extracted landmarks.
Works with MediaPipe Tasks API which returns 478 landmarks (includes iris).
"""

import numpy as np
from typing import Dict, Any, List


# ── Key landmark indices (MediaPipe 478-point set) ───────────────────────────
LEFT_EYE_OUTER  = 33
RIGHT_EYE_OUTER = 263
NOSE_TIP        = 1
NOSE_BRIDGE     = 6
CHIN            = 152
LEFT_JAW        = 234   # left cheek / jaw area
RIGHT_JAW       = 454   # right cheek / jaw area  (478-pt set has this)
LEFT_MOUTH      = 61
RIGHT_MOUTH     = 291
LEFT_CHEEK      = 116
RIGHT_CHEEK     = 345
FOREHEAD        = 10

FRONT_KEY_IDX = [
    LEFT_EYE_OUTER, RIGHT_EYE_OUTER,
    NOSE_TIP, NOSE_BRIDGE,
    CHIN,
    LEFT_JAW, RIGHT_JAW,
    LEFT_MOUTH, RIGHT_MOUTH,
    LEFT_CHEEK, RIGHT_CHEEK,
    FOREHEAD,
]

NOSE_IDX = [1, 2, 3, 4, 5, 6]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_index(landmarks: np.ndarray, idx: int) -> np.ndarray:
    """Return landmark at idx, or zeros if out of range."""
    if idx < len(landmarks):
        return landmarks[idx]
    return np.zeros(3, dtype=np.float32)


def _normalize(landmarks: np.ndarray) -> np.ndarray:
    """
    Scale-invariant normalisation:
    - Centre at midpoint of the two eye outer corners.
    - Scale by inter-ocular distance (IOD).
    """
    left  = _safe_index(landmarks, LEFT_EYE_OUTER)[:2]
    right = _safe_index(landmarks, RIGHT_EYE_OUTER)[:2]
    iod   = float(np.linalg.norm(right - left))
    if iod < 1e-6:
        iod = 1.0
    center = (left + right) / 2.0
    norm   = landmarks.copy()
    norm[:, :2] -= center
    norm         /= iod
    return norm


def _key_similarity(u: np.ndarray, t: np.ndarray, indices: List[int]) -> float:
    """Mean distance of key points → similarity 0-100."""
    valid = [i for i in indices if i < len(u) and i < len(t)]
    if not valid:
        return 50.0
    dists = [np.linalg.norm(u[i] - t[i]) for i in valid]
    mean_d = float(np.mean(dists))
    # distance 0 → 100%, distance ≥ 0.25 → 0%
    return float(max(0.0, 1.0 - mean_d / 0.25) * 100)


def _full_similarity(u: np.ndarray, t: np.ndarray) -> float:
    """All-landmark mean Euclidean distance → similarity 0-100."""
    n = min(len(u), len(t))
    mean_d = float(np.linalg.norm(u[:n] - t[:n], axis=1).mean())
    return float(max(0.0, 1.0 - mean_d / 0.20) * 100)


def _angle_at(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle at vertex b (degrees)."""
    ba = a - b;  bc = c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def _jawline_angle(lm: np.ndarray) -> float:
    chin      = _safe_index(lm, CHIN)[:2]
    left_jaw  = _safe_index(lm, LEFT_JAW)[:2]
    right_jaw = _safe_index(lm, RIGHT_JAW)[:2]
    return _angle_at(left_jaw, chin, right_jaw)


def _face_ratio(lm: np.ndarray) -> float:
    """Height / width ratio."""
    forehead = _safe_index(lm, FOREHEAD)[:2]
    chin     = _safe_index(lm, CHIN)[:2]
    l_jaw    = _safe_index(lm, LEFT_JAW)[:2]
    r_jaw    = _safe_index(lm, RIGHT_JAW)[:2]
    h = float(np.linalg.norm(forehead - chin)) + 1e-6
    w = float(np.linalg.norm(l_jaw    - r_jaw)) + 1e-6
    return h / w


def _nose_ratio(lm: np.ndarray) -> float:
    """Nose width / face width."""
    l_jaw = _safe_index(lm, LEFT_JAW)[:2]
    r_jaw = _safe_index(lm, RIGHT_JAW)[:2]
    fw    = float(np.linalg.norm(l_jaw - r_jaw)) + 1e-6
    # nose base points ~64 and ~294
    n_l   = lm[64][:2] if 64 < len(lm) else lm[NOSE_TIP][:2]
    n_r   = lm[294][:2] if 294 < len(lm) else lm[NOSE_TIP][:2]
    nw    = float(np.linalg.norm(n_l - n_r)) + 1e-6
    return nw / fw


def _ratio_similarity(ru: float, rt: float, tolerance: float = 0.3) -> float:
    diff = abs(ru - rt)
    return float(max(0.0, 1.0 - diff / tolerance) * 100)


# ── Insight builder ───────────────────────────────────────────────────────────

def _build_insights(
    front_sim: float,
    side_sim: float,
    jaw_match: float,
    nose_match: float,
    jaw_u: float,
    jaw_t: float,
    has_side: bool,
) -> List[str]:
    out: List[str] = []
    if front_sim >= 80:
        out.append("Front facial bone structure is highly similar between both faces.")
    elif front_sim >= 60:
        out.append("Moderate front facial structure overlap detected.")
    else:
        out.append("Front facial structures show notable differences.")

    if has_side and side_sim >= 75:
        out.append("Side profile contours align closely — similar face depth.")
    elif has_side and side_sim >= 50:
        out.append("Side profiles show partial similarity in depth and projection.")

    jaw_diff = abs(jaw_u - jaw_t)
    if jaw_diff < 5:
        out.append(f"Jawline angles are nearly identical ({jaw_u:.1f}° vs {jaw_t:.1f}°).")
    elif jaw_diff < 15:
        out.append(f"Similar jaw geometry detected (difference: {jaw_diff:.1f}°).")
    else:
        out.append(f"Jawline angles differ significantly ({jaw_u:.1f}° vs {jaw_t:.1f}°).")

    if nose_match >= 75:
        out.append("Nose bridge slope and structure match closely.")
    elif nose_match >= 50:
        out.append("Nose structures show moderate similarity.")

    if not out:
        out.append("Overall facial structure shows moderate similarity.")
    return out


# ── Public API ────────────────────────────────────────────────────────────────

def compute_match(
    user_landmarks: Dict[str, Any],
    target_landmarks: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compute weighted facial similarity score.

    Weights:
        Front structure  45%
        Jaw/profile      35%
        Proportions      20%
    """
    u_front = _normalize(user_landmarks["front"]["landmarks"])
    t_front = _normalize(target_landmarks["front"]["landmarks"])

    # ── Front similarity (45%) ─────────────────────────────────────────────
    front_sim = _key_similarity(u_front, t_front, FRONT_KEY_IDX)

    # ── Side profile similarity (part of 35%) ─────────────────────────────
    side_sims: List[float] = []
    for side in ("left", "right"):
        if side in user_landmarks and side in target_landmarks:
            u_s = _normalize(user_landmarks[side]["landmarks"])
            t_s = _normalize(target_landmarks[side]["landmarks"])
            side_sims.append(_full_similarity(u_s, t_s))

    has_side = len(side_sims) > 0
    side_profile_sim = float(np.mean(side_sims)) if has_side else front_sim * 0.85

    # ── Jawline angle (part of 35%) ────────────────────────────────────────
    jaw_u = _jawline_angle(u_front)
    jaw_t = _jawline_angle(t_front)
    jaw_diff = abs(jaw_u - jaw_t)
    jaw_match = float(max(0.0, 1.0 - jaw_diff / 30.0) * 100)

    # Combine jaw + side as profile score
    profile_score = (jaw_match * 0.5 + side_profile_sim * 0.5)

    # ── Proportions & ratios (20%) ─────────────────────────────────────────
    ratio_scores = [
        _ratio_similarity(_face_ratio(u_front), _face_ratio(t_front), 0.4),
        _ratio_similarity(_nose_ratio(u_front), _nose_ratio(t_front), 0.3),
    ]
    nose_structure_match = _key_similarity(u_front, t_front, NOSE_IDX)
    ratio_scores.append(nose_structure_match)
    proportions_score = float(np.mean(ratio_scores))

    # ── Weighted overall ───────────────────────────────────────────────────
    overall = (
        0.45 * front_sim +
        0.35 * profile_score +
        0.20 * proportions_score
    )

    insights = _build_insights(
        front_sim, side_profile_sim, jaw_match,
        nose_structure_match, jaw_u, jaw_t, has_side,
    )

    return {
        "overall_match_percentage": round(float(overall), 2),
        "breakdown": {
            "front_view_similarity":   round(front_sim, 2),
            "side_profile_similarity": round(side_profile_sim, 2),
            "jawline_angle_match":     round(jaw_match, 2),
            "nose_structure_match":    round(nose_structure_match, 2),
        },
        "insights": insights,
    }

import numpy as np
from numpy.typing import ArrayLike
from typing import Tuple, Optional

from simulator import RaceTrack

# ---- Utilities -----------------------------------------------------------------

def _as_array(a: ArrayLike) -> np.ndarray:
    return np.asarray(a, dtype=float)


def find_closest_point(state: ArrayLike, path: ArrayLike) -> Tuple[int, float]:
    s = _as_array(state)
    p = _as_array(path)
    car_pos = s[:2]
    dists = np.linalg.norm(p - car_pos, axis=1)
    idx = int(np.argmin(dists))
    return idx, float(dists[idx])


def find_lookahead_point(state: ArrayLike, path: ArrayLike, lookahead_distance: float) -> Tuple[int, np.ndarray]:
    idx_closest, _ = find_closest_point(state, path)
    p = _as_array(path)
    cumulative = 0.0
    n = len(p)
    for step in range(1, n):
        i_prev = (idx_closest + step - 1) % n
        i_next = (idx_closest + step) % n
        cumulative += np.linalg.norm(p[i_next] - p[i_prev])
        if cumulative >= lookahead_distance:
            return i_next, p[i_next]
    return idx_closest, p[idx_closest]


def estimate_curvature(path: ArrayLike, idx: int, step: int = 5) -> float:
    p = _as_array(path)
    n = len(p)
    i1 = (idx - step) % n
    i2 = idx % n
    i3 = (idx + step) % n

    p1, p2, p3 = p[i1], p[i2], p[i3]

    # area of triangle * 0.5
    area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))

    a = np.linalg.norm(p2 - p1)
    b = np.linalg.norm(p3 - p2)
    c = np.linalg.norm(p1 - p3)
    denom = a * b * c
    if denom < 1e-9:
        return 0.0
    return float(4.0 * area / denom)


# ---- Speed Reference (S1) helpers ---------------------------------------------

def compute_safe_corner_speed(curvature: float, v_max: float, raceline_mode: bool = False) -> float:
    if curvature < 1e-6:
        return v_max

    if raceline_mode:
        if curvature < 0.008:      a_lat = 21.0
        elif curvature < 0.020:    a_lat = 18.5
        elif curvature < 0.035:    a_lat = 16.0
        else:                      a_lat = 12.0
    else:
        if curvature < 0.008:      a_lat = 17.0
        elif curvature < 0.020:    a_lat = 14.5
        elif curvature < 0.035:    a_lat = 12.0
        else:                      a_lat = 9.5

    v_safe = np.sqrt(a_lat / curvature)
    return min(v_safe, v_max)


def compute_braking_distance(v_current: float, v_target: float, a_brake: float) -> float:
    if v_current <= v_target:
        return 0.0
    return max((v_current * v_current - v_target * v_target) / (2.0 * a_brake), 0.0)


def _safety_multiplier(curv: float, raceline_mode: bool) -> float:
    if raceline_mode:
        if curv > 0.030: return 0.50
        if curv > 0.020: return 0.68
        if curv > 0.012: return 0.76
        if curv > 0.006: return 0.90
        if curv > 0.003: return 0.985
        return 0.998
    else:
        if curv > 0.030: return 0.76
        if curv > 0.020: return 0.82
        if curv > 0.012: return 0.88
        if curv > 0.006: return 0.94
        if curv > 0.003: return 0.97
        return 0.995


def compute_reference_velocity(
    state: ArrayLike,
    path: ArrayLike,
    parameters: ArrayLike,
    raceline_mode: bool = False
) -> float:
    params = _as_array(parameters)
    v_min = float(params[2])
    v_max = float(params[5])
    a_max = float(params[10])

    if v_min <= 20:
        v_min = 20.0

    v = abs(float(state[3]))
    closest_idx, _ = find_closest_point(state, path)

    base_la = 40
    if v > 60:
        extra = ((v - 60.0) ** 1.3) / 2.0
        lookahead_pts = int(min(base_la + v / 2.5 + extra, 150))
    else:
        lookahead_pts = int(min(base_la + v / 2.5, 80))

    path_arr = _as_array(path)
    n = len(path_arr)

    min_safe = v_max
    dist_to_corner = 0.0
    found = False

    last_sign = None
    last_sign_dist = None
    cumulative = 0.0

    for i in range(lookahead_pts):
        idx = (closest_idx + i) % n
        if i > 0:
            prev = (closest_idx + i - 1) % n
            cumulative += np.linalg.norm(path_arr[idx] - path_arr[prev])

        curv = estimate_curvature(path_arr, idx, step=3)
        if curv <= 0.0008:
            continue

        safe_speed = compute_safe_corner_speed(curv, v_max, raceline_mode)
        turn_sign = 0.0

        if curv > 0.010:
            p_prev = path_arr[(idx - 1) % n]
            p_next = path_arr[(idx + 1) % n]
            v1 = path_arr[idx] - p_prev
            v2 = p_next - path_arr[idx]
            cross = v1[0] * v2[1] - v1[1] * v2[0]
            if abs(cross) > 1e-9:
                turn_sign = np.sign(cross)

        safe_speed *= _safety_multiplier(curv, raceline_mode)

        if turn_sign != 0.0 and last_sign is not None and last_sign_dist is not None and (turn_sign * last_sign) < 0 and (cumulative - last_sign_dist) < 40.0:
            safe_speed *= 0.90

        if turn_sign != 0.0:
            last_sign = turn_sign
            last_sign_dist = cumulative

        if safe_speed < min_safe:
            min_safe = safe_speed
            dist_to_corner = cumulative
            found = True

    if not found or min_safe >= v:
        return float(np.clip(v_max, v_min, v_max))

    brake_accel = 0.85 * a_max
    brake_dist = compute_braking_distance(v, min_safe, brake_accel)
    brake_dist *= 1.40

    if dist_to_corner <= brake_dist:
        v_ref = min_safe
    elif dist_to_corner < 1.3 * brake_dist:
        v_ref = v
    else:
        v_ref = v_max

    return float(np.clip(v_ref, v_min, v_max))


# ---- Steering Reference (S2) -------------------------------------------------

def _transform_to_vehicle_frame(sx: float, sy: float, heading: float, px: float, py: float) -> Tuple[float, float]:
    dx = px - sx
    dy = py - sy
    cos_h = np.cos(-heading)
    sin_h = np.sin(-heading)
    lx = dx * cos_h - dy * sin_h
    ly = dx * sin_h + dy * cos_h
    return lx, ly


def compute_reference_steering(
    state: ArrayLike,
    centerline: ArrayLike,
    parameters: ArrayLike,
    raceline_mode: bool = False
) -> float:
    params = _as_array(parameters)
    wheelbase = float(params[0])
    delta_max = float(params[4])

    sx, sy = float(state[0]), float(state[1])
    delta = float(state[2])
    v = abs(float(state[3]))
    heading = float(state[4])

    speed = max(v, 1.0)

    if speed < 50:
        Ld_target = 8.0 + 0.30 * speed
    else:
        Ld_target = 23.0 + 0.45 * (speed - 50.0)

    if raceline_mode:
        Ld_target *= 0.9

    _, look_pt = find_lookahead_point(state, centerline, Ld_target)
    lx, ly = _transform_to_vehicle_frame(sx, sy, heading, look_pt[0], look_pt[1])

    lx = max(lx, 0.5)
    Ld = max(np.hypot(lx, ly), 1.0)
    alpha = np.arctan2(ly, lx)

    pure_pursuit = np.arctan2(2.0 * wheelbase * np.sin(alpha), Ld)

    closest_idx, _ = find_closest_point(state, centerline)
    next_idx = (closest_idx + 1) % len(centerline)
    seg = centerline[next_idx] - centerline[closest_idx]
    seg_len = np.linalg.norm(seg)
    seg = seg / seg_len if seg_len > 1e-6 else np.array([1.0, 0.0])

    to_car = np.array([sx, sy]) - centerline[closest_idx]
    cross_err = seg[0] * to_car[1] - seg[1] * to_car[0]

    k = 0.8 if raceline_mode else 0.5
    stanley = np.arctan(k * cross_err / max(speed, 2.0))

    w = 0.22 if raceline_mode else 0.15
    delta_ref = pure_pursuit + w * stanley

    return float(np.clip(delta_ref, -0.9 * delta_max, 0.9 * delta_max))


# ---- Low-level controllers (C1, C2) -----------------------------------------

def velocity_controller(state: ArrayLike, reference_velocity: float, parameters: ArrayLike) -> float:
    v = float(state[3])
    v_ref = float(reference_velocity)
    a_max = float(_as_array(parameters)[10])

    error = v_ref - v

    if error < 0:
        if v < 50:
            kp = 6.5
        elif v > 90:
            kp = 6.5
        else:
            kp = 5.0 + ((v - 50.0) / 40.0) * 1.3
    else:
        kp = 3.5

    a_cmd = float(np.clip(kp * error, -a_max, a_max))
    return a_cmd


def steering_controller(
    state: ArrayLike,
    reference_steering: float,
    parameters: ArrayLike,
    prev_error: float = 0.0,
    dt: float = 0.1
) -> Tuple[float, float]:
    delta = float(state[2])
    delta_ref = float(reference_steering)
    v = abs(float(state[3]))

    params = _as_array(parameters)
    v_min = float(params[7])
    v_max = float(params[9])

    error = delta_ref - delta
    d_error = (error - float(prev_error)) / float(dt)

    if v < 50:
        kp, kd = 3.5, 0.50
    elif v > 70:
        kp, kd = 2.1, 0.85
    else:
        α = (v - 20.0) / 50.0
        kp = 2.4 - 0.3 * α
        kd = 0.35 + 0.5 * α

    v_delta = kp * error + kd * d_error
    v_delta = float(np.clip(v_delta, v_min, v_max))

    return v_delta, error


# ---- High-level interface ----------------------------------------------------

def compute_safe_raceline(
    raceline: ArrayLike,
    racetrack: RaceTrack,
    safety_margin: float = 0.7
) -> np.ndarray:
    rl = _as_array(raceline).copy()
    for i, p in enumerate(rl):
        d_center = np.linalg.norm(racetrack.centerline - p, axis=1)
        idx = int(np.argmin(d_center))
        right_pt = racetrack.right_boundary[idx]
        left_pt = racetrack.left_boundary[idx]
        center = racetrack.centerline[idx]

        d_r = np.linalg.norm(p - right_pt)
        d_l = np.linalg.norm(p - left_pt)

        if d_r < safety_margin or d_l < safety_margin:
            direction = center - p
            norm = np.linalg.norm(direction) + 1e-6
            direction /= norm
            shift = safety_margin - min(d_r, d_l)
            rl[i] = p + direction * shift
    return rl


def controller(
    state: ArrayLike,
    parameters: ArrayLike,
    racetrack: RaceTrack,
    raceline: Optional[ArrayLike] = None
) -> np.ndarray:
    use_raceline = raceline is not None
    path = raceline if use_raceline else racetrack.centerline

    v_ref = compute_reference_velocity(state, path, parameters, raceline_mode=use_raceline)
    delta_ref = compute_reference_steering(state, path, parameters, raceline_mode=use_raceline)

    return np.array([float(delta_ref), float(v_ref)], dtype=float)


# ---- Lower-level interface that converts references into actuator commands ----

_prev_steering_error = 0.0


def reset_steering_history() -> None:
    global _prev_steering_error
    _prev_steering_error = 0.0


def lower_controller(state: ArrayLike, desired: ArrayLike, parameters: ArrayLike) -> np.ndarray:
    global _prev_steering_error
    desired = _as_array(desired)
    assert desired.shape == (2,)

    delta_ref, v_ref = float(desired[0]), float(desired[1])

    v_delta, err = steering_controller(state, delta_ref, parameters, prev_error=_prev_steering_error)
    _prev_steering_error = err

    a_cmd = velocity_controller(state, v_ref, parameters)

    return np.array([float(v_delta), float(a_cmd)], dtype=float)

import numpy as np
from numpy.typing import ArrayLike
from simulator import RaceTrack

# ============================================================================
# Path Utility Functions
# ============================================================================

def find_closest_point(state: ArrayLike, path: ArrayLike) -> tuple[int, float]:
    """
    Return the index of the path point closest to the vehicle and the distance.
    state: [sx, sy, δ, v, ϕ]
    path:  N×2 array of [x, y]
    """
    car_pos = state[:2]
    distances = np.linalg.norm(path - car_pos, axis=1)
    idx = int(np.argmin(distances))
    return idx, float(distances[idx])


def find_lookahead_point(
    state: ArrayLike,
    path: ArrayLike,
    lookahead_distance: float
) -> tuple[int, ArrayLike]:
    """
    Starting from the closest path point, move forward until the cumulative
    arc length reaches the target lookahead_distance.
    """
    closest_idx, _ = find_closest_point(state, path)
    cumulative = 0.0
    lookahead_idx = closest_idx

    for i in range(1, len(path)):
        prev_idx = (closest_idx + i - 1) % len(path)
        next_idx = (closest_idx + i) % len(path)
        seg_dist = np.linalg.norm(path[next_idx] - path[prev_idx])
        cumulative += seg_dist

        if cumulative >= lookahead_distance:
            lookahead_idx = next_idx
            break

    return lookahead_idx, path[lookahead_idx]


def estimate_curvature(path: ArrayLike, idx: int, step: int = 5) -> float:
    """
    Estimate curvature at `idx` using the Menger curvature formula.
    """
    N = len(path)
    i1 = (idx - step) % N
    i2 = idx % N
    i3 = (idx + step) % N

    p1, p2, p3 = path[i1], path[i2], path[i3]

    # Triangle area (absolute oriented area × 0.5)
    area = 0.5 * abs(
        (p2[0] - p1[0]) * (p3[1] - p1[1])
        - (p3[0] - p1[0]) * (p2[1] - p1[1])
    )

    a = np.linalg.norm(p2 - p1)
    b = np.linalg.norm(p3 - p2)
    c = np.linalg.norm(p1 - p3)
    denom = a * b * c

    if denom < 1e-9:
        return 0.0

    return float(4.0 * area / denom)


# ============================================================================
# Reference Generators (Velocity S1 and Steering S2)
# ============================================================================

def compute_safe_corner_speed(
    curvature: float,
    v_max: float,
    raceline_mode: bool = False
) -> float:
    """
    Compute a safe speed given curvature, with different lateral g-limits
    for raceline vs. centerline mode.
    """
    if curvature < 1e-6:
        return v_max

    # Grip profiles
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
    """
    Compute the braking distance needed to reduce from `v_current` to `v_target`.
    """
    if v_current <= v_target:
        return 0.0
    return max((v_current**2 - v_target**2) / (2 * a_brake), 0.0)


def compute_reference_velocity(
    state: ArrayLike,
    path: ArrayLike,
    parameters: ArrayLike,
    raceline_mode: bool = False
) -> float:
    """
    S1: Physics-driven velocity planning.
    Scan ahead for upcoming high-curvature segments, compute their safe speed,
    then determine whether braking is required now.
    """
    v_min = float(parameters[2])
    v_max = float(parameters[5])
    a_max = float(parameters[10])

    if v_min <= 20:
        v_min = 20

    v = abs(float(state[3]))
    closest_idx, _ = find_closest_point(state, path)

    # Speed-based lookahead
    base_la = 40
    if v > 60:
        extra = ((v - 60) ** 1.3) / 2
        lookahead_pts = int(min(base_la + v / 2.5 + extra, 150))
    else:
        lookahead_pts = int(min(base_la + v / 2.5, 80))

    min_safe = v_max
    dist_to_corner = 0.0
    found = False

    last_sign = None
    last_sign_dist = None
    cumulative = 0.0

    for i in range(lookahead_pts):
        idx = (closest_idx + i) % len(path)

        if i > 0:
            prev = (closest_idx + i - 1) % len(path)
            cumulative += np.linalg.norm(path[idx] - path[prev])

        curv = estimate_curvature(path, idx, step=3)
        if curv <= 0.0008:
            continue

        safe = compute_safe_corner_speed(curv, v_max, raceline_mode)

        # Determine corner direction for S-curve accounting
        turn_sign = 0.0
        if curv > 0.010:
            p_prev = path[(idx - 1) % len(path)]
            p_next = path[(idx + 1) % len(path)]
            v1 = path[idx] - p_prev
            v2 = p_next - path[idx]
            cross = v1[0] * v2[1] - v1[1] * v2[0]
            if abs(cross) > 1e-9:
                turn_sign = np.sign(cross)

        # Safety factors
        if raceline_mode:
            if curv > 0.030: safe *= 0.50
            elif curv > 0.020: safe *= 0.68
            elif curv > 0.012: safe *= 0.76
            elif curv > 0.006: safe *= 0.90
            elif curv > 0.003: safe *= 0.985
            else:              safe *= 0.998
        else:
            if   curv > 0.030: safe *= 0.76
            elif curv > 0.020: safe *= 0.82
            elif curv > 0.012: safe *= 0.88
            elif curv > 0.006: safe *= 0.94
            elif curv > 0.003: safe *= 0.97
            else:              safe *= 0.995

        # Mild S-curve penalty
        if (
            turn_sign != 0.0
            and last_sign is not None
            and last_sign_dist is not None
            and (turn_sign * last_sign) < 0
            and (cumulative - last_sign_dist) < 40.0
        ):
            safe *= 0.90

        if turn_sign != 0:
            last_sign = turn_sign
            last_sign_dist = cumulative

        if safe < min_safe:
            min_safe = safe
            dist_to_corner = cumulative
            found = True

    # No slower corner found or already below limit
    if not found or min_safe >= v:
        return float(np.clip(v_max, v_min, v_max))

    # Braking model
    brake_accel = 0.85 * a_max
    brake_dist = compute_braking_distance(v, min_safe, brake_accel)
    brake_dist *= 1.40   # safety margin

    if dist_to_corner <= brake_dist:
        v_ref = min_safe
    elif dist_to_corner < 1.3 * brake_dist:
        v_ref = v
    else:
        v_ref = v_max

    return float(np.clip(v_ref, v_min, v_max))


def compute_reference_steering(
    state: ArrayLike,
    centerline: ArrayLike,
    parameters: ArrayLike,
    raceline_mode: bool = False
) -> float:
    """
    S2: Pure pursuit steering with a small Stanley cross-track correction.
    """
    wheelbase = float(parameters[0])
    delta_max = float(parameters[4])

    v = abs(float(state[3]))
    speed = max(v, 1.0)

    # Adaptive lookahead
    if speed < 50:
        Ld_target = 8.0 + 0.30 * speed
    else:
        Ld_target = 23.0 + 0.45 * (speed - 50)

    if raceline_mode:
        Ld_target *= 0.9

    _, look_pt = find_lookahead_point(state, centerline, Ld_target)
    sx, sy, _, _, heading = state

    # Transform into vehicle coordinates
    dx, dy = look_pt[0] - sx, look_pt[1] - sy
    cos_h, sin_h = np.cos(-heading), np.sin(-heading)
    lx = dx * cos_h - dy * sin_h
    ly = dx * sin_h + dy * cos_h

    lx = max(lx, 0.5)
    Ld = max(np.hypot(lx, ly), 1.0)
    alpha = np.arctan2(ly, lx)

    # Pure pursuit
    pp = np.arctan2(2 * wheelbase * np.sin(alpha), Ld)

    # Stanley correction
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
    delta_ref = pp + w * stanley

    return float(np.clip(delta_ref, -0.9 * delta_max, 0.9 * delta_max))


# ============================================================================
# Low-Level Controllers (C1 Velocity, C2 Steering Rate)
# ============================================================================

def velocity_controller(
    state: ArrayLike,
    reference_velocity: float,
    parameters: ArrayLike
) -> float:
    """
    C1: Longitudinal PD controller on velocity.
    """
    v = float(state[3])
    v_ref = float(reference_velocity)
    a_max = float(parameters[10])

    error = v_ref - v

    # Speed-dependent gain when braking
    if error < 0:
        if v < 50:
            kp = 6.5
        elif v > 90:
            kp = 6.5
        else:
            kp = 5.0 + ((v - 50) / 40) * 1.3
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
) -> tuple[float, float]:
    """
    C2: PD steering rate controller.
    """
    delta = float(state[2])
    delta_ref = float(reference_steering)
    v = abs(float(state[3]))

    v_min = float(parameters[7])
    v_max = float(parameters[9])

    error = delta_ref - delta
    d_error = (error - prev_error) / dt

    # Gains based on speed
    if v < 50:
        kp, kd = 3.5, 0.50
    elif v > 70:
        kp, kd = 2.1, 0.85
    else:
        α = (v - 20) / 50
        kp = 2.4 - 0.3 * α
        kd = 0.35 + 0.5 * α

    v_delta = kp * error + kd * d_error
    v_delta = float(np.clip(v_delta, v_min, v_max))

    return v_delta, error


# ============================================================================
# High-Level and Low-Level Interfaces
# ============================================================================

def compute_safe_raceline(
    raceline: ArrayLike,
    racetrack: RaceTrack,
    safety_margin: float = 0.7
) -> ArrayLike:
    """
    (Unused, optional)
    Modify raceline to enforce a margin from track boundaries.
    """
    safe = raceline.copy()

    for i, p in enumerate(raceline):
        d_center = np.linalg.norm(racetrack.centerline - p, axis=1)
        idx = np.argmin(d_center)

        right_pt = racetrack.right_boundary[idx]
        left_pt  = racetrack.left_boundary[idx]
        center   = racetrack.centerline[idx]

        d_r = np.linalg.norm(p - right_pt)
        d_l = np.linalg.norm(p - left_pt)

        if d_r < safety_margin or d_l < safety_margin:
            direction = center - p
            direction /= np.linalg.norm(direction) + 1e-6
            shift = safety_margin - min(d_r, d_l)
            safe[i] = p + direction * shift

    return safe


def controller(
    state: ArrayLike,
    parameters: ArrayLike,
    racetrack: RaceTrack,
    raceline: ArrayLike | None = None
) -> ArrayLike:
    """
    High-level controller that selects velocity + steering references.
    """
    use_raceline = raceline is not None
    path = raceline if use_raceline else racetrack.centerline

    v_ref = compute_reference_velocity(state, path, parameters, raceline_mode=use_raceline)
    δ_ref = compute_reference_steering(state, path, parameters, raceline_mode=use_raceline)

    return np.array([δ_ref, v_ref])


_prev_steering_error = 0.0

def lower_controller(
    state: ArrayLike,
    desired: ArrayLike,
    parameters: ArrayLike
) -> ArrayLike:
    """
    Low-level controller converting desired [δ_ref, v_ref]
    into actual commands [steering_rate, acceleration].
    """
    global _prev_steering_error
    assert desired.shape == (2,)

    δ_ref, v_ref = map(float, desired)

    v_delta, err = steering_controller(
        state, δ_ref, parameters, _prev_steering_error
    )
    _prev_steering_error = err

    a = velocity_controller(state, v_ref, parameters)

    return np.array([v_delta, a])

import numpy as np
from numpy.typing import ArrayLike

from simulator import RaceTrack

# Controller state (for tracking progress and PID)
_controller_state = {
    'last_closest_idx': 0,
    'velocity_integral': 0.0,
    'velocity_prev_error': 0.0,
    'steering_integral': 0.0,
    'steering_prev_error': 0.0
}

def lower_controller(
    state : ArrayLike, desired : ArrayLike, parameters : ArrayLike
) -> ArrayLike:
    # [steer angle, velocity] -> [steering_rate, acceleration]
    assert(desired.shape == (2,))
    
    # PID gains for steering rate control
    Kp_steer = 2.6
    Ki_steer = 0.07
    Kd_steer = 0.4
    
    # PID gains for velocity/acceleration control
    Kp_vel = 5
    Ki_vel = 0.02
    Kd_vel = 0.3
    
    # Current values
    current_steer = state[2]
    current_vel = state[3]
    
    # Desired values
    desired_steer = desired[0]
    desired_vel = desired[1]
    
    # Steering PID control
    steer_error = desired_steer - current_steer
    steer_error = np.arctan2(np.sin(steer_error), np.cos(steer_error))
    
    _controller_state['steering_integral'] += steer_error
    _controller_state['steering_integral'] = np.clip(_controller_state['steering_integral'], -2.0, 2.0)
    
    steer_derivative = steer_error - _controller_state['steering_prev_error']
    steering_rate = Kp_steer * steer_error + Ki_steer * _controller_state['steering_integral'] + Kd_steer * steer_derivative
    _controller_state['steering_prev_error'] = steer_error
    
    # Velocity PID control
    vel_error = desired_vel - current_vel
    _controller_state['velocity_integral'] += vel_error
    _controller_state['velocity_integral'] = np.clip(_controller_state['velocity_integral'], -50.0, 50.0)
    
    vel_derivative = vel_error - _controller_state['velocity_prev_error']
    acceleration = Kp_vel * vel_error + Ki_vel * _controller_state['velocity_integral'] + Kd_vel * vel_derivative
    _controller_state['velocity_prev_error'] = vel_error
    
    # Clip to parameter limits
    steering_rate = np.clip(steering_rate, parameters[7], parameters[9])
    acceleration = np.clip(acceleration, parameters[8], parameters[10])
    
    return np.array([steering_rate, acceleration])

def controller(
    state : ArrayLike, parameters : ArrayLike, racetrack : RaceTrack
) -> ArrayLike:
    """
    Pure Pursuit + Stanley hybrid controller for bicycle model path following.
    State: [x, y, steering_angle, velocity, heading]
    """
    car_pos = state[0:2]
    car_heading = state[4]
    current_vel = state[3]
    wheelbase = parameters[0]
    num_points = len(racetrack.centerline)
    
    # Find closest point (looped)
    start_idx = (_controller_state['last_closest_idx'] - 10) % num_points
    end_idx = (_controller_state['last_closest_idx'] + 50) % num_points
    if start_idx < end_idx:
        search_range = racetrack.centerline[start_idx:end_idx]
        search_indices = np.arange(start_idx, end_idx)
    else:
        search_range = np.vstack((racetrack.centerline[start_idx:], racetrack.centerline[:end_idx]))
        search_indices = np.concatenate((np.arange(start_idx, num_points), np.arange(0, end_idx)))
    
    distances = np.linalg.norm(search_range - car_pos, axis=1)
    local_closest_idx = np.argmin(distances)
    closest_idx = search_indices[local_closest_idx] % num_points
    _controller_state['last_closest_idx'] = closest_idx
    closest_point = racetrack.centerline[closest_idx]
    
    # Cross-track error
    next_idx = (closest_idx + 1) % num_points
    path_vec = racetrack.centerline[next_idx] - racetrack.centerline[closest_idx]
    path_length = np.linalg.norm(path_vec)
    path_vec = path_vec / path_length if path_length > 1e-6 else np.array([1.0, 0.0])
    to_car = car_pos - closest_point
    cross_track_error = path_vec[0] * to_car[1] - path_vec[1] * to_car[0]
    
    # Path heading
    path_heading = np.arctan2(
        racetrack.centerline[next_idx][1] - racetrack.centerline[closest_idx][1],
        racetrack.centerline[next_idx][0] - racetrack.centerline[closest_idx][0]
    )
    heading_error = np.arctan2(np.sin(path_heading - car_heading), np.cos(path_heading - car_heading))
    
    # Pure Pursuit
    lookahead_distance = max(8.0, min(25.0, 0.3 * current_vel + 5.0))
    accumulated_dist = 0.0
    for i in range(num_points):
        idx1 = (closest_idx + i) % num_points
        idx2 = (closest_idx + i + 1) % num_points
        segment_dist = np.linalg.norm(racetrack.centerline[idx2] - racetrack.centerline[idx1])
        if accumulated_dist + segment_dist >= lookahead_distance:
            remaining = lookahead_distance - accumulated_dist
            lookahead_point = racetrack.centerline[idx1] + (remaining / segment_dist) * (racetrack.centerline[idx2] - racetrack.centerline[idx1])
            break
        accumulated_dist += segment_dist
    else:
        lookahead_point = racetrack.centerline[closest_idx]
    
    to_lookahead = lookahead_point - car_pos
    lookahead_dist = np.linalg.norm(to_lookahead)
    if lookahead_dist > 1e-6:
        alpha = np.arctan2(to_lookahead[1], to_lookahead[0]) - car_heading
        alpha = np.arctan2(np.sin(alpha), np.cos(alpha))
        pure_pursuit_steer = np.arctan2(2.0 * wheelbase * np.sin(alpha), lookahead_dist)
    else:
        pure_pursuit_steer = 0.0
    
    # Stanley controller
    k_stanley = 2.5
    stanley_correction = np.arctan(k_stanley * cross_track_error / max(current_vel, 0.5))
    desired_steer = pure_pursuit_steer + 0.3 * stanley_correction
    desired_steer = np.clip(desired_steer, parameters[1], parameters[4])
    
    # Velocity
    base_velocity = 75.0
    
    # Curvature calculation (looped five points, S-curve handling)
    lookahead_offsets = [-5,-3 ,0 ,5 ,10, 15, 20]  # offsets from closest_idx
    points = [racetrack.centerline[(closest_idx + offset) % num_points] for offset in lookahead_offsets]
    total_angle = 0.0
    total_length = 0.0
    for i in range(len(points) - 2):
        v1 = points[i+1] - points[i]
        v2 = points[i+2] - points[i+1]
        l1 = np.linalg.norm(v1)
        l2 = np.linalg.norm(v2)
        if l1 > 1e-6 and l2 > 1e-6:
            cos_angle = np.dot(v1, v2) / (l1 * l2)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.arccos(cos_angle)
            total_angle += angle
            total_length += l1
    curvature = total_angle / (total_length + 1e-6) if total_length > 1e-6 else 0.0
    
    # Adjust velocity
    curvature_factor = 1.0 / (1.0 + 70.0 * curvature)
    cross_track_factor = 1.0 / (1.0 + 0.4 * abs(cross_track_error))
    desired_velocity = base_velocity * curvature_factor * cross_track_factor
    desired_velocity = np.clip(desired_velocity, 15.0, parameters[5])
    
    return np.array([desired_steer, desired_velocity])

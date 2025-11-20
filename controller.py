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
    Kp_steer = 2.5
    Ki_steer = 0.05
    Kd_steer = 0.7
    
    # PID gains for velocity/acceleration control
    Kp_vel = 4
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
    # Normalize steering error to [-pi, pi] range
    steer_error = np.arctan2(np.sin(steer_error), np.cos(steer_error))
    
    _controller_state['steering_integral'] += steer_error
    # Anti-windup: limit integral term
    _controller_state['steering_integral'] = np.clip(_controller_state['steering_integral'], -2.0, 2.0)
    
    steer_derivative = steer_error - _controller_state['steering_prev_error']
    steering_rate = Kp_steer * steer_error + Ki_steer * _controller_state['steering_integral'] + Kd_steer * steer_derivative
    _controller_state['steering_prev_error'] = steer_error
    
    # Velocity PID control
    vel_error = desired_vel - current_vel
    _controller_state['velocity_integral'] += vel_error
    # Anti-windup: limit integral term
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
    # Extract state variables
    car_pos = state[0:2]
    car_heading = state[4]
    current_vel = state[3]
    wheelbase = parameters[0]  # L in bicycle model
    
    # Find closest point on centerline (start search near last position for efficiency)
    start_idx = max(0, _controller_state['last_closest_idx'] - 10)
    end_idx = min(len(racetrack.centerline), _controller_state['last_closest_idx'] + 50)
    
    search_range = racetrack.centerline[start_idx:end_idx]
    if len(search_range) == 0:
        search_range = racetrack.centerline
    
    distances = np.linalg.norm(search_range - car_pos, axis=1)
    local_closest_idx = np.argmin(distances)
    closest_idx = start_idx + local_closest_idx
    
    # Update last closest index
    _controller_state['last_closest_idx'] = closest_idx
    
    closest_point = racetrack.centerline[closest_idx]
    
    # Calculate cross-track error (lateral distance from path)
    if closest_idx < len(racetrack.centerline) - 1:
        # Vector along the path
        path_vec = racetrack.centerline[closest_idx + 1] - racetrack.centerline[closest_idx]
        path_length = np.linalg.norm(path_vec)
        if path_length > 1e-6:
            path_vec = path_vec / path_length
        else:
            path_vec = np.array([1.0, 0.0])
        
        # Vector from closest point to car
        to_car = car_pos - closest_point
        
        # Cross-track error: perpendicular distance from path
        # Use cross product to determine sign (left/right of path)
        # 2D cross product: x1*y2 - y1*x2
        cross_track_error = path_vec[0] * to_car[1] - path_vec[1] * to_car[0]
    else:
        # At end of path, use simple distance
        cross_track_error = np.linalg.norm(car_pos - closest_point)
        path_vec = np.array([np.cos(car_heading), np.sin(car_heading)])
    
    # Calculate path heading at closest point
    if closest_idx < len(racetrack.centerline) - 1:
        path_heading = np.arctan2(
            racetrack.centerline[closest_idx + 1][1] - racetrack.centerline[closest_idx][1],
            racetrack.centerline[closest_idx + 1][0] - racetrack.centerline[closest_idx][0]
        )
    else:
        # Use previous segment or car heading
        if closest_idx > 0:
            path_heading = np.arctan2(
                racetrack.centerline[closest_idx][1] - racetrack.centerline[closest_idx - 1][1],
                racetrack.centerline[closest_idx][0] - racetrack.centerline[closest_idx - 1][0]
            )
        else:
            path_heading = car_heading
    
    # Heading error (difference between car heading and path heading)
    heading_error = path_heading - car_heading
    heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))  # Normalize to [-pi, pi]
    
    # Pure Pursuit: Find lookahead point
    # Adaptive lookahead distance based on velocity
    lookahead_distance = max(8.0, min(25.0, 0.3 * current_vel + 5.0))
    
    # Find lookahead point along centerline
    lookahead_idx = closest_idx
    accumulated_dist = 0.0
    
    for i in range(closest_idx, len(racetrack.centerline)):
        if i < len(racetrack.centerline) - 1:
            segment_dist = np.linalg.norm(racetrack.centerline[i+1] - racetrack.centerline[i])
            if accumulated_dist + segment_dist >= lookahead_distance:
                # Interpolate to get exact lookahead point
                remaining = lookahead_distance - accumulated_dist
                if segment_dist > 1e-6:
                    t = remaining / segment_dist
                    lookahead_point = racetrack.centerline[i] + t * (racetrack.centerline[i+1] - racetrack.centerline[i])
                else:
                    lookahead_point = racetrack.centerline[i]
                break
            accumulated_dist += segment_dist
            lookahead_idx = i + 1
        else:
            # Reached end, use last point
            lookahead_point = racetrack.centerline[-1]
            break
    
    # Pure Pursuit steering calculation
    # Vector from car to lookahead point
    to_lookahead = lookahead_point - car_pos
    lookahead_dist = np.linalg.norm(to_lookahead)
    
    if lookahead_dist > 1e-6:
        # Angle from car heading to lookahead point
        alpha = np.arctan2(to_lookahead[1], to_lookahead[0]) - car_heading
        alpha = np.arctan2(np.sin(alpha), np.cos(alpha))  # Normalize
        
        # Pure Pursuit formula: delta = atan(2*L*sin(alpha) / ld)
        # where L = wheelbase, ld = lookahead distance, alpha = angle to lookahead
        # This gives the steering angle needed to reach the lookahead point
        if abs(lookahead_dist) > 1e-6:
            pure_pursuit_steer = np.arctan2(2.0 * wheelbase * np.sin(alpha), lookahead_dist)
        else:
            pure_pursuit_steer = 0.0
    else:
        pure_pursuit_steer = 0.0
    
    # Stanley controller component: add cross-track error correction
    # Stanley: delta = heading_error + atan(k * cross_track_error / v)
    # where k is gain, v is velocity
    k_stanley = 2.5
    if abs(current_vel) > 0.5:  # Avoid division by zero
        stanley_correction = np.arctan(k_stanley * cross_track_error / current_vel)
    else:
        stanley_correction = k_stanley * cross_track_error / 0.5  # Linear approximation at low speed
    
    # Combine Pure Pursuit and Stanley
    desired_steer = pure_pursuit_steer + 0.3 * stanley_correction
    
    # Clip steering angle to physical limits
    desired_steer = np.clip(desired_steer, parameters[1], parameters[4])
    
    # Velocity control: adaptive based on curvature and cross-track error
    base_velocity = 70.0  # m/s
    
    # Calculate path curvature (approximate)
    curvature = 0.0
    if closest_idx < len(racetrack.centerline) - 2:
        p1 = racetrack.centerline[closest_idx]
        p2 = racetrack.centerline[closest_idx + 1]
        p3 = racetrack.centerline[closest_idx + 2]
        
        # Use three-point circle method to estimate curvature
        v1 = p2 - p1
        v2 = p3 - p2
        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)
        
        if v1_norm > 1e-6 and v2_norm > 1e-6:
            # Angle between segments
            cos_angle = np.dot(v1, v2) / (v1_norm * v2_norm)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.arccos(cos_angle)
            
            # Approximate curvature: larger angle change = higher curvature
            curvature = angle / (v1_norm + v2_norm + 1e-6)
    
    # Reduce speed based on curvature and cross-track error
    curvature_factor = 1.0 / (1.0 + 40.0 * curvature)
    cross_track_factor = 1.0 / (1.0 + 0.4 * abs(cross_track_error))
    
    desired_velocity = base_velocity * curvature_factor * cross_track_factor
    desired_velocity = np.clip(desired_velocity, 25.0, parameters[5])  # Min 25 m/s, max from parameters
    
    return np.array([desired_steer, desired_velocity])
from dataclasses import dataclass
import numpy as np
import math


@dataclass
class Trajectory:
    number_of_points: int
    time_points: np.ndarray
    positions: np.ndarray
    velocities: np.ndarray
    orientation_rpy: np.ndarray
    orientation_rpy_rates: np.ndarray

################################################################################

def resample(orig_path, max_distance=0.2):
    """
    Insert intermediate linearly interpolated waypoints such that the maximum distance between two consecutive waypoints is <max_distance>
    """
    if orig_path.shape[0] > 0:
        number_points = 1
        for i in range(1, orig_path.shape[0]):
            additonal_points = int(np.floor(np.linalg.norm(
                orig_path[i, :] - orig_path[i-1, :]) / max_distance))
            number_points += additonal_points + 1

        resampled_path = np.zeros([number_points, 3])
        resampled_path[0, :] = orig_path[0, :]

        p = 1
        for i in range(1, orig_path.shape[0]):
            additonal_points = int(np.floor(np.linalg.norm(
                orig_path[i, :] - orig_path[i-1, :]) / max_distance))
            for j in range(1, additonal_points+1):
                resampled_path[p, :] = orig_path[i-1, :] + \
                    (orig_path[i, :] - orig_path[i-1, :]) * \
                    j / (additonal_points + 1)
                p += 1
            resampled_path[p, :] = orig_path[i, :]
            p += 1

        return resampled_path

################################################################################

def time_parametrize_const_velocity(path, velocity=0.1):
    """
    Calculate time point of each waypoint given a constant velocity.
    """
    t = np.zeros(path.shape[0])
    for i in range(1, path.shape[0]):
        t[i] = t[i-1] + np.linalg.norm(path[i, :] - path[i-1, :]) / velocity
    return path, t

################################################################################

def get_velocties_from_path(positions, time_points):
    """
    Calculate the instantanous velocity at each waypoint given by each pair of <positions> and <time_points>.

    Returns
    velocities: ndarray
            (3,positions.shape[1])-shaped array of floats containing the velocities at the given positions
    """
    if positions.shape[1] == 1:
        return np.array([0, 0, 0])

    velocities = np.zeros_like(positions)
    velocities[:, 0] = (positions[:, 1] - positions[:, 0]) / \
        (time_points[1] - time_points[0])
    for i in range(1, positions.shape[1]-1):
        pre_vel = (positions[:, i] - positions[:, i-1]) / \
            (time_points[i] - time_points[i-1])
        post_vel = (positions[:, i+1] - positions[:, i]) / \
            (time_points[i+1] - time_points[i])
        velocities[:, i] = pre_vel + post_vel / 2
    velocities[:, positions.shape[1]-1] = np.array([0, 0, 0])
    return velocities

################################################################################

def bang_bang_velocity_profile(sampling_time, start_point, end_point, start_stop_speed, max_speed, max_acceleration):
    """
    Sample path between <start_point> and <end_point> with a bang-bang velocity profile at times seperated by <sampling_time>-

    sampling_time: float
    start_point: ndarray
            (3)-shaped array of floats containing the start position
    end_point: ndarray
            (3)-shaped array of floats containing the end position
    start_stop_speed: float
            the speed at the start and end position
    max_speed: float
        the maximum speed during the profile
    max_acceleration: float
        the maximum acceleration (which is applied at beginning and end of profile)

    Returns
    t: ndarray
            (N)-shaped array of floats containing the times of the sampled points
    point: ndarray
            (3,N)-shaped array of floats containing sampled positions
    v: ndarray
            (N)-shaped array of floats containing the velocities at the sampled points
    """
    t_acc = (max_speed - start_stop_speed) / max_acceleration
    d_acc = 0.5 * max_acceleration * t_acc**2 + start_stop_speed * t_acc

    d = np.linalg.norm(end_point - start_point)
    d_const_speed = d - 2*d_acc

    # acceleration + decceration phase is longer than total distance
    if d_const_speed < 0:
        t_acc = (-start_stop_speed + math.sqrt(start_stop_speed **
                 2 + max_acceleration * d)) / max_acceleration
        d_acc = d/2  # = t_acc * start_stop_speed + 0.5 * max_acceleration * t_acc**2
        d_const_speed = 0
        max_speed = max_acceleration * t_acc + start_stop_speed

    t_const_speed = d_const_speed / max_speed
    total_time = t_acc * 2 + t_const_speed
    t = np.hstack([np.arange(0, total_time, sampling_time),
                  math.ceil(total_time / sampling_time) * sampling_time])

    unit_dir = (end_point - start_point) / d
    point = np.zeros((3, t.size))
    v = np.zeros((3, t.size))
    v[:, 0] = unit_dir * start_stop_speed
    point[:, 0] = start_point
    v[:, t.size - 1] = unit_dir * start_stop_speed
    point[:, t.size-1] = end_point

    for i in range(1, t.size-1):
        if t[i] < t_acc:
            v[:, i] = unit_dir * (start_stop_speed + t[i] * max_acceleration)
            point[:, i] = start_point + unit_dir * \
                (start_stop_speed * t[i] + 0.5 * max_acceleration * t[i]**2)
        elif t[i] < t_acc + t_const_speed:
            v[:, i] = unit_dir * (max_speed)
            point[:, i] = start_point + unit_dir * \
                (d_acc + max_speed * (t[i] - t_acc))
        else:
            v[:, i] = unit_dir * \
                (max_speed - (t[i] - t_const_speed - t_acc) * max_acceleration)
            point[:, i] = start_point + unit_dir * (d_acc + d_const_speed + max_speed * (
                t[i] - t_acc - t_const_speed) - 0.5 * max_acceleration * (t[i] - t_acc - t_const_speed)**2)

    return t, point, v

################################################################################

def trajectory_from_path_bang_bang(target_path, max_velocity, sampling_time, min_speed=0):
    """
    Apply bang_bang_velocity_profile(...) function to each consecutive waypoint-pair to obtain whole trajectory.
    """
    time_points = np.array([0])
    positions = np.zeros((3, 1))
    velocities = np.zeros((3, 1))

    for i in range(target_path.shape[0]-1):
        t, p, v = bang_bang_velocity_profile(
            sampling_time, target_path[i], target_path[i+1], min_speed, max_velocity, 2)
        time_points = np.hstack([time_points[0:-1], time_points[-1] + t])
        positions = np.hstack([positions[:, 0:-1], p])
        velocities = np.hstack([velocities[:, 0:-1], v])
    velocities[:, -1] = np.array([0, 0, 0])

    number_of_points = time_points.size
    orientation_rpy = np.zeros_like(positions)
    orientation_rpy_rates = np.zeros_like(positions)

    return Trajectory(number_of_points, time_points, positions, velocities, orientation_rpy, orientation_rpy_rates)

################################################################################

def trajectory_from_path_const_vel(target_path, max_velocity, sampling_time):
    """
    Create trajectory from given path with a constant velocity <max_velocity> (can be less than that for edge cases) and return waypoint
    samples spaced <sampling_time> seconds apart.
    """
    positions = resample(
        target_path, max_distance=max_velocity*sampling_time).transpose()
    number_of_points = positions.shape[1]

    time_points = np.arange(number_of_points) * sampling_time + 1
    target_path, target_time = time_parametrize_const_velocity(
        target_path, velocity=max_velocity)

    velocities = np.zeros_like(positions)
    velocities = get_velocties_from_path(positions, time_points)
    number_of_points = time_points.size
    orientation_rpy = np.zeros_like(positions)
    orientation_rpy_rates = np.zeros_like(positions)

    return Trajectory(number_of_points, time_points, positions, velocities, orientation_rpy, orientation_rpy_rates)

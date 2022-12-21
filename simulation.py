"""Script demonstrating the joint use of simulation and control.

The simulation is run by a `CtrlAviary` or `VisionAviary` environment.
The control is given by the PID implementation in `DSLPIDControl`.

Example
-------
In a terminal, run as:

    $ python fly.py

Notes
-----
The drones move, at different altitudes, along cicular trajectories 
in the X-Y plane, around point (0, -.3).

"""
import os
import time
import argparse
from datetime import datetime
import pdb
import math
import random
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt

from dataclasses import dataclass
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.envs.VisionAviary import VisionAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.control.SimplePIDControl import SimplePIDControl
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool

from mpc_controller import *

DEFAULT_DRONES = DroneModel("cf2x")
DEFAULT_NUM_DRONES = 1
DEFAULT_PHYSICS = Physics("pyb")
DEFAULT_VISION = False
DEFAULT_GUI = True
DEFAULT_RECORD_VISION = False
DEFAULT_PLOT = False
DEFAULT_USER_DEBUG_GUI = False
DEFAULT_AGGREGATE = True
DEFAULT_OBSTACLES = False
DEFAULT_SIMULATION_FREQ_HZ = 240
DEFAULT_CONTROL_FREQ_HZ = 45
DEFAULT_DURATION_SEC = 60
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False
DEFAULT_LOG = False


@dataclass
class SimulationSettings:
    drone: DroneModel = DEFAULT_DRONES
    num_drones: int = DEFAULT_NUM_DRONES
    physics: Physics = DEFAULT_PHYSICS
    vision: bool = DEFAULT_VISION
    gui: bool = DEFAULT_GUI
    record_video: bool = DEFAULT_RECORD_VISION
    plot: bool = DEFAULT_PLOT
    user_debug_gui: bool = DEFAULT_USER_DEBUG_GUI
    aggregate: bool = DEFAULT_AGGREGATE
    obstacles: bool = DEFAULT_OBSTACLES
    simulation_freq_hz: float = DEFAULT_SIMULATION_FREQ_HZ
    control_freq_hz: float = DEFAULT_CONTROL_FREQ_HZ
    duration_sec: int = DEFAULT_DURATION_SEC
    output_folder: str = DEFAULT_OUTPUT_FOLDER
    colab: bool = DEFAULT_COLAB
    log: bool = DEFAULT_LOG

@dataclass
class Trajectory:
    number_of_points : int
    time_points : np.ndarray
    positions : np.ndarray
    velocities : np.ndarray
    orientation_rpy : np.ndarray
    orientation_rpy_rates : np.ndarray


def setup_simulation(settings: SimulationSettings, initial_xyzs: np.array, initial_rpys: np.array):
    aggr_phy_steps = int(settings.simulation_freq_hz /
                         settings.control_freq_hz) if settings.aggregate else 1

    # Create the environment with or without video capture
    if settings.vision:
        env = VisionAviary(drone_model=settings.drone,
                           num_drones=settings.num_drones,
                           initial_xyzs=initial_xyzs,
                           initial_rpys=initial_rpys,
                           physics=settings.physics,
                           neighbourhood_radius=10,
                           freq=settings.simulation_freq_hz,
                           aggregate_phy_steps=aggr_phy_steps,
                           gui=settings.gui,
                           record=settings.record_video,
                           obstacles=settings.obstacles
                           )
    else:
        env = CtrlAviary(drone_model=settings.drone,
                         num_drones=settings.num_drones,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=settings.physics,
                         neighbourhood_radius=10,
                         freq=settings.simulation_freq_hz,
                         aggregate_phy_steps=aggr_phy_steps,
                         gui=settings.gui,
                         record=settings.record_video,
                         obstacles=settings.obstacles,
                         user_debug_gui=settings.user_debug_gui
                         )

    # Obtain the PyBullet Client ID from the environment
    pyb_client = env.getPyBulletClient()

    # Initialize the logger
    logger = Logger(logging_freq_hz=int(settings.simulation_freq_hz/aggr_phy_steps),
                    num_drones=settings.num_drones,
                    output_folder=settings.output_folder,
                    colab=settings.colab
                    )
    return env, pyb_client, logger, aggr_phy_steps


# Insert intermediate linearly interpolated waypoints with a maximum distance
def resample(orig_path, max_distance = 0.2):
    if orig_path.shape[0] > 0:
        number_points = 1
        for i in range(1,orig_path.shape[0]):
            additonal_points = int(np.floor(np.linalg.norm(orig_path[i,:] - orig_path[i-1,:]) / max_distance))
            number_points += additonal_points + 1

        resampled_path = np.zeros([number_points, 3])
        resampled_path[0,:] = orig_path[0,:]

        p = 1
        for i in range(1,orig_path.shape[0]):
            additonal_points = int(np.floor(np.linalg.norm(orig_path[i,:] - orig_path[i-1,:]) / max_distance))
            for j in range(1, additonal_points+1):
                resampled_path[p,:] = orig_path[i-1,:] + (orig_path[i,:] - orig_path[i-1,:]) * j / (additonal_points + 1)
                p += 1
            resampled_path[p,:] = orig_path[i,:]
            p += 1

        return resampled_path


def time_parametrize_const_velocity(path, velocity = 0.1):
    t = np.zeros(path.shape[0])
    for i in range(1, path.shape[0]):
        t[i] = t[i-1] + np.linalg.norm(path[i,:] - path[i-1,:]) / velocity
    return path, t

def trajectory_from_path(target_path, max_velocity, controller_time_step):
    positions = resample(target_path, max_distance=max_velocity*controller_time_step).transpose()
    number_of_points = positions.shape[1]

    time_points = np.arange(number_of_points) * controller_time_step + 1
    # target_path, target_time = time_parametrize_const_velocity(target_path, velocity=max_velocity)

    velocities = np.zeros_like(positions)
    orientation_rpy = np.zeros_like(positions)
    orientation_rpy_rates = np.zeros_like(positions)

    return Trajectory(number_of_points, time_points, positions, velocities, orientation_rpy, orientation_rpy_rates)

def run(settings: SimulationSettings):
    # spherical obstacles (x,y,z,radius)
    sphereObstacles = [(1., 1., 1., .5), (3., 4., 5., 1.),
                       (4, 2, 3, 1), (6, 3, 1, 2), (6, 1, 1, 2), (8, 1, 4, 1)]


    target_path = np.array([(0.0, 0.0, 0.0),
                           (-0.2650163269488669,
                            0.537122213385431, 0.8007909055043438),
                           (0.08267007495844414,
                            0.8806578958716949, 0.9170031402950918),
                           (4.797320063086116, 3.7664814014520025, 4.9663196295733085),
                           (10.0, 5.0, 5.0)])
    # target_path = np.flip(target_path, 0)

    max_velocity = 1 # meters
    controller_time_step = 0.5 # seconds
    trajectory = trajectory_from_path(target_path, max_velocity=max_velocity, controller_time_step=controller_time_step)

    init_xyzs = np.array([trajectory.positions[:, 0]
                         for i in range(settings.num_drones)])
    init_rpys = np.array([[0, 0,  i * (np.pi/2)/settings.num_drones]
                         for i in range(settings.num_drones)])

    env, pyb_client, logger, aggr_phy_steps = setup_simulation(
        settings, init_xyzs, init_rpys)

    # Load obstacles
    for o in sphereObstacles:
        colSphereId = p.createCollisionShape(p.GEOM_SPHERE, radius=o[3])
        sphere = p.createMultiBody(0, colSphereId, -1, o[0:3])

    # Plot waypoints (only for debugging)
    tmp = p.createVisualShape(p.GEOM_SPHERE, radius=0.025)
    for i in range(0, trajectory.number_of_points, math.ceil(trajectory.number_of_points/50)):
        p.createMultiBody(0, -1, tmp, trajectory.positions[:,i])

    ctrl = [MPCControl(drone_model=settings.drone, timestep_reference=controller_time_step, timestep_mpc_stages=controller_time_step)
                for i in range(settings.num_drones)]


    # Run the simulation
    action = {str(i): np.array([0, 0, 0, 0])
              for i in range(settings.num_drones)}
    CTRL_EVERY_N_STEPS = int(np.floor(env.SIM_FREQ/settings.control_freq_hz))
    START = time.time()
    TIMESTEP = 1 / env.SIM_FREQ * aggr_phy_steps
    for i in range(0, int(settings.duration_sec*env.SIM_FREQ), aggr_phy_steps):

        # Step the simulation
        obs, _, _, _ = env.step(action)

        # Compute control at the desired frequency
        if i % CTRL_EVERY_N_STEPS == 0:
            # Compute control for the current way point
            for j in range(settings.num_drones):

                # compute control action
                action[str(j)], next_pos, _ = ctrl[j].computeControlFromState(control_timestep=CTRL_EVERY_N_STEPS*env.TIMESTEP,
                                                                       state=obs[str(
                                                                           j)]["state"],
                                                                       current_time = i * TIMESTEP,
                                                                       target_time = trajectory.time_points,
                                                                       target_pos = trajectory.positions,
                                                                       target_rpy = trajectory.orientation_rpy,
                                                                       target_vel = trajectory.velocities,
                                                                       target_rpy_rates = trajectory.orientation_rpy_rates)

        # Log the simulation
        if settings.log:
            for j in range(settings.num_drones):
                logger.log(drone=j,
                        timestamp=i/env.SIM_FREQ,
                        state=obs[str(j)]["state"],
                        # control=np.hstack(
                            # [target_path[wp_counters[j], 0:2], init_xyzs[j, 2], init_rpys[j, :], np.zeros(6)])
                        # control=np.hstack([INIT_XYZS[j, :]+TARGET_POS[wp_counters[j], :], INIT_RPYS[j, :], np.zeros(6)])
                        )

        # Printout
        if i % env.SIM_FREQ == 0:
            env.render()
            # Print matrices with the images captured by each drone #
            if settings.vision:
                for j in range(settings.num_drones):
                    print(obs[str(j)]["rgb"].shape, np.average(obs[str(j)]["rgb"]),
                          obs[str(j)]["dep"].shape, np.average(
                              obs[str(j)]["dep"]),
                          obs[str(j)]["seg"].shape, np.average(
                              obs[str(j)]["seg"])
                          )

        # Sync the simulation
        if settings.gui:
            sync(i, START, env.TIMESTEP)

    # Finish simulation and save results
    env.close()
    if settings.log:
        logger.save()
        logger.save_as_csv("pid")  # Optional CSV save
    if settings.plot:
        logger.plot()


if __name__ == "__main__":
    # Define and parse (optional) arguments for the script
    parser = argparse.ArgumentParser(
        description='Helix flight script using CtrlAviary or VisionAviary and DSLPIDControl')
    parser.add_argument('--drone',              default=DEFAULT_DRONES,     type=DroneModel,
                        help='Drone model (default: CF2X)', metavar='', choices=DroneModel)
    parser.add_argument('--num_drones',         default=DEFAULT_NUM_DRONES,
                        type=int,           help='Number of drones (default: 3)', metavar='')
    parser.add_argument('--physics',            default=DEFAULT_PHYSICS,      type=Physics,
                        help='Physics updates (default: PYB)', metavar='', choices=Physics)
    parser.add_argument('--vision',             default=DEFAULT_VISION,      type=str2bool,
                        help='Whether to use VisionAviary (default: False)', metavar='')
    parser.add_argument('--gui',                default=DEFAULT_GUI,       type=str2bool,
                        help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video',       default=DEFAULT_RECORD_VISION,
                        type=str2bool,      help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--plot',               default=DEFAULT_PLOT,       type=str2bool,
                        help='Whether to plot the simulation results (default: True)', metavar='')
    parser.add_argument('--user_debug_gui',     default=DEFAULT_USER_DEBUG_GUI,      type=str2bool,
                        help='Whether to add debug lines and parameters to the GUI (default: False)', metavar='')
    parser.add_argument('--aggregate',          default=DEFAULT_AGGREGATE,       type=str2bool,
                        help='Whether to aggregate physics steps (default: True)', metavar='')
    parser.add_argument('--obstacles',          default=DEFAULT_OBSTACLES,       type=str2bool,
                        help='Whether to add obstacles to the environment (default: True)', metavar='')
    parser.add_argument('--simulation_freq_hz', default=DEFAULT_SIMULATION_FREQ_HZ,
                        type=int,           help='Simulation frequency in Hz (default: 240)', metavar='')
    parser.add_argument('--control_freq_hz',    default=DEFAULT_CONTROL_FREQ_HZ,
                        type=int,           help='Control frequency in Hz (default: 48)', metavar='')
    parser.add_argument('--duration_sec',       default=DEFAULT_DURATION_SEC,         type=int,
                        help='Duration of the simulation in seconds (default: 5)', metavar='')
    parser.add_argument('--output_folder',     default=DEFAULT_OUTPUT_FOLDER, type=str,
                        help='Folder where to save logs (default: "results")', metavar='')
    parser.add_argument('--colab',              default=DEFAULT_COLAB, type=bool,
                        help='Whether example is being run by a notebook (default: "False")', metavar='')
    parser.add_argument('--log',              default=DEFAULT_LOG, type=bool,
                        help='Whether to save logs (default: "False")', metavar='')
    args = parser.parse_args()
    settings = SimulationSettings(args.drone,
                                  args.num_drones,
                                  args.physics,
                                  args.vision,
                                  args.gui,
                                  args.record_video,
                                  args.plot,
                                  args.user_debug_gui,
                                  args.aggregate,
                                  args.obstacles,
                                  args.simulation_freq_hz,
                                  args.control_freq_hz,
                                  args.duration_sec,
                                  args.output_folder,
                                  args.colab,
                                  args.log)
    run(settings)

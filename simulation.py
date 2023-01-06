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
from time_parametrization import *
from plotting import plot_translation_from_logger, plot_3d_from_logger

DEFAULT_DRONES = DroneModel("cf2x")
DEFAULT_NUM_DRONES = 1
DEFAULT_PHYSICS = Physics("pyb")
DEFAULT_VISION = False
DEFAULT_GUI = True
DEFAULT_RECORD_VISION = False
DEFAULT_PLOT = True
DEFAULT_USER_DEBUG_GUI = False
DEFAULT_AGGREGATE = True
DEFAULT_OBSTACLES = False
DEFAULT_SIMULATION_FREQ_HZ = 240
DEFAULT_CONTROL_FREQ_HZ = 45
DEFAULT_DURATION_SEC = 8
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

def run(settings: SimulationSettings, target_path, sphere_obstacles):
    if not(isinstance(target_path, list) and len(target_path) >= 2 and \
        all(isinstance(x,tuple) and len(x) == 3 for x in target_path)):
        raise ValueError("Target path is incorrect type/dimension")
    target_path = np.array(target_path)

    if not(isinstance(sphere_obstacles, list) and \
        all(isinstance(x,tuple) and len(x) == 4 for x in sphere_obstacles)):
        raise ValueError("Obstacles are incorrect type/dimension")

    # Create time parametrized trajectory from given path
    max_velocity = 3 # m/s
    controller_time_step = 0.5
    trajectory_time_step = controller_time_step
    trajectory = trajectory_from_path_bang_bang(target_path, max_velocity=max_velocity, sampling_time=trajectory_time_step, min_speed=0.5, max_acceleration=1.5)
    # trajectory = trajectory_from_path_const_vel(target_path, max_velocity=max_velocity, sampling_time=trajectory_time_step)


    # Inital drone position(s)
    init_xyzs = np.array([trajectory.positions[:, 0]
                         for i in range(settings.num_drones)])
    init_xyzs[:,2] = 0.05
    init_rpys = np.array([[0, 0,  i * (np.pi/2)/settings.num_drones]
                         for i in range(settings.num_drones)])

    # Setup simulation
    env, pyb_client, logger, aggr_phy_steps = setup_simulation(
        settings, init_xyzs, init_rpys)

    # Load obstacles
    for o in sphere_obstacles:
        colSphereId = p.createCollisionShape(p.GEOM_SPHERE, radius=o[3])
        p.createMultiBody(0, colSphereId, -1, o[0:3])

    # Plot waypoints (only for debugging)
    tmp = p.createVisualShape(p.GEOM_SPHERE, radius=0.025)
    for i in range(0, trajectory.number_of_points):
        p.createMultiBody(0, -1, tmp, trajectory.positions[:,i])

    # Controller(s)
    ctrl = [MPCControl(drone_model=settings.drone, timestep_reference=trajectory_time_step, timestep_mpc_stages=controller_time_step)
                for i in range(settings.num_drones)]


    # Run the simulation
    action = {str(i): np.array([0, 0, 0, 0])
              for i in range(settings.num_drones)}
    CTRL_EVERY_N_STEPS = int(np.floor(env.SIM_FREQ/settings.control_freq_hz))
    START = time.time()
    TIMESTEP = 1 / env.SIM_FREQ
    trans_error = 0
    target_index = 0
    i = 0
    # while True:
    while target_index != trajectory.number_of_points-1 or trans_error > 0.05:

        # Step the simulation
        obs, _, _, _ = env.step(action)

        # log inital
        if i == 0 and (settings.log or settings.plot):
                logger.log(drone=0,
                        timestamp=i * TIMESTEP,
                        state=obs[str(0)]["state"],
                        control= np.hstack([trajectory.positions[:,0], trajectory.velocities[:,0], trajectory.orientation_rpy[:,0], trajectory.orientation_rpy_rates[:,0]])
                        )

        # Compute control at the desired frequency
        if i % CTRL_EVERY_N_STEPS == 0:
            # Compute control for each drone
            for j in range(settings.num_drones):

                # compute control action
                action[str(j)], trans_error, current_target, target_index = ctrl[j].computeControlFromState(
                                                                       control_timestep=CTRL_EVERY_N_STEPS*env.TIMESTEP,
                                                                       state=obs[str(
                                                                           j)]["state"],
                                                                       current_time = i * TIMESTEP,
                                                                       target_time = trajectory.time_points,
                                                                       target_pos = trajectory.positions,
                                                                       target_rpy = trajectory.orientation_rpy,
                                                                       target_vel = trajectory.velocities,
                                                                       target_rpy_rates = trajectory.orientation_rpy_rates)

        # Log the simulation
        if settings.log or settings.plot:
            for j in range(settings.num_drones):
                # print(current_target)
                logger.log(drone=j,
                        timestamp=i * TIMESTEP,
                        state=obs[str(j)]["state"],
                        control= current_target
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

        # increment
        i = i + aggr_phy_steps

    # Finish simulation and save results
    env.close()
    if settings.log:
        logger.save()
        logger.save_as_csv("pid")  # Optional CSV save
    if settings.plot:
        plot_translation_from_logger(logger, trajectory)
        plot_3d_from_logger(logger, sphere_obstacles)
        input()

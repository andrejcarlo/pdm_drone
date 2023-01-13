import os
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler
from datetime import datetime
from src.visualisation import plot_obstacles
import math
from matplotlib.animation import FuncAnimation, PillowWriter


def plot_all_from_logger(logger):
    """Plot from logger"""

    #### Loop over colors and line styles ######################
    plt.rc(
        "axes",
        prop_cycle=(
            cycler("color", ["g", "b", "y", "r"])
            + cycler("linestyle", ["-", "--", ":", "-."])
        ),
    )
    fig, axs = plt.subplots(10, 2)
    t = np.arange(0, logger.timestamps.shape[1]) * 1 / logger.LOGGING_FREQ_HZ

    #### Column ################################################
    col = 0

    #### XYZ ###################################################
    row = 0
    for j in range(logger.NUM_DRONES):
        axs[row, col].plot(t, logger.states[j, 0, :], label="drone_" + str(j))
        axs[row, col].plot(t, logger.controls[j, 0, :], label="control_" + str(j))
    axs[row, col].set_xlabel("time")
    axs[row, col].set_ylabel("x (m)")

    row = 1
    for j in range(logger.NUM_DRONES):
        axs[row, col].plot(t, logger.states[j, 1, :], label="drone_" + str(j))
        axs[row, col].plot(t, logger.controls[j, 1, :], label="control_" + str(j))
    axs[row, col].set_xlabel("time")
    axs[row, col].set_ylabel("y (m)")

    row = 2
    for j in range(logger.NUM_DRONES):
        axs[row, col].plot(t, logger.states[j, 2, :], label="drone_" + str(j))
        axs[row, col].plot(t, logger.controls[j, 2, :], label="control_" + str(j))
    axs[row, col].set_xlabel("time")
    axs[row, col].set_ylabel("z (m)")

    #### RPY ###################################################
    row = 3
    for j in range(logger.NUM_DRONES):
        axs[row, col].plot(t, logger.states[j, 6, :], label="drone_" + str(j))
        axs[row, col].plot(t, logger.controls[j, 6, :], label="control_" + str(j))
    axs[row, col].set_xlabel("time")
    axs[row, col].set_ylabel("r (rad)")
    row = 4
    for j in range(logger.NUM_DRONES):
        axs[row, col].plot(t, logger.states[j, 7, :], label="drone_" + str(j))
        axs[row, col].plot(t, logger.controls[j, 7, :], label="control_" + str(j))
    axs[row, col].set_xlabel("time")
    axs[row, col].set_ylabel("p (rad)")
    row = 5
    for j in range(logger.NUM_DRONES):
        axs[row, col].plot(t, logger.states[j, 8, :], label="drone_" + str(j))
        axs[row, col].plot(t, logger.controls[j, 8, :], label="control_" + str(j))
    axs[row, col].set_xlabel("time")
    axs[row, col].set_ylabel("y (rad)")

    #### Ang Vel ###############################################
    row = 6
    for j in range(logger.NUM_DRONES):
        axs[row, col].plot(t, logger.states[j, 9, :], label="drone_" + str(j))
        axs[row, col].plot(t, logger.controls[j, 9, :], label="control_" + str(j))
    axs[row, col].set_xlabel("time")
    axs[row, col].set_ylabel("wx")
    row = 7
    for j in range(logger.NUM_DRONES):
        axs[row, col].plot(t, logger.states[j, 10, :], label="drone_" + str(j))
        axs[row, col].plot(t, logger.controls[j, 10, :], label="control_" + str(j))
    axs[row, col].set_xlabel("time")
    axs[row, col].set_ylabel("wy")
    row = 8
    for j in range(logger.NUM_DRONES):
        axs[row, col].plot(t, logger.states[j, 11, :], label="drone_" + str(j))
        axs[row, col].plot(t, logger.controls[j, 11, :], label="control_" + str(j))
    axs[row, col].set_xlabel("time")
    axs[row, col].set_ylabel("wz")

    #### Time ##################################################
    row = 9
    axs[row, col].plot(t, t, label="time")
    axs[row, col].set_xlabel("time")
    axs[row, col].set_ylabel("time")

    #### Column ################################################
    col = 1

    #### Velocity ##############################################
    row = 0
    for j in range(logger.NUM_DRONES):
        axs[row, col].plot(t, logger.states[j, 3, :], label="drone_" + str(j))
        axs[row, col].plot(t, logger.controls[j, 3, :], label="control_" + str(j))
    axs[row, col].set_xlabel("time")
    axs[row, col].set_ylabel("vx (m/s)")
    row = 1
    for j in range(logger.NUM_DRONES):
        axs[row, col].plot(t, logger.states[j, 4, :], label="drone_" + str(j))
        axs[row, col].plot(t, logger.controls[j, 4, :], label="control_" + str(j))
    axs[row, col].set_xlabel("time")
    axs[row, col].set_ylabel("vy (m/s)")
    row = 2
    for j in range(logger.NUM_DRONES):
        axs[row, col].plot(t, logger.states[j, 5, :], label="drone_" + str(j))
        axs[row, col].plot(t, logger.controls[j, 5, :], label="control_" + str(j))
    axs[row, col].set_xlabel("time")
    axs[row, col].set_ylabel("vz (m/s)")

    #### RPY Rates #############################################
    row = 3
    for j in range(logger.NUM_DRONES):
        rdot = np.hstack(
            [
                0,
                (logger.states[j, 6, 1:] - logger.states[j, 6, 0:-1])
                * logger.LOGGING_FREQ_HZ,
            ]
        )
        axs[row, col].plot(t, rdot, label="drone_" + str(j))
    axs[row, col].set_xlabel("time")
    axs[row, col].set_ylabel("rdot (rad/s)")
    row = 4
    for j in range(logger.NUM_DRONES):
        pdot = np.hstack(
            [
                0,
                (logger.states[j, 7, 1:] - logger.states[j, 7, 0:-1])
                * logger.LOGGING_FREQ_HZ,
            ]
        )
        axs[row, col].plot(t, pdot, label="drone_" + str(j))
    axs[row, col].set_xlabel("time")
    axs[row, col].set_ylabel("pdot (rad/s)")
    row = 5
    for j in range(logger.NUM_DRONES):
        ydot = np.hstack(
            [
                0,
                (logger.states[j, 8, 1:] - logger.states[j, 8, 0:-1])
                * logger.LOGGING_FREQ_HZ,
            ]
        )
        axs[row, col].plot(t, ydot, label="drone_" + str(j))
    axs[row, col].set_xlabel("time")
    axs[row, col].set_ylabel("ydot (rad/s)")

    #### RPMs ##################################################
    row = 6
    for j in range(logger.NUM_DRONES):
        axs[row, col].plot(t, logger.states[j, 12, :], label="drone_" + str(j))
    axs[row, col].set_xlabel("time")
    axs[row, col].set_ylabel("RPM0")
    row = 7
    for j in range(logger.NUM_DRONES):
        axs[row, col].plot(t, logger.states[j, 13, :], label="drone_" + str(j))
    axs[row, col].set_xlabel("time")
    axs[row, col].set_ylabel("RPM1")
    row = 8
    for j in range(logger.NUM_DRONES):
        axs[row, col].plot(t, logger.states[j, 14, :], label="drone_" + str(j))
    axs[row, col].set_xlabel("time")
    axs[row, col].set_ylabel("RPM2")
    row = 9
    for j in range(logger.NUM_DRONES):
        axs[row, col].plot(t, logger.states[j, 15, :], label="drone_" + str(j))
    axs[row, col].set_xlabel("time")
    axs[row, col].set_ylabel("RPM3")

    #### Drawing options #######################################
    for i in range(10):
        for j in range(2):
            axs[i, j].grid(True)
            axs[i, j].legend(loc="upper right", frameon=True)
    fig.subplots_adjust(
        left=0.06, bottom=0.05, right=0.99, top=0.98, wspace=0.15, hspace=0.0
    )
    plt.show()


def plot_translation_from_logger(logger, target):
    """Plot actual and reference position and velocity from logger"""
    #### Loop over colors and line styles ######################
    plt.rc(
        "axes",
        prop_cycle=(
            cycler("color", ["g", "b", "y", "r"])
            + cycler("linestyle", ["-", ":", "--", "-."])
        ),
    )
    fig, axs = plt.subplots(3, 2)
    # t = np.arange(0, logger.timestamps.shape[1]) * 1/logger.LOGGING_FREQ_HZ
    t = np.squeeze(logger.timestamps)

    #### XYZ ###################################################
    row = 0
    col = 0
    for j in range(logger.NUM_DRONES):
        axs[row, col].plot(t, logger.states[j, 0, :], label=r"$x$")
        # axs[row, col].plot(t, logger.controls[j,0, :], label=r'$x_{ref}$')
        axs[row, col].plot(
            target.time_points, target.positions[0, :], label=r"$x_{ref, 2}$"
        )
    axs[row, col].set_ylabel(r"$x \;(m)$")

    row = 1
    for j in range(logger.NUM_DRONES):
        axs[row, col].plot(t, logger.states[j, 1, :], label=r"$y$")
        # axs[row, col].plot(t, logger.controls[j,1, :], label=r'$y_{ref}$')
        axs[row, col].plot(
            target.time_points, target.positions[1, :], label=r"$y_{ref, 2}$"
        )
    axs[row, col].set_ylabel(r"$y \;(m)$")

    row = 2
    for j in range(logger.NUM_DRONES):
        axs[row, col].plot(t, logger.states[j, 2, :], label=r"$z$")
        # axs[row, col].plot(t, logger.controls[j,2, :], label=r'$z_{ref}$')
        axs[row, col].plot(
            target.time_points, target.positions[2, :], label=r"$z_{ref, 2}$"
        )
    axs[row, col].set_xlabel(r"$t \;(s)$")
    axs[row, col].set_ylabel(r"$z \;(m)$")

    #### Velocity ##############################################
    row = 0
    col = 1
    for j in range(logger.NUM_DRONES):
        axs[row, col].plot(t, logger.states[j, 3, :], label=r"actual")
        # axs[row, col].plot(t, logger.controls[j,3, :], label=r'$v_{x,ref}$')
        axs[row, col].plot(
            target.time_points, target.velocities[0, :], label=r"reference"
        )
    axs[row, col].set_ylabel(r"$v_x \;(m/s)$")
    row = 1
    for j in range(logger.NUM_DRONES):
        axs[row, col].plot(t, logger.states[j, 4, :], label=r"$v_y$")
        # axs[row, col].plot(t, logger.controls[j,4, :], label=r'$v_{y,ref}$')
        axs[row, col].plot(
            target.time_points, target.velocities[1, :], label=r"$v_{y, ref, 2}$"
        )
    axs[row, col].set_ylabel(r"$v_y \;(m/s)$")
    row = 2
    for j in range(logger.NUM_DRONES):
        axs[row, col].plot(t, logger.states[j, 5, :], label=r"$v_z$")
        # axs[row, col].plot(t, logger.controls[j,5, :], label=r'$v_{z,ref}$')
        axs[row, col].plot(
            target.time_points, target.velocities[2, :], label=r"$v_{z, ref, 2}$"
        )
    axs[row, col].set_xlabel(r"$t \;(s)$")
    axs[row, col].set_ylabel(r"$v_z \;(m/s)$")

    #### Drawing options #######################################
    for i in range(3):
        for j in range(2):
            axs[i, j].grid(True)

    axs[0, 1].legend(loc="upper right", frameon=True)

    # fig.subplots_adjust(left=0.1,
    #                     bottom=0.05,
    #                     right=0.99,
    #                     top=0.98,
    #                     wspace=0.3,
    #                     hspace=0.0
    #                     )

    fig.subplots_adjust(right=0.99, wspace=0.35, top=0.98, bottom=0.1, hspace=0.0)

    plt.savefig(
        os.path.join(
            "saved_results_simulation",
            datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "output.png",
        )
    )
    fig.show()


def set_axes_equal(ax: plt.Axes):
    """Set 3D plot axes to equal scale.

    Make axes of 3D plot have equal scale so that spheres appear as
    spheres and cubes as cubes.  Required since `ax.axis('equal')`
    and `ax.set_aspect('equal')` don't work on 3D.
    """
    limits = np.array(
        [
            ax.get_xlim3d(),
            ax.get_ylim3d(),
            ax.get_zlim3d(),
        ]
    )
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    _set_axes_radius(ax, origin, radius)


def _set_axes_radius(ax, origin, radius):
    x, y, z = origin
    ax.set_xlim3d([x - radius, x + radius])
    ax.set_ylim3d([y - radius, y + radius])
    ax.set_zlim3d([z - radius, z + radius])


def cuboid_data(o, size=(1, 1, 1)):
    # code taken from
    # https://stackoverflow.com/a/35978146/4124317
    # suppose axis direction: x: to left; y: to inside; z: to upper
    # get the length, width, and height
    l, w, h = size
    x = [
        [o[0], o[0] + l, o[0] + l, o[0], o[0]],
        [o[0], o[0] + l, o[0] + l, o[0], o[0]],
        [o[0], o[0] + l, o[0] + l, o[0], o[0]],
        [o[0], o[0] + l, o[0] + l, o[0], o[0]],
    ]
    y = [
        [o[1], o[1], o[1] + w, o[1] + w, o[1]],
        [o[1], o[1], o[1] + w, o[1] + w, o[1]],
        [o[1], o[1], o[1], o[1], o[1]],
        [o[1] + w, o[1] + w, o[1] + w, o[1] + w, o[1] + w],
    ]
    z = [
        [o[2], o[2], o[2], o[2], o[2]],
        [o[2] + h, o[2] + h, o[2] + h, o[2] + h, o[2] + h],
        [o[2], o[2], o[2] + h, o[2] + h, o[2]],
        [o[2], o[2], o[2] + h, o[2] + h, o[2]],
    ]
    return np.array(x), np.array(y), np.array(z)


def plot_3d_at_percentage(ax, logger, obstacles, completion_rate=1):
    # add reference and obstacles and actual at certain completion_rate
    # to a plot
    j = 0
    ax.plot(
        logger.controls[j, 0, :],
        logger.controls[j, 1, :],
        logger.controls[j, 2, :],
        label="reference",
        color="b",
        linestyle="--",
        zorder = 10,
    )
    i = int(completion_rate * logger.states[j].shape[1])
    ax.plot(
        logger.states[j, 0, :i],
        logger.states[j, 1, :i],
        logger.states[j, 2, :i],
        label="actual",
        color="g",
        linestyle="-",
        zorder = 10,
    )
    plot_obstacles(obstacles, ax)

    ax.margins(0.01)
    ax.set_box_aspect((1, 1, 1))
    set_axes_equal(ax)
    ax.get_xaxis().set_ticklabels([])
    ax.get_yaxis().set_ticklabels([])
    ax.get_zaxis().set_ticklabels([])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend()


def plot_3d_from_logger(logger, obstacles, createAnimation=False):
    """
    Plot actual and reference path in a 3D figure
    """
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    fig.tight_layout()

    if not createAnimation:
        plot_3d_at_percentage(ax, logger, obstacles, completion_rate=1)
        ax.view_init(elev=12.0, azim=-20)
        plt.savefig(
            os.path.join(
                "saved_results_simulation",
                datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "output_2.png",
            )
        )
        fig.show()
    else:
        max_frame = 100
        az_start = -30
        az_end = 10

        def animate(frame):
            p = (frame + 1) / max_frame
            ax.clear()
            plot_3d_at_percentage(ax, logger, obstacles, completion_rate=p)
            ax.view_init(elev=12.0, azim=az_start + (az_end - az_start) * p)

        ani = FuncAnimation(fig, animate, frames=max_frame, interval=25, repeat=True)
        ani.save(
            os.path.join(
                "saved_results_simulation",
                datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "output_3.gif",
            ),
            dpi=300,
            writer=PillowWriter(fps=25),
        )


def get_trajectory_from_logger(logger):
    t = np.squeeze(logger.timestamps)
    pos = logger.states[0, 0:3, :]
    return pos, t

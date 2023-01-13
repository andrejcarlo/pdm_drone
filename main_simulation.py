from src.simulation import *

from src.utils import dijkstra, distance, expand_obstacles, EnhancedJSONEncoder
from src.rrt import iRRT_s, RRT, RRT_s
from src.prm import PRM
from gym_pybullet_drones.utils.utils import str2bool
import argparse
import numpy as np
import json


if __name__ == "__main__":
    # Define and parse (optional) arguments for the script
    parser = argparse.ArgumentParser(
        description="Simulation combining path planning with various planning algorithms and MPC control."
    )
    parser.add_argument(
        "--planner",
        choices=["RRT", "RRT_s", "iRRT_s", "PRM"],
        default="iRRT_s",
        type=str,
        help="Which planning algorithm to use (default: iRRT_s)",
        metavar="",
    )
    parser.add_argument(
        "--room",
        choices=[0, 1, 2, 3],
        default=3,
        type=int,
        help="Which room environment to use (0: little cluttered, ..., 3: very clutterd)",
        metavar="",
    )
    parser.add_argument(
        "--plot",
        default=DEFAULT_PLOT,
        type=str2bool,
        help="Whether to plot the simulation results (default: True)",
        metavar="",
    )
    parser.add_argument(
        "--anim",
        default=False,
        type=str2bool,
        help="Whether to generate an animation from simulation results (default: False)",
        metavar="",
    )
    parser.add_argument(
        "--gui",
        default=DEFAULT_GUI,
        type=str2bool,
        help="Whether to use PyBullet GUI (default: True)",
        metavar="",
    )
    parser.add_argument(
        "--user_debug_gui",
        default=DEFAULT_USER_DEBUG_GUI,
        type=str2bool,
        help="Whether to add debug lines and parameters to the GUI (default: False)",
        metavar="",
    )
    args = parser.parse_args()
    settings = SimulationSettings(
        gui=args.gui,
        plot=args.plot,
        animation=args.anim,
        user_debug_gui=args.user_debug_gui,
    )

    # ----------- ROOM SELECTION ------------
    if args.room == 0:
        obstacles = [(10.0, 6, 2.5, 3, "sphere")]
    elif args.room == 1:
        obstacles = [
            (7.0, 5.5, 5.0, 1, "sphere"),  # O3
            (5.0, 10.0, 5.0, 2, "sphere"),  # O4
            ([1.0, 2.0, 1.0], [9.0, 4.0, 9.0], "cube"),  # O1
            ([11.0, 6.0, 1.0], [17.0, 12.0, 9.0], "cube"),  # O2
            ([5.0, 14.0, 1.0], [7.5, 14.5, 9.0], "cube"),  # O5
            ([0.0, 0.0, 0], [20.0, 15.0, 0.5], "cube"),  # bottom_plate
            ([0.0, 0.0, 10.0], [20.0, 15.0, 10.5], "cube"),  # top_plate
        ]
    elif args.room == 2:
        obstacles = [
            (7.0, 6.5, 5.0, 1, "sphere"),  # O3
            (5.0, 10.0, 5.0, 2, "sphere"),  # O4
            (8.5, 8.0, 8.0, 1.5, "sphere"),  # O6
            (13.5, 3.5, 3.0, 2, "sphere"),  # O7
            (18.5, 2.0, 2.0, 1.5, "sphere"),  # O8
            ([9.5, 8.5, 2.0], [11.5, 4.5, 6.5], "cube"),  # O9
            ([2.0, 3.0, 2.0], [8.0, 5.0, 8.0], "cube"),  # O1
            ([12.0, 7.0, 2.0], [16.0, 11.0, 8.0], "cube"),  # O2
            ([6.0, 14.0, 2.0], [8.0, 14.5, 8.0], "cube"),  # O5
            ([0.0, 0.0, 0], [20.0, 15.0, 0.5], "cube"),  # bottom_plate
            ([0.0, 0.0, 10.0], [20.0, 15.0, 10.5], "cube"),  # top_plate
        ]
    elif args.room == 3:
        obstacles = [
            (7.0, 6.5, 5.0, 1, "sphere"),  # O3
            (10.0, 12.5, 5.0, 2, "sphere"),  # O4
            (8.5, 8.0, 8.0, 1.5, "sphere"),  # O6
            (13.5, 3.5, 3.0, 2, "sphere"),  # O7
            (18.5, 2.0, 2.0, 1.5, "sphere"),  # O8
            ([9.5, 8.5, 2.0], [11.5, 4.5, 6.5], "cube"),  # O9
            ([2.0, 3.0, 2.0], [8.0, 5.0, 8.0], "cube"),  # O1
            ([12.0, 7.0, 2.0], [16.0, 11.0, 8.0], "cube"),  # O2
            ([6.0, 14.0, 2.0], [8.0, 14.5, 8.0], "cube"),  # O5
            ([0.0, 0.0, 0], [20.0, 15.0, 0.5], "cube"),  # bottom_plate
            ([0.0, 0.0, 10.0], [20.0, 15.0, 10.5], "cube"),  # top_plate
            ([0.5, 10.0, 0.5], [3.5, 14.0, 4.0], "cube"),  # O11
            ([15.0, 12.0, 2.0], [18.0, 14.5, 9.0], "cube"),  # O12
            ([9.0, 0.5, 0.5], [11.0, 4.0, 3.5], "cube"),  # O13
            ([17.0, 4.0, 5.5], [20.0, 7.0, 9.0], "cube"),  # O14
            ([4.0, 9.0, 2.5], [7.0, 12.0, 6.5], "cube"),  # O15
        ]

    # add obstacle margin
    margin = 0.5
    obstacles_expanded = expand_obstacles(obstacles, margin)

    # --------- PLANNER PARAMS ----------

    endposition = (19.0, 0.0, 2)
    startposition = (2.5, 8.6, 7.7)

    # Threshold to goal (how close to stop to the goal)
    threshold = 2.0
    # Distance between sampled points
    stepsize = 1.0
    # Set goal (path_length/optimum) to achieve, where optimum is euclidean distance start-end
    goal = 1.75

    # Number of iterations
    iterations = 1000
    # Use obstacle bias
    obstacle_bias = False
    # How much bias towards sampling next to obstacles
    bias = 0.2
    # Distance from obstacle to sample with bias
    rand_radius = 0.5

    # number of samples for PRM to use
    n_samples_prm = 5000

    config = {
        "Start": startposition,
        "Goal": endposition,
        "Iterations": iterations,
        "Threshold": threshold,
        "Step Size": stepsize,
        "Length/Optimum Goal": goal,
        "Using Bias": obstacle_bias,
        "Bias": bias,
        "Sample dist from obstacle": rand_radius,
    }

    print(f"Running planner {args.planner} in room {args.room} with parameters:")
    print(json.dumps(config, indent=4))  # use json to pretty print config

    # ----------- SELECT PLANNER -------------
    if args.planner == "PRM":
        prm_planner = PRM(
            n_samples_prm,
            obstacles=obstacles_expanded,
            start=startposition,
            destination=endposition,
            iterations=iterations,
            goal=goal,
        )
        prm_planner.runPRM()

        if prm_planner.solutionFound:
            # no need to run dijkstra for PRM, PRM does it internally
            target_path = prm_planner.found_path
        else:
            raise RuntimeError("No path found")
    else:
        if args.planner == "RRT":
            G = RRT(
                startposition,
                endposition,
                obstacles_expanded,
                iterations,
                threshold,
                rand_radius,
                bias,
                obstacle_bias,
                stepsize,
                goal,
            )
        elif args.planner == "RRT_s":
            G = RRT_s(
                startposition,
                endposition,
                obstacles_expanded,
                iterations,
                threshold,
                rand_radius,
                bias,
                obstacle_bias,
                stepsize,
                goal,
            )
        elif args.planner == "iRRT_s":
            G = iRRT_s(
                startposition,
                endposition,
                obstacles_expanded,
                iterations,
                threshold,
                rand_radius,
                bias,
                obstacle_bias,
                stepsize,
                goal,
            )

        if G.found_path:
            target_path = dijkstra(G)
        else:
            raise RuntimeError("No path found")

    print("\nPath has been found!")
    print("Now running MPC with the target_path found\n")

    # --------- RUN MPC ----------
    position, t = run(settings, target_path, obstacles)

    # compare length of planned & taken path
    length_planned = sum(
        [
            distance(target_path[i], target_path[i + 1])
            for i in range(len(target_path) - 1)
        ]
    )
    length_actual = sum(
        [distance(position[i], position[i + 1]) for i in range(len(position) - 1)]
    ) + distance(position[-1], target_path[-1])

    print("\n \n ============= FINAL RESULTS ARE ============= ")
    print(
        f"Length MPC path is {length_actual/length_planned * 100}% of length of planned path"
    )

    # keep figures open
    input()

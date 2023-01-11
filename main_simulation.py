from src.simulation import *

from src.utils import dijkstra
from src.rrt import iRRT_s, RRT, RRT_s
from src.prm import PRM

if __name__ == "__main__":
    # Define and parse (optional) arguments for the script
    parser = argparse.ArgumentParser(
        description="Helix flight script using CtrlAviary or VisionAviary and DSLPIDControl"
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
        "--gui",
        default=DEFAULT_GUI,
        type=str2bool,
        help="Whether to use PyBullet GUI (default: True)",
        metavar="",
    )
    parser.add_argument(
        "--record_video",
        default=DEFAULT_RECORD_VISION,
        type=str2bool,
        help="Whether to record a video (default: False)",
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
        "--user_debug_gui",
        default=DEFAULT_USER_DEBUG_GUI,
        type=str2bool,
        help="Whether to add debug lines and parameters to the GUI (default: False)",
        metavar="",
    )
    args = parser.parse_args()
    settings = SimulationSettings(
        gui=args.gui,
        record_video=args.record_video,
        plot=args.plot,
        user_debug_gui=args.user_debug_gui,
    )

    startposition = (0.0, 0.0, 0.0)
    endposition = (10.0, 5.0, 5.0)

    threshold = 2.0
    stepsize = 1.0
    bias = 0.2
    rand_radius = 0.5
    sample_size = 3

    goal = 1.5
    iterations = 200
    obstacle_bias = False

    # spherical obstacles (x,y,z,radius)
    sphere_obstacles = [
        (1.0, 0.5, 1.0, 0.5),
        (3.0, 4.0, 5.0, 1.0),
        (4, 2, 3, 1),
        (6, 3, 1, 2),
        (6, 1, 1, 2),
        (8, 1, 4, 1),
    ]
    sphere_obstacles_margin = sphere_obstacles

    # add obstacle margin
    obstacle_margin = 0.0
    for idx, x in enumerate(sphere_obstacles_margin):
        sphere_obstacles_margin[idx] = (x[0], x[1], x[2], x[3] + obstacle_margin)

    if args.planner == "PRM":
        prm_planner = PRM(
            iterations,
            obstacles=sphere_obstacles_margin,
            start=startposition,
            destination=endposition,
            goal=goal,
        )
        prm_planner.runPRM()

        if prm_planner.solutionFound:
            target_path = prm_planner.found_path
        else:
            raise RuntimeError("No path found")
    else:
        if args.planner == "RRT":
            G = RRT(
                startposition,
                endposition,
                sphere_obstacles_margin,
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
                sphere_obstacles_margin,
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
                sphere_obstacles_margin,
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

    # target_path = [(0.0, 0.0, 0.0),
    #                 (-0.2650163269488669,
    #                 0.537122213385431, 0.8007909055043438),
    #                 (0.08267007495844414,
    #                 0.8806578958716949, 0.9170031402950918),
    #                 (4.797320063086116, 3.7664814014520025, 4.9663196295733085),
    #                 (10.0, 5.0, 5.0)]

    run(settings, target_path, sphere_obstacles)

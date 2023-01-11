import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


from src.utils import distance

# Plotting function
def plot_graph_simple(Graph, obstacles, startposition, endposition, found_path=None):
    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection="3d")
    vertices = np.asarray(Graph.vertices)

    size_x = max(vertices[:, 0:1]) - min(vertices[:, 0:1])
    size_y = max(vertices[:, 1:2]) - min(vertices[:, 1:2])
    size_z = max(vertices[:, 2:]) - min(vertices[:, 2:])

    vex_x = [x for x, y, z in Graph.vertices]
    vex_y = [y for x, y, z in Graph.vertices]
    vex_z = [z for x, y, z in Graph.vertices]

    for obstacle in range(len(obstacles)):
        u, v = np.mgrid[0 : 2 * np.pi : 20j, 0 : np.pi : 10j]
        x = obstacles[obstacle][0] + obstacles[obstacle][3] * np.cos(u) * np.sin(v)
        y = obstacles[obstacle][1] + obstacles[obstacle][3] * np.sin(u) * np.sin(v)
        z = obstacles[obstacle][2] + obstacles[obstacle][3] * np.cos(v)
        ax.plot_surface(x, y, z, color="r")

    ax.scatter(vex_x, vex_y, vex_z, s=10, color="b")
    ax.scatter(
        Graph.startposition[0],
        Graph.startposition[1],
        Graph.startposition[2],
        s=50,
        color="y",
    )
    ax.scatter(
        Graph.endposition[0],
        Graph.endposition[1],
        Graph.endposition[2],
        s=50,
        color="y",
    )

    for edge in Graph.edges:
        x = np.array([Graph.vertices[edge[0]][0], Graph.vertices[edge[1]][0]])
        y = np.array([Graph.vertices[edge[0]][1], Graph.vertices[edge[1]][1]])
        z = np.array([Graph.vertices[edge[0]][2], Graph.vertices[edge[1]][2]])
        ax.plot3D(x, y, z, color="b", linewidth=0.3)

    if found_path is not None:
        length = 0.0
        for path in range(len(found_path) - 1):
            x = np.array([found_path[path][0], found_path[path + 1][0]])
            y = np.array([found_path[path][1], found_path[path + 1][1]])
            z = np.array([found_path[path][2], found_path[path + 1][2]])
            ax.plot3D(x, y, z, color="r", linewidth=3, alpha=0.5)

    ax.set_box_aspect(aspect=(size_x / size_z, size_y / size_z, 1))

    plt.show()


# Plotting function
def plot_graph(
    Graph,
    obstacles,
    startposition,
    endposition,
    RRT_time,
    found_path=None,
    dijkstra_time=None,
    visualize_all=False,
):
    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection="3d")
    vertices = np.asarray(Graph.vertices)

    size_x = max(vertices[:, 0:1]) - min(vertices[:, 0:1])
    size_y = max(vertices[:, 1:2]) - min(vertices[:, 1:2])
    size_z = max(vertices[:, 2:]) - min(vertices[:, 2:])

    vex_x = [x for x, y, z in Graph.vertices]
    vex_y = [y for x, y, z in Graph.vertices]
    vex_z = [z for x, y, z in Graph.vertices]

    plot_obstacles(obstacles, ax)
    if visualize_all:
        ax.scatter(vex_x, vex_y, vex_z, s=10, color="b", zorder=2)
        # for i, j, k in zip(vex_x, vex_y, vex_z):
        #     ax.text3D(i + 0.1, j+0.1, k +0.1, '({}, {}, {})'.format(round(i,2), round(j,2), round(k,2)))

    ax.scatter(
        Graph.startposition[0],
        Graph.startposition[1],
        Graph.startposition[2],
        s=50,
        color="y",
    )
    ax.scatter(
        Graph.endposition[0],
        Graph.endposition[1],
        Graph.endposition[2],
        s=50,
        color="y",
    )

    for edge in Graph.edges:
        x = np.array([Graph.vertices[edge[0]][0], Graph.vertices[edge[1]][0]])
        y = np.array([Graph.vertices[edge[0]][1], Graph.vertices[edge[1]][1]])
        z = np.array([Graph.vertices[edge[0]][2], Graph.vertices[edge[1]][2]])
        if visualize_all:
            ax.plot3D(x, y, z, color="g", linewidth=1.0, zorder=2)

    if found_path is not None:
        length = 0.0
        for path in range(len(found_path) - 1):
            x = np.array([found_path[path][0], found_path[path + 1][0]])
            y = np.array([found_path[path][1], found_path[path + 1][1]])
            z = np.array([found_path[path][2], found_path[path + 1][2]])
            length = length + distance(
                (found_path[path][0], found_path[path][1], found_path[path][2]),
                (
                    found_path[path + 1][0],
                    found_path[path + 1][1],
                    found_path[path + 1][2],
                ),
            )
            ax.plot3D(x, y, z, color="r", linewidth=3, alpha=0.9, zorder=3)
        print("A path was found, with length:", round(length, 2))
        print(
            "The direct path is of length:",
            round(distance(startposition, endposition), 2),
        )
        print("The RRT generation took: " + str(round(RRT_time, 2)) + "s")
        print("The Dijkstra path finding took: " + str(dijkstra_time) + "s")

    if found_path is None:
        print("No path was found.")
        print("The RRT generation took: " + str(round(RRT_time, 2)) + "s")

    # print(size)
    ax.set_box_aspect(aspect=(size_x / size_z, size_y / size_z, 1))
    ax.set_xlabel("$X$", fontsize=20)
    ax.set_ylabel("$Y$", fontsize=20)
    ax.set_zlabel("$Z$", fontsize=20)
    # ax.set_xlim3d(0, 20)
    # ax.set_ylim3d(0, 10)

    # ax.set_zlim3d(0, 10)
    # ax.can_zoom()
    # ax.set_aspect('equal', adjustable='box')
    plt.show()


def plot_obstacles_surface(obstacles, ax):
    for obstacle in range(len(obstacles)):
        if obstacles[obstacle][-1] == "sphere":
            u, v = np.mgrid[0 : 2 * np.pi : 20j, 0 : np.pi : 10j]
            x = obstacles[obstacle][0] + obstacles[obstacle][3] * np.cos(u) * np.sin(v)
            y = obstacles[obstacle][1] + obstacles[obstacle][3] * np.sin(u) * np.sin(v)
            z = obstacles[obstacle][2] + obstacles[obstacle][3] * np.cos(v)
            ax.plot_surface(x, y, z, color="r", alpha=0.25, zorder=1)
        elif obstacles[obstacle][-1] == "cube":
            sizes = np.array(obstacles[obstacle][1]) - np.array(obstacles[obstacle][0])
            X, Y, Z = cuboid_data_surface(tuple(obstacles[obstacle][0]), sizes)
            ax.plot_surface(X, Y, Z, color="b", alpha=0.25, zorder=5)


def plot_obstacles(obstacles, ax):
    for obstacle in range(len(obstacles)):
        if obstacles[obstacle][-1] == "sphere":
            u, v = np.mgrid[0 : 2 * np.pi : 20j, 0 : np.pi : 10j]
            x = obstacles[obstacle][0] + obstacles[obstacle][3] * np.cos(u) * np.sin(v)
            y = obstacles[obstacle][1] + obstacles[obstacle][3] * np.sin(u) * np.sin(v)
            z = obstacles[obstacle][2] + obstacles[obstacle][3] * np.cos(v)
            ax.plot_surface(x, y, z, color="r", alpha=0.5, zorder=5)
        elif obstacles[obstacle][-1] == "cube":
            sizes = np.array(obstacles[obstacle][1]) - np.array(obstacles[obstacle][0])
            data = cuboid_data(tuple(obstacles[obstacle][0]), sizes)
            pc = Poly3DCollection(
                data, facecolors="k", linewidths=1, edgecolors="k", alpha=0.25, zorder=5
            )
            ax.add_collection3d(pc)


def plot_obstacle_map(obstacles, start, goal, set_limits=True):

    fig = plt.figure(figsize=(10, 10))

    ax = plt.axes(projection="3d")
    ax.set_xlabel("$X$", fontsize=20)
    ax.set_ylabel("$Y$", fontsize=20)
    ax.set_zlabel("$Z$", fontsize=20)
    if set_limits:
        ax.set_xlim3d(0, 5)
        ax.set_ylim3d(0, 5)

    ax.set_zlim3d(0, 5)

    # plot start and goal
    ax.scatter(start[0], start[1], start[2], s=50, color="y")
    ax.scatter(goal[0], goal[1], goal[2], s=50, color="y")

    plot_obstacles(obstacles, ax)

    plt.show()


def cuboid_data(o, size=(1, 1, 1)):
    X = [
        [[0, 1, 0], [0, 0, 0], [1, 0, 0], [1, 1, 0]],
        [[0, 0, 0], [0, 0, 1], [1, 0, 1], [1, 0, 0]],
        [[1, 0, 1], [1, 0, 0], [1, 1, 0], [1, 1, 1]],
        [[0, 0, 1], [0, 0, 0], [0, 1, 0], [0, 1, 1]],
        [[0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0]],
        [[0, 1, 1], [0, 0, 1], [1, 0, 1], [1, 1, 1]],
    ]
    X = np.array(X).astype(float)
    for i in range(3):
        X[:, :, i] *= size[i]
    X += np.array(o)
    return X


def cuboid_data_surface(pos, size=(1, 1, 1)):
    # code taken from
    # https://stackoverflow.com/a/35978146/4124317
    # suppose axis direction: x: to left; y: to inside; z: to upper
    # get the (left, outside, bottom) point
    o = [a - b / 2 for a, b in zip(pos, size)]
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

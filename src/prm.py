from src.graph import Graph, Line
from src.utils import (
    distance,
    nearest_node,
    reshape,
    dijkstra,
    in_obstacle,
    trough_obstacle,
    in_ellipsoid,
)

import numpy as np
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm


class PRM:
    def __init__(
        self,
        numOfRandomCoordinates,
        obstacles,
        start,
        destination,
        iterations=5000,
        goal=None,
        fix_room_size=False,
    ):
        self.numOfCoords = numOfRandomCoordinates
        self.coordsList = np.array([])
        self.allObs = obstacles
        self.start = np.array(start)
        self.destination = np.array(destination)
        self.graph = Graph(start, destination, fix_room_size=fix_room_size)
        self.solutionFound = False
        self.found_path = None
        self.maxNumIterations = iterations
        self.goal = goal
        self.optimum_path_length = distance(start, destination)
        self.iterations = 0

    def runPRM(self, visualise=False):
        # do an initial check if start and end are inside obstacles
        if in_obstacle(self.allObs, self.start) or in_obstacle(
            self.allObs, self.destination
        ):
            raise Exception("Start/Goal have been initialized inside obstacles!")

        for i in tqdm(range(self.maxNumIterations)):
            # generate n random samples in the search area
            self.sample_points()

            # filter out all the samples in collision
            self.get_collision_free()

            # link every sample to k neighbours and check if the connections are collision free.
            self.generate_connections()

            # compute the shortest path
            self.found_path = dijkstra(self.graph)

            # check if a solution has been found
            if len(self.found_path) > 1:
                length = 0
                for path in range(len(self.found_path) - 1):
                    length = length + distance(
                        (
                            self.found_path[path][0],
                            self.found_path[path][1],
                            self.found_path[path][2],
                        ),
                        (
                            self.found_path[path + 1][0],
                            self.found_path[path + 1][1],
                            self.found_path[path + 1][2],
                        ),
                    )

                if self.goal is not None:
                    # print(f"Ratio length/optimum is {length/self.optimum_path_length}")
                    # Check if a solution satistfies path length requirement
                    if (length / self.optimum_path_length) <= self.goal:
                        self.solutionFound = True
                        break
                else:
                    self.solutionFound = True
                    break

            if i < (self.maxNumIterations - 1):
                # reset and reiterate
                np.random.seed(np.random.randint(1, 100000))
                self.coordsList = np.array([])
                self.graph = Graph(tuple(self.start), tuple(self.destination))

        # track number of iterations used
        self.iterations = i

    def sample_points(self):
        self.coordsList = np.random.rand(self.numOfCoords, 3)

        fix_room_factor = 1.0 if self.graph.fixed_room_size else -1.0

        # map each resampled of the random sampled points (0.,1.) to the searchbox area
        self.coordsList[:, 0] = (
            self.graph.searchbegin_x
            + fix_room_factor * self.coordsList[:, 0] * self.graph.searchboxsize_x * 2
        )
        self.coordsList[:, 1] = (
            self.graph.searchbegin_y
            + fix_room_factor * self.coordsList[:, 1] * self.graph.searchboxsize_y * 2
        )
        self.coordsList[:, 2] = (
            self.graph.searchbegin_z
            + fix_room_factor * self.coordsList[:, 2] * self.graph.searchboxsize_z * 2
        )

        # add the start and goal to the list of vertices
        start = self.start.reshape(1, 3)
        dest = self.destination.reshape(1, 3)
        self.coordsList = np.concatenate((self.coordsList, start, dest), axis=0)

    def get_collision_free(self, visualisation=False):
        self.collisionFreePoints = np.array([])
        for point in self.coordsList:
            collision = in_obstacle(self.allObs, point)
            if not collision:
                if self.collisionFreePoints.size == 0:
                    self.collisionFreePoints = point
                else:
                    self.collisionFreePoints = np.vstack(
                        [self.collisionFreePoints, point]
                    )

    def generate_connections(self, k=5):
        X = self.collisionFreePoints
        knn = NearestNeighbors(n_neighbors=k)
        knn.fit(X)
        distances, indices = knn.kneighbors(X)
        self.collisionFreePaths = np.empty((1, 3), int)

        for i, p in enumerate(X):
            # ignore first nearest neighbour  as it is the point itself
            for j, neighbour in enumerate(X[indices[i][1:]]):
                start_line = p
                end_line = neighbour
                # check if point is inside an obstacle
                if not in_obstacle(self.allObs, start_line) and not in_obstacle(
                    self.allObs, end_line
                ):
                    # check if connection is collision free with all obstacles
                    if not trough_obstacle(self.allObs, Line(start_line, end_line)):
                        self.collisionFreePaths = np.concatenate(
                            (
                                self.collisionFreePaths,
                                p.reshape(1, 3),
                                neighbour.reshape(1, 3),
                            ),
                            axis=0,
                        )

                        new_start_idx = self.graph.add_vertex(p)
                        new_end_idx = self.graph.add_vertex(neighbour)
                        self.graph.add_edge(
                            new_start_idx, new_end_idx, distances[i, j + 1]
                        )

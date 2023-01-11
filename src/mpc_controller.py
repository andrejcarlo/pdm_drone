import math
import cvxpy as cp
import numpy as np
from gym_pybullet_drones.control.BaseControl import BaseControl
from gym_pybullet_drones.utils.enums import DroneModel
import pybullet as p


class MPCControl(BaseControl):

    ################################################################################

    def __init__(self,
                 drone_model: DroneModel,
                 g: float = 9.81,
                 N=30,
                 timestep_reference=None,
                 timestep_mpc_stages = 0.25
                 ):
        """Common control classes __init__ method.

        Parameters
        ----------
        drone_model : DroneModel
            The type of drone to control (detailed in an .urdf file in folder `assets`).
        g : float, optional
            The gravitational acceleration in m/s^2.
        N : optimization horizon

        """
        super().__init__(drone_model=drone_model, g=g)
        self.t_s = timestep_mpc_stages  # time step per stage
        self.N = N
        self._buildModelMatrices()
        self._buildMPCProblem(timestep_reference)
        self.reset()

    ################################################################################

    def reset(self):
        """Resets the control classes.

        The previous step's and integral errors for both position and attitude are set to zero.

        """
        super().reset()

    ################################################################################

    def _buildModelMatrices(self):
        I_x = self._getURDFParameter('ixx')
        I_y = self._getURDFParameter('iyy')
        I_z = self._getURDFParameter('izz')
        l = self._getURDFParameter('arm')
        I_r = 6e-05
        k_f = self._getURDFParameter('kf')
        k_m = self._getURDFParameter('km')
        m = self._getURDFParameter('m')
        g = 9.81
        k_tx = 0
        k_ty = 0
        k_tz = 0
        k_rx = 0
        k_ry = 0
        k_rz = 0
        w_r = 0

        # matrix to convert inputs (=forces) to rpm^2
        # rpm^2 = K * u
        self.K = np.array([[1/(4*k_f), 0, 1/(2*k_f), 1/(4*k_m)],
                           [1/(4*k_f), -1/(2*k_f), 0, -1/(4*k_m)],
                           [1/(4*k_f), 0, -1/(2*k_f), 1/(4*k_m)],
                           [1/(4*k_f), 1/(2*k_f), 0, -1/(4*k_m)]])


        # operating point for linearization
        self.x_op = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0])
        self.hover_rpm = np.full(4, math.sqrt(m*g / (4*k_f)))
        self.u_op = np.matmul(np.linalg.inv(self.K), np.square(
            self.hover_rpm))

        x = self.x_op
        u = self.u_op

        self.A = np.array([[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, -k_tx/m, 0, 0, (u[0]*(math.cos(x[6])*math.sin(x[8]) - math.cos(x[8])*math.sin(x[6])*math.sin(x[7])))/m, (u[0]*math.cos(
                               x[6])*math.cos(x[7])*math.cos(x[8]))/m, (u[0]*(math.cos(x[8])*math.sin(x[6]) - math.cos(x[6])*math.sin(x[7])*math.sin(x[8])))/m, 0, 0, 0],
                           [0, 0, 0, 0, -k_ty/m, 0, -(u[0]*(math.cos(x[6])*math.cos(x[8]) + math.sin(x[6])*math.sin(x[7])*math.sin(x[8])))/m, (u[0]*math.cos(
                               x[6])*math.cos(x[7])*math.sin(x[8]))/m, (u[0]*(math.sin(x[6])*math.sin(x[8]) + math.cos(x[6])*math.cos(x[8])*math.sin(x[7])))/m, 0, 0, 0],
                           [0, 0, 0, 0, 0, -k_tz/m, -(u[0]*math.cos(x[7])*math.sin(x[6]))/m, -(
                               u[0]*math.cos(x[6])*math.sin(x[7]))/m, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, x[10]*math.cos(x[6])*math.tan(x[7]) - x[11]*math.sin(x[6])*math.tan(x[7]), x[11]*math.cos(x[6])*(math.tan(
                               x[7])**2 + 1) + x[10]*math.sin(x[6])*(math.tan(x[7])**2 + 1), 0, 1, math.sin(x[6])*math.tan(x[7]), math.cos(x[6])*math.tan(x[7])],
                           [0, 0, 0, 0, 0, 0, - x[11]*math.cos(x[6]) - x[10]*math.sin(
                               x[6]), 0, 0, 0, math.cos(x[6]), -math.sin(x[6])],
                           [0, 0, 0, 0, 0, 0, (x[10]*math.cos(x[6]))/math.cos(x[7]) - (x[11]*math.sin(x[6]))/math.cos(x[7]), (x[11]*math.cos(x[6])*math.sin(x[7]))/math.cos(
                               x[7])**2 + (x[10]*math.sin(x[6])*math.sin(x[7]))/math.cos(x[7])**2, 0, 0, math.sin(x[6])/math.cos(x[7]), math.cos(x[6])/math.cos(x[7])],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, -k_rx/I_x, -
                            (I_r*w_r - I_y*x[11] + I_z*x[11])/I_x, (I_y*x[10] - I_z*x[10])/I_x],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0,
                            (I_r*w_r - I_x*x[11] + I_z*x[11])/I_y, -k_ry/I_y, -(I_x*x[9] - I_z*x[9])/I_y],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, (I_x*x[10] - I_y*x[10])/I_z, (I_x*x[9] - I_y*x[9])/I_z, -k_rz/I_z]])

        self.B = np.array([[0, 0, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0],
                           [(math.sin(x[6])*math.sin(x[8]) + math.cos(x[6])
                             * math.cos(x[8])*math.sin(x[7]))/m, 0, 0, 0],
                           [-(math.cos(x[8])*math.sin(x[6]) - math.cos(x[6])
                              * math.sin(x[7])*math.sin(x[8]))/m, 0, 0, 0],
                           [(math.cos(x[6])*math.cos(x[7]))/m, 0, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0],
                           [0, -l/I_x, 0, 0],
                           [0, 0, -l/I_y, 0],
                           [0, 0, 0, -l/I_z]])

        # time discretize system
        self.A = self.t_s * self.A + np.identity(12)
        self.B = self.t_s * self.B

        # weight cost matrices
        self.W_output = np.diag(
            [1, 1, 1, 1, 1, 1, 0.001, 0.001, 0.001, 0.05, 0.05, 0.05])
        self.W_input = np.identity(4)*0.01

    ################################################################################

    def _buildMPCProblem(self, timestep_reference=None):
        if timestep_reference == None:
            timestep_reference = self.t_s
        elif timestep_reference % self.t_s != 0:
            raise Exception(
                "MPC Controller: timestep_reference must be whole-number multiple of the optimization time step")

        cost = 0.
        constraints = []

        # Parameters
        x_ref = cp.Parameter((12, self.N), name="x_ref")
        x_init = cp.Parameter((12), name="x_init")

        # Create the optimization variables
        x = cp.Variable((12, self.N + 1), name="x")
        u = cp.Variable((4, self.N), name="u")

        # For each stage in k = 0, ..., N-1
        for k in range(self.N):
            cost += cp.quad_form(x[:, k+1] - x_ref[:, k], self.W_output)

            # Cost
            cost += cp.quad_form(u[:, k], self.W_input)

            # System dynamics
            constraints += [x[:, k+1] == self.A@x[:, k] + self.B@u[:, k]]

            # Constraints
            constraints += [x[6:9, k] >=
                            np.array([-math.pi, -math.pi/2, -math.pi])]
            constraints += [x[6:9, k] <=
                            np.array([math.pi, math.pi/2, math.pi])]
            constraints += [self.K @ u[:, k] >= -np.matmul(self.K, self.u_op)]

        # Inital condition
        constraints += [x[:, 0] == x_init]

        self.problem = cp.Problem(cp.Minimize(cost), constraints)

    ################################################################################

    def _computeRPMfromInputs(self, u_delta):
        """
        Computes the rpm given the small-signal u_delta around the operating point.
        """
        return np.sqrt(np.matmul(self.K, self.u_op + u_delta))

    ################################################################################

    def _getNextGoalIndices(self, current_time, target_times, cur_pos, target_pos, select_spatially_closest=False):
        """
        Computes the upcoming next self.N waypoints to target and returns their indices.
        current_time:
            float with current time
        target_times:
            (n)-shaped float array with desired arrival times of waypoints
        """
        next_goal_indices = np.zeros(self.N, dtype=int)

        if select_spatially_closest:
            upcoming_goal_index = min(np.linalg.norm(target_pos - np.reshape(cur_pos, (3,1)), axis=0).argmin()+1, target_times.shape[0]-1)
        else:
            delta_times = target_times - current_time
            if (delta_times <= 0).all():
                upcoming_goal_index = target_times.shape[0]-1
            else:
                upcoming_goal_index = np.where(
                    delta_times > 0, delta_times, np.inf).argmin()

        remaining_goals_count = target_times.shape[0] - upcoming_goal_index
        if remaining_goals_count >= self.N:
            next_goal_indices = np.arange(
                upcoming_goal_index, upcoming_goal_index + self.N, dtype=int)
        else:
            next_goal_indices[0:remaining_goals_count] = np.arange(
                upcoming_goal_index, target_times.shape[0], dtype=int)
            next_goal_indices[remaining_goals_count:] = int(target_times.shape[0]-1)

        return next_goal_indices

    ################################################################################

    def computeControlFromState(self,
                                control_timestep,
                                state,
                                target_pos,
                                current_time,
                                target_time,
                                target_rpy=np.zeros(3),
                                target_vel=np.zeros(3),
                                target_rpy_rates=np.zeros(3)
                                ):
        """
        target_time : ndarray
            (1,n)-shaped array of floats containing the desired arrival times of the given waypoints.
        """
        return self.computeControl(control_timestep,
                                   state[0:3],
                                   state[3:7],
                                   state[10:13],
                                   state[13:16],
                                   current_time,
                                   target_time,
                                   target_pos=target_pos,
                                   target_rpy=target_rpy,
                                   target_vel=target_vel,
                                   target_rpy_rates=target_rpy_rates
                                   )

    ################################################################################

    def computeControl(self,
                       control_timestep,
                       cur_pos,
                       cur_quat,
                       cur_vel,
                       cur_ang_vel,
                       current_time,
                       target_time,
                       target_pos,
                       target_rpy=[None],
                       target_vel=None,
                       target_rpy_rates=None
                       ):
        """Computes the control action (as RPMs) for a single drone.

        Parameters
        ----------
        control_timestep : float
            The time step at which control is computed.
        cur_pos : ndarray
            (3,1)-shaped array of floats containing the current position.
        cur_quat : ndarray
            (4,1)-shaped array of floats containing the current orientation as a quaternion.
        cur_vel : ndarray
            (3,1)-shaped array of floats containing the current velocity.
        cur_ang_vel : ndarray
            (3,1)-shaped array of floats containing the current angular velocity.
        current_time:
            float with current time
        target_times:
            (n)-shaped float array with desired arrival times of waypoints
        target_pos : ndarray
            (3,n)-shaped array of floats containing the desired position.
        target_rpy : ndarray, optional
            (3,n)-shaped array of floats containing the desired orientation as roll, pitch, yaw.
        target_vel : ndarray, optional
            (3,n)-shaped array of floats containing the desired velocity.
        target_rpy_rates : ndarray, optional
            (3,n)-shaped array of floats containing the desired roll, pitch, and yaw rates.

        Returns
        -------
        ndarray
            (4,1)-shaped array of integers containing the RPMs to apply to each of the 4 motors.
        ndarray
            (3,1)-shaped array of floats containing the next predicted state
        float
            Nothing

        """
        # Check inputs
        if target_pos.shape[0] != 3:
            print("\n[ERROR] MPCController reference has incorrect dimension (1)")
        if target_vel is None:
            target_vel = np.zeros_like(target_pos)
        if target_rpy is None:
            target_rpy = np.zeros_like(target_pos)
        if target_rpy_rates is None:
            target_rpy_rates = np.zeros_like(target_pos)
        if any(s != target_pos.shape for s in [target_vel.shape, target_rpy.shape, target_rpy_rates.shape]):
            print("\n[ERROR] MPCController reference has incorrect dimension (2)")

        # Extract next self.N_ref goals from target path
        next_goal_indices = self._getNextGoalIndices(current_time, target_time, cur_pos, target_pos)

        # Current state
        cur_state = np.zeros(12,)
        cur_state[0:3] = cur_pos
        cur_state[3:6] = cur_vel
        cur_state[6:9] = p.getEulerFromQuaternion(cur_quat)
        cur_state[9:12] = cur_ang_vel

        # Solve MPC
        self.problem.param_dict["x_ref"].value = np.vstack(
            [target_pos, target_vel, target_rpy, target_rpy_rates])[:, next_goal_indices]
        self.problem.param_dict["x_init"].value = cur_state
        self.problem.solve(solver=cp.ECOS)

        # Convert small-signal u into large-signal rpm
        rpm = self._computeRPMfromInputs(
            self.problem.var_dict["u"].value[:, 0])

        translation_error = np.linalg.norm(self.problem.var_dict["x"].value[0:3, 0] - self.problem.param_dict["x_ref"].value[0:3, 0])
        return rpm, translation_error, self.problem.param_dict["x_ref"].value[:, 0], next_goal_indices[0]

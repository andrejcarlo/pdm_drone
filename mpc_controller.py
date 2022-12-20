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
                 N=30
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
        self.N = N
        self._buildModelMatrices()
        self._buildMPCProblem()
        self.reset()

    ################################################################################

    def reset(self):
        """Resets the control classes.

        The previous step's and integral errors for both position and attitude are set to zero.

        """
        super().reset()

    ################################################################################

    def _buildModelMatrices(self):
        # I_x = 0.0075
        # I_y = 0.0075
        # I_z = 0.0075
        I_x = self._getURDFParameter('ixx')
        I_y = self._getURDFParameter('iyy')
        I_z = self._getURDFParameter('izz')
        # l = 0.23
        l = self._getURDFParameter('arm')
        I_r = 6e-05
        # k_f = 3.13e-05
        # k_m = 7.5e-07
        # m = 0.65
        k_f = self._getURDFParameter('kf')
        k_m = self._getURDFParameter('km')
        m = self._getURDFParameter('m')
        g = 9.81
        # k_tx = 0.1
        # k_ty = 0.1
        # k_tz = 0.1
        # k_rx = 0.1
        # k_ry = 0.1
        # k_rz = 0.1
        k_tx = 0
        k_ty = 0
        k_tz = 0
        k_rx = 0
        k_ry = 0
        k_rz = 0
        w_r = 0

        self.K = np.array([[1/(4*k_f), 0, 1/(2*k_f), 1/(4*k_m)],
                           [1/(4*k_f), -1/(2*k_f), 0, -1/(4*k_m)],
                           [1/(4*k_f), 0, -1/(2*k_f), 1/(4*k_m)],
                           [1/(4*k_f), 1/(2*k_f), 0, -1/(4*k_m)]])

        self.hover_rpm = np.full(4, math.sqrt(m*g / (4*k_f)))

        # operating point for linearization
        self.x_op = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0])
        self.u_op = np.matmul(np.linalg.inv(self.K), np.square(
            self.hover_rpm))  * 0.8# np.array([m*g, 0, 0, 0])
        x = self.x_op
        u = self.u_op
        # u = np.zeros(4)

        t_s = 0.1 # time step per stage

        # k_ry k_rz drag made negative
        self.A = np.array([[0,0,0,1,0,0,0,0,0,0,0,0],
        [0,0,0,0,1,0,0,0,0,0,0,0],
        [0,0,0,0,0,1,0,0,0,0,0,0],
        [0,0,0,-k_tx/m,0,0,(u[0]*(math.cos(x[6])*math.sin(x[8]) - math.cos(x[8])*math.sin(x[6])*math.sin(x[7])))/m,(u[0]*math.cos(x[6])*math.cos(x[7])*math.cos(x[8]))/m,(u[0]*(math.cos(x[8])*math.sin(x[6]) - math.cos(x[6])*math.sin(x[7])*math.sin(x[8])))/m,0,0,0],
        [0,0,0,0,-k_ty/m,0,-(u[0]*(math.cos(x[6])*math.cos(x[8]) + math.sin(x[6])*math.sin(x[7])*math.sin(x[8])))/m,(u[0]*math.cos(x[6])*math.cos(x[7])*math.sin(x[8]))/m,(u[0]*(math.sin(x[6])*math.sin(x[8]) + math.cos(x[6])*math.cos(x[8])*math.sin(x[7])))/m,0,0,0],
        [0,0,0,0,0,-k_tz/m,-(u[0]*math.cos(x[7])*math.sin(x[6]))/m,-(u[0]*math.cos(x[6])*math.sin(x[7]))/m,0,0,0,0],
        [0,0,0,0,0,0,x[10]*math.cos(x[6])*math.tan(x[7]) - x[11]*math.sin(x[6])*math.tan(x[7]),x[11]*math.cos(x[6])*(math.tan(x[7])**2 + 1) + x[10]*math.sin(x[6])*(math.tan(x[7])**2 + 1),0,1,math.sin(x[6])*math.tan(x[7]),math.cos(x[6])*math.tan(x[7])],
        [0,0,0,0,0,0,- x[11]*math.cos(x[6]) - x[10]*math.sin(x[6]),0,0,0,math.cos(x[6]),-math.sin(x[6])],
        [0,0,0,0,0,0,(x[10]*math.cos(x[6]))/math.cos(x[7]) - (x[11]*math.sin(x[6]))/math.cos(x[7]),(x[11]*math.cos(x[6])*math.sin(x[7]))/math.cos(x[7])**2 + (x[10]*math.sin(x[6])*math.sin(x[7]))/math.cos(x[7])**2,0,0,math.sin(x[6])/math.cos(x[7]),math.cos(x[6])/math.cos(x[7])],
        [0,0,0,0,0,0,0,0,0,-k_rx/I_x,-(I_r*w_r - I_y*x[11] + I_z*x[11])/I_x,(I_y*x[10] - I_z*x[10])/I_x],
        [0,0,0,0,0,0,0,0,0,(I_r*w_r - I_x*x[11] + I_z*x[11])/I_y,-k_ry/I_y,-(I_x*x[9] - I_z*x[9])/I_y],
        [0,0,0,0,0,0,0,0,0,(I_x*x[10] - I_y*x[10])/I_z,(I_x*x[9] - I_y*x[9])/I_z,-k_rz/I_z]])

        self.B = np.array([[0,0,0,0],
        [0,0,0,0],
        [0,0,0,0],
        [(math.sin(x[6])*math.sin(x[8]) + math.cos(x[6])*math.cos(x[8])*math.sin(x[7]))/m,0,0,0],
        [-(math.cos(x[8])*math.sin(x[6]) - math.cos(x[6])*math.sin(x[7])*math.sin(x[8]))/m,0,0,0],
        [(math.cos(x[6])*math.cos(x[7]))/m,0,0,0],
        [0,0,0,0],
        [0,0,0,0],
        [0,0,0,0],
        [0,-l/I_x,0,0],
        [0,0,-l/I_y,0],
        [0,0,0,-l/I_z]])

        self.A = np.matmul(np.diag([t_s, t_s, t_s, t_s, t_s, t_s, t_s, t_s, t_s, t_s, t_s, t_s]), self.A)
        self.B = np.matmul(np.diag([t_s, t_s, t_s, t_s, t_s, t_s, t_s, t_s, t_s, t_s, t_s, t_s]), self.B)
        self.A += np.identity(12)

        print(self.A)
        print(self.B)

        self.W_output = np.diag([1, 1, 10, 1, 1, 1, 1, 1, 1, 0.1, 0.1, 0.1])
        self.W_input = np.identity(4)*0.5
    def _buildMPCProblem(self):
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
            # Cost
            cost += cp.quad_form(x[:, k+1] - x_ref[:, 0], self.W_output)
            cost += cp.quad_form(u[:, k], self.W_input)

            # System dynamics
            constraints += [x[:, k+1] == self.A@x[:, k] + self.B@u[:, k]]

            # Constraints
            # constraints += [x[3:6, k] <=
            #                 np.array([0.1, 0.1, 0.1])]
            constraints += [x[6:9, k] >=
                            np.array([-math.pi, -math.pi/2, -math.pi])]
            constraints += [x[6:9, k] <=
                            np.array([math.pi, math.pi/2, math.pi])]
            # constraints += [u[:, k] >= np.array([-0.1, -0.1, -0.1, -0.1])]
            constraints += [u[:, k] <= np.array([0.1, 0.1, 0.1, 0.1])]
            constraints += [self.K @ u[:, k]  >= -np.matmul(self.K, self.u_op)]
            # constraints += [u[:, k] >= np.array([-5, -5, -5, -5]) 

        # Inital condition
        # cost += cp.quad_form(x[:, 1] - x_ref[:, 0], self.W_output)
        constraints += [x[:, 0] == x_init]

        # print("Constraint u")
        # print(-np.matmul(self.K, self.u_op))
        # print(self.K)
        self.problem = cp.Problem(cp.Minimize(cost), constraints)

 ################################################################################

    def _computeRPMfromInputs(self, u_delta):
        # tmp = np.matmul(self.K, u)
        # m = tmp < 0
        # # tmp[m] = -tmp[m]
        # rpm = np.sqrt(tmp)
        # rpm[m] = 0
        return np.sqrt(np.matmul(self.K, self.u_op + u_delta))

 ################################################################################

    def computeControl(self,
                       control_timestep,
                       cur_pos,
                       cur_quat,
                       cur_vel,
                       cur_ang_vel,
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
        target_pos : ndarray
            (3,1)-shaped array of floats containing the desired position.
        target_rpy : ndarray, optional
            (3,1)-shaped array of floats containing the desired orientation as roll, pitch, yaw.
        target_vel : ndarray, optional
            (3,1)-shaped array of floats containing the desired velocity.
        target_rpy_rates : ndarray, optional
            (3,1)-shaped array of floats containing the desired roll, pitch, and yaw rates.

        Returns
        -------
        ndarray
            (4,1)-shaped array of integers containing the RPMs to apply to each of the 4 motors.
        ndarray
            (3,1)-shaped array of floats containing the current XYZ position error.
        float
            The current yaw error.

        """
        target_pos = np.zeros((3, self.N))
        target_pos[0, :] = 0.5
        target_pos[1, :] = 1
        target_pos[2, :] = 2
        # if target_vel is None:
        target_vel = np.zeros((3, self.N))
        # if target_rpy is None:
        target_rpy = np.zeros((3, self.N))
        # if target_rpy_rates is None:
        target_rpy_rates = np.zeros((3, self.N))

        cur_state = np.zeros(12,)
        cur_state[0:3] = cur_pos
        cur_state[3:6] = cur_vel
        cur_state[6:9] = p.getEulerFromQuaternion(cur_quat)
        cur_state[9:12] = cur_ang_vel

        if any(s != (3, self.N) for s in [target_pos.shape, target_vel.shape, target_rpy.shape, target_rpy_rates.shape]):
            print("\n[ERROR] MPCController reference has incorrect dimension")

        self.problem.param_dict["x_ref"].value = np.vstack(
            [target_pos, target_vel, target_rpy, target_rpy_rates])
        self.problem.param_dict["x_init"].value = cur_state
        self.problem.solve(solver=cp.ECOS)
        print(self.problem.status)
        print(self.problem.value)

        rpm = self._computeRPMfromInputs(
            self.problem.var_dict["u"].value[:, 0])
        print("u ")
        print(self.problem.var_dict["u"].value[:, 0])
        print(rpm)
        print(target_pos[:, 0])
        print(cur_pos)

        # rpm[:] += 14500
        return rpm, self.problem.var_dict["x"].value[0:3, 1] + cur_state[0:3], 0

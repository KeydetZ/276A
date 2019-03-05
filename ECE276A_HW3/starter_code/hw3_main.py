from utils import *
import scipy.linalg as sp

class localizer():
    def __init__(self):
        self.pose_mu = np.eye(4)
        self.pose_sigma = np.eye(6)


    def hat3(self, v):
        result = np.array([[0, -v[2], v[1]],
                          [v[2], 0, -v[0]],
                          [-v[0], v[1], 0]])
        return result

    def twist_hat(self, u):
        result = np.zeros((4, 4))
        omega_hat = self.hat3(u[3:])
        result[:3, :3] = omega_hat
        result[:3, 3] = u[:3]
        return result

    def curvy_hat(self, u):
        result = np.zeros((u.shape[0], u.shape[0]))

        # linear velocity
        v_hat = self.hat3(u[:3])

        # angular_velocity
        omega_hat = self.hat3(u[3:])
        result[:3, :3] = omega_hat
        result[:3, 3:] = v_hat
        result[3:, 3:] = omega_hat
        return result

    def predict_mu(self, dt, u):
        tran = sp.expm(-dt * self.twist_hat(u))
        self.pose_mu = tran@self.pose_mu

    def predict_sigma(self, dt, u):
        tran = sp.expm(-dt * self.curvy_hat(u))
        self.pose_sigma = tran@self.pose_sigma@tran.T

def pose_inverse(pose):
    """
    :param pose: 4x4
    :return: inverse pose 4x4
    """
    result = np.eye(4, 4)
    result[:3, :3] = pose[:3, :3].T
    result[:3, 3] = -np.matmul(pose[:3, :3].T, pose[:3, 3])
    return result

class update():
    def __init__(self, M, D):
        # mu: 4x1
        self.mu = np.array([0, 0, 0, 1]).T
        self.sigma = np.eye(3)
        self.M = M
        self.D = D
        #self.q = cam_T_imu @ self.mu
        self.z = self.z_hat()

    def z_hat(self, Tt):
        # 4x1
        q = cam_T_imu @ Tt @ self.mu
        return self.M @ projection_d(q)

    def kalman_gain(self):
        # 3x3 @ 3X4 @ (4X3 @ 3x3 @ 3x4 + 4x4) = 3x4
        Q = np.eye(4)
        H = self.jacobian()
        Kt = self.sigma @ H.T @ (H @ self.sigma @ H.T + Q)
        return Kt

    def jacobian(self, Tt):
        q = cam_T_imu @ Tt @ self.mu
        # 4x4 @ 4x4 @ 4X3 = 4X3
        H = self.M @ projection_d(q) @ self.D
        return H

    def predict_mu(self, feature):
        zt = feature
        Kt = self.kalman_gain()
        self.mu = self.mu + self.D @ Kt @ (zt - self.z)
        # self.q = cam_T_imu @ self.mu

    def predict_sigma(self):
        H = self.jacobian()
        I = np.eye(3)
        Kt = self.kalman_gain()
        self.sigma = (I - Kt @ H) @ self.sigma

    def update_param(self):
        #self.q = cam_T_imu @ self.mu



def projection_d(q):
    # projection function and its derivative:
    derivative_matrix = [[1, 0, -q[0] / [2], 0],
                         [0, 1, -q[1] / q[2], 0],
                         [0, 0, 0, 0],
                         [0, 0, -q[3] / q[2], 1]]
    proj_d = (1 / q[3]) @ derivative_matrix
    return proj_d

def valid_features(features):
    indValid = np.empty((features.shape))
    for pixel in range(len(features)):
        for feature in range(len(features[0])):
            for epoch in range(len(features[0][0])):
                # shape: 4 x n x time
                indValid[pixel][feature][epoch] = features[pixel][feature][epoch] > 0
    # result = np.where(indValid=True)
    return indValid

if __name__ == '__main__':
    """
        t: time stamp
                with shape 1*t: t = 500
            features: visual feature point coordinates in stereo images,
                with shape 4*n*t, where n is number of features 4x104x500
            linear_velocity: IMU measurements in IMU frame
                with shape 3*t
            rotational_velocity: IMU measurements in IMU frame
                with shape 3*t
            K: (left)camera intrinsic matrix
                [fx  0 cx
                  0 fy cy
                  0  0  1]
                with shape 3*3
            b: stereo camera baseline
                with shape 1
            cam_T_imu: extrinsic matrix from IMU to (left)camera, in SE(3).
                close to
                [ 0 -1  0 t1
                  0  0 -1 t2
                  1  0  0 t3
                  0  0  0  1]
                with shape 4*4
    """
    filename = "./data/0027.npz"
    t, features, linear_velocity, rotational_velocity, K, b, cam_T_imu = load_data(filename)
    # pose = [0, 0, 0]
    # prev_pose = [0, 0, 0]
    # t_interval = 0.1
    u = np.vstack((linear_velocity, rotational_velocity)).T
    pose = np.empty((4, 4))

    # (a) IMU Localization via EKF Prediction
    prediction = localizer()
    for i in range(len(t[0]) - 1):
        print('predict time epoch %d' %i)
        t_interval = t[0][i + 1] - t[0][i]
        prediction.predict_mu(t_interval, u[i])
        prediction.predict_sigma(t_interval, u[i])

        new_pose = pose_inverse(prediction.pose_mu)
        pose = np.dstack((pose, new_pose))

    #print(pose.shape)
    visualize_trajectory_2d(pose)

    # (b) Landmark Mapping via EKF Update
    # initialize:
    update = update()
    for i in range(len(t[0]) - 1):
        print('update time epoch %d' %i)


    # (c) Visual-Inertial SLAM (Extra Credit)

    # You can use the function below to visualize the robot pose over time
    # visualize_trajectory_2d(world_T_imu,show_ori=True)

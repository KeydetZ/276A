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
    def __init__(self, M, D, size_N, size_M, Tt, valid_index):
        # mu: 4xN
        #print(size_N)
        #print(size_M)
        self.mu = np.zeros((3, size_M))
        self.mu = np.vstack((self.mu, np.ones((1, size_M))))

        # 3M x 3M
        self.sigma = np.eye(3 * size_M) * 0.001

        # 4x4
        self.M = M
        # 4x3
        self.D = D
        # 4xN
        self.z = self.z_hat(Tt, valid_index)

        self.size_N = size_N
        self.size_M = size_M


    def z_hat(self, Tt, valid_index):
        # 4x4 @ 4x4 @ 4xN = 4xN
        print(self.mu[:, valid_index].reshape((4, -1)))
        print('tt', Tt)
        q = cam_T_imu @ Tt @ self.mu[:, valid_index].reshape((4, -1))
        return self.M @ projection(q)

    def kalman_gain(self, Tt, valid_index):
        # 4x4
        Q = np.eye(4 * self.size_N) * 10000
        H = self.jacobian(Tt, valid_index)
        # 3x3 @ 3X4 @ (4X3 @ 3x3 @ 3x4 + 4x4) = 3x4
        # print(self.sigma.shape)
        # print(H.shape)
        # print(Q.shape)
        Kt = self.sigma @ H.T @ (H @ self.sigma @ H.T + Q)
        return Kt

    def jacobian(self, Tt, valid_index):
        # 4x4 @ 4x4 @ 4xN = 4xN
        q = cam_T_imu @ Tt @ self.mu
        H = np.zeros((4 * self.size_N, 3 * size_M))
        for i in range(self.size_N):
            #print(self.size_N)
            #print(q[:, i])
            H_feature = self.M @ projection_d(q[:, i]) @ self.D
            index = valid_index[i]
            H[4*i:4*i+4, 3*index:3*index+3] = H_feature
        # 4N x 3M
        return H

    def predict_mu(self, feature, Tt, valid_index):
        # 4xN
        zt = feature
        zt = zt.reshape((-1, 1))
        z_temp = self.z
        z_temp = z_temp.reshape((-1, 1))
        mu_temp = self.mu
        mu_temp = mu_temp.reshape((-1, 1))

        # print(zt.shape)
        # print(z_temp.shape)
        # print(mu_temp.shape)
        # print(zt)
        # print(z_temp)

        # 3Mx4N
        Kt = self.kalman_gain(Tt, valid_index)
        #print(Kt.shape)
        D_temp = np.eye(3 * self.size_M)
        D_temp = np.vstack((D_temp, np.zeros((self.size_M, 3 * self.size_M))))
        #print(D_temp.shape)
        # 4xN + 4Mx3M @ 3Mx4N @ (4xN - 4xN) = 4xN
        self.mu = (mu_temp + (D_temp @ Kt @ (zt - z_temp))).reshape((4, -1))
        # self.q = cam_T_imu @ self.mu

    def predict_sigma(self, Tt, valid_index):
        # 4x3
        H = self.jacobian(Tt, valid_index)
        I = np.eye(3 * self.size_M)
        Kt = self.kalman_gain(Tt, valid_index)
        self.sigma = (I - Kt @ H) @ self.sigma

    def update_param(self, Tt, N, valid_index):
        self.z = self.z_hat(Tt, valid_index)
        self.size_N = N

def projection_d(q):
    # projection function and its derivative:
    q1 = q[0]
    q2 = q[1]
    q3 = q[2]
    q4 = q[3]
    derivative_matrix = np.array([[1, 0, -q1 / q3, 0],
                         [0, 1, -q2 / q3, 0],
                         [0, 0, 0, 0],
                         [0, 0, -q4 / q3, 1]])
    proj_d = (1 / q3) * derivative_matrix
    return proj_d

def valid_features(feature):
    feature = feature.T
    indValid = np.where(feature[:, 0] > 0)
    return indValid

def projection(q):
    return (1/q[2, :]) * q

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
    pose = np.eye(4)
    pose_before_inverse = np.eye(4)

    # (a) IMU Localization via EKF Prediction
    prediction = localizer()
    for i in range(len(t[0]) - 1):
        print('predict time epoch %d' %i)
        t_interval = t[0][i + 1] - t[0][i]
        prediction.predict_mu(t_interval, u[i])
        prediction.predict_sigma(t_interval, u[i])

        pose_before_inverse = np.dstack((pose_before_inverse, prediction.pose_mu))
        new_pose = pose_inverse(prediction.pose_mu)
        pose = np.dstack((pose, new_pose))
    pose_before_inverse = np.delete(pose_before_inverse, 0, 2)
    pose = np.delete(pose, 0, 2)
    #print(pose.shape)
    #visualize_trajectory_2d(pose)

    # (b) Landmark Mapping via EKF Update
    fx = K[0][0]
    fy = K[1][1]
    cx = K[0][2]
    cy = K[1][2]

    M = [[fx, 0, cx, 0],
         [0, fy, cy, 0],
         [fx, 0, cx, -fx * b],
         [0, fy, cy, 0]]

    D = [[1, 0, 0],
         [0, 1, 0],
         [0, 0, 1],
         [0, 0, 0]]

    # initialize:
    current_features = features[:, :, 0]
    #print(current_features.shape)
    valid_index = valid_features(current_features)[0]
    #print(valid_index)
    num_valid_feature_init = len(valid_index)
    landmarks = np.empty((4, 1))
    size_M = len(features[0])

    update = update(M=M, D=D, size_N=num_valid_feature_init, size_M=size_M, Tt=pose_before_inverse[:, :, 0], valid_index=valid_index)
    for j in range(len(t[0]) - 1):
        #input("press enter.....")
        print('update time epoch %d' %j)
        current_features = features[:, :, j]
        valid_index = valid_features(current_features)[0]
        z = current_features[:, valid_index].reshape((4, -1))
        if (len(valid_index) >= 1):
            update.update_param(pose_before_inverse[:, :, j], N=len(valid_index), valid_index=valid_index)
            update.predict_mu(feature=z, Tt=pose_before_inverse[:, :, j], valid_index=valid_index)
            update.predict_sigma(Tt=pose_before_inverse[:, :, j], valid_index=valid_index)
            landmarks = update.mu
        # landmarks = np.dstack((landmarks, landmark))
    print(landmarks)
    fig, ax = visualize_trajectory_2d(pose, landmarks)


    # (c) Visual-Inertial SLAM (Extra Credit)

    # You can use the function below to visualize the robot pose over time
    # visualize_trajectory_2d(world_T_imu,show_ori=True)

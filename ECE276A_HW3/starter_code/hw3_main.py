from utils import *
import scipy.linalg as sp


class localizer():
    def __init__(self):
        self.pose_mu = np.eye(4)
        self.pose_sigma = np.eye(6)

    def hat(self, v):
        result = np.array([[0, -v[2], v[1]],
                           [v[2], 0, -v[0]],
                           [-v[0], v[1], 0]])
        return result

    def twist_hat(self, u):
        result = np.zeros((4, 4))
        omega_hat = self.hat(u[3:])
        result[:3, :3] = omega_hat
        result[:3, 3] = u[:3]
        return result

    def curvy_hat(self, u):
        result = np.zeros((u.shape[0], u.shape[0]))

        # linear velocity
        v_hat = self.hat(u[:3])

        # angular_velocity
        omega_hat = self.hat(u[3:])
        result[:3, :3] = omega_hat
        result[:3, 3:] = v_hat
        result[3:, 3:] = omega_hat
        return result

    def predict_mu(self, dt, u):
        tran = sp.expm(-dt * self.twist_hat(u))
        self.pose_mu = tran @ self.pose_mu

    def predict_sigma(self, dt, u):
        tran = sp.expm(-dt * self.curvy_hat(u))
        self.pose_sigma = tran @ self.pose_sigma @ tran.T + dt**2 * 1000 * np.eye(6)


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
        # print(size_N)
        # print(size_M)
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
        Q = np.eye(4 * self.size_N) * 1000000
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
        H = np.zeros((4 * self.size_N, 3 * self.size_M))
        for i in range(self.size_N):
            # print(self.size_N)
            # print(q[:, i])
            H_feature = self.M @ projection_d(q[:, i]) @ self.D
            index = valid_index[i]
            H[4 * i:4 * i + 4, 3 * index:3 * index + 3] = H_feature
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

        # 3Mx4N
        Kt = self.kalman_gain(Tt, valid_index)
        # print(Kt.shape)
        D_temp = np.eye(3 * self.size_M)
        D_temp = np.vstack((D_temp, np.zeros((self.size_M, 3 * self.size_M))))
        # print(D_temp.shape)
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
    derivative_matrix = np.array([[1, 0, -q[0] / q[2], 0],
                                  [0, 1, -q[1] / q[2], 0],
                                  [0, 0, 0, 0],
                                  [0, 0, -q[3] / q[2], 1]])
    proj_d = (1 / q[2]) * derivative_matrix
    return proj_d


def valid_features(feature):
    feature = feature.T
    indValid = np.where(feature[:, 0] > 0)
    return indValid


def projection(q):
    return (1 / q[2, :]) * q


def W_T_O(zt, M, cam_T_imu, pose):
    zt.reshape(4, 1)
    x1 = (zt[0] - M[0][2]) / M[0][0]
    x2 = (zt[1] - M[1][2]) / M[0][0]
    x4 = (zt[0] - zt[2]) / -M[2][3]

    q3 = 1 / x4
    q1 = x1 * q3
    q2 = x2 * q3
    q4 = 1
    opt_v = np.array([q1, q2, q3, q4]).reshape((4, -1))
    # print(pose.shape)
    world = np.linalg.inv(cam_T_imu @ pose) @ opt_v
    return world.reshape((-1, 4))[0]


def circle_dot_operation(imu_m):
    matrix = imu_m.reshape((4, 1))
    s = matrix[0:3, :]
    param_lambda = matrix[3, :]
    # 3x3
    lambda_i = param_lambda * np.eye(3)
    s_hat = -1 * hat(s)
    # 3x6
    circle = np.concatenate((lambda_i, s_hat), axis=1)
    # 4x6
    circle = np.concatenate((circle, np.zeros((1, 6))), axis=0)
    return circle


def twist_hat(v, w):
    result = np.zeros((4, 4))
    omega_hat = hat(w)
    result[:3, :3] = omega_hat
    result[:3, 3] = v.T
    return result


def hat(w):
    result = np.array([[0, -w[2], w[1]],
                       [w[2], 0, -w[0]],
                       [-w[1], w[0], 0]])
    return result


def cur_hat(v, w):
    result = np.zeros((u.shape[0], u.shape[0]))

    # linear velocity
    v_hat = hat(v)

    # angular_velocity
    omega_hat = hat(w)
    result[:3, :3] = omega_hat
    result[:3, 3:] = v_hat
    result[3:, 3:] = omega_hat
    return result


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
    filename = "./data/0042.npz"
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
        print('predict time epoch %d' % i)
        t_interval = t[0][i + 1] - t[0][i]
        prediction.predict_mu(t_interval, u[i])
        prediction.predict_sigma(t_interval, u[i])

        pose_before_inverse = np.dstack((pose_before_inverse, prediction.pose_mu))
        new_pose = pose_inverse(prediction.pose_mu)
        pose = np.dstack((pose, new_pose))
    pose_before_inverse = np.delete(pose_before_inverse, 0, 2)
    pose_all = np.delete(pose, 0, 2)
    # print(pose.shape)
    # visualize_trajectory_2d(pose)

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

    num_landmarks = len(features[0])
    # initialize landmark x,y,0,1 to 0,0,0,1
    landmarks = np.zeros((4, num_landmarks))
    landmarks[3, :] = 1

    # initialize sigma
    sigma = 0.001 * np.eye(3 * num_landmarks)

    # boolean flag
    flag = np.zeros((num_landmarks, 1), dtype=bool)

    for i in range(len(t[0]) - 1):
        pose = pose_before_inverse[:, :, i]
        # print(pose.shape)
        w = rotational_velocity[:, i]
        v = linear_velocity[:, i]
        # 4xM
        feature = features[:, :, i]
        indValid = valid_features(feature)
        N = len(indValid[0])
        if N >= 1:
            # 4xN
            z = feature[:, indValid].reshape((4, -1))
            for num in range(N):
                idx = indValid[0][num]
                # if never seen this feature then update and set it to seen
                if flag[idx] == False:
                    landmarks[:, idx] = W_T_O(z[:, num], M, cam_T_imu, pose)
                    flag[idx] = True
            # 4xN
            landmark_i = landmarks[:, indValid[0]].reshape((4, -1))
            # 4xN
            q = cam_T_imu @ pose @ landmark_i
            proj = projection(q)
            # 4xN
            z_hat = M @ proj
            # compute Jacobian
            # 4Nx3M
            H = np.zeros((4 * N, 3 * num_landmarks))
            for j in range(N):
                proj_der = projection_d(q[:, j])
                idx = indValid[0][j]
                # 4x3
                proj_der = M @ proj_der @ cam_T_imu @ pose @ D
                H[4 * j:4 * j + 4, 3 * idx:3 * idx + 3] = proj_der

            # perform the EKF update
            Kt = sigma @ H.T @ np.linalg.inv((H @ sigma @ H.T) + (1000000 * np.eye(4 * N)))

            # update homogeneous coordinates
            z_hat = z_hat.reshape((4 * N, 1))
            z = z.reshape((4 * N, 1))
            landmarks = landmarks.reshape((4 * num_landmarks, 1))
            # 4Mx3M
            D_temp = np.concatenate((np.eye(3 * num_landmarks), np.zeros((num_landmarks, 3 * num_landmarks))),
                                    axis=0)
            # 4xM
            landmarks = (landmarks + (D_temp @ Kt @ (z - z_hat))).reshape((4, num_landmarks))

            sigma = (np.eye(3 * num_landmarks) - (Kt @ H)) @ sigma
        else:
            print('No landmark')

    # fig, ax = visualize_trajectory_2d(pose_all, landmarks)

    # (c) Visual-Inertial SLAM (Extra Credit)
    local_sigma = 0.001
    pose = np.eye(4)
    sigma_next = local_sigma * np.eye(6)
    pose_all_update = np.zeros((4, 4, len(t[0])))

    landmarks_update = np.zeros((4, num_landmarks))
    landmarks_update[3, :] = 1

    # initialize sigma
    map_sigma = 0.001
    sigma = map_sigma * np.eye(3 * num_landmarks)
    # initialize a seen matrix
    flag_m = np.zeros((num_landmarks, 1), dtype=bool)

    for i in range(len(t[0]) - 1):
        pose = pose_before_inverse[:, :, i]
        # print(pose.shape)
        w = rotational_velocity[:, i]
        v = linear_velocity[:, i]
        # 4xM
        feature = features[:, :, i]
        indValid = valid_features(feature)
        N = len(indValid[0])
        if N >= 1:
            # 4xN
            z = feature[:, indValid].reshape((4, -1))
            for num in range(N):
                idx = indValid[0][num]
                # if never seen this feature then update and set it to seen
                if flag[idx] == False:
                    landmarks[:, idx] = W_T_O(z[:, num], M, cam_T_imu, pose)
                    flag[idx] = True
            # 4xN
            landmark_i = landmarks[:, indValid[0]].reshape((4, -1))
            # 4xN
            q = cam_T_imu @ pose @ landmark_i
            proj = projection(q)
            # 4xN
            z_hat = M @ proj
            # compute Jacobian
            # 4Nx3M
            H = np.zeros((4 * N, 3 * num_landmarks))

            # build Jacobian
            for j in range(N):
                proj_der = projection_d(q[:, j])
                idx = indValid[0][j]
                # 4x3
                proj_der = M @ proj_der @ cam_T_imu @ pose @ D
                H[4 * j:4 * j + 4, 3 * idx:3 * idx + 3] = proj_der

            # perform the EKF update
            Kt = sigma @ H.T @ np.linalg.inv((H @ sigma @ H.T) + (1000000 * np.eye(4 * N)))

            # update homogeneous coordinates
            z_hat = z_hat.reshape((4 * N, 1))
            z = z.reshape((4 * N, 1))
            landmarks = landmarks.reshape((4 * num_landmarks, 1))
            # 4Mx3M
            D_temp = np.concatenate((np.eye(3 * num_landmarks), np.zeros((num_landmarks, 3 * num_landmarks))),
                                    axis=0)
            # 4xM
            landmarks = (landmarks + (D_temp @ Kt @ (z - z_hat))).reshape((4, num_landmarks))

            sigma = (np.eye(3 * num_landmarks) - (Kt @ H)) @ sigma
        else:
            print('No landmark')

        # get pose from imu at time i.
        pose = pose_before_inverse[:, :, i]

        # update the IMU pose
        feature_next = features[:, :, i + 1]
        indValid_next = valid_features(feature_next)
        N_next = len(indValid_next[0])

        if N_next >= 1:
            z_plus = feature_next[:, indValid_next].reshape((4, -1))
            for num in range(N_next):
                idx = indValid_next[0][num]
                if flag_m[idx] == False:
                    landmarks_update[:, idx] = W_T_O(z_plus[:, num], M, cam_T_imu, pose)
                    flag_m[idx] = True
            # 4xN
            landmark_next = landmarks_update[:, indValid_next[0]].reshape((4, -1))
            q_next = cam_T_imu @ pose @ landmark_next
            proj = projection(q_next)
            z_hat_plus = M @ proj

            H_next = np.zeros((4 * N_next, 6))

            # build Jacobian
            for j in range(N_next):
                proj_der = projection_d(q_next[:, j])
                # 4x4 @ 4x1 = 4x1
                imu_pose = pose @ landmark_next[:, j]
                # 4x6 compute Jacobian evaluated at mu next
                proj_der = M @ proj_der @ cam_T_imu @ circle_dot_operation(imu_pose)
                # 4Nx6
                H_next[4 * j:4 * j + 4, :] = proj_der
            # perform the EKF update
            local_v = 100000
            # 6x4N
            Kt_next = sigma_next @ H_next.T @ np.linalg.inv(
                (H_next @ sigma_next @ H_next.T) + local_v * np.eye(4 * N_next))
            # pose update
            temp = Kt_next @ (z_plus.reshape((4 * N_next, 1)) - z_hat_plus.reshape((4 * N_next, 1)))
            temp_hat = twist_hat(temp[0:3, :], temp[3:6, :])
            pose = sp.expm(temp_hat) @ pose
            sigma_next = (np.eye(6) - (Kt_next @ H_next)) @ sigma_next
        pose_all_update[:, :, i + 1] = np.linalg.inv(pose)

    fig, ax = visualize_trajectory_2d(pose_all_update, landmarks, pose_all, landmarks_update)

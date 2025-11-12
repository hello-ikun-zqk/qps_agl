import numpy as np
import matplotlib.pyplot as plt



def generate_multi_robot_target_tracking(num_robots= 9,num_malicious= 6,T = 10,dims=4):
    cls_nums=1
    A = np.eye(dims)  # Dynamics matrix
    Q = generate_random_covariance_matrices(T)  # Process noise covariance
    R = generate_random_covariance_matrices(T,num_robots,2)  # Observation noise covariance
    bar_x0 = np.zeros(dims)  # Initial state (position and velocity)
    bar_P0 = np.eye(dims) * 0.1
    x0=np.random.multivariate_normal(mean=bar_x0, cov=bar_P0)
    x0=np.expand_dims(x0,1)
    C=np.random.uniform(1,num_robots-num_malicious,(T,num_robots,2,dims))
    # Main simulation
    target_trajectory = generate_target_trajectory(A, x0, T,Q,dims)
    observations = generate_observations(target_trajectory,C,T,num_robots,R)

    bar_x0=np.expand_dims(bar_x0,1)

    
    return A,Q,R,C,bar_x0,bar_P0,x0,target_trajectory,observations,dims,cls_nums


def generate_random_covariance_matrices(T=None, agent_nums=None,dims=4):
    if T and agent_nums:
        cov_matrices = np.zeros((T, agent_nums, dims, dims))  # 初始化形状为 (T, agent_nums, 4, 4) 的数组
        
        for t in range(T):
            for i in range(agent_nums):
                A = np.random.rand(dims, dims)  # 生成一个 4x4 的随机矩阵
                cov_matrices[t, i] = A.T @ A  # 计算协方差矩阵
    elif T:
        cov_matrices = np.zeros((T, dims, dims))  # 初始化形状为 (T, agent_nums, 4, 4) 的数组
        
        for t in range(T):
            A = np.random.rand(dims, dims)  # 生成一个 4x4 的随机矩阵
            cov_matrices[t] = A.T @ A  # 计算协方差矩阵
    elif agent_nums:
        cov_matrices = np.zeros((agent_nums, dims, dims))  # 初始化形状为 (T, agent_nums, 4, 4) 的数组

        for i in range(agent_nums):
            A = np.random.rand(dims, dims)  # 生成一个 4x4 的随机矩阵
            cov_matrices[i] = A.T @ A  # 计算协方差矩阵
    else:
        A = np.random.rand(dims, dims)  # 生成一个 4x4 的随机矩阵
        cov_matrices = A.T @ A  # 计算协方差矩阵
    return cov_matrices


def generate_target_trajectory(A,x0, T,Q,dims):
    """Generate the target trajectory."""
    trajectory = np.zeros((T, dims, 1))
    trajectory[0] = x0
    for t in range(1, T):
        noise = np.random.multivariate_normal(mean=np.zeros(4), cov=Q[t])
        noise=np.expand_dims(noise,1)
        trajectory[t] = A @ trajectory[t-1] + noise
    return trajectory

def generate_observations(target_trajectory,C,T,num_robots,R):
    """Generate noisy observations from robots."""
    observations = np.zeros((T,num_robots,2,1))
    for t in range(T):
        obs=np.zeros((num_robots,2,1))
        for i in range(num_robots):
            observation_noise = np.random.multivariate_normal(mean=np.zeros(2), cov=R[t,i])
            observation_noise=np.expand_dims(observation_noise,1)
            obs[i]=C[t,i]@target_trajectory[t] + observation_noise  # 2D position
        observations[t]=obs
    return observations


if __name__=="__main__":
    generate_multi_robot_target_tracking()

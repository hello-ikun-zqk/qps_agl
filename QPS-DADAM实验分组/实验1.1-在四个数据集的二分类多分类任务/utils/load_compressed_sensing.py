import numpy as np
from scipy.linalg import orth

def generate_sin_wave(dims,sparsity_level,signal_strength):
    t = np.linspace(0, 1, dims, endpoint=False)  # 生成时间向量，1秒长，512个点
    t=np.expand_dims(t,axis=1)
    f = 5  # 正弦波的频率（Hz）
    # 生成正弦信号
    sin_wave = signal_strength*np.sin(2 * np.pi * f * t)

    zero_indices = np.random.choice(dims,dims-sparsity_level, replace=False)
    sin_wave[zero_indices] = 0  # 随机生成非零值
    return sin_wave

def generate_random_wave(dims,sparsity_level,signal_strength):
    x_true=np.zeros((dims,1))
    nonzero_indices = np.random.choice(dims,dims-sparsity_level, replace=False)
    x_true[nonzero_indices] = (np.random.normal(scale=signal_strength,size=(sparsity_level,1)))  # 随机生成非零值
    return x_true

def generate_compressed_sensing_data(n, dims, m, p, sparsity_level,sense_strength=10,signal_strength=10, noise_variance=0.01):
    """
    生成压缩感知问题的数据。

    参数:
    n -- 节点数量
    dims -- 信号维度
    m -- 每个节点的测量次数
    p -- 每个测量的维度
    sparsity_level -- 稀疏信号的非零元素数量
    noise_variance -- 噪声的方差

    返回:
    A -- 感测矩阵列表，每个元素对应一个节点的测量
    y -- 测量值列表
    x_true -- 原始稀疏信号
    """
    # 生成稀疏信号，只有sparsity_level个非零元素
    x_true=generate_sin_wave(dims,sparsity_level,signal_strength)

    # 生成感测矩阵
    A=np.zeros((n,m,p, dims))
    # 测量值
    y = np.zeros((n,m,p,1))
    
    for i in range(n):
        for j in range(m):
            # 归一化测量矩阵的每一列
            tmp=(np.random.normal(scale=sense_strength,size=(p, dims)))*sense_strength
            A[i,j]=orth(tmp.T).T
            # A[i,j]=tmp / np.linalg.norm(tmp, axis=0)
            y[i,j]=A[i,j] @ x_true + (np.random.normal(scale=noise_variance,size=(p,1)))

    return A,y,x_true

if __name__=="__main__":
    # 使用函数生成数据
    n = 20
    dims = 1000
    m = 10
    p = 45
    sparsity_level = 180
    noise_variance = 1.0
    signal_strength=1.0
    A, y, x_true = generate_compressed_sensing_data(n, dims, m, p, sparsity_level, noise_variance)

    # 打印一些信息来验证数据
    print(A[0,0])

    # a=generate_sin_wave(dims,sparsity_level,signal_strength)
    # print(a)
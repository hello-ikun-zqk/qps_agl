import numpy as np
from scipy.optimize import linprog

def project_onto_box(x, a, b):
    """
    将 NumPy 数组 x 投影到区间 [a, b] 上。
    
    参数:
    x (np.ndarray): 待投影的数组
    a (float or np.ndarray): 区间下界
    b (float or np.ndarray): 区间上界
    
    返回:
    np.ndarray: 投影后的数组
    """
    return np.clip(x, a, b)

def project_onto_l2_ball(x, r):
    """
    将 NumPy 数组 x 投影到以原点为中心，半径为 r 的二范数球上。
    
    参数:
    x (np.ndarray): 待投影的数组
    r (float): 二范数球的半径
    
    返回:
    np.ndarray: 投影后的数组
    """
    norm_x = np.linalg.norm(x, ord=2)  # 计算 x 的二范数
    if norm_x <= r:
        return x  # 如果 x 已经在二范数球内，直接返回 x
    else:
        return (r / norm_x) * x  # 否则将 x 缩放到球面上

def project_onto_sphere(x, r):
    """
    将 NumPy 数组 x 投影到以原点为中心、半径为 r 的球体上。
    
    参数:
    x (np.ndarray): 待投影的数组
    r (float): 球体的半径
    
    返回:
    np.ndarray: 投影后的数组
    """
    norm_x = np.linalg.norm(x)
    if norm_x > r:
        return (r / norm_x) * x
    return x

def project_onto_simplex(x):
    """
    将 NumPy 数组 x 投影到概率单纯形上。
    
    参数:
    x (np.ndarray): 待投影的数组
    
    返回:
    np.ndarray: 投影后的数组
    """
    if np.sum(x) == 1 and np.all(x >= 0):
        return x
    sorted_x = np.sort(x)[::-1]
    tmp_sum = 0
    for i in range(len(x)):
        tmp_sum += sorted_x[i]
        t = (tmp_sum - 1) / (i + 1)
        if t >= sorted_x[i]:
            break
    return np.maximum(x - t, 0)


def project_onto_l1_ball(x, r):
    """
    将 NumPy 数组 x 投影到 L1 球上。
    
    参数:
    x (np.ndarray): 待投影的数组
    r (float): L1 球的半径
    
    返回:
    np.ndarray: 投影后的数组
    """
    u = np.abs(x)
    if np.sum(u) <= r:
        return x
    w = np.sort(u)[::-1]
    sv = np.cumsum(w)
    rho = np.where(w > (sv - r) / np.arange(1, len(x) + 1))[0][-1]
    theta = (sv[rho] - r) / (rho + 1)
    return np.sign(x) * np.maximum(u - theta, 0)

def project_onto_polytope(x, A, b):
    """
    将 NumPy 数组 x 投影到一个简单多面体上。
    
    参数:
    x (np.ndarray): 待投影的数组
    A (np.ndarray): 多面体的线性不等式系数矩阵
    b (np.ndarray): 多面体的线性不等式右侧常数
    
    返回:
    np.ndarray: 投影后的数组
    """
    res = linprog(c=x, A_ub=A, b_ub=b, bounds=(None, None))
    if res.success:
        return res.x
    else:
        raise ValueError("线性规划求解失败")
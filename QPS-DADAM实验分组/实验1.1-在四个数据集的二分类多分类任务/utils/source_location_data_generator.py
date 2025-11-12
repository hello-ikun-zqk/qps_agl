import numpy as np
import math
import random


def get_source_location_generator(
    agents_num,
    dims,
    t,
    m,
    r,
    gamma=1,
    A=100,
    fix=False,
    noise_variance=0.00001,
    gen_variance=100,
):
    noise_stantard = math.sqrt(noise_variance)  # 噪声的标准差
    gen_stantard = math.sqrt(gen_variance)  # source的标准差
    agents_pos_ls = np.zeros((agents_num, dims,1))  # agent 位置
    target_ls = np.zeros((t, dims, 1))  # 目标位置
    signal_strengeth_ls = np.zeros((agents_num, m, t,1))  # 信号强度

    for i in range(agents_num):
        agents_pos_ls[i] = gen_stantard * np.random.randn(dims,1)

    center = gen_stantard * np.random.randn(dims,1)

    for i in range(t):
        target = center  # (dims,1)
        if dims == 1 and not fix:
            target = center + r * math.sin(i / 200)
        elif dims == 2 and not fix:
            target = center + r * np.array([[math.cos(i / 200)], [math.sin(i / 200)]])
        elif dims == 3 and not fix:
            target = center + r * np.array(
                [
                    [math.sin(i / 300) * math.cos(i / 200)],
                    [math.sin(i / 300) * math.sin(i / 200)],
                    [math.cos(i / 300)],
                ]
            ) 
        target_ls[i] = target

        # 计算agents的信号强度
        for j in range(agents_num):
            for k in range(m):
                norm_gamma=np.linalg.norm(target - agents_pos_ls[j], gamma)
                w = (
                    A / (norm_gamma)
                    + noise_stantard *np.random.randn(1)
                )
                signal_strengeth_ls[j, k, i] = w

    return signal_strengeth_ls, target_ls, agents_pos_ls


if __name__ == "__main__":
    (x, y, agents_pos_ls) = get_source_location_generator(
        agents_num=30,
        dims=2,
        t=100,
        m=5,
        r=1,
        gamma=2,
        A=100,
        fix=False,
        noise_variance=0.00001,
        gen_variance=100,
    )
    print("")

import numpy as np

# def probabilistic_quantizer(v, kt,use_quantize=True):
#     # 如果不使用量化，則直接輸出
#     if not use_quantize or kt<=0:
#         return v
    
#     def quantization(vi):
#         floori = np.floor(vi * kt) / kt
#         ceili = np.ceil(vi * kt) / kt
#         r = np.random.rand(1)
#         if r < (vi - floori) * kt:
#             return floori
#         else:
#             return ceili

#     if v.ndim==2:
#         q_v = np.array([[quantization(vij) for vij in vi] for vi in v])
#     elif v.ndim==3:
#         q_v = np.array([[[quantization(vijk) for vijk in vij] for vij in vi] for vi in v])
#     return q_v

def probabilistic_quantizer(v, kt, use_quantize=True):
    # 如果不使用量化，則直接輸出
    if not use_quantize or kt <= 0:
        return v
    
    # 计算 floor 和 ceil 的值
    floor_v = np.floor(v * kt) / kt
    ceil_v = np.ceil(v * kt) / kt
    
    # 计算随机数
    r = np.random.rand(*v.shape)
    
    # 计算 floor 和 ceil 之间的差值
    diff = v - floor_v
    
    # 计算阈值
    threshold = diff * kt
    
    # 使用矢量化操作来确定是向下取整还是向上取整
    mask = r < threshold
    
    # 应用 mask 来选择 floor 或 ceil
    q_v = np.where(mask, floor_v, ceil_v)
    
    return q_v


def probabilistic_quantizer_preset(v, kt,r=None,use_quantize=True):
    # 如果不使用量化，則直接輸出
    if not use_quantize or kt<=0:
        return v


    def quantization(vi,ri):
        floori = np.floor(vi * kt) / kt
        ceili = np.ceil(vi * kt) / kt
        if ri < (vi - floori) * kt:
            return floori
        else:
            return ceili

    if v.ndim==2:
        q_v = np.array([[quantization(vij,rij) for vij,rij in zip(vi,ri)] for vi,ri in zip(v,r)])

    return q_v



def stochastic_k_level_quantizer(v, kt, use_quantize=True):
    # 如果不使用量化，则直接返回原始值
    if not use_quantize or kt <= 0:
        return v

    max_v = np.max(v)
    min_v = np.min(v)

    # 注意：当 kt==1 时，这里的 (kt-1) 会导致除0错误，所以需要确保 kt>1
    # 如果 kt==1，则直接返回 v 即可（或返回固定量化值）
    if kt == 1:
        return v

    # 构造量化区间（共 kt+1 个分界点）
    Iv = min_v + np.arange(0, kt + 1) * (max_v - min_v) / (kt - 1)

    def quantization(vi):
        # 获取原始索引
        idx = np.digitize(vi, Iv, right=False)
        # 将 idx 限制在 [1, len(Iv)-1] 范围内，避免越界
        idx = np.clip(idx, 1, len(Iv) - 1)
        lower = Iv[idx - 1]
        upper = Iv[idx]
        # 如果上下界相同（可能因浮点数精度问题），直接返回其中一个
        if upper - lower == 0:
            return lower
        # 计算比例阈值
        threshold = (vi - lower) / (upper - lower)
        r = np.random.rand()
        return lower if r < threshold else upper

    # 对 v 中的每个元素都进行量化
    q_v = np.array([[quantization(vij) for vij in vi] for vi in v])
    return q_v




if __name__=="__main__":
    # v=np.random.rand(748,10)*100
    # print(v)
    # kt=9
    # res=probabilistic_quantizer(v,kt)
    # print(res)
    # print(res.shape)

    v=np.array([[1,2,3],[3,4,5]])
    kt=5
    res=stochastic_k_level_quantizer(v,kt)
    print(res)
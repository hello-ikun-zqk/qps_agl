import numpy as np

def uniform_quantizer_method(v, lower_bound, upper_bound, K, use_quantize=True):
    # 如果不使用量化，則直接輸出
    if not use_quantize:
        return v
    
    # 确保输入是numpy数组
    v = np.asarray(v)
    lower_bound = np.asarray(lower_bound)
    upper_bound = np.asarray(upper_bound)
    
    # 检查维度是否一致
    if v.shape != lower_bound.shape or v.shape != upper_bound.shape:
        raise ValueError("The dimensions of v, lower_bound, and upper_bound must be the same.")
    
    # 初始化输出数组
    q_v = np.zeros_like(v)
    
    # 遍历数组中的每个元素
    it = np.nditer(v, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        # 计算量化间隔
        quantization_interval = (upper_bound[idx] - lower_bound[idx]) / K
        # 生成量化级别
        quantization_bins = np.arange(lower_bound[idx], upper_bound[idx] + quantization_interval, quantization_interval)
        # 量化当前元素
        q_v[idx] = quantization_bins[np.digitize(v[idx], quantization_bins, right=True)-1]
        
        it.iternext()
    
    return q_v

class PeriodicDynamicQuantization:

    def __init__(self,tau,sigma_n,D_eta,rho2=0.5,K=29) -> None:
        self.tau=tau
        self.sigma_n=sigma_n
        self.D_eta=D_eta
        self.rho2=rho2
        self.K=K
        self.t=0
        self.bound=None

    def get_theta_t(self):
        return (1/(self.t+1))**self.rho2
    
    def reset(self):
        self.t=0
        self.bound=None

    def quantization(self,v, grad,z,alpha=None,kt=None,use_quantize=True):
        if not use_quantize:
                return v
        if alpha is None:
            alpha=(1+self.t)**0.5
        if self.t%self.tau==0 or self.bound is None:
            G_i=np.max(grad)
            gamma_i=G_i*self.tau/self.sigma_n*alpha+np.sqrt(2*self.D_eta/self.sigma_n)*self.tau*self.get_theta_t()
            self.bound=[z-gamma_i,z+gamma_i]
        if kt is None:
            kt=self.K
        self.t+=1
        return uniform_quantizer_method(v,self.bound[0],self.bound[1],kt)

    


if __name__=="__main__":
    v=np.random.rand(3,3)*100
    print(v)
    # a=uniform_quantizer_method(v, 10,20,30,use_quantize=True)
    grad=np.random.rand(3,3)*100
    quantizer=PeriodicDynamicQuantization(10,1,1)
    a=quantizer.quantization(v,grad)
    print(a)
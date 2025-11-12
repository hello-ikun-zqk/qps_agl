import numpy as np
import math
import matplotlib.pyplot as plt
import random


def get_targets_tracking_data(agents_num,signal_num,t,dynamic_C=False, w_ls=None,k_ls=None,v_ls=None,method=None):
    if method is None:
        if k_ls is None:
            k_ls=np.random.rand(3)*3
        if v_ls is None:
            v_ls=np.random.rand(3)*np.pi
        if w_ls is None:
            w_ls=np.ones(3)*(2*np.pi*10)
    elif method=="region_random":
        def get_random_ls(low, high, is_int=False):
            p = list()
            for i in range(signal_num):
                rd = np.random.rand()
                v = rd * (high - low) + low
                if is_int:
                    v = int(v)
                p.append(v)
            return p
        if k_ls is None:
            # k_ls=[0.2, 0.6, 1]
            k_ls=[0.5*(i+1) for i in range(signal_num)]
        if v_ls is None:
            v_ls=get_random_ls(0, math.pi)
        if w_ls is None:
            w_ls=get_random_ls(2*math.pi / 4000 , 2*math.pi / 2000)

    print("k_ls:",k_ls)
    print("v_ls:",v_ls)
    print("w_ls:",w_ls)
    dims=signal_num*2
    
    if dynamic_C:
        C_ls=np.random.randn(t,agents_num,1,dims)
    else:
        # C_const=np.ones((agents_num,1,dims))
        C_const=np.random.randn(agents_num,1,dims)
        C_ls=np.tile(C_const,(t,1,1,1))

    w_opt_ls=np.zeros((t,dims,1))
    strength_ls=np.zeros((t,agents_num,1))
    for j in range(t):
        w_opt_it=np.zeros((dims,1))
        for i in range(signal_num):
            w_opt_it[2*i]=k_ls[i]*math.sin(w_ls[i]*j+v_ls[i])
            w_opt_it[2*i+1]=w_ls[i]*k_ls[i]*math.cos(w_ls[i]*j+v_ls[i])
        w_opt_ls[j]=w_opt_it
        strength_ls[j]=np.dot(C_ls[j],w_opt_it).reshape(agents_num,1)

    return C_ls, w_opt_ls, strength_ls,dims

def grad_fn(x,y,C):
    if len(y.shape)==1:
        y=np.expand_dims(y,axis=1)
    # 计算C * x - y
    diff = np.dot(C, x) - y
    # 计算梯度
    gradient = np.dot(diff, C).T
    return gradient

def loss_fn( x, y, C):
    num=y.shape[0]
    predicted_value=np.zeros(y.shape)
    x=np.expand_dims(x,axis=0)
    x=np.tile(x,(num,1,1))
    # 计算预测值
    for i in range(num):
        predicted_value[i] = np.dot(C[i], x[i])
    # 计算平方差并除以2
    squared_error = 0.5 * np.linalg.norm(predicted_value-y,ord=2)
    return squared_error

        
if __name__=="__main__":

    
    
    # (C_ls, x, y,dims)=get_targets_tracking_data(agents_num=10,signal_num=3, t=10, w_ls=[1,2,3], k_ls=[2,3,4],v_ls=[3,4,5])
    agents_num=5
    signal_num=3
    t=1000
    dynamic_C=False

    (C, x, y,dims)=get_targets_tracking_data(agents_num=agents_num,signal_num=signal_num, t=t,dynamic_C=dynamic_C, method="region_random")
    
    # print(C_ls)

    x_data = x[0,:]
    y_data = y[0,1,:]+1
    C_ls=C[0]
    w_init = np.random.randn(dims, 1)
    g=grad_fn(x_data,y_data,C_ls[1])
    print(g)
    print(x_data)

    for i in range(t):
        for ii in range(agents_num):
            x_data = x[i,:]
            y_data = y[i,ii,:]
            C_data=C[i,ii]

            g=grad_fn(x_data,y_data,C_data)
            print(g)

    # x_data = x[0]
    # y_data = y[0,1,:]
    
    # l=loss_fn(x_data,y_data,C_ls[1])
    # print(l)





import numpy as np
import math
import matplotlib.pyplot as plt
import random

def get_dynamic_sparse_signal_recovery_data(t,agents_num,dims,m,use_sparse=False):

    def get_random_matrix(low,high,shape):
        return (high-low)*np.random.rand(*shape)+low
    
    def create_random_sparse_matrix(rows, cols, num_non_zero_elements):
        sparse_matrix = np.zeros((rows, cols), dtype=float)  # 创建全零的矩阵
        
        for _ in range(num_non_zero_elements):
            row = random.randint(0, rows - 1)
            col = random.randint(0, cols - 1)
            value = random.uniform(0.1, 1.0)
            sparse_matrix[row, col] = value

        return sparse_matrix

    
    z_ls=np.zeros((t,agents_num,dims))
    P_ls=np.zeros((t,agents_num,dims,m))
    w_ls=np.zeros((t,m))

    noise=0.01

    for i in range(t):
        z_t=np.zeros((agents_num,dims))
        w_t=get_random_matrix(0,1,(m,1))
        P_t=np.zeros((agents_num,dims,m))
        for j in range(agents_num):
            if use_sparse:
                P_t[j]=create_random_sparse_matrix(dims, m, random.randint(1, max(int(dims*m/2 - 1),1)))
            else:
                P_t[j]=get_random_matrix(0,1,(dims,m))
            # P_t[j]=np.ones((dims,m))
            z_t[j]=(np.dot(P_t[j],w_t)+noise*get_random_matrix(-1,1,(dims,1))).squeeze(axis=1)

        z_ls[i]=z_t
        P_ls[i]=P_t
        w_ls[i]=w_t.squeeze(axis=1)
    return z_ls,w_ls,P_ls


def loss_fn(x,predict,P,rho=1):
    diff=x - np.dot(P,predict)
    loss=np.linalg.norm(diff, 2)+rho*np.linalg.norm(predict,1)

    return loss

def grad_fn(x,predict,P,rho=1):
    if len(predict.shape)>1:
        predict=predict.squeeze(axis=1)
    diff=x - np.dot(P,predict)
    g=2*np.dot(P.T,diff)+rho*np.sign(predict)

    return g


if __name__=="__main__":
    x,y,P=get_dynamic_sparse_signal_recovery_data(100,20,3,10,use_sparse=True)
    print("test")
    # print(loss_fn(x[0,0],y[0],P[0,0],rho=0))

    print(loss_fn(x[0],y[0],P[0],rho=0))

    print(grad_fn(x[0,0],y[0],P[0,0],rho=0))
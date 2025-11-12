import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import random 

random_point = None

def generate_point_on_hyperplane(w_opt,a,use_random_point=True):
    global random_point
    # Generate a random point pc in the (dims+1)-dimensional space
    dims = w_opt.shape[0] - 1
    if not use_random_point:
        if random_point is None:
            random_point = a*np.random.rand(dims + 1, 1) 
    else:
        random_point = a*np.random.rand(dims + 1, 1) 
    
    
    # Project pc onto the hyperplane defined by w_opt
    pc_proj = np.dot(w_opt.T, random_point) / np.dot(w_opt.T, w_opt) * w_opt
    point_on_hyperplane = random_point - pc_proj

    return point_on_hyperplane

def get_circle_data(agents_num, dims, t, m, r, a,cricle_T=100,use_sklearn_wopt=False):
    w_opt = np.random.rand(dims+1, 1) * a - a / 2
    w_opt_ls = np.zeros((t, dims+1, 1))
    for i in range(t):
        w_opt_ls[i] = w_opt
    
    x = np.zeros((agents_num, m, dims + 1, t))
    y = np.zeros((agents_num, m, 1, t))

    for i in range(agents_num):
        for j in range(m):
            tmp_rand_val=random.random()
            pc =generate_point_on_hyperplane(w_opt,a)+a*np.array([[math.cos(2*math.pi*tmp_rand_val)], [math.sin(2*math.pi*tmp_rand_val )],[0]])  # Project w_opt onto itself
            for k in range(t):
                add_pt = pc + r * np.array([[math.cos(k / cricle_T)], [math.sin(k / cricle_T)],[0]])
                add_pt[-1]=1
                x[i, j, :, k] = add_pt.squeeze()
                if np.dot(w_opt_ls[k].T, add_pt) >= 0:
                    y[i, j, :, k] = 1
                else:
                    y[i, j, :, k] = 0
    
    if use_sklearn_wopt:
        print("---- 开始生成w_opt ----")
        w_opt_ls=get_w_opt_ls_by_sklearn(x,y,w_opt_ls,agents_num,m,t)

    return x, y, w_opt_ls,pc


def is_point_on_hyperplane(point, w, b, tol=1e-10):
    # Check if w and point have the same dimensionality
    if len(w) != len(point):
        raise ValueError("The dimensions of the point and the hyperplane normal vector must be the same.")

    # Calculate the value of w^T * point + b
    value = np.dot(w.T, point) + b

    # Check if the value is within the tolerance of zero
    return abs(value) < tol


def visualize_data_and_hyperplane(x, y, w_opt_ls, pc):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Plot data points
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            points = x[i, j]
            pos_index= y[i, j, 0, :] == 1
            neg_index=y[i, j, 0, :] == 0
            ax.scatter(points[0,pos_index], points[1,pos_index], points[2,pos_index], c="r", marker='o')
            ax.scatter(points[0,neg_index], points[1,neg_index], points[2,neg_index], c="g", marker='o')

    # Plot hyperplane
    xx, yy = np.meshgrid(np.linspace(-10, 10, 10), np.linspace(-10, 10, 10))
    for k in range(w_opt_ls.shape[0]):
        w_opt = w_opt_ls[k].squeeze()
        b = w_opt[-1]  # Extract constant term b
        zz = (-w_opt[0] * xx - w_opt[1] * yy - b) / w_opt[2]
        ax.plot_surface(xx, yy, zz, color='b', alpha=0.3)

    # Plot point pc on the hyperplane
    ax.scatter(pc[0], pc[1], pc[2], c='k', marker='X', s=500)

    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    ax.set_title('Data Points and Hyperplane')
    plt.show()

def get_w_opt_ls_by_sklearn(x,y,w_opt_ls,agents_num,m,t):

    x=np.reshape(x,(agents_num*m,-1,t))
    y=np.reshape(y,(agents_num*m,-1,t))

    # print(x.shape)
    # print(y.shape)

    for curr_t in range(t):
        # 创建并训练逻辑回归模型
        model = LogisticRegression()
        model.fit(x[:,:,curr_t], y[:,:,curr_t].ravel())

        # print(model.coef_)
        w_opt_ls[curr_t]=model.coef_.T

        # # 在测试集上进行预测
        # y_pred = model.predict(x[:,:,curr_t])

        # # 评估性能
        # accuracy = accuracy_score(y[:,:,curr_t], y_pred)
        # conf_matrix = confusion_matrix(y[:,:,curr_t], y_pred)
        # print("准确率:", accuracy)
        # print("混淆矩阵:\n", conf_matrix)

    return w_opt_ls

if __name__=="__main__":
    agents_num=50   # agents数量
    org_dims=2      # 原始维度
    dims=org_dims+1 # 实际维度
    t=1000   # 时间长度
    m=10    # 每个时刻，每个agent得到的数据个数
    r=1     # 圆的半径长度
    a=3     # 随机数的范围

    (x, y, w_opt_ls,pc)=get_circle_data(agents_num=agents_num,dims=org_dims,t=t,m=m,r=r,a=a,cricle_T=t/(2*math.pi))
    print(np.count_nonzero(y))
    print(w_opt_ls.shape)

    print(f"正样本数量为：{np.count_nonzero(y)}")
    print(f"负样本数量为：{y.size-np.count_nonzero(y)}")
    # visualize_data_and_hyperplane(x, y, w_opt_ls,pc)


    # x=np.reshape(x,(agents_num*m,-1,t))
    # y=np.reshape(y,(agents_num*m,-1,t))

    # print(x.shape)
    # print(y.shape)

    # for curr_t in range(t):
    #     # 创建并训练逻辑回归模型
    #     model = LogisticRegression()
    #     model.fit(x[:,:,curr_t], y[:,:,curr_t].ravel())

    #     # print(model.coef_)
    #     w_opt_ls[curr_t]=model.coef_.T

    #     # # 在测试集上进行预测
    #     # y_pred = model.predict(x[:,:,curr_t])

    #     # # 评估性能
    #     # accuracy = accuracy_score(y[:,:,curr_t], y_pred)
    #     # conf_matrix = confusion_matrix(y[:,:,curr_t], y_pred)
    #     # print("准确率:", accuracy)
    #     # print("混淆矩阵:\n", conf_matrix)

    # print(w_opt_ls)


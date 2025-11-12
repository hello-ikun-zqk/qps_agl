import matplotlib.pyplot as plt
import numpy as np
import math
import matplotlib.font_manager as fm

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    confusion_matrix,
    classification_report,
)

import itertools


# 设置字体优先级：中文用宋体，英文用 Times New Roman

plt.rcParams['font.sans-serif'] = ['Times New Roman']  # ✅ 宋体+Times New Roman
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示异常

# 指定中英文字体路径
chinese_font_path = 'C:/Windows/Fonts/simsun.ttc'  # 替换为你的宋体路径
english_font_path = 'C:/Windows/Fonts/times.ttf'   # 替换为 Times New Roman 路径

chinese_font = fm.FontProperties(fname=chinese_font_path)
english_font = fm.FontProperties(fname=english_font_path)


def my_plot_confusion_matrix(
    cm, classes, normalize=False, title=None, cmap=plt.cm.Blues,save_path=None,use_CN=False
):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normalization")
    print(cm)
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    if title is not None:
        plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )
    if use_CN:
        plt.ylabel("真实标签",fontsize=16,fontproperties=chinese_font)
        plt.xlabel("预测标签",fontsize=16,fontproperties=chinese_font)
        ext="png"
    else:
        plt.ylabel("True label",fontsize=16)
        plt.xlabel("Predicted label",fontsize=16)
        ext="pdf"
    if save_path is not None:
        
        plt.savefig(
        fname=save_path,
        format=ext,
        bbox_inches='tight',
        dpi=300,  # ✅ 分辨率倍增
        pad_inches=0.1,  # ✅ 边界留白
        facecolor='auto',  # ✅ 自动背景填充
        metadata={'Creator': ''},  # ✅ 清除元数据
        quality=95 if ext == 'jpg' else None  # ✅ 格式定制
    )


def plot_lr_all_info(y, predicted,classes=[0, 1],save_path=None,use_CN=False):
    print(classification_report(y, predicted, digits=8))
    cnf_matrix = confusion_matrix(y, predicted)
    my_plot_confusion_matrix(cnf_matrix, classes=classes, title=None,save_path=save_path,use_CN=use_CN)

def plot_dynamic_semilogy(x, y: dict):
    for (key,item) in y.items():
        plt.semilogy(x, np.array(item), label=key)
    plt.xlabel("T")  # 梯度下降的次数
    plt.ylabel("loss")  # 损失值
    plt.title("loss trend")  # 损失值随着W不断更新，不断变化的趋势
    plt.legend()  # 图形图例
    plt.show()


def plot_static_semilogy(x, y: dict):
    for (key,item) in y.items():
        plt.semilogy(x, np.array(item), label=key)
    plt.xlabel("Epoch")  # 梯度下降的次数
    plt.ylabel("loss")  # 损失值
    plt.title("loss trend")  # 损失值随着W不断更新，不断变化的趋势
    plt.legend()  # 图形图例
    plt.show()


def plot_source_localization_contour(x,y,agents_pos_ls,dims,allw,loss_fn:callable,trasform_func:callable,source_t=0,start=-25,end=25,nums=100,track_agents_num=10,time_nums=20,base_gap=3):
    # 计算等高线
    x_grid=np.linspace(start,end,nums)
    y_grid=np.linspace(start,end,nums)

    x_grid, y_grid = np.meshgrid(x_grid, y_grid)
    loglikelihood = np.zeros((nums,nums))

    for a in range(nums):
        for b in range(nums):
            predict=np.array([[x_grid[a,b]],[y_grid[a,b]]])
            x_all_data_per_t = trasform_func(x,source_t)
            y_all_data_per_t = trasform_func(y,source_t)

            aver_cost = loss_fn(x_all_data_per_t, y_all_data_per_t, predict, agents_pos_ls)
            loglikelihood[a,b]=math.log(aver_cost)
    
    plt.figure(figsize=(10, 10))
    plt.contour(x_grid, y_grid,loglikelihood, 20)
    for i in range(track_agents_num):
        # 零时刻
        evolution=np.zeros((dims,time_nums))
        evolution[:,0]=allw[0][i].squeeze()
        for j in range(time_nums-1):
            index=int((j+1)*base_gap)
            evolution[:,j+1]=allw[index][i].squeeze()
        plt.plot(evolution[0],evolution[1],marker='o',linestyle='--')
    plt.scatter(agents_pos_ls[:,0],agents_pos_ls[:,1],c='black',marker='x',s=30)
    plt.scatter(y[0,0],y[0,1],c='r',marker='*',s=100)
    plt.show()


def plot_target_tracking(x, w, colors=None, markers=None, signal_num=3, save_fig_path=None):
    names = list()
    alexs = list()
    indices = list()
    
    if not colors:
        colors = ["#77AC30", "#EDB123", "#0072BD"]
    if not markers:
        markers = ["o", "v", "^", "<", ">"]
        
    for i in range(signal_num):
        names.append(r"$x_{%d1,t}^{*}$" % (i+1))
        alexs.append(r"$x_{%d1,t}$" % (i+1))
        indices.append(2*i)
    
    markers = markers[0:len(names)]
    marker_num = 15
    measure_data_len = x.shape[0]

    color_index = 0

    plt.figure(figsize=(10, 5))
    
    # Calculate the max and min values for dynamic y-limits
    y_min = min(np.min(x), np.min(w))
    y_max = max(np.max(x), np.max(w))

    for name, alex, indice, marker in zip(names, alexs, indices, markers):
        line = x[:, indice]
        line_track = w[:, indice]
        interval = 40
        
        # Plot actual target
        plt.plot(np.arange(len(line))[0:len(line):interval], line[0:len(line):interval], 
                 linewidth=2, markersize=4, color=colors[color_index], 
                 marker=marker, markevery=int(measure_data_len / interval / marker_num),
                 label=name)
        
        # Plot tracking line
        plt.plot(np.arange(len(line))[0:len(line):interval], line_track[0:len(line):interval], 
                 linewidth=2, markersize=4, color=colors[color_index], linestyle='--', 
                 marker=marker, markevery=int(measure_data_len / interval / marker_num),
                 label=alex)
        
        color_index += 1

    # Set axis labels, grid, and legend
    plt.legend(loc="upper right", fontsize=14)
    plt.grid(True)
    plt.xlabel(r'$t$', fontsize=22)
    plt.ylabel(r'$x_{a1,t}^{*}$ and $x_{a1,t}$', fontsize=22)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlim(0, len(x))

    # Dynamically adjust y-limits based on data
    plt.ylim(np.floor(y_min) - 0.5, np.ceil(y_max) + 0.5)

    # Save or show the figure
    if save_fig_path is not None:
        plt.savefig(save_fig_path, bbox_inches='tight')
    plt.show()

def plot_target_tracking_speed(x,w,signal_num=3,save_fig_path=None):
    names=list()
    alexs=list()
    indices =list()
    # colors = ['brown','blue','olive','peru','slategrey','burlywood','tan']
    colors= ["#00a8e1", "#fcd300", "#EB003B"]
    markers = ['s','o','d','^','v','<','>']
    for i in range(signal_num):
        names.append(r"$x_{%d2,t}^{*}$" % (i+1))
        alexs.append(r"$x_{%d2,t}$" % (2*(i+1)))
        indices.append(2*i+1)
    
    markers = markers[0:len(names)]
    color_index = 0

    plt.figure(figsize=(10, 5))
    for name, alex, indice in zip(names, alexs,  indices):
        line=x[:,indice]
        line_track = w[:,indice]
        interval = 40
        plt.plot(np.arange(len(line))[0:len(line):interval], line[0:len(line):interval], 
                    linewidth=1.5, markersize=4, color=colors[color_index], 
                    label=name)
        plt.plot(np.arange(len(line))[0:len(line):interval], line_track[0:len(line):interval], 
                    linewidth=1.0, markersize=4, color=colors[color_index], linestyle='--',
                    label=alex)
        color_index+=1

    plt.legend(loc="upper right",fontsize=13)
    plt.grid(True)
    plt.xlabel('Iteration', fontsize=20, fontname='Times New Roman')
    plt.ylabel(r'$x_{a2,t}^{*}$ and $x_{a1,t}$', fontsize=16, fontname='Times New Roman')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlim(0, len(x))
    plt.ylim(5e-3,-5e-3)

    plt.title(f'Trajectories of the targets', fontsize=16, fontname='Times New Roman')
    if save_fig_path is not None:
        plt.savefig(save_fig_path)
    plt.show()


if __name__ == "__main__":
    plt.rcParams['font.sans-serif'] = 'times new roman'
    
    def test_plot_target_tracking():
        x=np.load('./temp/tt_x.npy')
        meanw=np.load("./temp/tt_meanw.npy")

        plot_target_tracking(x,meanw[1:],signal_num=3)

    def test_plot_lr_all_info():
        y_all_data=np.load('./temp/dc_y_all_data.npy')
        predicted=np.load('./temp/dc_predicted.npy')

        plot_lr_all_info(y_all_data.squeeze(), predicted,save_path="./figs/qdadam_dc.pdf")
    
    test_plot_lr_all_info()
from scipy.io import loadmat
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from matplotlib.colors import Normalize
import numpy as np
import os
import re
import gzip

# 归一化函数，将图像从0-255范围映射到0-1范围
def normalize_image(image):
    norm = Normalize(vmin=None, vmax=None)
    return norm(image)

# 逆归一化函数，将图像从0-1范围映射回0-255范围
def denormalize_image(image):
    return (image * 255.0).astype(np.uint8)

def unpickle(file):
    import pickle

    with open(file, "rb") as fo:
        dict = pickle.load(fo, encoding="bytes")
    return dict


def load_lr_data(data_path):
    data = loadmat(data_path, mat_dtype=True)
    x_train = data["x_train"]
    y_train = data["y_train"]
    y_train[y_train == -1] = 0
    x_test = data["x_test"]
    y_test = data["y_test"]
    y_test[y_test == -1] = 0
    w_opt = data["w_star"]
    lambda_val = data["lambda"]

    (cls_nums, nums) = y_train.shape
    (dims, nums) = x_train.shape

    cls_nums = cls_nums + 1 if cls_nums == 1 else cls_nums

    return (x_train, y_train, x_test, y_test, w_opt, lambda_val, dims, cls_nums, nums)


def load_mnist_data(mnist_path):
    data = loadmat(mnist_path, mat_dtype=True)
    x_train = data["x_trn"]
    y_train = data["y_trn"]
    x_test = data["x_tst"]
    y_test = data["y_tst"]
    w_opt = data["w_star"]
    lambda_val = data["lambda"]

    (cls_nums, nums) = y_train.shape
    (dims, nums) = x_train.shape
    w_opt = np.reshape(w_opt, (cls_nums, dims))

    cls_nums = cls_nums + 1 if cls_nums == 1 else cls_nums

    return (x_train, y_train, x_test, y_test, w_opt.T, lambda_val, dims, cls_nums, nums)

def load_fashion_mnist_data(data_path,use_nomalization=True):
    def load_mnist(path, kind='train'):
        """Load MNIST data from `path`"""
        labels_path = os.path.join(path,
                                '%s-labels-idx1-ubyte.gz'
                                % kind)
        images_path = os.path.join(path,
                                '%s-images-idx3-ubyte.gz'
                                % kind)

        with gzip.open(labels_path, 'rb') as lbpath:
            labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                                offset=8)

        with gzip.open(images_path, 'rb') as imgpath:
            images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                                offset=16).reshape(len(labels), 784)

        return images, labels
    x_train, y_train = load_mnist(data_path, kind='train')
    x_test, y_test = load_mnist(data_path, kind='t10k')

    y_train = np.expand_dims(y_train, axis=1)
    y_test = np.expand_dims(y_test, axis=1)
    # 创建 OneHotEncoder 对象
    encoder = OneHotEncoder(categories="auto")
    # 对数据进行独热编码
    y_train = encoder.fit_transform(y_train).toarray().T
    y_test = encoder.fit_transform(y_test).toarray().T
    x_train=x_train.T
    x_test=x_test.T

    lambda_val =1
    (cls_nums, nums) = y_train.shape
    (dims, nums) = x_train.shape
    w_opt = np.zeros(((cls_nums, dims)))

    cls_nums = cls_nums + 1 if cls_nums == 1 else cls_nums

    if use_nomalization:
        x_train=normalize_image(x_train)
        x_test=normalize_image(x_test)


    return (x_train, y_train, x_test, y_test, w_opt.T, lambda_val, dims, cls_nums, nums)


def load_mushroom_data(mushroom_path, test_size=0.2):
    data = loadmat(mushroom_path, mat_dtype=True)
    x = data["X"]
    y = data["y"]
    y[y == -1] = 0
    # 划分训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=42
    )

    # 导入w_opt
    mushroom_w_path = re.sub("mushroom.mat", "mushroom_w_star.mat", mushroom_path)
    w_opt = loadmat(mushroom_w_path, mat_dtype=True)["w_star"]

    # 转置
    x_train = x_train.T
    y_train = y_train.T
    x_test = x_test.T
    y_test = y_test.T

    (cls_nums, nums) = y_train.shape
    (dims, nums) = x_train.shape

    cls_nums = cls_nums + 1 if cls_nums == 1 else cls_nums

    return (x_train, y_train, x_test, y_test, w_opt, dims, cls_nums, nums)


def load_covtype_data(covtype_path="~/scikit_learn_data", test_size=0.2):
    (
        x_train,
        y_train,
        x_test,
        y_test,
        w_opt,
        dims,
        cls_nums,
        nums,
    ) = load_covtype_data_for_sklearn(covtype_path=covtype_path, test_size=test_size)
    
    x_train = x_train.T
    x_test = x_test.T

    y_train = np.expand_dims(y_train, axis=1)
    y_test = np.expand_dims(y_test, axis=1)
    # 创建 OneHotEncoder 对象
    encoder = OneHotEncoder(categories="auto")
    # 对数据进行独热编码
    y_train = encoder.fit_transform(y_train).toarray().T
    y_test = encoder.fit_transform(y_test).toarray().T

    return (x_train, y_train, x_test, y_test, w_opt, dims, cls_nums, nums)

def load_breast_cancer_data(test_size=0.2, random_state=42,oneHot=False):
    data = load_breast_cancer()
    X = data.data.T  # 转置以符合你的格式
    y = data.target.reshape(-1, 1)  # 变为列向量
    
    # 划分训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(
        X.T, y, test_size=test_size, random_state=random_state
    )
    
    x_train = x_train.T
    x_test = x_test.T
    
    # One-hot 编码
    if oneHot:
        encoder = OneHotEncoder(categories='auto', sparse=False)
        y_train = encoder.fit_transform(y_train)
        y_test = encoder.transform(y_test)
    
    y_train=y_train.T
    y_test=y_test.T
    # 其他返回值
    w_opt = None  # 这里没有已知的最优权重
    dims = x_train.shape[0]  # 特征维度
    cls_nums = len(np.unique(data.target))  # 类别数
    nums = x_train.shape[1]  # 数据样本总数
    
    return x_train, y_train, x_test, y_test, w_opt, dims, cls_nums, nums


def load_cifar10_data(cifar10_path):
    (
        x_train,
        y_train,
        x_test,
        y_test,
        w_opt,
        dims,
        cls_nums,
        nums,
    ) = load_cifar10_data_for_sklearn(cifar10_path)
    x_train = x_train.T
    x_test = x_test.T

    y_train = np.expand_dims(y_train, axis=1)
    y_test = np.expand_dims(y_test, axis=1)
    # 创建 OneHotEncoder 对象
    encoder = OneHotEncoder(categories="auto")
    # 对数据进行独热编码
    y_train = encoder.fit_transform(y_train).toarray().T
    y_test = encoder.fit_transform(y_test).toarray().T
    
    return (x_train, y_train, x_test, y_test, w_opt, dims, cls_nums, nums)


def load_lr_data_for_sklearn(data_path):
    (
        x_train,
        y_train,
        x_test,
        y_test,
        w_opt,
        lambda_val,
        dims,
        cls_nums,
        nums,
    ) = load_lr_data(data_path)
    x_train = x_train.T
    y_train = y_train.squeeze()
    x_test = x_test.T
    y_test = y_test.squeeze()

    return (x_train, y_train, x_test, y_test, w_opt, lambda_val, dims, cls_nums, nums)


def load_mnist_data_for_sklearn(mnist_path):
    (
        x_train,
        y_train,
        x_test,
        y_test,
        w_opt,
        lambda_val,
        dims,
        cls_nums,
        nums,
    ) = load_mnist_data(mnist_path)
    x_train = x_train.T
    y_train = np.argmax(y_train.T, axis=1)
    x_test = x_test.T
    y_test = np.argmax(y_test.T, axis=1)

    return (x_train, y_train, x_test, y_test, w_opt, lambda_val, dims, cls_nums, nums)


def load_mushroom_data_for_sklearn(mushroom_path, test_size=0.2):
    x_train, y_train, x_test, y_test, w_opt, dims, cls_nums, nums = load_mushroom_data(
        mushroom_path, test_size=test_size
    )
    x_train = x_train.T
    x_test = x_test.T
    y_train = y_train.squeeze()
    y_test = y_test.squeeze()

    return (x_train, y_train, x_test, y_test, w_opt, dims, cls_nums, nums)


def load_covtype_data_for_sklearn(covtype_path="~/scikit_learn_data", test_size=0.2):
    from sklearn.datasets import fetch_covtype

    x, y = fetch_covtype(data_home=covtype_path, return_X_y=True)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=42
    )
    (nums, dims) = x_train.shape
    cls_nums = 7

    w_opt = None

    return (x_train, y_train, x_test, y_test, w_opt, dims, cls_nums, nums)


def load_cifar10_data_for_sklearn(cifar10_path):
    x_ls = []
    y_ls = []
    # 读取训练样本
    for i in range(1, 6):
        batch_path = os.path.join(cifar10_path, f"data_batch_{i}")
        batch_dict = unpickle(batch_path)
        train_batch = batch_dict[b"data"].astype("float")
        train_labels = np.array(batch_dict[b"labels"])
        x_ls.append(train_batch)
        y_ls.append(train_labels)

    x_train = np.concatenate(x_ls)  # (50000, 3072) 50000*(32*32*3)
    y_train = np.concatenate(y_ls)  # (50000,)

    # 读取测试样本
    test_dict = unpickle(os.path.join(cifar10_path, "test_batch"))
    x_test = test_dict[b"data"].astype("float")
    y_test = np.array(test_dict[b"labels"])

    nums, dims = x_train.shape
    cls_nums = 10

    w_opt = None

    return (x_train, y_train, x_test, y_test, w_opt, dims, cls_nums, nums)



if __name__ == "__main__":
    # mushroom_path = r".\data\mushroom\mushroom.mat"
    # load_mushroom_data(mushroom_path, test_size=0.2)
    # fashion_mnist_path = r".\data\fashion_mnist"
    # load_fashion_mnist_data(fashion_mnist_path)
    def test_load_mnist_data():
        mnist_path = r".\data\mnist\6000_data_0.001.mat"
        load_mnist_data(mnist_path)
        (
        x_train,
        y_train,
        x_test,
        y_test,
        w_opt,
        lambda_val,
        dims,
        cls_nums,
        nums,
        ) = load_mnist_data(mnist_path)
        print(np.min(x_train),np.max(x_train))
    test_load_mnist_data()

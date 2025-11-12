from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from matplotlib.colors import Normalize
import numpy as np
import os
import cv2

# 归一化函数，将图像从0-255范围映射到0-1范围
def normalize_image(image):
    norm = Normalize(vmin=None, vmax=None)
    return norm(image)

# 逆归一化函数，将图像从0-1范围映射回0-255范围
def denormalize_image(image):
    return (image * 255.0).astype(np.uint8)

def load_a_picture(data_path,pic_name='ikun.jpg',use_nomalization=True,to_grayscale=True):
    img = cv2.imread(os.path.join(data_path,pic_name))
    # cv2 读取的图像是 BGR 颜色空间，需要转换为 RGB 颜色空间
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 将图像转换为灰度图
    if to_grayscale:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        image_dim_x,image_dim_y=img.shape
    else:
        image_dim_x,image_dim_y,image_channel=img.shape
        
    if use_nomalization:
        img=normalize_image(img)

    return (img.astype(np.float32),image_dim_x,image_dim_y,1)

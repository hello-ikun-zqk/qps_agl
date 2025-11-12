import numpy as np
import cv2
import math
import random 

def convolution2d(image, kernel):
    """
    实现二维卷积操作
    Args:
        image: 输入图像，二维numpy数组（灰度图）或三维numpy数组（彩色图），
               对于彩色图像，最后一个维度表示通道数
        kernel: 卷积核，二维numpy数组
    Returns:
        卷积结果，与输入图像相同大小的numpy数组
    """
    # 确定输入图像的维度
    if image.ndim == 2:  # 灰度图像
        image_height, image_width = image.shape
        num_channels = 1
    elif image.ndim == 3:  # 彩色图像
        image_height, image_width, num_channels = image.shape
    else:
        raise ValueError("Unsupported image shape")
    
    # 获取卷积核的尺寸
    kernel_height, kernel_width = kernel.shape
    
    # 计算卷积结果的尺寸
    output_height = image_height - kernel_height + 1
    output_width = image_width - kernel_width + 1
    
    # 初始化卷积结果数组
    if num_channels == 1:  # 灰度图像
        output = np.zeros((output_height, output_width))
    elif num_channels > 1:  # 彩色图像
        output = np.zeros((output_height, output_width, num_channels))
    
    # 执行卷积操作
    for c in range(num_channels):
        for i in range(output_height):
            for j in range(output_width):
                # 从输入图像中提取与卷积核大小相同的区域
                if num_channels == 1:  # 灰度图像
                    region = image[i:i+kernel_height, j:j+kernel_width]
                elif num_channels > 1:  # 彩色图像
                    region = image[i:i+kernel_height, j:j+kernel_width, c]
                # 计算卷积结果的单个像素值
                output[i, j, c] = np.sum(region * kernel)
    
    return output


def generate_motion_blur_kernels(degrees, angles):
    """
    生成多个用于模拟运动模糊效果的模糊核
    Args:
        degrees: 运动模糊的程度列表，每个元素表示一个模糊核的大小
        angles: 运动模糊的角度列表，每个元素表示一个模糊核的角度
    Returns:
        卷积核列表，包含多个二维numpy数组
    """
    kernels = []
    for degree, angle in zip(degrees, angles):
        # 生成旋转矩阵
        M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
        # 生成对角线核
        motion_blur_kernel = np.diag(np.ones(degree))
        # 进行仿射变换
        motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))
        # 归一化
        motion_blur_kernel = motion_blur_kernel / degree
        kernels.append(motion_blur_kernel)
    return kernels

def default_generate_motion_blur_kernels(n,image_shape=None,degrees_ls=None,angles_ls=None,strategy=None,max_bound=25,aggregation=None):
    if strategy is None:
        strategy="variation"
    if strategy=="given":
        assert(degrees_ls is not None and angles_ls is not None)
        degrees=degrees_ls
        angles=angles_ls
    elif strategy=="variation":
        assert(image_shape is not None)
        min_dim=min(image_shape[0],image_shape[1])
        degrees_ls=list(range(1,int(min_dim/50)*3))
        angles_ls=list(range(0,360,45))
        degrees=[degrees_ls[i%len(degrees_ls)] for i in range(n)]
        angles = [angles_ls[i%len(angles_ls)] for i in range(n)]
    elif strategy=="fixed":
        degrees=[5]*n
        angles = [3]*n
    elif strategy=="variation_angle":
        assert(image_shape is not None)
        min_dim=min(image_shape[0],image_shape[1])
        angles_ls=list(range(0,360,45))
        degrees=[5]*n
        angles = [random.choice(angles_ls)  for _ in range(n)]
    elif strategy=="range":
        degrees=list(range(1,n+1))
        angles = [math.floor(d * 0.1 % 3+1) for d in degrees]
    elif strategy=="range_max":
        if aggregation:
            degrees=[]
            count=0
            for a in aggregation.labels:
                if a=="L":
                    degrees.append(int(count*1.0/aggregation.legitimate_nums*max_bound+1))
                    count+=1
                else:
                    degrees.append(random.randint(1,max_bound))
            angles = [math.floor(d * 0.1 % 3+1) for d in degrees]
        else:
            degrees=[int(i/n*max_bound+1) for i in range(n)]
            angles = [math.floor(d * 0.1 % 3+1) for d in degrees]

    kernels = generate_motion_blur_kernels(degrees, angles)
    return kernels


def default_generate_gaussian_noise(n,shape,mean=0, var=0.01):
    # 生成高斯噪声
    noise_ls=[np.random.normal(mean, np.sqrt(var), shape) for i in range(n)]
    return noise_ls


if __name__=="__main__":
    default_generate_motion_blur_kernels(32)
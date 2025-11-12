
import numpy as np
import cv2
from utils.image_calculate import convolution2d
 
 # 归一化函数，将图像从0-255范围映射到0-1范围
def normalize_image(image):
    return image / 255.0

# 逆归一化函数，将图像从0-1范围映射回0-255范围
def denormalize_image(image):
    return (image * 255.0).astype(np.uint8)

def motion_blur(image, motion_blur_kernel):
    blurred = cv2.filter2D(image, -1, motion_blur_kernel)
    return blurred

def motion_blur_by_convolution2d(image, degree=35, angle=3):
    image = np.array(image)
    
    # 生成任意角度的运动模糊kernel矩阵， degree越大，模糊程度越高
    M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
    motion_blur_kernel = np.diag(np.ones(degree))
    motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))
    motion_blur_kernel = motion_blur_kernel / degree
    blurred = convolution2d(image, motion_blur_kernel)

    return blurred

def add_gaussian_noise(image, mean=0, var=0.01):
    image = np.array(image)
        
    # 生成高斯噪声
    noise = np.random.normal(mean, np.sqrt(var), image.shape)
    # 将噪声添加到图像
    noisy_image = image + noise
    # 将值限制在0到1之间
    noisy_image = np.clip(noisy_image, 0, 1)

    return noisy_image

def add_given_gaussian_noise(image, noise):
    # 将噪声添加到图像
    noisy_image = image + noise
    # 将值限制在0到1之间
    # noisy_image = np.clip(noisy_image, 0, 1)

    return noisy_image
 
if __name__=="__main__":
    img = cv2.imread('./data/test/ikun.jpg')
    def test_motion_blur():
        img_ = motion_blur(img)
        cv2.imshow('Source image',img)
        cv2.imshow('blur image',img_)
        cv2.waitKey()
    def test_motion_blur_by_convolution2d():
        img_ = motion_blur_by_convolution2d(img)
        cv2.imshow('Source image',img)
        cv2.imshow('blur image',img_)
        cv2.waitKey()
    def test_add_gaussian_noise():
        img_ = add_gaussian_noise(img)
        cv2.imshow('Source image',img)
        cv2.imshow('blur image',img_)
        cv2.waitKey()
    def test_motion_blur_and_add_gaussian_noise():
        img_ = motion_blur(img)
        img_ = add_gaussian_noise(img_)
        cv2.imshow('Source image',img)
        cv2.imshow('blur image',img_)
        cv2.waitKey()
    test_motion_blur()

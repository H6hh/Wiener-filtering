import cv2
import numpy as np
from matplotlib import pyplot as plt

# 读取图像和噪声自相关函数
img = cv2.imread('njust.png',0)
noise_acf = cv2.imread('njust.png',0)

# 计算噪声功率谱
noise_acf = np.float32(noise_acf)
noise_power_spectrum = np.fft.fftshift(np.fft.fft2(noise_acf))
noise_power_spectrum_abs = np.abs(noise_power_spectrum) ** 2

# 计算图像功率谱
img_power_spectrum = np.fft.fftshift(np.abs(np.fft.fft2(img))) ** 2

# 计算维纳滤波器
K = 0.01  # 维纳滤波器调整系数
H = noise_power_spectrum_abs / (img_power_spectrum + noise_power_spectrum_abs/K)  # 维纳滤波器
H_shift = np.fft.ifftshift(H)  # 取中心

# 进行滤波
filtered_img = np.fft.ifft2(np.fft.fft2(img) * H_shift)

# 显示结果
plt.subplot(121), plt.imshow(img, cmap='gray')
plt.title('Original Image')
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(np.uint8(filtered_img), cmap='gray')
plt.title('Restored Image with Wiener Filter')
plt.xticks([]), plt.yticks([])
plt.show()
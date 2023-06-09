import numpy as np
import scipy.signal as ss
import matplotlib.pyplot as plt
from PIL import Image
import math  # 导入数学库，用于计算噪声标准差的缩放因子

# 载入图像
im = np.array(Image.open('njust.png').convert('L'))
im_size = im.shape
plt.imshow(im, cmap='gray')
plt.show()

# 生成高斯白噪声
noise = np.random.normal(scale=10, size=im_size)

# 计算信噪比
SNR = 5
noise_variance = np.var(im) / (10**(SNR/10))
# 缩放噪声的标准差
noise *= math.sqrt(noise_variance) / np.std(noise)
# 添加噪声到原图像
im_noise = im + noise
plt.imshow(im_noise, cmap='gray')
plt.show()

# 计算FFT
im_fft = np.fft.fft2(im_noise)

# 声明滤波器半径，这里使用图像尺寸的四分之一
radius = int(min(im_size) / 4)

# 计算PSF（点扩散函数）
PSF = np.zeros(im_size)
PSF[int(im_size[0]/2-radius):int(im_size[0]/2+radius),
    int(im_size[1]/2-radius):int(im_size[1]/2+radius)] = 1
PSF = ss.fftconvolve(PSF, PSF)
PSF /= PSF.sum()
OTF = np.fft.fft2(PSF)

# 计算维纳滤波器
K = 0.01
NSR = noise_variance / np.var(im)
WNR = 1.0 / (NSR + abs(OTF)**2 / K)
im_restore = np.real(np.fft.ifft2(im_fft * WNR))

# 显示恢复后的图像
plt.imshow(im_restore, cmap='gray')
plt.show()

# 保存恢复后的图像
im_restore = Image.fromarray(im_restore.astype(np.uint8))
im_restore.save('lena_wnr_restored_known.png')
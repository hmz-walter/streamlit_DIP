from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt


def fourier_transform(image):
    """对图像进行傅里叶变换并返回频域图像。"""
    image_np = np.array(image)
    if len(image_np.shape) == 3:
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_np
    # dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    return dft_shift


def inverse_fourier_transform(dft_shift):
    """对傅里叶变换后的图像进行逆变换。"""
    f_ishift = np.fft.ifftshift(dft_shift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
    img_back = np.uint8(np.clip(img_back, 0, 255))
    return img_back


def ideal_lowpass_filter(shape, D0):
    """生成理想低通滤波器掩膜。"""
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    U, V = np.meshgrid(np.arange(cols), np.arange(rows))
    D = np.sqrt((U - ccol) ** 2 + (V - crow) ** 2)
    H = np.zeros((rows, cols), dtype=np.float32)
    H[D <= D0] = 1
    # H = H[:, :, np.newaxis]
    # H = np.repeat(H, 2, axis=2)
    return H


def ideal_highpass_filter(shape, D0):
    """生成理想高通滤波器掩膜。"""
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    U, V = np.meshgrid(np.arange(cols), np.arange(rows))
    D = np.sqrt((U - ccol) ** 2 + (V - crow) ** 2)
    H = np.ones((rows, cols), dtype=np.float32)
    H[D <= D0] = 0
    # H = H[:, :, np.newaxis]
    # H = np.repeat(H, 2, axis=2)
    return H


def butterworth_lowpass_filter(shape, D0, n):
    """生成巴特沃斯低通滤波器掩膜。"""
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    U, V = np.meshgrid(np.arange(cols), np.arange(rows))
    D = np.sqrt((U - ccol) ** 2 + (V - crow) ** 2)
    H = 1 / (1 + (D / D0) ** (2 * n))
    # H = H[:, :, np.newaxis]
    # H = np.repeat(H, 2, axis=2)
    return H


def butterworth_highpass_filter(shape, D0, n):
    """生成巴特沃斯高通滤波器掩膜。"""
    H_low = butterworth_lowpass_filter(shape, D0, n)
    H_high = 1 - H_low
    return H_high


def gaussian_lowpass_filter(shape, D0):
    """生成高斯低通滤波器掩膜。"""
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    U, V = np.meshgrid(np.arange(cols), np.arange(rows))
    D2 = (U - ccol) ** 2 + (V - crow) ** 2
    H = np.exp(-D2 / (2 * (D0 ** 2)))
    # H = H[:, :, np.newaxis]
    # H = np.repeat(H, 2, axis=2)
    return H


def gaussian_highpass_filter(shape, D0):
    """生成高斯高通滤波器掩膜。"""
    H_low = gaussian_lowpass_filter(shape, D0)
    H_high = 1 - H_low
    return H_high


def apply_frequency_filter(dft_shift, filter_type="ideal", pass_type="low", cutoff=30, order=2, n=2):
    """应用频率域滤波器到傅里叶变换后的图像。

    参数:
        dft_shift (ndarray): 已经中心化的傅里叶变换图像，形状为 (rows, cols, 2)。
        filter_type (str): 滤波器类型（"ideal", "butterworth", "gaussian"）。
        pass_type (str): 通过类型（"low", "high"）。
        cutoff (int): 截止频率。
        order (int): 巴特沃斯滤波器的阶数。
        n (float): 高斯滤波器的标准差。

    返回:
        ndarray: 过滤后的频域图像，形状为 (rows, cols, 2)。
    """
    shape = dft_shift.shape[:2]

    if filter_type == "理想滤波器":
        if pass_type == "低通滤波":
            H = ideal_lowpass_filter(shape, cutoff)
        else:
            H = ideal_highpass_filter(shape, cutoff)
    elif filter_type == "巴特沃斯滤波器":
        if pass_type == "低通滤波":
            H = butterworth_lowpass_filter(shape, cutoff, order)
        else:
            H = butterworth_highpass_filter(shape, cutoff, order)
    elif filter_type == "高斯滤波器":
        if pass_type == "低通滤波":
            H = gaussian_lowpass_filter(shape, cutoff)
        else:
            H = gaussian_highpass_filter(shape, cutoff)
    else:
        H = np.ones((shape[0], shape[1], 2), dtype=np.float32)  # 默认不滤波

    # 应用掩膜
    fshift_filtered = dft_shift * H

    return fshift_filtered


def apply_frequency_filter_to_image(image, filter_type="ideal", pass_type="low", cutoff=30, order=2, n=2):
    """对图像应用频率域滤波器并进行逆变换。

    参数:
        image (PIL.Image): 输入图像。
        filter_type (str): 滤波器类型（"ideal", "butterworth", "gaussian"）。
        pass_type (str): 通过类型（"low", "high"）。
        cutoff (int): 截止频率。
        order (int): 巴特沃斯滤波器的阶数。
        n (float): 高斯滤波器的标准差。

    返回:
        PIL.Image: 过滤后的图像。
        ndarray: 原始频域图像的幅值谱。
        ndarray: 过滤后频域图像的幅值谱。
    """
    # 进行傅里叶变换
    # dft_shift = fourier_transform(image)
    dft_shift = np.fft.fftshift(np.fft.fft2(image))

    # 生成掩膜并应用滤波
    fshift_filtered = apply_frequency_filter(
        dft_shift,
        filter_type=filter_type,
        pass_type=pass_type,
        cutoff=cutoff,
        order=order,
        n=n
    )

    # 计算逆傅里叶变换
    # img_back = inverse_fourier_transform(fshift_filtered)
    img_back = np.fft.ifft2(np.fft.ifftshift(fshift_filtered))

    # 确保 img_back 为 uint8 类型
    if img_back.dtype != np.uint8:
        img_back = np.uint8(np.clip(img_back, 0, 255))

    # 转换为 PIL.Image 并确保模式为 'L'
    img_back_pil = Image.fromarray(img_back).convert('L')

    # 计算频域图像的幅值谱用于可视化
    # f_image = np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]) + 1)
    # f_image_filtered = np.log(cv2.magnitude(fshift_filtered[:, :, 0], fshift_filtered[:, :, 1]) + 1)
    f_image = np.log(np.abs(dft_shift))
    f_image_filtered = np.log(np.abs(fshift_filtered))

    return img_back_pil, f_image, f_image_filtered


# 在 frequency_transformations.py 中扩展支持多通道频域滤波

def apply_frequency_filter_to_image_multichannel(image, filter_type="ideal", pass_type="low", cutoff=30, order=2, n=2):
    """对彩色图像的每个通道应用频率域滤波器并进行逆变换。

    参数:
        image (PIL.Image): 输入图像。
        filter_type (str): 滤波器类型。
        pass_type (str): 通过类型。
        cutoff (int): 截止频率。
        order (int): 巴特沃斯滤波器的阶数。
        n (float): 高斯滤波器的标准差。

    返回:
        PIL.Image: 过滤后的图像。
        ndarray: 原始频域图像的幅值谱。
        ndarray: 过滤后频域图像的幅值谱。
    """
    image_np = np.array(image)
    channels = cv2.split(image_np)
    filtered_channels = []
    f_image_total = []
    f_image_filtered_total = []

    for ch in channels:
        dft = cv2.dft(np.float32(ch), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)

        fshift_filtered = apply_frequency_filter(
            dft_shift,
            filter_type=filter_type,
            pass_type=pass_type,
            cutoff=cutoff,
            order=order,
            n=n
        )

        img_back = inverse_fourier_transform(fshift_filtered)
        img_back = np.uint8(np.clip(img_back, 0, 255))
        filtered_channels.append(img_back)

        # 计算频域图像的幅值谱
        f_image = np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]) + 1)
        f_image_filtered = np.log(cv2.magnitude(fshift_filtered[:, :, 0], fshift_filtered[:, :, 1]) + 1)
        f_image_total.append(f_image)
        f_image_filtered_total.append(f_image_filtered)

    # 合并通道
    filtered_image_np = cv2.merge(filtered_channels)
    filtered_image = Image.fromarray(filtered_image_np).convert(image.mode)

    # 合并频域图像的幅值谱
    f_image_combined = np.mean(f_image_total, axis=0)
    f_image_filtered_combined = np.mean(f_image_filtered_total, axis=0)

    return filtered_image, f_image_combined, f_image_filtered_combined

# image_processing/frequency_transformations.py

import numpy as np
import cv2
from PIL import Image
import pywt
from matplotlib import pyplot as plt
# from scipy.fft import dct, idct

def apply_fft(image):
    """
    对图像应用傅里叶变换并返回变换后的图像和幅度谱。

    参数:
        image (PIL.Image): 输入图像。

    返回:
        transformed_image (PIL.Image): 傅里叶变换后的图像（频域表示）。
        magnitude_spectrum (PIL.Image): 幅度谱图像。
    """
    image_np = np.array(image.convert('L'))
    f = np.fft.fft2(image_np)
    fshift = np.fft.fftshift(f)

    magnitude_spectrum = abs(fshift)
    phase_spectrum = np.angle(fshift)

    magnitude_spectrum_pil = Image.fromarray(magnitude_spectrum)
    phasor_spectrum_pil = Image.fromarray(phase_spectrum)

    return magnitude_spectrum_pil, phasor_spectrum_pil


def inverse_fft(magnitude=None, phase=None):
    """
    根据幅度和/或相位谱重建图像。

    参数:
        magnitude (PIL.Image or None): 幅度谱图像。
        phase (PIL.Image or None): 相位谱图像。

    返回:
        inverse_image (PIL.Image): 重建后的图像。
    """
    if magnitude is None and phase is None:
        raise ValueError("至少需要提供幅度谱或相位谱。")

    if magnitude is not None:
        magnitude = np.array(magnitude).astype(float)
    if phase is not None:
        phase = np.array(phase).astype(float)

    if magnitude is not None and phase is not None:
        # 使用幅度和相位重建复数频域表示
        f_ishift = magnitude * np.exp(1j * phase)
    elif magnitude is not None:
        # 仅使用幅度，假设相位为0
        # template_image = np.zeros_like(magnitude)
        # h, w = template_image.shape
        # template_image[h // 2 - h // 8:h // 2 + h // 8, w // 2 - w // 32:w // 2 + w // 32] = 255
        # f_ishift = np.fft.fftshift(np.fft.fft2(template_image))
        # angle_template = np.angle(f_ishift)
        # f_ishift = magnitude * np.exp(1j * angle_template)
        f_ishift = magnitude * np.exp(1j * 0)
    elif phase is not None:
        # 仅使用相位，使用大小相同的模板图像获得相位，图像中央有一个矩形
        # template_image = np.zeros_like(phase)
        # h, w = template_image.shape
        # template_image[h // 2 - h // 8:h // 2 + h // 8, w // 2 - w // 32:w // 2 + w // 32] = 255
        # f_ishift = np.fft.fftshift(np.fft.fft2(template_image))
        # magnitude_template = np.abs(f_ishift)
        # f_ishift = magnitude_template * np.exp(1j * phase)
        f_ishift = 1e5 * np.exp(1j * phase)

    # 逆移位
    f_ishift = np.fft.ifftshift(f_ishift)
    # 进行逆傅里叶变换
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.uint8(np.clip(img_back, 0, 255))

    inverse_image = Image.fromarray(img_back).convert('L')
    return inverse_image


def apply_dct(image):
    """
    对图像应用离散余弦变换（DCT）并返回变换后的图像和DCT幅度谱。

    参数:
        image (PIL.Image): 输入图像。

    返回:
        transformed_image (PIL.Image): DCT变换后的图像。
        dct_spectrum (PIL.Image): DCT幅度谱图像。
    """
    image_np = np.array(image.convert('L')).astype(float)
    dct_transformed = cv2.dct(image_np)
    # dct_shift = np.fft.fftshift(dct_transformed)
    # 计算DCT幅度谱
    # dct_spectrum = abs(dct_transformed)
    dct_spectrum_pil = Image.fromarray(dct_transformed)
    return dct_spectrum_pil


def inverse_dct(dct_spectrum, threshold=100):
    """
    对离散余弦变换后的幅度谱应用逆DCT重建图像。

    参数:
        dct_spectrum (PIL.Image): DCT幅度谱图像。

    返回:
        inverse_image (PIL.Image): 逆DCT变换后的图像。
    """
    dct_spectrum_np = np.array(dct_spectrum).astype(float)
    # dct_spectrum_np = abs(dct_spectrum_np)
    dct_spectrum_np[abs(dct_spectrum_np) < threshold] = 0
    # 进行逆DCT
    img_back = cv2.idct(dct_spectrum_np)
    # img_back = np.abs(img_back)
    img_back = np.uint8(np.clip(img_back, 0, 255))
    # img_back = np.uint8(img_back)

    inverse_image = Image.fromarray(img_back).convert('L')

    return inverse_image


def apply_wavelet(image, wavelet_type='haar', level=2):
    """
    对图像应用小波变换并返回变换后的图像和小波幅度谱。

    参数:
        image (PIL.Image): 输入图像。
        wavelet_type (str): 小波类型（如 'haar', 'db1', 'db2', 'sym2'）。
        level (int): 分解层数。

    返回:
        transformed_image (PIL.Image): 小波变换后的图像（重构图像）。
        wavelet_spectrum (dict): 包含幅度谱和小波系数信息的字典。
    """
    image_np = np.array(image.convert('L'))
    coeffs = pywt.wavedec2(image_np, wavelet=wavelet_type, level=level)
    coeff_arr, coeff_slices = pywt.coeffs_to_array(coeffs)

    # # 计算幅度谱
    # magnitude_spectrum = 20 * np.log(np.abs(coeff_arr) + 1)
    # # 归一化幅度谱到0-255
    # magnitude_spectrum_norm = np.uint8(255 * magnitude_spectrum / np.max(magnitude_spectrum))

    wavelet_spectrum_pil = Image.fromarray(coeff_arr)

    # # 重构图像（无损）
    # reconstructed = pywt.waverec2(coeffs, wavelet=wavelet_type)
    # reconstructed = np.clip(reconstructed, 0, 255).astype(np.uint8)
    # transformed_image_pil = Image.fromarray(reconstructed)

    return {
        "spectrum": wavelet_spectrum_pil,
        "coeffs": coeffs,
        "slices": coeff_slices,
        "wavelet": wavelet_type,
        "level": level
    }


def inverse_wavelet(coeffs, slices=None, wavelet='haar', level=2, selected_coeffs=None):
    """
    对小波变换后的系数应用逆小波变换。

    参数:
        coeffs (ndarray): 小波变换的系数数组。
        wavelet (str): 小波类型。
        level (int): 分解层数。

    返回:
        inverse_image (PIL.Image): 逆小波变换后的图像。
    """
    if slices is None:
        raise ValueError("缺少小波变换的切片信息。")

    modified_coeffs = list(coeffs)
    if selected_coeffs is not None:
        # coeffs 列表格式: [cA_n, (cH_n, cV_n, cD_n), ..., (cH1, cV1, cD1)]
        for i in range(1, level + 1):
            if i in selected_coeffs:
                details = selected_coeffs[i]
                cH = details.get('cH', True)
                cV = details.get('cV', True)
                cD = details.get('cD', True)

                # 获取当前层级的细节系数
                cH_coeff, cV_coeff, cD_coeff = modified_coeffs[i]

                # 根据用户选择，保留或舍弃细节系数
                if not cH:
                    cH_coeff = np.zeros_like(cH_coeff)
                if not cV:
                    cV_coeff = np.zeros_like(cV_coeff)
                if not cD:
                    cD_coeff = np.zeros_like(cD_coeff)

                # 更新系数列表
                modified_coeffs[i] = (cH_coeff, cV_coeff, cD_coeff)

    # 将数组和切片重新转换为系数
    # coeffs_from_arr = pywt.array_to_coeffs(coeffs, slices, output_format='wavedec2')
    # 进行逆小波变换
    reconstructed = pywt.waverec2(modified_coeffs, wavelet=wavelet)
    reconstructed = np.clip(reconstructed, 0, 255).astype(np.uint8)
    inverse_image = Image.fromarray(reconstructed)

    return inverse_image

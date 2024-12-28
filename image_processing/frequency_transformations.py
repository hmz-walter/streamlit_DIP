# # image_processing/frequency_transformations.py
#
# import numpy as np
# from PIL import Image
# import pywt
# from matplotlib import pyplot as plt
#
# def apply_fft(image):
#     """
#     对图像应用傅里叶变换并返回变换后的图像和幅度谱。
#
#     参数:
#         image (PIL.Image): 输入图像。
#
#     返回:
#         transformed_image (PIL.Image): 傅里叶变换后的图像（频域表示）。
#         magnitude_spectrum (PIL.Image): 幅度谱图像。
#     """
#     image_np = np.array(image.convert('L'))
#     f = np.fft.fft2(image_np)
#     fshift = np.fft.fftshift(f)
#     magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
#
#     # 归一化幅度谱到0-255
#     magnitude_spectrum = np.uint8(255 * magnitude_spectrum / np.max(magnitude_spectrum))
#
#     # 创建频域图像
#     transformed_image = np.abs(fshift)
#     transformed_image = np.log(transformed_image + 1)
#     transformed_image = np.uint8(255 * transformed_image / np.max(transformed_image))
#     transformed_image_pil = Image.fromarray(transformed_image)
#
#     magnitude_spectrum_pil = Image.fromarray(magnitude_spectrum)
#
#     return transformed_image_pil, magnitude_spectrum_pil
#
# def inverse_fft(transformed_image):
#     """
#     对傅里叶变换后的图像应用逆傅里叶变换。
#
#     参数:
#         transformed_image (PIL.Image): 傅里叶变换后的图像。
#
#     返回:
#         inverse_image (PIL.Image): 逆傅里叶变换后的图像。
#     """
#     transformed_np = np.array(transformed_image)
#     # 反对数变换
#     transformed_np = np.exp(transformed_np / 255 * np.max(np.log(transformed_np + 1)))
#     f_ishift = np.fft.ifftshift(transformed_np)
#     img_back = np.fft.ifft2(f_ishift)
#     img_back = np.abs(img_back)
#     img_back = np.clip(img_back, 0, 255).astype(np.uint8)
#     inverse_image = Image.fromarray(img_back)
#
#     return inverse_image
#
# def apply_dct(image):
#     """
#     对图像应用离散余弦变换（DCT）并返回变换后的图像和DCT幅度谱。
#
#     参数:
#         image (PIL.Image): 输入图像。
#
#     返回:
#         transformed_image (PIL.Image): DCT变换后的图像。
#         dct_spectrum (PIL.Image): DCT幅度谱图像。
#     """
#     image_np = np.array(image.convert('L'))
#     dct = np.fft.dct(np.fft.dct(image_np.T, norm='ortho').T, norm='ortho')
#     dct_shift = np.fft.fftshift(dct)
#     dct_spectrum = 20 * np.log(np.abs(dct_shift) + 1)
#
#     # 归一化DCT幅度谱到0-255
#     dct_spectrum = np.uint8(255 * dct_spectrum / np.max(dct_spectrum))
#
#     # 归一化DCT系数到0-255
#     dct_normalized = np.uint8(255 * (dct_shift - np.min(dct_shift)) / (np.max(dct_shift) - np.min(dct_shift)))
#     transformed_image_pil = Image.fromarray(dct_normalized)
#
#     dct_spectrum_pil = Image.fromarray(dct_spectrum)
#
#     return transformed_image_pil, dct_spectrum_pil
#
# def inverse_dct(transformed_image):
#     """
#     对离散余弦变换后的图像应用逆DCT。
#
#     参数:
#         transformed_image (PIL.Image): DCT变换后的图像。
#
#     返回:
#         inverse_image (PIL.Image): 逆DCT变换后的图像。
#     """
#     transformed_np = np.array(transformed_image)
#     # 反归一化
#     transformed_np = transformed_np.astype(float)
#     dct_shift = transformed_np / 255 * (np.max(transformed_np) - np.min(transformed_np)) + np.min(transformed_np)
#     dct_ishift = np.fft.ifftshift(dct_shift)
#     img_back = np.fft.idct(np.fft.idct(dct_ishift.T, norm='ortho').T, norm='ortho')
#     img_back = np.abs(img_back)
#     img_back = np.clip(img_back, 0, 255).astype(np.uint8)
#     inverse_image = Image.fromarray(img_back)
#
#     return inverse_image
#
# def apply_wavelet(image, wavelet_type='haar', level=2):
#     """
#     对图像应用小波变换并返回变换后的图像和小波幅度谱。
#
#     参数:
#         image (PIL.Image): 输入图像。
#         wavelet_type (str): 小波类型（如 'haar', 'db1', 'db2', 'sym2'）。
#         level (int): 分解层数。
#
#     返回:
#         transformed_image (PIL.Image): 小波变换后的图像（重构图像）。
#         wavelet_spectrum (PIL.Image): 小波幅度谱图像。
#     """
#     image_np = np.array(image.convert('L'))
#     coeffs = pywt.wavedec2(image_np, wavelet=wavelet_type, level=level)
#     coeff_arr, coeff_slices = pywt.coeffs_to_array(coeffs)
#     magnitude_spectrum = 20 * np.log(np.abs(coeff_arr) + 1)
#
#     # 归一化幅度谱到0-255
#     magnitude_spectrum = np.uint8(255 * magnitude_spectrum / np.max(magnitude_spectrum))
#
#     # 重构图像（无损）
#     reconstructed = pywt.waverec2(coeffs, wavelet=wavelet_type)
#     reconstructed = np.clip(reconstructed, 0, 255).astype(np.uint8)
#     transformed_image_pil = Image.fromarray(reconstructed)
#
#     wavelet_spectrum_pil = Image.fromarray(magnitude_spectrum)
#
#     return transformed_image_pil, wavelet_spectrum_pil
#
# def inverse_wavelet(transformed_image, wavelet_type='haar', level=2):
#     """
#     对小波变换后的图像应用逆小波变换。
#
#     参数:
#         transformed_image (PIL.Image): 小波变换后的图像。
#         wavelet_type (str): 小波类型。
#         level (int): 分解层数。
#
#     返回:
#         inverse_image (PIL.Image): 逆小波变换后的图像。
#     """
#     # 假设传入的transformed_image是重构后的图像，无需逆变换
#     # 如果需要从频域系数逆变换，需要传递更多信息
#     # 这里只返回原图
#     return transformed_image

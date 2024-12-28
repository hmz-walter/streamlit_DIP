# image_processing/filters_and_noise.py

from PIL import Image
import numpy as np
import cv2

def add_noise(image, noise_type="gaussian", mean=0, sigma=25, amount=0.02, s_vs_p=0.5, speckle_scale=0.5):
    """
    向图像添加指定类型的噪声。

    参数:
        image (PIL.Image): 输入的图像。
        noise_type (str): 噪声类型（"gaussian", "salt and pepper", "poisson", "speckle"）。
        mean (float): 高斯噪声的均值。
        sigma (float): 高斯噪声的标准差。
        amount (float): 盐和胡椒噪声的比例。
        s_vs_p (float): 盐和胡椒噪声中盐的比例。
        speckle_scale (float): 斑点噪声的比例。

    返回:
        PIL.Image: 添加噪声后的图像。
    """
    try:
        image_np = np.array(image)
        noise_type = noise_type.lower()

        # 处理灰度图像和RGB图像的通用逻辑
        if len(image_np.shape) == 2:  # 灰度图像
            is_gray = True
            image_np = image_np[:, :, np.newaxis]
        else:
            is_gray = False

        if noise_type == "gaussian":
            row, col, ch = image_np.shape
            gaussian_noise = np.random.normal(mean, sigma, (row, col, ch))
            noisy_image = np.clip(image_np + gaussian_noise, 0, 255).astype(np.uint8)

        elif noise_type == "salt and pepper":
            out = np.copy(image_np)
            num_salt = int(np.ceil(amount * image_np.shape[0] * image_np.shape[1] * s_vs_p))
            num_pepper = int(np.ceil(amount * image_np.shape[0] * image_np.shape[1] * (1. - s_vs_p)))

            # Salt noise
            salt_coords = [np.random.randint(0, i, num_salt) for i in image_np.shape[:2]]
            out[salt_coords[0], salt_coords[1], :] = 255

            # Pepper noise
            pepper_coords = [np.random.randint(0, i, num_pepper) for i in image_np.shape[:2]]
            out[pepper_coords[0], pepper_coords[1], :] = 0

            noisy_image = out

        elif noise_type == "poisson":
            vals = len(np.unique(image_np))
            vals = 2 ** np.ceil(np.log2(vals))
            noisy_image = np.random.poisson(image_np * vals) / float(vals)
            noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)

        elif noise_type == "speckle":
            row, col, ch = image_np.shape
            speckle_noise = np.random.randn(row, col, ch) * speckle_scale
            noisy_image = np.clip(image_np + image_np * speckle_noise, 0, 255).astype(np.uint8)

        else:
            print(f"Unsupported noise type: {noise_type}. Returning original image.")
            noisy_image = image_np

        # 如果是灰度图像，去掉多余的通道
        if is_gray:
            noisy_image = noisy_image[:, :, 0]

        return Image.fromarray(noisy_image)
    except Exception as e:
        print(f"Error in add_noise: {e}")
        return image

def mean_filter(image, kernel_size=3):
    """应用均值滤波器平滑图像。

    参数:
        image (PIL.Image): 输入的图像。
        kernel_size (int): 滤波器大小。

    返回:
        PIL.Image: 滤波后的图像。
    """
    try:
        image_np = np.array(image)
        if kernel_size % 2 == 0 or kernel_size < 1:
            raise ValueError("Kernel size must be a positive odd integer.")
        blurred = cv2.blur(image_np, (kernel_size, kernel_size))
        return Image.fromarray(blurred)
    except Exception as e:
        print(f"Error in mean_filter: {e}")
        return image

def median_filter(image, kernel_size=3):
    """应用中值滤波器去除噪声。

    参数:
        image (PIL.Image): 输入的图像。
        kernel_size (int): 滤波器大小。

    返回:
        PIL.Image: 滤波后的图像。
    """
    try:
        image_np = np.array(image)
        if kernel_size % 2 == 0 or kernel_size < 1:
            raise ValueError("Kernel size must be a positive odd integer.")
        median_blurred = cv2.medianBlur(image_np, kernel_size)
        return Image.fromarray(median_blurred)
    except Exception as e:
        print(f"Error in median_filter: {e}")
        return image

def gaussian_filter(image, kernel_size=3, sigma=0):
    """应用高斯滤波器平滑图像。

    参数:
        image (PIL.Image): 输入的图像。
        kernel_size (int): 滤波器大小。
        sigma (float): 高斯核的标准差。

    返回:
        PIL.Image: 滤波后的图像。
    """
    try:
        image_np = np.array(image)
        if kernel_size % 2 == 0 or kernel_size < 1:
            raise ValueError("Kernel size must be a positive odd integer.")
        blurred_image = cv2.GaussianBlur(image_np, (kernel_size, kernel_size), sigma)
        return Image.fromarray(blurred_image)
    except Exception as e:
        print(f"Error in gaussian_filter: {e}")
        return image

def bilateral_filter(image, diameter=9, sigma_color=75, sigma_space=75):
    """应用双边滤波器保留边缘的同时平滑图像。

    参数:
        image (PIL.Image): 输入的图像。
        diameter (int): 每个像素邻域的直径。
        sigma_color (float): 颜色空间的标准差。
        sigma_space (float): 坐标空间的标准差。

    返回:
        PIL.Image: 滤波后的图像。
    """
    try:
        image_np = np.array(image)
        if diameter < 1:
            raise ValueError("Diameter must be a positive integer.")
        filtered_image = cv2.bilateralFilter(image_np, diameter, sigma_color, sigma_space)
        return Image.fromarray(filtered_image)
    except Exception as e:
        print(f"Error in bilateral_filter: {e}")
        return image

def sharpen_filter(image, method="laplacian", alpha=1.0, direction="综合", kernel=None):
    """
    对图像进行锐化处理。

    参数:
        image (PIL.Image): 输入的图像。
        method (str): 锐化方法（laplacian, sobel, high_boost, custom）。
        alpha (float): 锐化强度因子（用于高提升和拉普拉斯）。
        direction (str): Sobel方向（水平、垂直、综合）。
        kernel (np.array): 自定义核矩阵（仅适用于自定义算子）。

    返回:
        PIL.Image: 锐化后的图像。
    """
    try:
        image_np = np.array(image.convert("L"))  # 转灰度图像
        if method == "laplacian":
            kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
            laplacian = cv2.filter2D(image_np, -1, kernel)
            sharpened = np.clip(image_np + alpha * laplacian, 0, 255).astype(np.uint8)

        elif method == "sobel":
            if direction == "水平":
                sobelx = cv2.Sobel(image_np, cv2.CV_64F, 1, 0, ksize=3)
                sharpened = np.uint8(np.clip(sobelx, 0, 255))
            elif direction == "垂直":
                sobely = cv2.Sobel(image_np, cv2.CV_64F, 0, 1, ksize=3)
                sharpened = np.uint8(np.clip(sobely, 0, 255))
            else:  # 综合
                sobelx = cv2.Sobel(image_np, cv2.CV_64F, 1, 0, ksize=3)
                sobely = cv2.Sobel(image_np, cv2.CV_64F, 0, 1, ksize=3)
                sobel = np.sqrt(sobelx ** 2 + sobely ** 2)
                sharpened = np.uint8(np.clip(sobel, 0, 255))

        # elif method == "high_boost":
        #     kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        #     base_image = cv2.filter2D(image_np, -1, kernel)
        #     sharpened = np.clip(alpha * image_np - base_image, 0, 255).astype(np.uint8)

        elif method == "custom" and kernel is not None:
            sharpened = cv2.filter2D(image_np, -1, kernel)

        else:
            raise ValueError("Unsupported sharpening method or kernel is None.")

        return Image.fromarray(sharpened)
    except Exception as e:
        print(f"Error in sharpen_image: {e}")
        return image

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

def apply_frequency_filter(fft_shift, filter_type="ideal", pass_type="low", cutoff=30, order=2, n=2):
    """应用频率域滤波器到傅里叶变换后的图像。

    参数:
        fft_shift (ndarray): 已经中心化的傅里叶变换图像，形状为 (rows, cols, 2)。
        filter_type (str): 滤波器类型（"ideal", "butterworth", "gaussian"）。
        pass_type (str): 通过类型（"low", "high"）。
        cutoff (int): 截止频率。
        order (int): 巴特沃斯滤波器的阶数。
        n (float): 高斯滤波器的标准差。

    返回:
        ndarray: 过滤后的频域图像，形状为 (rows, cols, 2)。
    """
    shape = fft_shift.shape[:2]

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
    fshift_filtered = fft_shift * H

    return fshift_filtered

def homomorphic_filter(image, low_cutoff=30, high_cutoff=80, alpha=0.5, beta=2.0, filter_type="高斯滤波器"):
    """
    同态滤波，增强高频分量并压制低频分量。

    参数:
        image (PIL.Image): 输入的图像。
        low_cutoff (float): 低频截止值。
        high_cutoff (float): 高频截止值。
        alpha (float): 低频增益（建议 <1）。
        beta (float): 高频增益（建议 >1）。
        filter_type (str): 滤波器类型（"理想滤波器", "巴特沃斯滤波器", "高斯滤波器"）。

    返回:
        PIL.Image: 滤波后的图像。
    """
    try:
        # 转灰度图并对数变换
        image_np = np.array(image.convert("L"), dtype=np.float32)
        image_log = np.log1p(image_np)

        # 傅里叶变换
        dft = np.fft.fft2(image_log)
        dft_shift = np.fft.fftshift(dft)

        # 构造滤波器
        rows, cols = image_np.shape
        u = np.arange(-rows // 2, rows // 2)
        v = np.arange(-cols // 2, cols // 2)
        U, V = np.meshgrid(v, u)  # 注意：meshgrid 的顺序
        D = np.sqrt(U ** 2 + V ** 2)

        if filter_type == "理想滤波器":
            H = np.ones_like(D)
            H[D <= low_cutoff] = alpha
            H[D > low_cutoff] = beta  # 修改为对所有 D > low_cutoff 赋值 beta

        elif filter_type == "巴特沃斯滤波器":
            n = 2  # 默认阶数
            H = (beta - alpha) / (1 + (D / high_cutoff) ** (2 * n)) + alpha

        elif filter_type == "高斯滤波器":
            H = (beta - alpha) * (1 - np.exp(-(D ** 2) / (2 * (high_cutoff ** 2)))) + alpha

        else:
            raise ValueError(f"Unsupported filter type: {filter_type}")

        # 可视化滤波器（可选）
        # plt.imshow(H, cmap='gray')
        # plt.title(f"{filter_type} Filter")
        # plt.show()

        # 滤波
        filtered_dft = dft_shift * H
        f_ishift = np.fft.ifftshift(filtered_dft)
        filtered_image = np.fft.ifft2(f_ishift)
        filtered_image = np.real(filtered_image)
        filtered_image = np.expm1(filtered_image)  # 指数逆变换
        filtered_image = np.clip(filtered_image, 0, 255).astype(np.uint8)

        return Image.fromarray(filtered_image)
    except Exception as e:
        print(f"Error in homomorphic_filter: {e}")
        return image

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
    fft_shift = np.fft.fftshift(np.fft.fft2(image))

    # 生成掩膜并应用滤波
    fshift_filtered = apply_frequency_filter(
        fft_shift,
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
    f_image = np.log(np.abs(fft_shift)+1)
    f_image_filtered = np.log(np.abs(fshift_filtered)+1)

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
        fft_shift = np.fft.fftshift(np.fft.fft2(ch))

        fshift_filtered = apply_frequency_filter(
            fft_shift,
            filter_type=filter_type,
            pass_type=pass_type,
            cutoff=cutoff,
            order=order,
            n=n
        )
        img_back = np.fft.ifft2(np.fft.ifftshift(fshift_filtered))
        if img_back.dtype != np.uint8:
            img_back = np.uint8(np.clip(img_back, 0, 255))

        filtered_channels.append(img_back)

        # 计算频域图像的幅值谱
        f_image = np.log(np.abs(fft_shift) + 1)
        f_image_filtered = np.log(np.abs(fshift_filtered) + 1)
        f_image_total.append(f_image)
        f_image_filtered_total.append(f_image_filtered)

    # 合并通道
    filtered_image_np = cv2.merge(filtered_channels)
    filtered_image = Image.fromarray(filtered_image_np).convert(image.mode)

    # 合并频域图像的幅值谱
    f_image_combined = np.mean(f_image_total, axis=0)
    f_image_filtered_combined = np.mean(f_image_filtered_total, axis=0)

    return filtered_image, f_image_combined, f_image_filtered_combined


# # 频域低通滤波
# def low_pass_filter(image, cutoff=30):
#     image = np.array(image)
#     # 转换到灰度图像
#     gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     dft = cv2.dft(np.float32(gray_image), flags=cv2.DFT_COMPLEX_OUTPUT)
#     dft_shift = np.fft.fftshift(dft)
#
#     # 创建理想低通滤波器
#     rows, cols = gray_image.shape
#     crow, ccol = rows // 2, cols // 2
#     mask = np.zeros((rows, cols, 2), np.uint8)
#     mask[crow - cutoff:crow + cutoff, ccol - cutoff:ccol + cutoff] = 1
#
#     # 应用低通滤波
#     fshift = dft_shift * mask
#     f_ishift = np.fft.ifftshift(fshift)
#     img_back = cv2.idft(f_ishift)
#     img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
#
#     return Image.fromarray(np.uint8(np.clip(img_back, 0, 255)))
#
#
# # 频域高通滤波
# def high_pass_filter(image, cutoff=30):
#     image = np.array(image)
#     # 转换到灰度图像
#     gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     dft = cv2.dft(np.float32(gray_image), flags=cv2.DFT_COMPLEX_OUTPUT)
#     dft_shift = np.fft.fftshift(dft)
#
#     # 创建理想高通滤波器
#     rows, cols = gray_image.shape
#     crow, ccol = rows // 2, cols // 2
#     mask = np.ones((rows, cols, 2), np.uint8)
#     mask[crow - cutoff:crow + cutoff, ccol - cutoff:ccol + cutoff] = 0
#
#     # 应用高通滤波
#     fshift = dft_shift * mask
#     f_ishift = np.fft.ifftshift(fshift)
#     img_back = cv2.idft(f_ishift)
#     img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
#
#     return Image.fromarray(np.uint8(np.clip(img_back, 0, 255)))

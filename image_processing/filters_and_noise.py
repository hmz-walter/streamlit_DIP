# image_processing/filters_and_noise.py

from PIL import Image
import numpy as np
import cv2

def add_noise(image, noise_type="gaussian", mean=0, sigma=25, amount=0.02, s_vs_p=0.5):
    """向图像添加指定类型的噪声。

    参数:
        image (PIL.Image): 输入的图像。
        noise_type (str): 噪声类型（"gaussian", "salt and pepper", "poisson", "speckle"）。
        mean (float): 高斯噪声的均值。
        sigma (float): 高斯噪声的标准差。
        amount (float): 盐和胡椒噪声的比例。
        s_vs_p (float): 盐和胡椒噪声中盐的比例。

    返回:
        PIL.Image: 添加噪声后的图像。
    """
    try:
        image_np = np.array(image)
        noise_type = noise_type.lower()

        if noise_type == "gaussian":
            row, col, ch = image_np.shape
            gaussian_noise = np.random.normal(mean, sigma, (row, col, ch))
            noisy_image = np.clip(image_np + gaussian_noise, 0, 255).astype(np.uint8)

        elif noise_type == "salt and pepper":
            out = np.copy(image_np)
            num_salt = int(np.ceil(amount * image_np.shape[0] * image_np.shape[1] * s_vs_p))
            num_pepper = int(np.ceil(amount * image_np.shape[0] * image_np.shape[1] * (1. - s_vs_p)))

            # Salt noise
            salt_coords = [np.random.randint(0, i - 1, num_salt) for i in image_np.shape[:2]]
            out[salt_coords[0], salt_coords[1], :] = 255

            # Pepper noise
            pepper_coords = [np.random.randint(0, i - 1, num_pepper) for i in image_np.shape[:2]]
            out[pepper_coords[0], pepper_coords[1], :] = 0

            noisy_image = out

        elif noise_type == "poisson":
            vals = len(np.unique(image_np))
            vals = 2 ** np.ceil(np.log2(vals))
            noisy_image = np.random.poisson(image_np * vals) / float(vals)
            noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)

        elif noise_type == "speckle":
            row, col, ch = image_np.shape
            speckle_noise = np.random.randn(row, col, ch)
            noisy_image = np.clip(image_np + image_np * speckle_noise, 0, 255).astype(np.uint8)

        else:
            print(f"Unsupported noise type: {noise_type}. Returning original image.")
            noisy_image = image_np

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



# 频域低通滤波
def low_pass_filter(image, cutoff=30):
    image = np.array(image)
    # 转换到灰度图像
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dft = cv2.dft(np.float32(gray_image), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    # 创建理想低通滤波器
    rows, cols = gray_image.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols, 2), np.uint8)
    mask[crow - cutoff:crow + cutoff, ccol - cutoff:ccol + cutoff] = 1

    # 应用低通滤波
    fshift = dft_shift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

    return Image.fromarray(np.uint8(np.clip(img_back, 0, 255)))


# 频域高通滤波
def high_pass_filter(image, cutoff=30):
    image = np.array(image)
    # 转换到灰度图像
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dft = cv2.dft(np.float32(gray_image), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    # 创建理想高通滤波器
    rows, cols = gray_image.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.ones((rows, cols, 2), np.uint8)
    mask[crow - cutoff:crow + cutoff, ccol - cutoff:ccol + cutoff] = 0

    # 应用高通滤波
    fshift = dft_shift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

    return Image.fromarray(np.uint8(np.clip(img_back, 0, 255)))


def edge_detection(image, method="canny", threshold1=100, threshold2=200):
    """对图像进行边缘检测。

    参数:
        image (PIL.Image): 输入的图像。
        method (str): 边缘检测方法（"canny", "sobel", "laplacian"）。
        threshold1 (int): Canny 边缘检测的第一个阈值。
        threshold2 (int): Canny 边缘检测的第二个阈值。

    返回:
        PIL.Image: 边缘检测后的图像。
    """
    try:
        image_np = np.array(image)
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

        if method == "canny":
            edges = cv2.Canny(gray, threshold1, threshold2)
            edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
            return Image.fromarray(edges_colored)

        elif method == "sobel":
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
            sobel = cv2.magnitude(sobelx, sobely)
            sobel = np.uint8(np.clip(sobel, 0, 255))
            sobel_colored = cv2.cvtColor(sobel, cv2.COLOR_GRAY2RGB)
            return Image.fromarray(sobel_colored)

        elif method == "laplacian":
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            laplacian = np.uint8(np.clip(np.abs(laplacian), 0, 255))
            laplacian_colored = cv2.cvtColor(laplacian, cv2.COLOR_GRAY2RGB)
            return Image.fromarray(laplacian_colored)

        else:
            print(f"Unsupported edge detection method: {method}. Returning original image.")
            return image
    except Exception as e:
        print(f"Error in edge_detection: {e}")
        return image

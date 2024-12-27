# image_processing/geometric_transformations.py

from PIL import Image
import numpy as np
import cv2

def rectangle_crop(image, left, top, right, bottom):
    """按指定边界裁剪图像。

    参数:
        image (PIL.Image): 输入的图像。
        left (int): 左边界。
        top (int): 上边界。
        right (int): 右边界。
        bottom (int): 下边界。

    返回:
        PIL.Image: 裁剪后的图像。
    """
    try:
        return image.crop((left, top, right, bottom))
    except Exception as e:
        print(f"Error in rectangle_crop: {e}")
        return image

def rotate_image(image, angle, interpolation=cv2.INTER_LINEAR, expand=True):
    """旋转图像至指定角度。

    参数:
        image (PIL.Image): 输入的图像。
        angle (float): 旋转角度。
        interpolation (int): 插值方法。
        expand (bool): 是否扩展图像以适应旋转后的图像。

    返回:
        PIL.Image: 旋转后的图像。
    """
    try:
        image_np = np.array(image)
        (h, w) = image_np.shape[:2]
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        if expand:
            # 计算新的边界
            cos = np.abs(M[0, 0])
            sin = np.abs(M[0, 1])
            new_w = int((h * sin) + (w * cos))
            new_h = int((h * cos) + (w * sin))
            M[0, 2] += (new_w / 2) - center[0]
            M[1, 2] += (new_h / 2) - center[1]
            rotated = cv2.warpAffine(image_np, M, (new_w, new_h), flags=interpolation)
        else:
            rotated = cv2.warpAffine(image_np, M, (w, h), flags=interpolation)
        return Image.fromarray(rotated)
    except Exception as e:
        print(f"Error in rotate_image: {e}")
        return image

def scale_image(image, scale_factor, interpolation=cv2.INTER_LINEAR):
    """按指定比例缩放图像。

    参数:
        image (PIL.Image): 输入的图像。
        scale_factor (float): 缩放比例。
        interpolation (int): 插值方法。

    返回:
        PIL.Image: 缩放后的图像。
    """
    try:
        image_np = np.array(image)
        height, width = image_np.shape[:2]
        new_dimensions = (int(width * scale_factor), int(height * scale_factor))
        scaled = cv2.resize(image_np, new_dimensions, interpolation=interpolation)
        return Image.fromarray(scaled)
    except Exception as e:
        print(f"Error in scale_image: {e}")
        return image

def translate_image(image, tx, ty, interpolation=cv2.INTER_LINEAR):
    """平移图像。

    参数:
        image (PIL.Image): 输入的图像。
        tx (int): 水平平移量。
        ty (int): 垂直平移量。
        interpolation (int): 插值方法。

    返回:
        PIL.Image: 平移后的图像。
    """
    try:
        image_np = np.array(image)
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        shifted = cv2.warpAffine(image_np, M, (image_np.shape[1], image_np.shape[0]), flags=interpolation)
        return Image.fromarray(shifted)
    except Exception as e:
        print(f"Error in translate_image: {e}")
        return image

def shear_image(image, shear_factor, interpolation=cv2.INTER_LINEAR):
    """剪切图像。

    参数:
        image (PIL.Image): 输入的图像。
        shear_factor (float): 剪切因子。
        interpolation (int): 插值方法。

    返回:
        PIL.Image: 剪切后的图像。
    """
    try:
        image_np = np.array(image)
        h, w = image_np.shape[:2]
        M = np.array([[1, shear_factor, 0],
                      [0, 1, 0]], dtype=np.float32)
        sheared = cv2.warpAffine(image_np, M, (w, h), flags=interpolation)
        return Image.fromarray(sheared)
    except Exception as e:
        print(f"Error in shear_image: {e}")
        return image

def mirror_image(image, mode='horizontal'):
    """镜像图像。

    参数:
        image (PIL.Image): 输入的图像。
        mode (str): 镜像方向（'horizontal', 'vertical'）。

    返回:
        PIL.Image: 镜像后的图像。
    """
    try:
        image_np = np.array(image)
        if mode == 'horizontal':
            mirrored = cv2.flip(image_np, 1)
        elif mode == 'vertical':
            mirrored = cv2.flip(image_np, 0)
        else:
            print(f"Unsupported mirror mode: {mode}. Using horizontal by default.")
            mirrored = cv2.flip(image_np, 1)
        return Image.fromarray(mirrored)
    except Exception as e:
        print(f"Error in mirror_image: {e}")
        return image

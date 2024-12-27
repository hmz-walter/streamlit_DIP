# image_processing/basic_operations.py

from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
import os
from image_processing.utils import validate_crop_coordinates

def convert_to_grayscale(image):
    """将彩色图像转换为灰度图像。"""
    return image.convert("L")


def convert_to_color(image):
    """将灰度图像转换为彩色图像。
    """
    if image.mode != "L":
        return image  # 只有灰度图像需要转换
    try:
        # 将灰度图像转换为RGB模式
        image_np = np.array(image)
        if len(image_np.shape) == 2:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
        return image.convert("RGB")
    except Exception as e:
        print(f"Error converting to color: {e}")
        return image

def color_style_change(image, colormap="Jet"):
    """改变图像的颜色风格。
    参数:
        image (PIL.Image): 输入的图像。
        colormap (str): 颜色风格（"Jet", "Hot", "Cool", "Rainbow"）。
    返回:
        PIL.Image: 颜色风格改变后的图像。
    """
    try:
        # 转换为OpenCV格式的RGB图像
        image_np = np.array(image)
        # 检查图像是否为灰度图像，如果是则转换为RGB
        if len(image_np.shape) == 2:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
        # 将RGB转换为BGR，因为OpenCV使用BGR
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        # 选择对应的OpenCV色彩映射
        colormap_dict = {
            "Jet": cv2.COLORMAP_JET,
            "Hot": cv2.COLORMAP_HOT,
            "Cool": cv2.COLORMAP_COOL,
            "Rainbow": cv2.COLORMAP_RAINBOW,
            "Bone": cv2.COLORMAP_BONE,
            "Winter": cv2.COLORMAP_WINTER,
            "Summer": cv2.COLORMAP_SUMMER,
            "Autumn": cv2.COLORMAP_AUTUMN,
            "Spring": cv2.COLORMAP_SPRING,
            "Ocean": cv2.COLORMAP_OCEAN,
            "Pink": cv2.COLORMAP_PINK,
            "HSV": cv2.COLORMAP_HSV
        }
        selected_colormap = colormap_dict.get(colormap, cv2.COLORMAP_JET)
        # 应用色彩映射
        colored_bgr = cv2.applyColorMap(image_bgr, selected_colormap)
        # 将BGR转换回RGB
        colored_rgb = cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(colored_rgb)
    except Exception as e:
        print(f"Error in color_style_change: {e}")
        return image

def enhance_hsi(image, hue_factor=1.0, saturation_factor=1.0, intensity_factor=1.0):
    try:
        image_np = np.array(image)
        hsi_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV).astype(float)

        h, s, v = cv2.split(hsi_image)

        h *= hue_factor
        s *= saturation_factor
        v *= intensity_factor

        h = np.clip(h, 0, 179)
        s = np.clip(s, 0, 255)
        v = np.clip(v, 0, 255)

        enhanced_hsi_image = cv2.merge([h, s, v])
        enhanced_rgb_image = cv2.cvtColor(enhanced_hsi_image.astype(np.uint8), cv2.COLOR_HSV2RGB)
        return Image.fromarray(enhanced_rgb_image)
    except Exception as e:
        print(f"Error enhancing HSI: {e}")
        return image


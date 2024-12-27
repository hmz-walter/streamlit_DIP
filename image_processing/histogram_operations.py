# image_processing/histogram_operations.py

from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import io

def compute_histogram(image):
    """计算图像的灰度直方图。

    参数:
        image (PIL.Image): 输入的图像。

    返回:
        list: 直方图数据。
    """
    try:
        gray_image = image.convert("L")
        image_np = np.array(gray_image)
        hist = cv2.calcHist([image_np], [0], None, [256], [0, 256])
        hist = hist.flatten().tolist()
        return hist
    except Exception as e:
        print(f"Error in compute_histogram: {e}")
        return []

def histogram_equalization(image):
    """对图像进行直方图均衡化。

    参数:
        image (PIL.Image): 输入的图像。

    返回:
        PIL.Image: 均衡化后的图像。
    """
    try:
        image_np = np.array(image.convert("L"))
        equalized = cv2.equalizeHist(image_np)
        return Image.fromarray(equalized)
    except Exception as e:
        print(f"Error in histogram_equalization: {e}")
        return image

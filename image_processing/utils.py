# image_processing/utils.py

from PIL import Image
import numpy as np

def validate_crop_coordinates(left, top, right, bottom, image_width, image_height):
    """验证裁剪坐标是否在图像范围内。

    参数:
        left (int): 左边界。
        top (int): 上边界。
        right (int): 右边界。
        bottom (int): 下边界。
        image_width (int): 图像宽度。
        image_height (int): 图像高度。

    返回:
        bool: 如果坐标有效，返回 True，否则返回 False。
    """
    if left < 0 or top < 0 or right > image_width or bottom > image_height:
        return False
    if left >= right or top >= bottom:
        return False
    return True

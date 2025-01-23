# image_processing/advanced_features.py

import numpy as np
import cv2
import os
from PIL import Image, ImageDraw, ImageFont
from rembg import remove
from io import BytesIO

def add_watermark(image, watermark_text="Sample Watermark", position=(10, 10), font_path=None, font_size=20, color=(255, 255, 255, 128)):
    """在图像上添加文本水印。

    参数:
        image (PIL.Image): 输入的图像。
        watermark_text (str): 水印文本。
        position (tuple): 水印位置。
        font_path (str): 字体文件路径。
        font_size (int): 字体大小。
        color (tuple): 字体颜色和透明度。

    返回:
        PIL.Image: 添加水印后的图像。
    """
    try:
        watermark = Image.new("RGBA", image.size)
        draw = ImageDraw.Draw(watermark)

        if font_path and os.path.exists(font_path):
            font = ImageFont.truetype(font_path, font_size)
        else:
            try:
                font = ImageFont.truetype("DejaVuSans.ttf", font_size)
            except IOError:
                font = ImageFont.load_default()

        draw.text(position, watermark_text, font=font, fill=color)
        combined = Image.alpha_composite(image.convert("RGBA"), watermark)
        return combined.convert("RGB")
    except Exception as e:
        print(f"Error in add_watermark: {e}")
        return image

def compress_image(image, quality=50, format="JPG"):
    """按指定质量和格式压缩图像。

    参数:
        image (PIL.Image): 输入的图像。
        quality (int): 压缩质量（0-100）。
        format (str): 保存格式（"JPEG", "PNG", "BMP"）。

    返回:
        PIL.Image: 压缩后的图像。
    """
    try:
        buffer = BytesIO()
        image.save(buffer, format=format, quality=quality)
        buffer.seek(0)
        compressed_image = Image.open(buffer)
        return compressed_image
    except Exception as e:
        print(f"Error in compress_image: {e}")
        return image

def edge_detection(image, method="canny", **kwargs):
    """检测图像边缘。"""
    try:
        image_np = np.array(image.convert("RGB"))
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

        if method == "canny":
            threshold1 = kwargs.get("threshold1", 100)
            threshold2 = kwargs.get("threshold2", 200)
            edges = cv2.Canny(gray, threshold1, threshold2)

        elif method == "sobel":
            ksize = kwargs.get("ksize", 3)
            scale = kwargs.get("scale", 1)
            delta = kwargs.get("delta", 0)
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize, scale=scale, delta=delta)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize, scale=scale, delta=delta)
            edges = np.sqrt(sobelx**2 + sobely**2)
            edges = np.uint8(np.clip(edges, 0, 255))

        elif method == "laplacian":
            ksize = kwargs.get("ksize", 3)
            scale = kwargs.get("scale", 1)
            delta = kwargs.get("delta", 0)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=ksize, scale=scale, delta=delta)
            edges = np.uint8(np.clip(laplacian, 0, 255))

        else:
            raise ValueError(f"Unsupported edge detection method: {method}")

        return Image.fromarray(edges).convert("L")
    except Exception as e:
        print(f"Error in edge_detection: {e}")
        return image


def remove_background(image, method="color_threshold", **kwargs):
    """去除图像背景。"""
    try:
        image_np = np.array(image)

        if method == "color_threshold":
            lower_hue = kwargs.get("lower_hue", 0)
            upper_hue = kwargs.get("upper_hue", 180)
            hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
            mask = cv2.inRange(hsv, (lower_hue, 50, 50), (upper_hue, 255, 255))
            output = cv2.bitwise_and(image_np, image_np, mask=~mask)

        elif method == "deep_learning":
            output = remove(image_np)

        else:
            raise ValueError(f"Unsupported background removal method: {method}")

        return Image.fromarray(output)
    except Exception as e:
        print(f"Error in remove_background: {e}")
        return image


def object_detection(image, cascade_path="haarcascade_frontalface_default.xml", scaleFactor=1.1, minNeighbors=4):
    """检测图像中的对象（如人脸），并在检测到的对象周围绘制矩形框。

    参数:
        image (PIL.Image): 输入的图像。
        cascade_path (str): Haar 级联分类器文件路径。
        scaleFactor (float): 图像尺寸缩放比例。
        minNeighbors (int): 每个目标至少检测到的邻近目标数。

    返回:
        PIL.Image: 标注后的图像。
    """
    try:
        import os
        image_np = np.array(image)
        gray_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        cascade_full_path = os.path.join(cv2.data.haarcascades, cascade_path)
        if not os.path.exists(cascade_full_path):
            raise FileNotFoundError(f"Cascade file not found: {cascade_full_path}")
        cascade = cv2.CascadeClassifier(cascade_full_path)
        objects = cascade.detectMultiScale(gray_image, scaleFactor=scaleFactor, minNeighbors=minNeighbors)

        for (x, y, w, h) in objects:
            cv2.rectangle(image_np, (x, y), (x + w, y + h), (255, 0, 0), 2)

        return Image.fromarray(image_np)
    except Exception as e:
        print(f"Error in object_detection: {e}")
        return image

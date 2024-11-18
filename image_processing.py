import numpy as np
import cv2
import io
from PIL import Image, ImageDraw, ImageFont


# 图像灰度转换
def convert_to_grayscale(image):
    return image.convert("L")


# HSI增强
def enhance_hsi(image, hue_factor=1.0, saturation_factor=1.0, intensity_factor=1.0):
    image = np.array(image)
    hsi_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hsi_image = np.array(hsi_image, dtype=float)

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


# 直方图均衡化
def histogram_equalization(image):
    image = np.array(image)
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    equalized_image = cv2.equalizeHist(gray_image)
    return Image.fromarray(equalized_image)


# 颜色风格改变
def color_style_change(image, colormap=cv2.COLORMAP_JET):
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if colormap == "Jet":
        colormap = cv2.COLORMAP_JET
    elif colormap == "Hot":
        colormap = cv2.COLORMAP_HOT
    elif colormap == "Cool":
        colormap = cv2.COLORMAP_COOL
    elif colormap == "Rainbow":
        colormap = cv2.COLORMAP_RAINBOW
    colored_image = cv2.applyColorMap(image, colormap)
    colored_image = cv2.cvtColor(colored_image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(colored_image)


# 裁剪功能
def rectangle_crop(image, x, y, w, h):
    image = np.array(image)
    cropped_image = image[y:y + h, x:x + w]
    return Image.fromarray(cropped_image)


# 缩放
def zoom_image(image, scale_factor=1.5):
    image = np.array(image)
    height, width = image.shape[:2]
    new_dimensions = (int(width * scale_factor), int(height * scale_factor))
    zoomed_image = cv2.resize(image, new_dimensions, interpolation=cv2.INTER_LINEAR)
    return Image.fromarray(zoomed_image)


# 旋转
def rotate_image(image, angle=90):
    image = np.array(image)
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, matrix, (w, h))
    return Image.fromarray(rotated_image)


# 均值滤波
def mean_filter(image, kernel_size=3):
    image = np.array(image)
    return Image.fromarray(cv2.blur(image, (kernel_size, kernel_size)))

# 中值滤波
def median_filter(image, kernel_size=3):
    image = np.array(image)
    return Image.fromarray(cv2.medianBlur(image, kernel_size))

# 高斯滤波
def gaussian_filter(image, kernel_size=3):
    image = np.array(image)
    return Image.fromarray(cv2.GaussianBlur(image, (kernel_size, kernel_size), 0))

# 双边滤波
def bilateral_filter(image, diameter=9, sigma_color=75, sigma_space=75):
    image = np.array(image)
    return Image.fromarray(cv2.bilateralFilter(image, diameter, sigma_color, sigma_space))

# 引导滤波
def guided_filter(image, guide_image, radius=5, epsilon=0.1):
    image = np.array(image)
    guide_image = np.array(guide_image)
    return Image.fromarray(cv2.ximgproc.guidedFilter(guide_image, image, radius, epsilon))


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

# 添加噪声的函数
def add_noise(image, noise_type="gaussian", mean=0, sigma=25, amount=0.02, s_vs_p=0.5):
    image = np.array(image)
    noise_type = noise_type.lower()

    if noise_type == "gaussian":
        # 高斯噪声
        row, col, ch = image.shape
        gaussian_noise = np.random.normal(mean, sigma, (row, col, ch))
        noisy_image = np.uint8(np.clip(image + gaussian_noise, 0, 255))

    elif noise_type == "salt and pepper":
        # 盐和胡椒噪声
        out = np.copy(image)
        # Salt noise
        num_salt = np.ceil(amount * image.size * s_vs_p)
        salt_coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
        out[salt_coords[0], salt_coords[1], :] = 1

        # Pepper noise
        num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
        pepper_coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
        out[pepper_coords[0], pepper_coords[1], :] = 0

        noisy_image = out

    elif noise_type == "poisson":
        # 泊松噪声
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy_image = np.random.poisson(image * vals) / float(vals)
        noisy_image = np.uint8(np.clip(noisy_image, 0, 255))

    elif noise_type == "speckle":
        # 斑点噪声
        row, col, ch = image.shape
        speckle_noise = np.random.randn(row, col, ch)
        speckle_noise = speckle_noise.reshape(row, col, ch)
        noisy_image = np.uint8(np.clip(image + image * speckle_noise, 0, 255))

    return Image.fromarray(noisy_image)


# 添加水印
def add_watermark(image, watermark_text="Sample Watermark", position=(10, 10)):
    image = image.convert("RGBA")
    watermark_image = Image.new("RGBA", image.size, (0, 0, 0, 0))
    watermark_draw = ImageDraw.Draw(watermark_image)
    font = ImageFont.load_default()
    # 大小为原图的1/10
    font_size = int(image.size[1] / 10)
    font = ImageFont.truetype("arial.ttf", font_size)
    watermark_draw.text(position, watermark_text, font=font, fill=(255, 255, 255, 128))
    watermarked_image = Image.alpha_composite(image.convert("RGBA"), watermark_image)
    return watermarked_image.convert("RGB")


# 图像压缩
def compress_image(image, quality=50):
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    return Image.open(buffer)


# 边缘检测
def edge_detection(image, method="canny"):
    image = np.array(image)
    if method == "canny":
        edges = cv2.Canny(image, 100, 200)
    return Image.fromarray(edges)


# 背景去除
def remove_background(image):
    from rembg import remove
    image = np.array(image)
    output_image = remove(image)
    return Image.fromarray(output_image)


# 目标检测
def object_detection(image, cascade_path="haarcascade_frontalface_default.xml"):
    image = np.array(image)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + cascade_path)
    faces = cascade.detectMultiScale(gray_image, 1.1, 4)

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

    return Image.fromarray(image)

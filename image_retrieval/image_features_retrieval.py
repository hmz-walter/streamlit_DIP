import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

def extract_color_histogram(image, bins=(8, 12, 3), spatial_grid=(2, 2)):
    """
    提取图像的颜色直方图特征，包括局部颜色直方图。

    参数：
    - image: 输入的图像（BGR颜色空间）。
    - bins: 每个颜色通道的bin数，默认为HSV空间下的 (8, 12, 3)。
    - spatial_grid: 图像划分的网格数，默认为2x2。

    返回：
    - 综合的颜色特征向量。
    """
    # 转换颜色空间到HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # 全局颜色直方图
    global_hist = cv2.calcHist([hsv], [0, 1, 2], None, bins,
                               [0, 180, 0, 256, 0, 256])
    cv2.normalize(global_hist, global_hist)
    global_hist = global_hist.flatten()

    # 局部颜色直方图
    h_grid, w_grid = spatial_grid
    height, width = hsv.shape[:2]
    local_hists = []
    for i in range(h_grid):
        for j in range(w_grid):
            # 定义区域
            start_y = int(i * height / h_grid)
            end_y = int((i + 1) * height / h_grid)
            start_x = int(j * width / w_grid)
            end_x = int((j + 1) * width / w_grid)
            roi = hsv[start_y:end_y, start_x:end_x]
            hist = cv2.calcHist([roi], [0, 1, 2], None, bins,
                                [0, 180, 0, 256, 0, 256])
            cv2.normalize(hist, hist)
            local_hists.append(hist.flatten())

    # 组合全局和局部直方图
    feature_vector = np.hstack([global_hist] + local_hists)
    return feature_vector

def extract_glcm_features(image, distances=[1], angles=[0], levels=256, symmetric=True, normed=True):
    """
    提取图像的GLCM纹理特征。

    参数：
    - image: 输入的图像（BGR颜色空间）。
    - distances: 距离列表，用于GLCM计算。
    - angles: 角度列表，用于GLCM计算。
    - levels: 灰度级数。
    - symmetric: 是否对称。
    - normed: 是否归一化。

    返回：
    - GLCM特征向量（对比度、相关性、能量、同质性）。
    """
    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 计算GLCM
    glcm = graycomatrix(gray, distances=distances, angles=angles, levels=levels, symmetric=symmetric, normed=normed)
    # 提取特征
    contrast = graycoprops(glcm, 'contrast').flatten()
    correlation = graycoprops(glcm, 'correlation').flatten()
    energy = graycoprops(glcm, 'energy').flatten()
    homogeneity = graycoprops(glcm, 'homogeneity').flatten()
    # 组合特征
    glcm_features = np.hstack([contrast, correlation, energy, homogeneity])
    return glcm_features

def extract_lbp_features(image, num_points=24, radius=3, method='uniform'):
    """
    提取图像的LBP纹理特征。

    参数：
    - image: 输入的图像（BGR颜色空间）。
    - num_points: LBP的邻域点数。
    - radius: LBP的半径。
    - method: LBP的方法。

    返回：
    - LBP直方图特征向量。
    """
    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 计算LBP
    lbp = local_binary_pattern(gray, num_points, radius, method)
    # 计算LBP直方图
    (hist, _) = np.histogram(lbp.ravel(),
                             bins=np.arange(0, num_points + 3),
                             range=(0, num_points + 2))
    # 归一化直方图
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    return hist
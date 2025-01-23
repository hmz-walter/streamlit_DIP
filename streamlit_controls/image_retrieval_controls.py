# streamlit_controls/image_retrieval_controls.py
import os
import streamlit as st
from image_retrieval.image_features_retrieval import extract_color_histogram,extract_glcm_features,extract_lbp_features
import numpy as np
from sklearn.neighbors import KDTree
import cv2
from matplotlib import pyplot as plt

def load_image(img_path):
    # 打开文件并读取为 byte 数据
    with open(img_path, 'rb') as f:
        file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return img
# 前端界面选择要提取的图像的特征
def image_retrieval_controls(query_image_file,scrap_image_files, selected_features, bins=(8, 12, 3), spatial_grid=(2, 2),
                     glcm_params={'distances': [1], 'angles': [0], 'levels': 256, 'symmetric': True, 'normed': True},
                     lbp_params={'num_points': 24, 'radius': 3, 'method': 'uniform'}):
    # 1. 提取待检索图像的特征
    query_image=load_image(query_image_file)
    query_feature = extract_features(query_image, selected_features, bins, spatial_grid,glcm_params,lbp_params)
    # 2. 提取爬取的图像集的特征
    scrap_features = []
    for img_path in scrap_image_files:
        # print("正在处理: " + img_path)
        if not os.path.exists(img_path):
            print(f"图像路径不存在: {img_path}")
            continue

        img = load_image(img_path)
        if img is None:
            print(f"无法加载图像: {img_path}")
            continue

        # 确保图像不是空的
        if img.size == 0:
            print("图像为空: " + img_path)
            continue
        scrap_feature = extract_features(img, selected_features, bins, spatial_grid,glcm_params,lbp_params)
        scrap_features.append(scrap_feature)
    return query_feature,scrap_features

# 根据选择的特征提取图像的特征
def extract_features(image, selected_features, bins=(8, 12, 3), spatial_grid=(2,2),
                     glcm_params={'distances': [1], 'angles': [0], 'levels': 256, 'symmetric': True, 'normed': True},
                     lbp_params={'num_points': 24, 'radius': 3, 'method': 'uniform'}):
    features_vector = []
    # 根据选择的特征执行相应操作
    if "颜色直方图特征" in selected_features:
        color_features = extract_color_histogram(image, bins, spatial_grid)
        features_vector.append(color_features)

    if "GLCM纹理特征" in selected_features:
        glcm_features = extract_glcm_features(image, **glcm_params)
        features_vector.append(glcm_features)


    if "LBP纹理特征" in selected_features:
        lbp_features = extract_lbp_features(image, **lbp_params)
        features_vector.append(lbp_features)

    #融合特征
    if len(features_vector) > 1:
        features_vector = np.hstack(features_vector)
    elif len(features_vector) == 1:
        features_vector = np.ravel(features_vector[0])
    return features_vector


def build_kdtree(features):
    """
    使用KD-Tree构建特征索引。

    参数：
    - features: 特征向量列表。

    返回：
    - KDTree对象。
    """
    feature_array = np.array(features)
    tree = KDTree(feature_array, leaf_size=40, metric='euclidean')
    return tree

def search_image_kdtree(query_feature, tree, top_k=5):
    """
    使用KD-Tree根据查询特征检索最相似的图像。

    参数：
    - query_feature: 查询图像的特征向量。
    - tree: KDTree对象。
    - image_paths: 图像路径列表。
    - top_k: 返回最相似的前k个图像。

    返回：
    - indices: 检索结果的索引列表。
    - distances: 检索结果的距离列表。
    """
    dist, ind = tree.query([query_feature], k=top_k)
    return ind[0], dist[0]


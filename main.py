import streamlit as st
from PIL import Image
import numpy as np
import cv2
from io import BytesIO
from image_processing import (
    convert_to_grayscale,
    enhance_hsi,
    histogram_equalization,
    color_style_change,
    rectangle_crop,
    zoom_image,
    rotate_image,
    add_noise,
    add_watermark,
    compress_image,
    edge_detection,
    remove_background,
    object_detection,
)
from streamlit_control import noise_control, spatial_filter_control, frequency_filter_control

# 设置网页标题
st.title("图像处理工具集")


# 上传图像部分
def upload_image():
    uploaded_file = st.file_uploader("上传一张图片", type=["jpg", "jpeg", "png"])
    return uploaded_file


# 主界面和功能选择部分
st.sidebar.title("功能选择")
option = st.sidebar.selectbox("请选择功能", (
    "主页", "图像处理", "水印与压缩", "目标检测", "背景去除"))

if option == "主页":
    st.subheader("欢迎使用图像处理工具集！")
    st.markdown("""
    这是一个集成图像处理工具的应用，提供以下功能：
    - 图像灰度、颜色调整、裁剪、旋转等常规图像处理功能。
    - 高级功能：目标检测、背景去除、水印添加、图像压缩等。
    """)

elif option == "图像处理":
    uploaded_file = upload_image()
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="上传的图像", use_container_width=True)

        # 选择图像处理功能
        st.subheader("选择图像处理功能")

        # 图像灰度处理
        if st.checkbox("转换为灰度图"):
            grayscale_image = convert_to_grayscale(image)
            st.image(grayscale_image, caption="灰度图", use_container_width=True)

        # HSI增强
        if st.checkbox("增强HSI"):
            hue_factor = st.slider("色调增强", 0.0, 2.0, 1.0)
            saturation_factor = st.slider("饱和度增强", 0.0, 2.0, 1.0)
            intensity_factor = st.slider("亮度增强", 0.0, 2.0, 1.0)
            enhanced_image = enhance_hsi(image, hue_factor, saturation_factor, intensity_factor)
            st.image(enhanced_image, caption="HSI增强后的图像", use_container_width=True)

        # 直方图均衡化
        if st.checkbox("直方图均衡化"):
            equalized_image = histogram_equalization(image)
            st.image(equalized_image, caption="均衡化后的图像", use_container_width=True)

        # 颜色风格改变
        if st.checkbox("颜色风格改变"):
            colormap = st.selectbox("选择风格", ["Jet", "Hot", "Cool", "Rainbow"])
            styled_image = color_style_change(image, colormap=colormap)
            st.image(styled_image, caption=f"风格变化: {colormap}", use_container_width=True)

        # 图像裁剪
        if st.checkbox("裁剪图像"):
            st.subheader("裁剪图像")
            left = st.slider("左边界", 0, image.width, 0)
            top = st.slider("上边界", 0, image.height, 0)
            right = st.slider("右边界", 0, image.width, image.width)
            bottom = st.slider("下边界", 0, image.height, image.height)
            cropped_image = image.crop((left, top, right, bottom))
            st.image(cropped_image, caption="裁剪后的图像", use_container_width=True)
            # # 手动裁剪（输入框裁剪）
            # st.checkbox("启用手动裁剪", key="manual_crop")
            # # 鼠标拖拽裁剪
            # st.checkbox("启用拖拽裁剪", key="drag_crop")
        # 图像旋转
        if st.checkbox("旋转图像"):
            angle = st.slider("旋转角度", 0, 360, 0)
            rotated_image = rotate_image(image, angle)
            st.image(rotated_image, caption=f"旋转角度: {angle}", use_container_width=True)
        # 图像缩放
        if st.checkbox("缩放图像"):
            zoom_factor = st.slider("缩放比例", 0.1, 10.0, 1.0)
            zoomed_image = zoom_image(image, zoom_factor)
            st.image(zoomed_image, caption=f"缩放比例: {zoom_factor}", use_container_width=True)
        # 低通滤波
        if st.checkbox("空域滤波"):
            spatial_filter_control(image)
        # if st.checkbox("频域滤波"):
        #     frequency_filter_control(image)
        # 图像加噪声
        if st.checkbox("图像加噪"):
            noise_control(image)
        # 边缘检测
        if st.checkbox("边缘检测"):
            edge_image = edge_detection(image)
            st.image(edge_image, caption="边缘检测后的图像", use_container_width=True)

elif option == "水印与压缩":
    uploaded_file = upload_image()
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="上传的图像", use_container_width=True)

        # 水印添加
        watermark_text = st.text_input("水印文本", "Sample Watermark")
        if st.button("添加水印"):
            watermarked_image = add_watermark(image, watermark_text)
            st.image(watermarked_image, caption="添加水印后的图像", use_container_width=True)

        # 图像压缩
        quality = st.slider("压缩质量", 0, 100, 50)
        if st.button("压缩图像"):
            compressed_image = compress_image(image, quality)
            st.image(compressed_image, caption="压缩后的图像", use_container_width=True)

# elif option == "目标检测":
#     uploaded_file = upload_image()
#     if uploaded_file:
#         image = Image.open(uploaded_file)
#         st.image(image, caption="上传的图像", use_container_width=True)
#
#         if st.button("开始目标检测"):
#             detected_image = object_detection(image)
#             st.image(detected_image, caption="目标检测结果", use_container_width=True)

elif option == "背景去除":
    uploaded_file = upload_image()
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="上传的图像", use_container_width=True)

        if st.button("去除背景"):
            output_image = remove_background(image)
            st.image(output_image, caption="去除背景后的图像", use_container_width=True)

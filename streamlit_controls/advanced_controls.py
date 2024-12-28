# streamlit_controls/advanced_features_control.py

import streamlit as st
from image_processing.advanced_features import remove_background, edge_detection, add_watermark, compress_image

def advanced_control(image_controller):
    """集成边缘检测、水印与压缩、背景去除的高级功能控制界面。"""
    st.sidebar.subheader("高级功能")

    # 调用各个高级功能控制模块，传递 image_controller
    edge_detection_control(image_controller)
    watermark_compress_control(image_controller)
    background_removal_control(image_controller)

def edge_detection_control(image_controller):
    """控制边缘检测功能的界面，如Canny、Sobel、Laplacian。"""
    if st.sidebar.checkbox("边缘检测"):
        method = st.sidebar.selectbox("选择边缘检测方法", ["Canny", "Sobel", "Laplacian"])

        if method == "Canny":
            threshold1 = st.sidebar.slider("Threshold1", 0, 500, 100, 1)
            threshold2 = st.sidebar.slider("Threshold2", 0, 500, 200, 1)
            if st.sidebar.button("应用Canny边缘检测"):
                image_controller.apply_operation(
                    edge_detection, method="canny", threshold1=threshold1, threshold2=threshold2
                )
                st.success("Canny边缘检测已应用。")

        elif method == "Sobel":
            ksize = st.sidebar.slider("核大小 (ksize)", 1, 31, 3, 2)
            scale = st.sidebar.slider("尺度 (scale)", 1, 10, 1)
            delta = st.sidebar.slider("偏移量 (delta)", 0, 100, 0)
            if st.sidebar.button("应用Sobel边缘检测"):
                image_controller.apply_operation(
                    edge_detection, method="sobel", ksize=ksize, scale=scale, delta=delta
                )
                st.success("Sobel边缘检测已应用。")

        elif method == "Laplacian":
            ksize = st.sidebar.slider("核大小 (ksize)", 1, 31, 3, 2)
            scale = st.sidebar.slider("尺度 (scale)", 1, 10, 1)
            delta = st.sidebar.slider("偏移量 (delta)", 0, 100, 0)
            if st.sidebar.button("应用Laplacian边缘检测"):
                image_controller.apply_operation(
                    edge_detection, method="laplacian", ksize=ksize, scale=scale, delta=delta
                )
                st.success("Laplacian边缘检测已应用。")


def background_removal_control(image_controller):
    """控制背景去除的界面。"""
    if st.sidebar.checkbox("背景去除"):
        method = st.sidebar.selectbox("选择背景去除方法", ["颜色阈值", "深度学习模型"])

        if method == "颜色阈值":
            lower_hue = st.sidebar.slider("低色调 (Hue)", 0, 180, 0)
            upper_hue = st.sidebar.slider("高色调 (Hue)", 0, 180, 180)
            if st.sidebar.button("去除背景"):
                image_controller.apply_operation(
                    remove_background, method="color_threshold", lower_hue=lower_hue, upper_hue=upper_hue
                )
                st.success("背景已去除。")

        elif method == "深度学习模型":
            if st.sidebar.button("去除背景"):
                st.info("请等待，正在进行背景去除...")
                image_controller.apply_operation(remove_background, method="deep_learning")
                st.success("背景已去除。")

def watermark_compress_control(image_controller):
    """控制水印添加和图像压缩的界面。"""
    # st.sidebar.subheader("水印与压缩")

    # operation = st.sidebar.selectbox("选择操作", ["无", "添加水印", "图像压缩"])

    if st.sidebar.checkbox("添加水印"):
        st.sidebar.markdown("**水印参数**")
        watermark_text = st.sidebar.text_input("水印文本", "Sample Watermark")
        font_size = st.sidebar.slider("字体大小", 10, 100, 20, 1)
        color = st.sidebar.color_picker("字体颜色", "#FFFFFF")
        transparency = st.sidebar.slider("透明度", 0, 255, 128, 1)
        position = st.sidebar.selectbox("水印位置", ["左上", "右上", "左下", "右下", "中心"])

        # 解析颜色，添加透明度
        color_rgb = tuple(int(color[i:i + 2], 16) for i in (1, 3, 5)) + (int(transparency),)  # RGBA

        # 获取当前图像尺寸
        current_image = image_controller.get_current_image()
        image_width, image_height = current_image.size

        # 定义水印位置坐标
        pos_dict = {
            "左上": (10, 10),
            "右上": (image_width - 100, 10),
            "左下": (10, image_height - 100),
            "右下": (image_width - 100, image_height - 100),
            "中心": (image_width // 2 - 100, image_height // 2 - 100)
        }
        selected_position = pos_dict.get(position, (10, 10))

        if st.sidebar.button("应用添加水印"):
            # description = f"添加水印: '{watermark_text}' 在 {position}, 字体大小={font_size}, 颜色={color}"
            image_controller.apply_operation(
                add_watermark,
                watermark_text=watermark_text,
                position=selected_position,
                font_size=font_size,
                color=color_rgb
            )
            st.success("水印已添加。")
            # st.image(image_controller.get_current_image(), caption="添加水印后的图像", use_container_width=True)

    if st.sidebar.checkbox("图像压缩"):
        st.sidebar.markdown("**图像压缩参数**")
        quality = st.sidebar.slider("压缩质量", 0, 100, 50, 1)
        format = st.sidebar.selectbox("选择保存格式", ["JPEG"])

        if st.sidebar.button("应用图像压缩"):
            # description = f"图像压缩: 格式={format}, 质量={quality}"
            image_controller.apply_operation(
                compress_image,
                quality=quality,
                format=format
            )
            st.success(f"图像已压缩，格式={format}, 质量={quality}。")
            # st.image(
            #     image_controller.get_current_image(),
            #     caption=f"压缩后的图像 (格式: {format}, 质量: {quality})",
            #     use_container_width=True
            # )

# streamlit_controls/basic_controls.py

import streamlit as st
from PIL import Image
from image_processing.basic_operations import convert_to_grayscale, convert_to_color, color_style_change, enhance_hsi

def basic_operations_control(image_controller):
    st.sidebar.subheader("色彩操作")
    # operation = st.sidebar.selectbox("选择色彩操作", ["无", "转换为灰度", "颜色风格更改", "HSI增强"])
    if st.sidebar.checkbox("转换为灰度"):
        if st.sidebar.button("应用转换为灰度"):
            if image_controller.apply_operation(convert_to_grayscale):
                st.success("已转换为灰度。")

    if st.sidebar.checkbox("颜色风格更改"):
        colormap = st.sidebar.selectbox("选择风格", ["Jet", "Hot", "Cool", "Rainbow", "Bone", "Winter", "Summer", "Autumn", "Spring", "Ocean", "Pink", "HSV"])
        if st.sidebar.button("应用颜色风格更改"):
            if image_controller.apply_operation(color_style_change, colormap=colormap):
                st.success("颜色风格已更改。")

    if st.sidebar.checkbox("HSI增强"):
        hue_factor = st.sidebar.slider("色调增强", 0.0, 2.0, 1.0)
        saturation_factor = st.sidebar.slider("饱和度增强", 0.0, 2.0, 1.0)
        intensity_factor = st.sidebar.slider("亮度增强", 0.0, 2.0, 1.0)

        st.session_state.hsi_params = {"hue": hue_factor, "saturation": saturation_factor, "intensity": intensity_factor}
        # 生成预览图像
        current_image = image_controller.get_current_image()
        preview_crop = current_image.copy()
        preview_crop = enhance_hsi(preview_crop, hue_factor, saturation_factor, intensity_factor)
        st.sidebar.image(preview_crop, caption="HSI增强预览", use_container_width=True)

        if st.sidebar.button("应用HSI增强"):
            if image_controller.apply_operation(enhance_hsi, hue_factor, saturation_factor, intensity_factor):
                st.success("HSI增强已应用。")

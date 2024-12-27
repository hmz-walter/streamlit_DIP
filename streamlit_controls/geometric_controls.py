# streamlit_controls/geometric_controls.py

import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw
from image_processing.geometric_transformations import rotate_image, scale_image, rectangle_crop, shear_image, translate_image, mirror_image
from image_processing.utils import validate_crop_coordinates

def geometric_operations_control(image_controller):
    """控制几何变换的界面，如裁剪、旋转、缩放等。"""
    st.sidebar.subheader("几何变换")

    # 存储预览图像
    current_image = image_controller.get_current_image()

    interpolation = st.sidebar.selectbox("插值方法", ["最近邻", "双线性", "三次样条"], index=1)
    interp_val = cv2.INTER_NEAREST if interpolation == "最近邻" \
        else cv2.INTER_LINEAR if interpolation == "双线性" else cv2.INTER_CUBIC

    if st.sidebar.checkbox("裁剪"):
        if current_image is None:
            st.warning("请先上传一张图像。")
            return
        image_width, image_height = image_controller.get_current_image().size
        # 使用会话状态存储裁剪参数
        if "crop_params" not in st.session_state:
            st.session_state.crop_params = {
                "left": 0,
                "top": 0,
                "right": image_width,
                "bottom": image_height
            }
        left = st.sidebar.slider("左边界", 0, image_width - 1, 0)
        top = st.sidebar.slider("上边界", 0, image_height - 1, 0)
        right = st.sidebar.slider("右边界", left + 1, image_width, image_width)
        bottom = st.sidebar.slider("下边界", top + 1, image_height, image_height)
        # 更新会话状态
        st.session_state.crop_params = {"left": left, "top": top, "right": right, "bottom": bottom}
        # 生成预览图像
        preview_crop = current_image.copy()
        draw = ImageDraw.Draw(preview_crop)
        draw.rectangle([left, top, right, bottom], outline="red", width=3)
        st.sidebar.image(preview_crop, caption="裁剪预览", use_container_width=True)

        if st.sidebar.button("应用裁剪"):
            if validate_crop_coordinates(left, top, right, bottom, image_width, image_height):
                if image_controller.apply_operation(rectangle_crop, left=left, top=top, right=right, bottom=bottom):
                    st.success(f"裁剪已应用: 左={left}, 上={top}, 右={right}, 下={bottom}。")
                    # 清除裁剪参数
                    st.session_state.crop_params = {
                        "left": 0,
                        "top": 0,
                        "right": image_width,
                        "bottom": image_height
                    }
            else:
                st.sidebar.error("裁剪坐标无效，请重新设置。")

    if st.sidebar.checkbox("旋转"):
        if current_image is None:
            st.warning("请先上传一张图像。")
            return
        # 使用会话状态存储旋转参数
        if "rotate_angle" not in st.session_state:
            st.session_state.rotate_angle = 0
        if "rotate_expand" not in st.session_state:
            st.session_state.rotate_expand = True
        angle = st.sidebar.slider("旋转角度", 0, 360, st.session_state.rotate_angle, 1)
        expand = st.sidebar.checkbox("扩展图像尺寸以适应旋转", value=st.session_state.rotate_expand)
        # # 更新会话状态
        # st.session_state.rotate_angle = angle
        # st.session_state.rotate_expand = expand
        # 生成预览图像
        preview_rotate = rotate_image(current_image, angle=angle, interpolation=interp_val, expand=expand)
        st.sidebar.image(preview_rotate, caption=f"旋转预览: {angle}°", use_container_width=True)

        if st.sidebar.button("应用旋转"):
            if image_controller.apply_operation(rotate_image, angle=angle, interpolation=interp_val, expand=expand):
                st.success(f"旋转已应用，角度={angle}°。")
                # 重置旋转参数
                st.session_state.rotate_angle = 0
                st.session_state.rotate_expand = True

    if st.sidebar.checkbox("缩放"):
        if current_image is None:
            st.warning("请先上传一张图像。")
            return
        # 使用会话状态存储缩放参数
        if "scale_factor" not in st.session_state:
            st.session_state.scale_factor = 1.0
        scale_factor = st.sidebar.slider("缩放比例", 0.1, 5.0, st.session_state.scale_factor, 0.1)
        # 更新会话状态
        # st.session_state.scale_factor = scale_factor
        # 生成预览图像
        preview_scale = scale_image(current_image, scale_factor=scale_factor, interpolation=interp_val)
        st.sidebar.image(preview_scale, caption=f"缩放预览: {scale_factor}x", use_container_width=True)

        if st.sidebar.button("应用缩放"):
            if image_controller.apply_operation(scale_image, scale_factor=scale_factor, interpolation=interp_val):
                st.success(f"缩放已应用，比例={scale_factor}x。")
                # st.image(image_controller.get_current_image(), caption=f"缩放比例: {scale_factor}x", use_container_width=True)
                # 重置缩放参数
                st.session_state.scale_factor = 1.0

    if st.sidebar.checkbox("剪切"):
        if current_image is None:
            st.warning("请先上传一张图像。")
            return
        # 使用会话状态存储剪切参数
        if "shear_factor" not in st.session_state:
            st.session_state.shear_factor = 0.0
        shear_factor = st.sidebar.slider("剪切因子", -1.0, 1.0, st.session_state.shear_factor, 0.1)
        # 更新会话状态
        # st.session_state.shear_factor = shear_factor
        # 生成预览图像
        preview_shear = shear_image(current_image, shear_factor=shear_factor, interpolation=interp_val)
        st.sidebar.image(preview_shear, caption=f"剪切预览: 因子={shear_factor}", use_container_width=True)

        if st.sidebar.button("应用剪切"):
            if image_controller.apply_operation(shear_image, shear_factor=shear_factor, interpolation=interp_val):
                st.success(f"剪切已应用，因子={shear_factor}。")
                # 重置剪切参数
                st.session_state.shear_factor = 0.0

    if st.sidebar.checkbox("平移"):
        if current_image is None:
            st.warning("请先上传一张图像。")
            return
        image_width, image_height = current_image.size
        # 使用会话状态存储平移参数
        if "translate_tx" not in st.session_state:
            st.session_state.translate_tx = 0
        if "translate_ty" not in st.session_state:
            st.session_state.translate_ty = 0
        tx = st.sidebar.slider("水平平移量", -image_width // 2, image_width // 2, st.session_state.translate_tx, 1)
        ty = st.sidebar.slider("垂直平移量", -image_height // 2, image_height // 2, st.session_state.translate_ty, 1)
        # 更新会话状态
        # st.session_state.translate_tx = tx
        # st.session_state.translate_ty = ty
        # 生成预览图像
        preview_translate = translate_image(current_image, tx=tx, ty=ty, interpolation=interp_val)
        st.sidebar.image(preview_translate, caption=f"平移预览: ({tx}, {ty})", use_container_width=True)

        if st.sidebar.button("应用平移"):
            if image_controller.apply_operation(translate_image, tx=tx, ty=ty, interpolation=interp_val):
                st.success(f"平移已应用，水平={tx}, 垂直={ty}。")
                # 重置平移参数
                st.session_state.translate_tx = 0
                st.session_state.translate_ty = 0

    if st.sidebar.checkbox("镜像"):
        if current_image is None:
            st.warning("请先上传一张图像。")
            return
        mode = st.sidebar.radio("选择镜像方向", ("水平", "垂直"))
        mode_val = 'horizontal' if mode == "水平" else 'vertical'
        if st.sidebar.button("应用镜像"):
            if image_controller.apply_operation(mirror_image, mode=mode_val):
                st.success(f"镜像已应用，方向={mode}。")
                # st.image(image_controller.get_current_image(), caption=f"镜像: {mode}", use_container_width=True)

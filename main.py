# main.py

import streamlit as st
from PIL import Image
from io import BytesIO

from streamlit_controls.basic_controls import basic_operations_control
from streamlit_controls.geometric_controls import geometric_operations_control
from streamlit_controls.histogram_controls import histogram_operations_control
from streamlit_controls.filters_noise_controls import filters_noise_control
# from streamlit_controls.frequency_transform_controls import frequency_transform_control
from streamlit_controls.advanced_controls import advanced_control
from streamlit_controls.total_controls import ImageController, history_control, image_to_bytes

# 初始化 ImageController
is_initialized = False

# 设置网页标题和配置
st.set_page_config(page_title="图像处理工具集", layout="wide")
st.title("图像处理工具集")

if not is_initialized:
    st.session_state.image_controller = ImageController()
    st.session_state.is_upload = False  # 上传状态
    is_initialized = True
    image_controller = st.session_state.image_controller

# 上传图像部分
def upload_image():
    uploaded_file = st.file_uploader("上传一张图片 (支持 BMP, JPG, JPEG, PNG)", type=["bmp", "jpg", "jpeg", "png"])
    if uploaded_file is None:
        image_controller.delete_image()
    if uploaded_file is not None and image_controller.get_current_image() is None:
        image = Image.open(uploaded_file).convert("RGB")
        image_controller.load_image(image)
        st.session_state.is_upload = True
    return uploaded_file

# 图像保存功能
def save_image():
    if image_controller.get_current_image():
        save_format = st.selectbox("选择保存格式", ["JPEG", "PNG", "BMP"], key="save_format_selectbox")
        buffer = BytesIO()
        if save_format.upper() == "JPEG":
            image_controller.get_current_image().convert("RGB").save(buffer, format=save_format, quality=95)
        else:
            image_controller.get_current_image().save(buffer, format=save_format)
        buffer.seek(0)
        st.download_button(
            label="下载图像",
            data=buffer,
            file_name=f"processed_image.{save_format.lower()}",
            mime=f"image/{save_format.lower()}"
        )
    else:
        st.warning("请先上传并处理一张图像。")

# 图像删除功能
def delete_image():
    if image_controller.get_current_image():
        if st.button("删除当前图像"):
            image_controller.delete_image()
            st.session_state.is_upload = False  # 更新上传状态
            st.rerun()
    else:
        st.warning("当前没有图像可删除。")

# 主界面和功能选择部分
st.sidebar.title("功能选择")
option = st.sidebar.selectbox("请选择功能", (
    "主页",
    "图像处理",
    "图像检索",
    # "保存图像",
    # "删除图像",
    # "像素值显示"
))

if option == "主页":
    st.subheader("欢迎使用图像处理工具集！")
    st.markdown("""
    这是一个集成图像处理工具的应用，提供以下功能：
    - **图像处理**：选择并应用不同的图像处理模块。
    - **图像检索**：待实现的图像检索功能。
    """)
    # - ** 保存图像 **：保存处理后的图像。
    # - ** 删除图像 **：删除当前处理的图像。

elif option == "图像处理":
    uploaded_file = upload_image()

    # 图像处理模块
    basic_operations_control(image_controller)
    geometric_operations_control(image_controller)
    histogram_operations_control(image_controller)
    filters_noise_control(image_controller)
    # frequency_transform_control(image_controller)
    advanced_control(image_controller)

    # 历史操作、保存和删除图像功能
    if uploaded_file:
        history_control(image_controller)
        image_controller.show_image()
        save_image()
        # delete_image()

elif option == "图像检索":
    st.subheader("图像检索功能待实现")
    st.markdown("此功能正在开发中，敬请期待。")

    # # 图像处理模块
    # basic_operations_control(image_controller)
    # geometric_operations_control(image_controller)
    # histogram_operations_control(image_controller)
    # filters_noise_control(image_controller)
    # advanced_control(image_controller)
    #
    # history_control(image_controller)
    # image_controller.show_image()

    # if image_controller.get_original_image() is not None:
    #     col1, col2 = st.columns([2, 2])
    #     with col1:
    #         st.subheader("上传的图像")
    #         st.image(image_controller.get_original_image(), use_container_width=True)
    #
    #     with col2:
    #         st.subheader("处理后的图像")
    #         current_image = image_controller.get_current_image()
    #         if current_image is not None:
    #             st.image(current_image, use_container_width=True)
    #         else:
    #             st.info("请进行图像处理操作。")

    # upload_image()
    #
    # if image_controller.get_original_image():
    #     col1, col2 = st.columns([1, 3])
    #     with col1:
    #         st.subheader("上传的图像")
    #         st.image(image_controller.get_original_image(), use_column_width=True)
    #
    #     with col2:
    #         st.subheader("处理后的图像")
    #         if image_controller.get_current_image():
    #             st.image(image_controller.get_current_image(), use_column_width=True)
    #         else:
    #             st.info("请进行图像处理操作。")
    #
    # if image_controller.get_current_image():
    #     st.sidebar.subheader("图像处理工具")
    #     # 图像处理模块
    #     basic_operations_control(image_controller)
    #     geometric_operations_control(image_controller)
    #     histogram_operations_control(image_controller)
    #     filters_noise_control(image_controller)
    #     advanced_control(image_controller)
    #
    #     st.markdown("### 操作历史")
    #     history_control(image_controller)

# elif option == "保存图像":
#     if image_controller.get_current_image():
#         save_format = st.selectbox("选择保存格式", ["JPEG", "PNG", "BMP"])
#         buffer = BytesIO()
#         if save_format.upper() == "JPEG":
#             image_controller.get_current_image().save(buffer, format=save_format, quality=95)
#         else:
#             image_controller.get_current_image().save(buffer, format=save_format)
#         buffer.seek(0)
#         st.download_button(
#             label="下载图像",
#             data=buffer,
#             file_name=f"processed_image.{save_format.lower()}",
#             mime=f"image/{save_format.lower()}"
#         )
#     else:
#         st.warning("请先上传并处理一张图像。")
#
# elif option == "删除图像":
#     if image_controller.get_current_image():
#         if st.button("删除当前图像"):
#             image_controller.delete()
#             st.success("图像已删除。")
#     else:
#         st.warning("当前没有图像可删除。")

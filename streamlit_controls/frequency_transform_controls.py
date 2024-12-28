# # streamlit_controls/frequency_transform_controls.py
#
# import streamlit as st
# import numpy as np
# from matplotlib import pyplot as plt
# from PIL import Image
# from image_processing.frequency_transformations import (
#     apply_fft, apply_dct, apply_wavelet, inverse_fft, inverse_dct, inverse_wavelet
# )
#
# def frequency_transform_control(image_controller):
#     """控制频域变换的界面，如FFT、DCT、小波变换等。"""
#     st.sidebar.subheader("频域变换")
#
#     current_image = image_controller.get_current_image()
#     if current_image is None:
#         st.sidebar.info("请先上传一张图像以使用频域变换操作。")
#         return
#
#     transform_type = st.sidebar.selectbox("选择变换类型", ["傅里叶变换 (FFT)", "离散余弦变换 (DCT)", "小波变换"])
#
#     if transform_type == "傅里叶变换 (FFT)":
#         if st.sidebar.button("应用傅里叶变换"):
#             transformed, magnitude_spectrum = apply_fft(current_image)
#             image_controller.set_transformed_image(transformed)
#             st.image(transformed, caption="傅里叶变换后的图像", use_container_width=True)
#             st.image(magnitude_spectrum, caption="幅度谱", use_container_width=True)
#
#         if st.sidebar.button("应用逆傅里叶变换"):
#             transformed_image = image_controller.get_transformed_image()
#             if transformed_image is not None:
#                 inverse_image = inverse_fft(transformed_image)
#                 image_controller.set_current_image(inverse_image)
#                 st.image(inverse_image, caption="逆傅里叶变换后的图像", use_container_width=True)
#             else:
#                 st.sidebar.info("请先进行傅里叶变换。")
#
#     elif transform_type == "离散余弦变换 (DCT)":
#         if st.sidebar.button("应用离散余弦变换"):
#             transformed, dct_spectrum = apply_dct(current_image)
#             image_controller.set_transformed_image(transformed)
#             st.image(transformed, caption="离散余弦变换后的图像", use_container_width=True)
#             st.image(dct_spectrum, caption="DCT幅度谱", use_container_width=True)
#
#         if st.sidebar.button("应用逆离散余弦变换"):
#             transformed_image = image_controller.get_transformed_image()
#             if transformed_image is not None:
#                 inverse_image = inverse_dct(transformed_image)
#                 image_controller.set_current_image(inverse_image)
#                 st.image(inverse_image, caption="逆离散余弦变换后的图像", use_container_width=True)
#             else:
#                 st.sidebar.info("请先进行离散余弦变换。")
#
#     elif transform_type == "小波变换":
#         wavelet_type = st.sidebar.selectbox("选择小波类型", ["haar", "db1", "db2", "sym2"])
#         level = st.sidebar.slider("分解层数", 1, 5, 2)
#
#         if st.sidebar.button("应用小波变换"):
#             transformed, wavelet_spectrum = apply_wavelet(current_image, wavelet_type, level)
#             image_controller.set_transformed_image(transformed)
#             st.image(transformed, caption="小波变换后的图像", use_container_width=True)
#             st.image(wavelet_spectrum, caption="小波幅度谱", use_container_width=True)
#
#         if st.sidebar.button("应用逆小波变换"):
#             transformed_image = image_controller.get_transformed_image()
#             if transformed_image is not None:
#                 inverse_image = inverse_wavelet(transformed_image, wavelet_type, level)
#                 image_controller.set_current_image(inverse_image)
#                 st.image(inverse_image, caption="逆小波变换后的图像", use_container_width=True)
#             else:
#                 st.sidebar.info("请先进行小波变换。")

# streamlit_controls/histogram_controls.py

import streamlit as st
import numpy as np
from image_processing.histogram_operations import compute_histogram, histogram_equalization
import matplotlib.pyplot as plt

def histogram_operations_control(image_controller):
    """控制直方图操作的界面，如直方图均衡化、直方图显示等。"""
    st.sidebar.subheader("直方图操作")

    current_image = image_controller.get_current_image()

    # 使用展开器组织界面
    with st.sidebar.expander("显示直方图", expanded=False):
        if st.button("计算并显示直方图"):
            if current_image is None:
                st.info("请先上传一张图像以使用直方图操作。")
                return
            hist = compute_histogram(current_image)
            if hist:
                st.subheader("灰度直方图")
                fig, ax = plt.subplots()
                bin_edges = np.arange(256)  # 离散的灰度值
                ax.bar(bin_edges, hist, width=1.0, color='gray')
                ax.set_title('Histogram of Grayscale Image')
                ax.set_xlabel('Gray Level')
                ax.set_ylabel('Frequency')
                ax.set_xlim([0, 255])
                st.pyplot(fig)
                st.success("直方图已计算并显示。")
            else:
                st.error("直方图计算失败。")

    with st.sidebar.expander("直方图均衡化", expanded=False):
        if st.button("应用直方图均衡化"):
            if current_image is None:
                st.info("请先上传一张图像以使用直方图操作。")
                return
            equalized_image = histogram_equalization(current_image)
            if equalized_image:
                image_controller.apply_operation(lambda _: equalized_image)
                st.success("直方图均衡化已应用。")
                # 自动显示均衡化后的直方图
                hist = compute_histogram(equalized_image)
                if hist:
                    st.subheader("均衡化后的灰度直方图")
                    fig, ax = plt.subplots()
                    bin_edges = np.arange(256)
                    ax.bar(bin_edges, hist, width=1.0, color='gray')
                    ax.set_title('Histogram Equalization')
                    ax.set_xlabel('Gray Level')
                    ax.set_ylabel('Frequency')
                    ax.set_xlim([0, 255])
                    st.pyplot(fig)
                    st.success("均衡化后的直方图已计算并显示。")
                else:
                    st.error("均衡化后的直方图计算失败。")
            else:
                st.error("直方图均衡化失败。")

    # 显示当前图像的直方图（可选）
    # 可以选择始终显示当前图像的直方图，或者提供一个复选框来控制是否显示
    # if st.sidebar.checkbox("显示当前图像的直方图"):
    #     hist = compute_histogram(current_image)
    #     if hist:
    #         st.subheader("当前图像的灰度直方图")
    #         fig, ax = plt.subplots()
    #         bin_edges = np.arange(256)
    #         ax.bar(bin_edges, hist, width=1.0, color='gray')
    #         ax.set_title('Histogram of Current Image')
    #         ax.set_xlabel('Gray Level')
    #         ax.set_ylabel('Frequency')
    #         ax.set_xlim([0, 255])
    #         st.pyplot(fig)
    #     else:
    #         st.error("当前图像的直方图计算失败。")
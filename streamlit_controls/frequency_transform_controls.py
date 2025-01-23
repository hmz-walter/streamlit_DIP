# streamlit_controls/frequency_transform_controls.py

import streamlit as st
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from image_processing.frequency_transformations import (
    apply_fft, apply_dct, apply_wavelet, inverse_fft, inverse_dct, inverse_wavelet
)

def frequency_transform_control(image_controller):
    """控制频域变换的界面，如FFT、DCT、小波变换等。"""
    st.sidebar.subheader("频域变换")

    current_image = image_controller.get_current_image()
    if current_image is None:
        st.sidebar.info("请先上传一张图像以使用频域变换操作。")
        return

    # transform_type = st.sidebar.selectbox("选择变换类型", ["傅里叶变换 (FFT)", "离散余弦变换 (DCT)", "小波变换"])

    # if transform_type == "傅里叶变换 (FFT)":
    if st.sidebar.checkbox("傅里叶变换 (FFT)"):
        if st.sidebar.button("应用傅里叶变换"):
            magnitude_spectrum, phase_spectrum = apply_fft(current_image)
            image_controller.set_transform_coeffs("fft", {
                "magnitude": magnitude_spectrum,
                "phase": phase_spectrum
            })

            magnitude_spectrum, phase_spectrum = np.array(magnitude_spectrum), np.array(phase_spectrum)
            magnitude_spectrum = 20 * np.log(np.abs(magnitude_spectrum) + 1)
            # 归一化幅度谱到0-255
            magnitude_spectrum = np.uint8(255 * magnitude_spectrum / np.max(magnitude_spectrum))
            phase_spectrum = np.uint8(255 * phase_spectrum / np.max(phase_spectrum))

            col1, col2 = st.columns([2, 2])
            with col1:
                st.image(magnitude_spectrum, caption="幅度谱", use_container_width=True)
            with col2:
                st.image(phase_spectrum, caption="相位谱", use_container_width=True)

        if st.sidebar.button("应用逆傅里叶变换"):
            fft_coeffs = image_controller.get_transform_coeffs("fft")
            if fft_coeffs is None:
                st.sidebar.info("请先进行傅里叶变换。")
                return
            reconstructed_image1 = inverse_fft(
                magnitude=fft_coeffs["magnitude"],
                phase=None
            )
            reconstructed_image2 = inverse_fft(
                magnitude=None,
                phase=fft_coeffs["phase"]
            )
            reconstructed_image3 = inverse_fft(
                magnitude=fft_coeffs["magnitude"],
                phase=fft_coeffs["phase"]
            )
            col1, col2, col3 = st.columns([2, 2, 2])
            with col1:
                st.image(reconstructed_image1, caption="仅幅度谱重建的图像", use_container_width=True)
            with col2:
                st.image(reconstructed_image2, caption="仅相位谱重建的图像", use_container_width=True)
            with col3:
                st.image(reconstructed_image3, caption="幅度谱 + 相位谱重建的图像", use_container_width=True)

    # elif transform_type == "离散余弦变换 (DCT)":
    if st.sidebar.checkbox("离散余弦变换 (DCT)"):
        threshold = st.sidebar.slider("逆变换阈值", 0, 500, 100, 1)
        if st.sidebar.button("应用离散余弦变换与逆变换"):
            dct_spectrum = apply_dct(current_image)
            inverse_image = inverse_dct(dct_spectrum, threshold)
            # image_controller.set_transform_coeffs("dct", dct_spectrum)

            dct_spectrum = 20 * np.log(np.abs(dct_spectrum) + 1)
            dct_spectrum = np.uint8(255 * dct_spectrum / np.max(dct_spectrum))
            col1, col2 = st.columns([2, 2])
            with col1:
                st.image(dct_spectrum, caption="DCT幅度谱", use_container_width=True)
            with col2:
                st.image(inverse_image, caption="逆离散余弦变换后的图像", use_container_width=True)

            # dct_spectrum = image_controller.get_transform_coeffs("dct")

            # if dct_spectrum is None:
            #     st.sidebar.info("请先进行离散余弦变换。")
            #     return

    # elif transform_type == "小波变换":
    if st.sidebar.checkbox("小波变换（WT）"):
        wavelet_type = st.sidebar.selectbox("选择小波类型", ["haar", "db1", "db2", "sym2"])
        level = st.sidebar.slider("分解层数", 1, 5, 2)
        selected_coeffs = {}
        # coeffs list: [cA_n, (cH_n, cV_n, cD_n), ..., (cH1, cV1, cD1)]
        for i in range(1, level + 1):
            st.sidebar.markdown(f"### Level {i}")
            cH = st.sidebar.checkbox(f"保留 cH_{i}", value=True)
            cV = st.sidebar.checkbox(f"保留 cV_{i}", value=True)
            cD = st.sidebar.checkbox(f"保留 cD_{i}", value=True)
            selected_coeffs[i] = {'cH': cH, 'cV': cV, 'cD': cD}

        if st.sidebar.button("应用小波变换与逆变换"):
            wavelet_spectrum = apply_wavelet(current_image, wavelet_type, level)
            image_controller.set_transform_coeffs("wavelet", {
                "coeffs": wavelet_spectrum["coeffs"],
                "slices": wavelet_spectrum["slices"],
                "wavelet": wavelet_type,
                "level": level
            })
            wavelet_spectrum_show = wavelet_spectrum["spectrum"].copy()
            wavelet_spectrum_show = np.array(wavelet_spectrum_show)
            wavelet_spectrum_show = np.uint8(255 * wavelet_spectrum_show / np.max(wavelet_spectrum_show))

            # st.image(transformed_image, caption="小波变换后的图像", use_container_width=False)
            # wavelet_coeffs = image_controller.get_transform_coeffs("wavelet")
            # if wavelet_coeffs is None:
            #     st.sidebar.info("请先进行小波变换。")
            #     return

            inverse_image = inverse_wavelet(
                coeffs=wavelet_spectrum["coeffs"],
                slices=wavelet_spectrum["slices"],
                wavelet=wavelet_type,
                level=level,
                selected_coeffs=selected_coeffs
            )

            col1, col2 = st.columns([2, 2])
            with col1:
                st.image(wavelet_spectrum_show, caption="小波幅度谱", use_container_width=True)
            with col2:
                st.image(inverse_image, caption="逆小波变换后的图像", use_container_width=True)

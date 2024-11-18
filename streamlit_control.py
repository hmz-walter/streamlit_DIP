import streamlit as st
from PIL import Image
from image_processing import (
    add_noise,
    mean_filter, median_filter, gaussian_filter, bilateral_filter, guided_filter,
    low_pass_filter, high_pass_filter)

# Streamlit 控件，允许用户控制噪声类型和参数
def noise_control(image):
    noise_type = st.selectbox("选择噪声类型", ("gaussian", "salt and pepper")) #, "poisson", "speckle"
    noise_type = noise_type.lower()
    mean, sigma, amount, s_vs_p = 0, 0, 0, 0

    # 控制噪声参数
    if noise_type == "gaussian":
        mean = st.slider("选择均值 (mean)", -50, 50, 0)
        sigma = st.slider("选择标准差 (sigma)", 1, 100, 25)

    elif noise_type == "salt and pepper":
        amount = st.slider("选择噪声比例 (amount)", 0.0, 0.1, 0.02)
        s_vs_p = st.slider("选择盐与胡椒噪声比例 (salt vs pepper)", 0.0, 1.0, 0.5)

    elif noise_type == "poisson":
        # 泊松噪声没有直接的可调参数，可以加入其他参数来影响
        pass

    elif noise_type == "speckle":
        pass

    # 添加噪声并显示
    noisy_image = add_noise(image, noise_type=noise_type, mean=mean, sigma=sigma, amount=amount, s_vs_p=s_vs_p)
    st.image(noisy_image, caption=f"添加{noise_type}噪声后的图像", use_container_width=True)

# 空域滤波控制
def spatial_filter_control(image):
    filter_type = st.selectbox("选择滤波类型", ("均值滤波", "中值滤波", "高斯滤波"))

    if filter_type == "均值滤波":
        kernel_size = st.slider("滤波器大小", 1, 51, 3)
        filtered_image = mean_filter(image, kernel_size)
        st.image(filtered_image, caption=f"均值滤波器大小: {kernel_size}", use_container_width=True)

    elif filter_type == "中值滤波":
        kernel_size = st.slider("滤波器大小", 1, 51, 3)
        filtered_image = median_filter(image, kernel_size)
        st.image(filtered_image, caption=f"中值滤波器大小: {kernel_size}", use_container_width=True)

    elif filter_type == "高斯滤波":
        kernel_size = st.slider("滤波器大小", 1, 51, 3)
        filtered_image = gaussian_filter(image, kernel_size)
        st.image(filtered_image, caption=f"高斯滤波器大小: {kernel_size}", use_container_width=True)

    elif filter_type == "双边滤波":
        diameter = st.slider("直径 (diameter)", 1, 25, 9)
        sigma_color = st.slider("颜色空间的标准差 (sigma_color)", 1, 150, 75)
        sigma_space = st.slider("坐标空间的标准差 (sigma_space)", 1, 150, 75)
        filtered_image = bilateral_filter(image, diameter, sigma_color, sigma_space)
        st.image(filtered_image,
                 caption=f"双边滤波：直径={diameter}, sigma_color={sigma_color}, sigma_space={sigma_space}",
                 use_container_width=True)
    # elif filter_type == "引导滤波":
    #     radius = st.slider("半径 (radius)", 1, 25, 5)
    #     eps = st.slider("eps", 0.01, 1.0, 0.1)
    #     filtered_image = guided_filter(image, image, radius, eps)
    #     st.image(filtered_image, caption=f"引到滤波：半径={radius}, eps={eps}", use_container_width=True)


# 频域滤波控制
def frequency_filter_control(image):
    filter_type = st.selectbox("选择滤波类型", ("低通滤波", "高通滤波"))

    if filter_type == "低通滤波":
        cutoff = st.slider("截止频率 (cutoff)", 1, 100, 30)
        low_pass_image = low_pass_filter(image, cutoff)
        st.image(low_pass_image, caption=f"低通滤波：截止频率={cutoff}", use_container_width=True)

    elif filter_type == "高通滤波":
        cutoff = st.slider("截止频率 (cutoff)", 1, 100, 30)
        high_pass_image = high_pass_filter(image, cutoff)
        st.image(high_pass_image, caption=f"高通滤波：截止频率={cutoff}", use_container_width=True)
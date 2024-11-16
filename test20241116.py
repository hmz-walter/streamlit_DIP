import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import cv2
import speech_recognition as sr
from rembg import remove

# 设置网页标题
st.title("图像处理工具集")


# 上传图像部分
def upload_image():
    uploaded_file = st.file_uploader("上传一张图片", type=["jpg", "jpeg", "png"])
    return uploaded_file


# 语音控制功能
def speech_to_text():
    recognizer = sr.Recognizer()

    # 使用麦克风获取语音
    with sr.Microphone() as source:
        st.info("请说话...")
        recognizer.adjust_for_ambient_noise(source)  # 调整噪音阈值
        audio = recognizer.listen(source)

    try:
        text = recognizer.recognize_google(audio, language="zh-CN")
        st.success(f"识别到的指令：{text}")
        return text
    except sr.UnknownValueError:
        st.error("无法识别语音，请再试一次")
        return None
    except sr.RequestError:
        st.error("无法连接到语音识别服务")
        return None


# 语音控制并执行图像处理操作
def execute_speech_command(command, image):
    # 解析语音命令并执行相关操作
    if "亮度" in command:
        brightness_value = float(command.split("亮度")[-1].strip()) if "亮度" in command else 1.0
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(brightness_value)
        st.image(image, caption=f"亮度调整: {brightness_value}", use_container_width=True)

    elif "对比度" in command:
        contrast_value = float(command.split("对比度")[-1].strip()) if "对比度" in command else 1.0
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(contrast_value)
        st.image(image, caption=f"对比度调整: {contrast_value}", use_container_width=True)

    elif "旋转" in command:
        command = command.replace("负", "-")
        angle = int(command.split("旋转")[-1].strip()) if "旋转" in command else 0
        image = image.rotate(angle)
        st.image(image, caption=f"旋转角度: {angle}°", use_container_width=True)

    elif "灰度" in command:
        image = image.convert("L")
        st.image(image, caption="灰度图", use_container_width=True)

    elif "裁剪" in command:
        # 假设语音命令为“裁剪 左边 0 上边 0 右边 300 下边 300”
        parts = command.split("裁剪")[-1].split()
        left = int(parts[0])
        top = int(parts[1])
        right = int(parts[2])
        bottom = int(parts[3])
        image = image.crop((left, top, right, bottom))
        st.image(image, caption="裁剪后的图像", use_container_width=True)

    elif "模糊" in command:
        blur_value = int(command.split("模糊")[-1].strip()) if "模糊" in command else 2
        image = image.filter(ImageFilter.GaussianBlur(blur_value))
        st.image(image, caption=f"模糊程度: {blur_value}", use_container_width=True)

    elif "去除背景" in command:
        st.info("开始进行抠图处理...")
        input_image = np.array(image)
        output_image = remove(input_image)
        output_image_pil = Image.fromarray(output_image)
        st.image(output_image_pil, caption="抠图后的图像", use_container_width=True)

    return image


# 图像处理功能
def image_processing_with_speech(uploaded_file):
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="上传的图像", use_container_width=True)

        # 转换为 OpenCV 格式，方便做更多操作
        opencv_image = np.array(image)
        opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_RGB2BGR)

        # 选择语音控制按钮
        if st.button("启动语音控制"):
            st.info("开始监听语音，请说话...")
            command = speech_to_text()
            if command:
                execute_speech_command(command, image)


def image_processing(uploaded_file):
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="上传的图像", use_container_width=True)

        # 转换为 OpenCV 格式，方便做更多操作
        opencv_image = np.array(image)
        opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_RGB2BGR)

        # 选择图像处理功能
        st.subheader("选择图像处理功能")

        # 1. 灰度处理
        if st.checkbox("转换为灰度图"):
            gray_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
            st.image(gray_image, caption="灰度图", use_container_width=True)

        # 2. 调整亮度
        if st.checkbox("调制亮度"):
            brightness = st.slider("调整亮度", 0.0, 2.0, 1.0)
            enhancer = ImageEnhance.Brightness(image)
            bright_image = enhancer.enhance(brightness)
            st.image(bright_image, caption=f"亮度调节: {brightness}", use_container_width=True)

        # 3. 旋转图像
        if st.checkbox("旋转图像"):
            angle = st.slider("旋转角度", -180, 180, 0)
            rotated_image = image.rotate(angle)
            st.image(rotated_image, caption=f"旋转角度: {angle}°", use_container_width=True)

        # 4. 图像裁剪
        if st.checkbox("裁剪图像"):
            st.subheader("裁剪图像")
            left = st.slider("左边界", 0, image.width, 0)
            top = st.slider("上边界", 0, image.height, 0)
            right = st.slider("右边界", 0, image.width, image.width)
            bottom = st.slider("下边界", 0, image.height, image.height)
            cropped_image = image.crop((left, top, right, bottom))
            st.image(cropped_image, caption="裁剪后的图像", use_container_width=True)

        # 5. 调整对比度
        if st.checkbox("调制对比度"):
            contrast = st.slider("调整对比度", 0.0, 2.0, 1.0)
            contrast_enhancer = ImageEnhance.Contrast(image)
            contrast_image = contrast_enhancer.enhance(contrast)
            st.image(contrast_image, caption=f"对比度调节: {contrast}", use_container_width=True)

        # 6. 模糊处理
        if st.checkbox("模糊处理"):
            blur = st.slider("模糊程度", 0, 10, 0)
            if blur > 0:
                blurred_image = image.filter(ImageFilter.GaussianBlur(blur))
                st.image(blurred_image, caption=f"模糊处理: {blur}", use_container_width=True)

        # 7. 边缘检测
        if st.checkbox("边缘检测"):
            edges = cv2.Canny(opencv_image, 100, 200)
            st.image(edges, caption="边缘检测图像", use_container_width=True)

        # 8. 保存处理后的图像
        if st.checkbox("去除背景"):
            st.info("开始进行抠图处理...")
            input_image = np.array(image)
            output_image = remove(input_image)
            output_image_pil = Image.fromarray(output_image)
            st.image(output_image_pil, caption="抠图后的图像", use_container_width=True)

# 主页与功能选择部分
st.sidebar.title("功能选择")
option = st.sidebar.selectbox("请选择功能", ("主页", "图像处理", "语音控制"))

if option == "主页":
    st.subheader("欢迎使用图像处理工具集！")
    st.markdown("""
    这是一个集成图像处理和语音控制功能的应用。你可以通过以下选择来进行图像处理：
    - 上传图像
    - 调整亮度、对比度、模糊、旋转等
    - 使用语音控制来调节图像处理参数
    """)

elif option == "图像处理":
    uploaded_file = upload_image()
    image_processing(uploaded_file)

elif option == "语音控制":
    uploaded_file = upload_image()
    image_processing_with_speech(uploaded_file)

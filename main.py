# main.py

import streamlit as st
from PIL import Image
from io import BytesIO
import random
import os
import tempfile
import shutil
import requests
from bs4 import BeautifulSoup
import requests
import tqdm
import numpy as np
import cv2
# import torch
# import torchvision.transforms as transforms
# from torchvision import models
# from sklearn.metrics.pairwise import cosine_similarity

from streamlit_controls.basic_controls import basic_operations_control
from streamlit_controls.geometric_controls import geometric_operations_control
from streamlit_controls.histogram_controls import histogram_operations_control
from streamlit_controls.filters_noise_controls import filters_noise_control
from streamlit_controls.frequency_transform_controls import frequency_transform_control
from streamlit_controls.advanced_controls import advanced_control
from streamlit_controls.total_controls import ImageController, history_control, image_to_bytes
from streamlit_controls.image_retrieval_controls import image_retrieval_controls,build_kdtree,search_image_kdtree
from coin.det import det_coin

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

#爬虫配置，这里爬取百度图片内容
def scrape_configs(search,page,number):
    url = 'https://image.baidu.com/search/acjson'
    params = {
        "tn": "resultjson_com",
        "logid": "11555092689241190059",
        "ipn": "rj",
        "ct": "201326592",
        "is": "",
        "fp": "result",
        "queryWord": search,
        "cl": "2",
        "lm": "-1",
        "ie": "utf-8",
        "oe": "utf-8",
        "adpicid": "",
        "st": "-1",
        "z": "",
        "ic": "0",
        "hd": "",
        "latest": "",
        "copyright": "",
        "word": search,
        "s": "",
        "se": "",
        "tab": "",
        "width": "",
        "height": "",
        "face": "0",
        "istype": "2",
        "qc": "",
        "nc": "1",
        "fr": "",
        "expermode": "",
        "force": "",
        "pn": str(60 * page),
        "rn": number,
        "gsm": "1e",
        "1617626956685": ""
    }
    return url, params

#获取图片
def getImg(url, idx, path,header,keyword):
    """

    :param url:
    :param idx:
    :param path:
    :return:
    """
    img = requests.get(url, headers=header)
    file = open(path + 'scrape_'+ keyword + str(idx + 1) + '.jpg', 'wb')
    file.write(img.content)
    file.close()

#爬取图片
def scrape_images(keyword, num_images,path):
    page=0
    header = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/66.0.3359.139 Safari/537.36'}
    bar = tqdm.tqdm(total=num_images)
    number=num_images

    while True:
        if num_images == 0:
            break
        url, params = scrape_configs(keyword, page, num_images*2) #每次多爬取2倍图像，再随机打乱返回，确保随机性
        response = requests.get(url, headers=header, params=params)
        try:
            result = response.json()
            if 'data' not in result:
                print("Unexpected response format. Skipping this page.")
                page += 1
                continue
        except ValueError:
            print("Failed to decode JSON, response content:")
            print(response.text)
            page += 1
            continue

        url_list = []
        for data in result['data'][:-1]:
            if 'thumbURL' in data:
                url_list.append(data['thumbURL'])

        #随机打乱
        random.shuffle(url_list)
        for i in range(len(url_list)):
            getImg(url_list[i], 60 * page + i, path,header,keyword)
            bar.update(1)
            num_images -= 1
            if num_images == 0:
                break
        page += 1
    # st.info("保存路径:"+path)
    st.success(f"成功爬取了 {number} 张图片！")

    def clear_directory(dir_path):
        if os.path.exists(dir_path):
            for filename in os.listdir(dir_path):
                file_path = os.path.join(dir_path, filename)
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
        else:
            os.makedirs(dir_path)

def clear_directory(dir_path):
    if os.path.exists(dir_path):
        for filename in os.listdir(dir_path):
            file_path = os.path.join(dir_path, filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
    else:
        os.makedirs(dir_path)

# 主界面和功能选择部分
st.sidebar.title("功能选择")
option = st.sidebar.selectbox("请选择功能", (
    "主页",
    "图像处理",
    "图像检索",
    "硬币检测与计数",
))

if option == "主页":
    st.subheader("欢迎使用图像处理工具集！")
    st.markdown("""
    这是一个集成图像处理工具的应用，提供以下功能：
    - **图像处理**：选择并应用不同的图像处理模块。
    - **图像检索**：本地/在线爬取相关图片，应用不同特征进行检索。
    - **硬币检测与计数**：检测硬币并进行计数。
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
    frequency_transform_control(image_controller)
    advanced_control(image_controller)

    # 历史操作、保存和删除图像功能
    if uploaded_file:
        history_control(image_controller)
        image_controller.show_image()
        save_image()
        # delete_image()

elif option == "图像检索":
    # 定义和初始化
    if 'images_upload' not in st.session_state:
        st.session_state['images_upload'] = False
    if 'upload_image_files' not in st.session_state:
        st.session_state['upload_image_files'] = []

    if 'images_crawled' not in st.session_state:
        st.session_state['images_crawled'] = False
    if 'scrap_image_files' not in st.session_state:
        st.session_state['scrap_image_files'] = []
    # 1. 上传待检索的图像
    st.subheader("上传待检索的图像")

    query_image_file = st.file_uploader("上传查询图像 (支持 BMP, JPG, JPEG, PNG)", type=["bmp", "jpg", "jpeg", "png"],
                                        key="query_image")
    query_image = None
    save_dir='./upload_imags'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir) #创建目录
    else:
        clear_directory(save_dir) #清空目录内容
    query_image_path=None
    if query_image_file is not None:
        query_image = Image.open(query_image_file).convert("RGB")
        # st.image(query_image, caption="待检索图像", use_container_width=True)
        st.image(query_image, caption="待检索图像", width=200)
        # 保存图像
        query_image_path = os.path.join(save_dir, query_image_file.name)
        with open(query_image_path, "wb") as f:
            f.write(query_image_file.getbuffer())

    if query_image_file is not None:
        upload_img_path=save_dir+"/"+query_image_file.name
    # 2. 选择检索数据源
    st.subheader("选择检索数据源")
    data_source_option=st.selectbox("选择数据源：", ["本地图片数据集", "在线爬取相关图片"], key="data_source_option")
    # 2.1 上传本地图片数据集
    # 2.2 在线爬取相关图片
    if data_source_option == "本地图片数据集":
        save_path = "upload_files/"
        # 判断路径是否存在，若不存在就新建路径，若存在则清空目录内容
        if not os.path.exists(save_path):
            os.makedirs(save_path)  # 创建目录
        clear_directory(save_path)
        upload_files=st.file_uploader("上传本地图片数据集", type=["bmp", "jpg", "jpeg", "png"], accept_multiple_files=True)
        # 保存上传的文件
        for uploaded_file in upload_files:
            with open(os.path.join(save_path, uploaded_file.name), "wb") as f:
                f.write(uploaded_file.getbuffer())

        st.session_state['images_upload'] = True    # 更新状态
        st.session_state['upload_image_files'] = [os.path.join(save_path, f) for f in os.listdir(save_path) if
                                                  f.endswith(('jpg', 'jpeg', 'png'))]
        # 3. 展示上传的图片
        st.sidebar.subheader("展示上传的图片")
        scrap_image_files = []
        if st.sidebar.button("展示上传的图片"):
            st.subheader("上传的图片")
            scrap_image_files = st.session_state['upload_image_files']
            st.info(f"上传图片数量：{len(scrap_image_files)}")
            if len(scrap_image_files) == 0:
                st.warning("没有找到上传的图片，请检查上传过程是否成功。")
            else:
                cloumn_num=int(len(scrap_image_files)/2)>10 and 10 or int(len(scrap_image_files)/2)
                for i in range(0,len(scrap_image_files),cloumn_num):
                    cols=st.columns(cloumn_num)
                    for j in range(cloumn_num):
                        if i+j<len(scrap_image_files):
                            cols[j].image(Image.open(scrap_image_files[i+j]),use_container_width=True)
    elif data_source_option == "在线爬取相关图片":
        save_path = "scrap_images/"
        # 判断路径是否存在，若不存在就新建路径，若存在则清空目录内容
        if not os.path.exists(save_path):
            os.makedirs(save_path)  # 创建目录
        # clear_directory(save_path)
        st.subheader("在线爬取相关图片")
        st.markdown("注意：在线爬取图片暂只支持爬取[百度图片](https://image.baidu.com/search/acjson)。输入的关键词尽可能详细！")
        keyword = st.text_input("输入关键词进行图片爬取，例如：白色的猫", key="scrape_keyword")
        num_images = st.number_input("输入要爬取的图片数量，最多爬取1000张", min_value=1, max_value=1000, value=10, step=1,
                                     key="num_images")


        if st.button("开始爬取图片"):
            if keyword:
                clear_directory(save_path)
                with st.spinner("正在爬取图片，请稍候..."):
                    scrape_images(keyword, num_images,save_path)
                st.session_state['images_crawled'] = True    # 更新状态
                st.session_state['scrap_image_files'] = [os.path.join(save_path, f) for f in os.listdir(save_path) if
                                                         f.endswith('jpg')]
            else:
                st.warning("请输入关键词以进行图片爬取。")
        # 3. 展示爬取的图片
        st.sidebar.subheader("展示爬取的图片")
        scrap_image_files = []
        if st.sidebar.button("展示爬取的图片"):
            st.subheader("爬取的图片")
            scrap_image_files = st.session_state['scrap_image_files']
            if len(scrap_image_files) == 0:
                st.warning("没有找到爬取的图片，请检查爬取过程是否成功。")
            else:
                cloumn_num=int(len(scrap_image_files)/2)>10 and 10 or int(len(scrap_image_files)/2)
                for i in range(0,len(scrap_image_files),cloumn_num):
                    cols=st.columns(cloumn_num)
                    for j in range(cloumn_num):
                        if i+j<len(scrap_image_files):
                            cols[j].image(Image.open(scrap_image_files[i+j]),use_container_width=True)
    #4. 执行检索

    st.sidebar.subheader("图像特征提取和检索")
    features = ["颜色直方图特征", "GLCM纹理特征", "LBP纹理特征"]
    selected_features = []
    # 创建多个单选框，每个对应一个特征
    for feature in features:
        selected = st.sidebar.checkbox(feature)
        if selected:
            selected_features.append(feature)

    scrap_image_files=[]
    if data_source_option == "本地图片数据集":
        scrap_image_files = st.session_state['upload_image_files']
    elif data_source_option == "在线爬取相关图片":
        scrap_image_files = st.session_state['scrap_image_files']
    # 最相似的k张图片：
    top_k = 5
    if len(scrap_image_files) > 0:
        top_k = st.sidebar.number_input("最相似的k张图片：", min_value=1, max_value=len(scrap_image_files), value=5, step=1, key="k")

    # 在选择了至少一个特征时显示执行按钮
    execute_extraction = False
    if selected_features:
        execute_extraction = st.sidebar.button("执行特征提取")
    else:
        st.sidebar.warning("请至少选择一种特征提取方法！")

    if execute_extraction:
        if len(scrap_image_files) == 0 or query_image_path is None:
            st.sidebar.warning("请先上传待检索的图像和爬取相关图片。")
        else:
            st.sidebar.success("开始进行特征提取...")
            #4.1 提取所有图像特征
            bins = (8, 12, 3)  # HSV颜色空间的bin数
            spatial_grid = (2,2)  # 图像划分的网格数
            glcm_params = {
                'distances': [1],
                'angles': [0],
                'levels': 256,
                'symmetric': True,
                'normed': True
            }

            lbp_params = {
                'num_points': 24,
                'radius': 3,
                'method': 'uniform'
            }
            query_feature,scrap_features=image_retrieval_controls(upload_img_path,scrap_image_files,selected_features,bins,spatial_grid)
            st.sidebar.success("所有特征提取成功")
            #4.2 构建KD-Tree索引
            tree = build_kdtree(scrap_features)
            print("KD-Tree构建完成。")
            #4.3 执行图像检索
            indices, distances = search_image_kdtree(query_feature, tree, top_k=top_k)

            st.write("检索结果:")
            column_num=int(len(indices)/2)
            for i in range(0,len(indices),column_num):
                cols=st.columns(column_num)
                for j,col in enumerate(cols):
                    if i+j<len(indices):
                        idx=indices[i+j]
                        distance=distances[i+j]
                        result_img_path=scrap_image_files[idx]
                        col.image(Image.open(result_img_path),use_container_width=True)
                        col.write(f"{i + j + 1}. 距离: {distance:.4f}")


elif option == "硬币检测与计数":
    st.title("硬币检测与计数")
    # File upload
    uploaded_file = st.file_uploader("上传一张图片 (支持 BMP, JPG, JPEG, PNG)", type=["bmp", "jpg", "jpeg", "png"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        coins_img = None
    
        if st.button("检测硬币"):
            image = np.array(img, dtype=np.uint8)
            # convert to BGR
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            coins_img = det_coin(image_bgr)

    
        if uploaded_file is not None:
            col1, col2 = st.columns([2, 2])
            with col1:
                st.image(img, caption="上传的图像", use_container_width=True)
    
            with col2:
                if coins_img is not None:
                    coins_img = cv2.cvtColor(coins_img, cv2.COLOR_BGR2RGB)
                    st.image(coins_img, caption="检测结果", use_container_width=True)



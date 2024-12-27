# streamlit_controls/total_controls.py

from PIL import Image, ImageDraw
import streamlit as st
import cv2
from streamlit_image_coordinates import streamlit_image_coordinates
from io import BytesIO
import base64

class ImageController:
    def __init__(self):
        if 'history' not in st.session_state:
            st.session_state.history = []
        if 'redo_stack' not in st.session_state:
            st.session_state.redo_stack = []
        if 'processed_image' not in st.session_state:
            st.session_state.processed_image = None
        if 'original_image' not in st.session_state:
            st.session_state.original_image = None  # For reset
        if 'is_refresh' not in st.session_state:
            st.session_state.is_refresh = True
        if 'is_show_coordinates' not in st.session_state:
            st.session_state.is_show_coordinates = False
        if 'points' not in st.session_state:
            st.session_state.points = None

    def load_image(self, image: Image.Image):
        st.session_state.processed_image = image.copy()
        st.session_state.original_image = image.copy()  # Save original image
        st.session_state.history = [image.copy()]  # Initialize history
        st.session_state.redo_stack = []  # Clear redo stack
        st.session_state.is_refresh = True  # Control image update

    def save_state(self):
        if st.session_state.processed_image:
            st.session_state.history.append(st.session_state.processed_image.copy())
            st.session_state.redo_stack = []  # Clear redo stack

    def apply_operation(self, operation_func, *args, description="", reset_history=False, **kwargs):
        if reset_history and self.get_original_image():
            self.reset_history()
        if st.session_state.processed_image:
            self.save_state()
            st.session_state.processed_image = operation_func(st.session_state.processed_image, *args, **kwargs)
            st.session_state.is_refresh = True
            return st.session_state.processed_image
        else:
            st.warning("请先上传一张图像。")
            return None

    def show_image(self):
        # original_image = self.get_original_image()
        # processed_image = self.get_current_image()
        #
        # if original_image and processed_image:
        #     col1, col2 = st.columns([2, 2])
        #
        #     # 原始图像显示和鼠标坐标捕获
        #     with col1:
        #         st.subheader("上传的图像")
        #         display_image_with_pixel_info(original_image, key="original")
        #         # 获取从 JS 传递到 Python 的值
        #         if "stcore_data" in st.session_state:
        #             data = st.session_state["pixel_data"]
        #             st.write(
        #                 f"鼠标坐标: ({data['x']}, {data['y']}), 像素值: (R: {data['r']}, G: {data['g']}, B: {data['b']})")
        #
        #     # 处理后图像显示和鼠标坐标捕获
        #     with col2:
        #         st.subheader("处理后的图像")
        #         display_image_with_pixel_info(processed_image, key="processed")
        #         # 获取从 JS 传递到 Python 的值
        #         if "stcore_data" in st.session_state:
        #             data = st.session_state["pixel_data"]
        #             st.write(
        #                 f"鼠标坐标: ({data['x']}, {data['y']}), 像素值: (R: {data['r']}, G: {data['g']}, B: {data['b']})")

        # original_image = self.get_original_image()
        # processed_image = self.get_current_image()
        # if original_image and processed_image:
        #     col1, col2 = st.columns([2, 2])
        #
        #     # 显示原始图像
        #     with col1:
        #         st.subheader("上传的图像")
        #         coords_orig = streamlit_image_coordinates(original_image, key="orig_image")
        #         # st.image(original_image, use_container_width=True)
        #         if coords_orig:
        #             x, y = int(coords_orig["x"]), int(coords_orig["y"])
        #             if 0 <= x < original_image.width and 0 <= y < original_image.height:
        #                 pixel_orig = original_image.getpixel((x, y))
        #                 st.text(f"原始图像 - 坐标: ({x}, {y}), 像素值: {pixel_orig}")
        #         st.image(original_image, use_container_width=True)
        #
        #     # 显示处理后的图像
        #     with col2:
        #         st.subheader("处理后的图像")
        #         coords_proc = streamlit_image_coordinates(processed_image, key="proc_image")
        #         # st.image(processed_image, use_container_width=True)
        #         if coords_proc:
        #             x, y = int(coords_proc["x"]), int(coords_proc["y"])
        #             if 0 <= x < processed_image.width and 0 <= y < processed_image.height:
        #                 pixel_proc = processed_image.getpixel((x, y))
        #                 st.text(f"处理后图像 - 坐标: ({x}, {y}), 像素值: {pixel_proc}")
        #         st.image(processed_image, use_container_width=True)

        # st.empty() # 先清空之前的图像显示
        current_image = self.get_current_image()
        if current_image is not None:
            if st.session_state.is_show_coordinates:
                current_image_width, current_image_height = current_image.width, current_image.height
                # 如果图像尺寸过大，则进行缩放
                if current_image_width > 2000 or current_image_height > 2000:
                    scale = 2000 / max(current_image_width, current_image_height)
                    width = int(current_image_width * scale)
                    height = int(current_image_height * scale)
                    image_to_display = current_image.resize((width, height))
                else:
                    image_to_display = current_image

                image_to_display_temp = image_to_display.copy()
                # 绘制十字标记到临时图像
                if st.session_state.points is not None:
                    x, y = st.session_state.points
                    draw = ImageDraw.Draw(image_to_display_temp)
                    line_length = 10
                    line_color = "red"
                    draw.line((x, y - line_length, x, y + line_length), fill=line_color, width=1)  # 垂直线
                    draw.line((x - line_length, y, x + line_length, y), fill=line_color, width=1)  # 水平线

                # 显示图像并捕获鼠标点击位置
                value = streamlit_image_coordinates(image_to_display_temp, key="interactive_image")
                if value is not None:
                    x = int(value["x"] / image_to_display.width * current_image_width)
                    y = int(value["y"] / image_to_display.height * current_image_height)
                    value["width"] = current_image_width
                    value["height"] = current_image_height
                    value["value"] = current_image.getpixel((value["x"], value["y"]))
                    if st.session_state.points is None:
                        st.session_state.points = (x, y)
                        st.rerun()  # 刷新页面
                    elif st.session_state.points != (x, y):
                        st.session_state.points = (x, y)
                        st.rerun()  # 刷新页面
                st.write(value)

                # # 绘制椭圆
                # def get_ellipse_coords(point):
                #     x, y = point
                #     a = 10
                #     return (x - a, y - a, x + a, y + a)
                # if st.session_state.points is not None:
                #     draw = ImageDraw.Draw(current_image)
                #     coords = get_ellipse_coords(st.session_state.points)
                #     draw.ellipse(coords, fill="red")

            col1, col2 = st.columns([2, 2])
            with col1:
                st.subheader("上传的图像")
                st.image(self.get_original_image(), use_container_width=True)

            with col2:
                st.subheader("处理后的图像")
                current_image = self.get_current_image()
                if current_image is not None:
                    st.image(current_image, use_container_width=True)
                else:
                    st.info("请进行图像处理操作。")

    def undo(self):
        if len(st.session_state.history) > 1:
            last_state = st.session_state.history.pop()
            st.session_state.redo_stack.append(last_state)
            st.session_state.processed_image = st.session_state.history[-1].copy()
            st.session_state.is_refresh = True
            st.success("撤销成功。")
        else:
            st.warning("无法撤销，已到达初始状态。")

    def redo(self):
        if st.session_state.redo_stack:
            state = st.session_state.redo_stack.pop()
            st.session_state.history.append(state.copy())
            st.session_state.processed_image = state.copy()
            st.session_state.is_refresh = True
            st.success("重做成功。")
        else:
            st.warning("无法重做，重做栈为空。")

    def reset(self):
        if st.session_state.original_image:
            st.session_state.processed_image = st.session_state.original_image.copy()
            st.session_state.history = [st.session_state.original_image.copy()]
            st.session_state.redo_stack = []
            st.session_state.is_refresh = True
            st.session_state.is_show_coordinates = False
            st.session_state.points = None
            st.success("图像已重置。")
        else:
            st.warning("没有原始图像可重置。")

    def reset_history(self):
        if self.get_original_image():
            st.session_state.history = [self.get_original_image().copy()]
            st.session_state.redo_stack = []

    def delete(self):
        st.session_state.history = []
        st.session_state.redo_stack = []
        st.session_state.processed_image = None
        st.session_state.original_image = None
        st.session_state.is_refresh = True
        st.session_state.is_show_coordinates = False
        st.session_state.points = None

    def get_current_image(self) -> Image.Image:
        return st.session_state.processed_image

    def get_original_image(self) -> Image.Image:
        return st.session_state.original_image

def history_control(image_controller):
    """显示撤销、重做、重置和下载按钮在图像下方。"""

    col1, col2, col3, col4, col5, col6, col7, col8 = st.columns(8)
    if image_controller.get_current_image():
        with col5:
            if st.button("撤销"):
                image_controller.undo()

        with col6:
            if st.button("重做"):
                image_controller.redo()

        with col7:
            if st.button("重置"):
                image_controller.reset()

        with col8:
            if st.button("查看"):
                st.session_state.is_show_coordinates = not st.session_state.is_show_coordinates
                # # Convert image to bytes
                # img_bytes = image_to_bytes(image_controller.get_current_image())
                # st.download_button(
                #     label="下载",
                #     data=img_bytes,
                #     file_name="processed_image.png",
                #     mime="image/png"
                # )

def image_to_bytes(image: Image.Image) -> bytes:
    """将 PIL 图像转换为字节流。"""
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer.getvalue()

def display_image_with_pixel_info(image, key):
    """
    显示图像，并捕获鼠标悬浮时的坐标和像素值。
    """
    # 将图像转换为 Base64 格式，供 HTML 显示
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    # 获取图像的宽高
    width, height = image.size

    # 自定义 HTML 和 JS 脚本
    html_code = f"""
        <div style="position: relative; display: inline-block;">
            <img src="data:image/png;base64,{img_str}" id="image-{key}" style="width: 100%; max-width: 100%;">
            <div id="coords-{key}" style="position: absolute; top: 10px; left: 10px; background-color: rgba(0, 0, 0, 0.5); color: white; padding: 5px; border-radius: 5px; font-size: 12px;">
                坐标: (x, y), 像素值: (R, G, B)
            </div>
        </div>
        <script>
            const img = document.getElementById("image-{key}");
            const coords = document.getElementById("coords-{key}");
            const canvas = document.createElement("canvas");
            canvas.width = {width};
            canvas.height = {height};
            const ctx = canvas.getContext("2d");
            const imgElement = new Image();
            imgElement.src = "data:image/png;base64,{img_str}";
            imgElement.onload = function() {{
                ctx.drawImage(imgElement, 0, 0);
            }};
            img.addEventListener("mousemove", function(event) {{
                const rect = img.getBoundingClientRect();
                const x = Math.floor((event.clientX - rect.left) / rect.width * {width});
                const y = Math.floor((event.clientY - rect.top) / rect.height * {height});
                if (x >= 0 && x < {width} && y >= 0 && y < {height}) {{
                    const pixel = ctx.getImageData(x, y, 1, 1).data;
                    coords.innerHTML = "坐标: (" + x + ", " + y + "), 像素值: (" + pixel[0] + ", " + pixel[1] + ", " + pixel[2] + ")";

                    // 将坐标和像素值发送到 Streamlit 后端
                    const streamlitData = {{ x: x, y: y, r: pixel[0], g: pixel[1], b: pixel[2] }};
                    fetch('/_stcore_data', {{
                        method: 'POST',
                        headers: {{ 'Content-Type': 'application/json' }},
                        body: JSON.stringify(streamlitData)
                    }});
                }} else {{
                    coords.innerHTML = "坐标: 超出图像范围";
                }}
            }});
        </script>
        """
    # 显示 HTML
    st.markdown(html_code, unsafe_allow_html=True)
    # 创建一个 JS 监听器，将数据从前端传递到后端
    st.write(
        """
        <script>
        document.addEventListener("sendStreamlitData", function(e) {
            const data = e.detail;
            fetch('/streamlit/report_event', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ event: "update_pixel_data", data: data })
            });
        });
        </script>
        """,
        unsafe_allow_html=True,
    )

# 自定义回调函数，用于处理事件更新
def update_pixel_data(event_data):
    st.session_state["pixel_data"] = event_data

def display_image_with_crosshair(image, key):
    """
    显示图像，并在点击时显示十字地标和坐标信息。
    """
    # 将图像转换为 Base64 格式，用于 HTML 显示
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    # 获取图像的宽高
    width, height = image.size

    # 自定义 HTML 和 JS 脚本
    html_code = f"""
    <div style="position: relative; display: inline-block;">
        <canvas id="canvas-{key}" width="{width}" height="{height}" style="border:1px solid black; cursor: crosshair;"></canvas>
        <script>
            const canvas = document.getElementById("canvas-{key}");
            const ctx = canvas.getContext("2d");
            const img = new Image();
            img.src = "data:image/png;base64,{img_str}";
            img.onload = function() {{
                ctx.drawImage(img, 0, 0);
            }};
            canvas.addEventListener("click", function(event) {{
                const rect = canvas.getBoundingClientRect();
                const x = Math.floor(event.clientX - rect.left);
                const y = Math.floor(event.clientY - rect.top);

                // 清除旧的十字地标
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                ctx.drawImage(img, 0, 0);

                // 绘制十字地标
                ctx.beginPath();
                ctx.moveTo(x, 0);
                ctx.lineTo(x, canvas.height);
                ctx.moveTo(0, y);
                ctx.lineTo(canvas.width, y);
                ctx.strokeStyle = "red";
                ctx.lineWidth = 1;
                ctx.stroke();

                // 将坐标发送到 Streamlit 后端
                const streamlitData = {{ x: x, y: y }};
                const event = new CustomEvent("sendStreamlitData", {{ detail: streamlitData }});
                document.dispatchEvent(event);
            }});
        </script>
    </div>
    """
    # 显示 HTML
    st.markdown(html_code, unsafe_allow_html=True)

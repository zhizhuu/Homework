import cv2
import numpy as np
import gradio as gr
import math

# 初始化全局变量，存储控制点和目标点
points_src = []
points_dst = []
image = None

# 上传图像时清空控制点和目标点
def upload_image(img):
    global image, points_src, points_dst
    points_src.clear()  # 清空控制点
    points_dst.clear()  # 清空目标点
    image = img
    return img

# 记录点击点事件，并标记点在图像上，同时在成对的点间画箭头
def record_points(evt: gr.SelectData):
    global points_src, points_dst, image
    x, y = evt.index[0], evt.index[1]  # 获取点击的坐标
    
    # 判断奇偶次来分别记录控制点和目标点
    if len(points_src) == len(points_dst):
        points_src.append([x, y])  # 奇数次点击为控制点
    else:
        points_dst.append([x, y])  # 偶数次点击为目标点
    
    # 在图像上标记点（蓝色：控制点，红色：目标点），并画箭头
    marked_image = image.copy()
    for pt in points_src:
        cv2.circle(marked_image, tuple(pt), 1, (255, 0, 0), -1)  # 蓝色表示控制点
    for pt in points_dst:
        cv2.circle(marked_image, tuple(pt), 1, (0, 0, 255), -1)  # 红色表示目标点
    
    # 画出箭头，表示从控制点到目标点的映射
    for i in range(min(len(points_src), len(points_dst))):
        cv2.arrowedLine(marked_image, tuple(points_src[i]), tuple(points_dst[i]), (0, 255, 0), 1)  # 绿色箭头表示映射
    
    return marked_image








class MLSImageWarping(object):
    def __init__(self, cp, cq, whether_color_q=False, point_size=3):
        self._cp = cp
        self._cq = cq
        self._whether_color_q = whether_color_q
        self._point_size = point_size
        self._num_cpoints = len(cp)
        self._maximum = 2**31-1

    def update_cp(self, cp):
        self._cp = cp

    def update_cq(self, cq):
        self._cq = cq

    def check_is_cq(self, x, y):
        for i in range(self._num_cpoints):
            if abs(x - self._cq[i][0]) <= self._point_size and abs(y - self._cq[i][1]) <= self._point_size:
                return True
        return False

    def get_weights(self, input_pixel):
        Weights = np.zeros(self._num_cpoints)

        for i in range(self._num_cpoints):
            cpx, cpy = self._cp[i][0], self._cp[i][1]
            x, y = input_pixel[1], input_pixel[0]
            if x != cpx or y != cpy:
                weight = 1 / ((cpx - x) * (cpx - x) + (cpy - y) * (cpy - y))
            else:
                weight = self._maximum

            Weights[i] = weight

        return Weights

    def getPStar(self, Weights):
        numerator = np.zeros(2)
        denominator = 0
        for i in range(self._num_cpoints):
            numerator[0] += Weights[i] * self._cp[i][0]
            numerator[1] += Weights[i] * self._cp[i][1]
            denominator += Weights[i]

        return numerator / denominator

    def getQStar(self, Weights):
        numerator = np.zeros(2)
        denominator = 0
        for i in range(self._num_cpoints):
            numerator[0] += Weights[i] * self._cq[i][0]
            numerator[1] += Weights[i] * self._cq[i][1]
            denominator += Weights[i]

        return numerator / denominator

    def getTransformMatrix(self, p_star, q_star, Weights):
        sum_pwp = np.zeros((2, 2))
        sum_wpq = np.zeros((2, 2))
        for i in range(self._num_cpoints):
            tmp_cp = (np.array(self._cp[i]) - np.array(p_star)).reshape(1, 2)
            tmp_cq = (np.array(self._cq[i]) - np.array(q_star)).reshape(1, 2)

            sum_pwp += np.matmul(tmp_cp.T*Weights[i], tmp_cp)
            sum_wpq += Weights[i] * np.matmul(tmp_cp.T, tmp_cq)

        try:
            inv_sum_pwp = np.linalg.inv(sum_pwp)
        except np.linalg.linalg.LinAlgError:
            if np.linalg.det(sum_pwp) < 1e-8:
                return np.identity(2)
            else:
                raise

        return inv_sum_pwp*sum_wpq


    def transfer(self, data):
        row, col, channel = data.shape
        res_data = np.zeros((row, col, channel), np.uint8)

        for j in range(col):
            for i in range(row):
                input_pixel = [i, j]
                Weights = self.get_weights(input_pixel)
                p_star = self.getPStar(Weights)
                q_star = self.getQStar(Weights)
                M = self.getTransformMatrix(p_star, q_star, Weights)

                ## 逆变换版本
                try:
                    inv_M = np.linalg.inv(M)
                except np.linalg.linalg.LinAlgError:
                    if np.linalg.det(M) < 1e-8:
                        inv_M = np.identity(2)
                    else:
                        raise

                pixel = np.matmul((np.array([input_pixel[1], input_pixel[0]]) - np.array(q_star)).reshape(1, 2),
                                  inv_M) + np.array(p_star).reshape(1, 2)

                pixel_x = pixel[0][0]
                pixel_y = pixel[0][1]

                if math.isnan(pixel_x):
                    pixel_x = 0
                if math.isnan(pixel_y):
                    pixel_y = 0

                # pixel_x, pixel_y = max(min(int(pixel_x), row-1), 0), max(min(int(pixel_y), col-1), 0)
                pixel_x, pixel_y = max(min(int(pixel_x), col - 1), 0), max(min(int(pixel_y), row - 1), 0)

                if self._whether_color_q == True:
                    if self.check_is_cq(j, i):
                        res_data[i][j] = np.array([255, 0, 0]).astype(np.uint8)
                    else:
                        res_data[i][j] = data[pixel_y][pixel_x]
                else:
                    res_data[i][j] = data[pixel_y][pixel_x]

        return res_data





# 执行仿射变换

def point_guided_deformation(image, source_pts, target_pts, alpha=1.0, eps=1e-8):
    """ 
    Return
    ------
        A deformed image.
    """
    
    warped_image = np.array(image)
    ### FILL: 基于MLS or RBF 实现 image warping
    # Change (x, y) to (row, col)
    mls = MLSImageWarping(source_pts, target_pts, True)

    res_img = mls.transfer(image)

    return res_img





def run_warping():
    global points_src, points_dst, image ### fetch global variables

    warped_image = point_guided_deformation(image, np.array(points_src), np.array(points_dst))

    return warped_image

# 清除选中点
def clear_points():
    global points_src, points_dst
    points_src.clear()
    points_dst.clear()
    return image  # 返回未标记的原图

# 使用 Gradio 构建界面
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(source="upload", label="上传图片", interactive=True, width=800, height=200)
            point_select = gr.Image(label="点击选择控制点和目标点", interactive=True, width=800, height=800)
            
        with gr.Column():
            result_image = gr.Image(label="变换结果", width=800, height=400)
    
    # 按钮
    run_button = gr.Button("Run Warping")
    clear_button = gr.Button("Clear Points")  # 添加清除按钮
    
    # 上传图像的交互
    input_image.upload(upload_image, input_image, point_select)
    # 选择点的交互，点选后刷新图像
    point_select.select(record_points, None, point_select)
    # 点击运行 warping 按钮，计算并显示变换后的图像
    run_button.click(run_warping, None, result_image)
    # 点击清除按钮，清空所有已选择的点
    clear_button.click(clear_points, None, point_select)
    
# 启动 Gradio 应用
demo.launch()


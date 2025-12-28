import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import os
import warnings

warnings.filterwarnings('ignore')


# ===================== 1. 预处理,分割函数 =====================
def preprocess(img_bgr):
    # 1. 灰度化
    if len(img_bgr.shape) == 3:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_bgr.copy()

    # 2. 高斯模糊去噪
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 3. Otsu 自适应二值化
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 4. 自动判断是否取反（适配白底黑字/黑底白字）
    white_pixels = np.sum(binary == 255)
    total_pixels = binary.shape[0] * binary.shape[1]
    if white_pixels > total_pixels * 0.5:
        binary = cv2.bitwise_not(binary)

    # 5. 形态学操作去噪+连接笔画
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    return binary

#从二值图中抠出每个数字
def segment_digits(binary, min_area=100, min_height=20, max_aspect_ratio=8.0):
    # 查找轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rois = []
    img_height = binary.shape[0]

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        if h < min_height:
            continue

        aspect_ratio = w / h if h > 0 else 0
        if aspect_ratio > max_aspect_ratio:
            continue

        if h < img_height * 0.05:
            continue

        # 扩展边界，避免裁剪过紧
        margin = 3
        x_start = max(0, x - margin)
        y_start = max(0, y - margin)
        x_end = min(binary.shape[1], x + w + margin)
        y_end = min(binary.shape[0], y + h + margin)

        roi = binary[y_start:y_end, x_start:x_end]
        if roi.size > 0:
            rois.append({
                'roi': roi,
                'bbox': (x_start, y_start, x_end - x_start, y_end - y_start)
            })

    # 过滤过多噪点
    if len(rois) > 20:
        rois.sort(key=lambda r: r['roi'].shape[0] * r['roi'].shape[1], reverse=True)
        rois = rois[:12]

    # 按 x 坐标从左到右排序
    rois.sort(key=lambda r: r['bbox'][0])

    return rois

# 把数字 ROI 转成模型能识别的格式
def roi_to_mnist_tensor(roi, target_size=28, padding=4):

    h, w = roi.shape

    # 计算缩放比例
    available_size = target_size - 2 * padding
    scale = min(available_size / h, available_size / w)
    scale = max(scale, 0.1)

    new_h = max(1, int(h * scale))
    new_w = max(1, int(w * scale))

    # 缩放 ROI
    if new_h > 0 and new_w > 0 and h > 0 and w > 0:
        resized = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        resized = roi.copy()
        new_h, new_w = resized.shape

    # 创建 28x28 画布并居中放置
    canvas = np.zeros((target_size, target_size), dtype=np.uint8)
    y_offset = (target_size - new_h) // 2
    x_offset = (target_size - new_w) // 2
    y_offset = max(0, y_offset)
    x_offset = max(0, x_offset)

    y_end = min(y_offset + new_h, target_size)
    x_end = min(x_offset + new_w, target_size)
    src_h = min(new_h, y_end - y_offset)
    src_w = min(new_w, x_end - x_offset)

    if src_h > 0 and src_w > 0:
        canvas[y_offset:y_end, x_offset:x_end] = resized[:src_h, :src_w]

    # 归一化 + 转 tensor + 标准化
    canvas_float = canvas.astype(np.float32) / 255.0
    tensor = torch.from_numpy(canvas_float).unsqueeze(0)
    normalize = transforms.Normalize((0.1307,), (0.3081,))
    tensor = normalize(tensor)

    return tensor


# ===================== 2. MNIST_CNN 模型 =====================
class MNIST_CNN(nn.Module):
    def __init__(self):
        super(MNIST_CNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


# ===================== 3.模型推理，结果标注 =====================
def recognize_student_id_with_toolkit(img_path, model_path="mnist_model.pth", save_annotated="annotated_id.jpg"):

    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. 加载模型
    model = MNIST_CNN()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # 2. 读取图片
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"无法读取图片：{img_path}")
    annotated_img = img.copy()  # 用于标注的原图

    # 3. 调用预处理函数
    binary_img = preprocess(img)

    # 4. 调用分割函数
    digit_rois = segment_digits(binary_img)
    if len(digit_rois) == 0:
        raise ValueError("未检测到任何数字！")

    # 5. 逐个识别数字
    final_id = []
    font = cv2.FONT_HERSHEY_SIMPLEX
    color_box = (0, 255, 0)  # 绿色框
    color_text = (0, 0, 255)  # 红色文字

    for i, roi_info in enumerate(digit_rois):
        # 获取 ROI 和 bounding box
        roi = roi_info['roi']
        x, y, w, h = roi_info['bbox']

        # 调用你的 tensor 转换函数
        mnist_tensor = roi_to_mnist_tensor(roi)
        mnist_tensor = mnist_tensor.unsqueeze(0).to(device)  # 增加batch维度 (1,1,28,28)

        # 模型推理
        with torch.no_grad():
            output = model(mnist_tensor)
            prob = torch.softmax(output, dim=1)
            pred_num = torch.argmax(prob, dim=1).item()
            confidence = prob[0][pred_num].item()

        # 只保留置信度>0.5的结果（可调整）
        if confidence > 0.5:
            final_id.append(str(pred_num))

            # 标注到原图
            # 绘制数字框
            cv2.rectangle(annotated_img, (x, y), (x + w, y + h), color_box, 2)
            # 绘制预测结果+置信度
            text = f"{pred_num} ({confidence:.2f})"
            cv2.putText(annotated_img, text, (x, y - 10), font, 0.6, color_text, 2)

    # 6. 标注学号
    final_id_str = ''.join(final_id)
    cv2.putText(annotated_img, f"Student ID: {final_id_str}", (20, 40), font, 1.0, (255, 0, 0), 3)

    # 7. 保存标注后的图片
    cv2.imwrite(save_annotated, annotated_img)
    print(f"标注后的图片已保存至：{save_annotated}")

    # 调试信息
    print(f"\n===== 识别结果 =====")
    print(f"检测到数字数量：{len(digit_rois)}")
    print(f"识别出的学号：{final_id_str}")

    return final_id_str, annotated_img


# ===================== 4. 调用示例 =====================
if __name__ == '__main__':
    IMG_PATH = "id.jpg"  # 手写学号照片
    MODEL_PATH = "mnist_model.pth"  # 训练好的模型
    SAVE_PATH = "annotated_student_id.jpg"  # 标注后保存路径

    try:
        # 执行完整识别流程
        result, img = recognize_student_id_with_toolkit(IMG_PATH, MODEL_PATH, SAVE_PATH)
        print(f"\n最终识别结果：{result}")

        # 显示结果
        cv2.imshow("Recognition Result", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"识别出错：{e}")
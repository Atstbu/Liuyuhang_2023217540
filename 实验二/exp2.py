import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# 字体配置
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


# ===================== 1. 初始化文件夹 =====================
def init_experiment_folder(folder_name="实验二结果"):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"已创建文件夹：{folder_name}")
    return folder_name


# ===================== 2. ROI =====================
def apply_crop_equivalent_roi(img, crop_ratio=0.1):
    """
    用ROI掩码实现“裁剪四周”的等效效果
    crop_ratio：原裁剪比例（如0.1=四周各屏蔽10%）
    """
    h, w = img.shape[:2]
    # 计算ROI的边界
    top = int(h * crop_ratio)
    bottom = int(h * (1 - crop_ratio))
    left = int(w * crop_ratio)
    right = int(w * (1 - crop_ratio))

    # 创建全0掩码，仅中间区域置255（保留中间道路）
    mask = np.zeros_like(img)
    if len(img.shape) == 3:  # 彩色图
        mask[top:bottom, left:right, :] = 255
    else:  # 灰度/边缘图
        mask[top:bottom, left:right] = 255

    # 应用掩码：仅保留中间区域，四周置0
    img_roi = cv2.bitwise_and(img, mask)
    return img_roi


# ===================== 3. 预处理：只保留白线强边缘 =====================
def preprocess_for_hough(img_bgr):
    """仅做灰度+双边滤波+Canny"""
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # 双边滤波：保留白线边缘，去除路面纹理干扰
    img_blur = cv2.bilateralFilter(img_gray, d=9, sigmaColor=75, sigmaSpace=75)
    # Canny边缘检测：提取白线
    img_canny = cv2.Canny(img_blur, threshold1=100, threshold2=200)
    return img_canny


# ===================== 4. 霍夫变换（ROI内检测白线，无区域限制） =====================
def hough_detect_white_lines(img_edges):
    """检测ROI处理后的白线"""
    lines = cv2.HoughLinesP(
        img_edges,
        rho=1,  # 极坐标步长
        theta=np.pi / 180,  # 角度步长
        threshold=30,  # 最小投票数
        minLineLength=120,  # 最小线长
        maxLineGap=15  # 合并零散线段
    )
    lines = lines if lines is not None else []
    return lines


# ===================== 5. 绘制绿色描边 =====================
def draw_green_lines(img_bgr, lines):
    """用绿色线条描出检测到的白线"""
    img_result = np.copy(img_bgr)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(img_result, (x1, y1), (x2, y2), (0, 255, 0), thickness=18)
    return img_result


# ===================== 6. 结果图 =====================
def concat_images(img_original, img_result):
    h1, w1 = img_original.shape[:2]
    h2, w2 = img_result.shape[:2]
    if h1 != h2:
        img_result = cv2.resize(img_result, (int(w2 * h1 / h2), h1))
    return np.hstack((img_original, img_result))


# ===================== 7. 主函数（等效ROI+纯霍夫变换） =====================
def lane_detection_roi_instead_crop(img_path):
    exp_folder = init_experiment_folder()

    # 1. 读取原始图像
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise ValueError(f"图像路径错误：{img_path}")

    # 2. 预处理（原始图）
    img_canny = preprocess_for_hough(img_bgr)
    plt.figure(figsize=(img_canny.shape[1] / 100, img_canny.shape[0] / 100), dpi=100)  # 匹配原图尺寸
    plt.imshow(img_canny, cmap='gray')  # 以灰度模式显示
    plt.axis('off')  # 关闭坐标轴
    # 保存：bbox_inches='tight'去除白边，pad_inches=0避免留白
    plt.savefig(os.path.join(exp_folder, "Canny边缘检测结果.jpg"),
                bbox_inches='tight', pad_inches=0, dpi=100)
    plt.close()  # 关闭画布

    # 3. 等效ROI（替换裁剪，屏蔽四周10%，保留中间道路）
    img_canny_roi = apply_crop_equivalent_roi(img_canny, crop_ratio=0.1)

    # 4. 纯霍夫变换检测白线（仅ROI区域有效）
    lines = hough_detect_white_lines(img_canny_roi)

    # 5. 绘制绿色描边（在原始尺寸图上）
    img_lane = draw_green_lines(img_bgr, lines)

    # 6. 拼接原始图+检测结果图（尺寸不变）
    img_concat = concat_images(img_bgr, img_lane)


    # 7. 可视化汇总
    plt.figure(figsize=(18, 8))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    plt.title("原始图")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(img_lane, cv2.COLOR_BGR2RGB))
    plt.title("ROI+霍夫变换+绿色描边结果")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(exp_folder, "实验二总结果图.jpg"), dpi=150)
    plt.close()

    # 结果汇总
    print(f"检测完成！")
    print(f"结果保存至：{exp_folder}")



# ===================== 运行入口 =====================
if __name__ == "__main__":
    # ********** 替换成你图片的本地绝对路径 **********
    IMAGE_PATH = "exp2.png"  # 例："C:/Users/xxx/Desktop/road.png"
    # 执行等效ROI+霍夫变换检测
    lane_detection_roi_instead_crop(IMAGE_PATH)
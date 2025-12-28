import cv2
import numpy as np
import matplotlib.pyplot as plt
import os



#常用SimHei/微软雅黑
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
# ===================== 1. 初始化实验文件夹 =====================
def init_experiment_folder(folder_name="实验一结果"):
    """
    创建实验结果文件夹，避免路径不存在
    """
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"已创建实验结果文件夹：{folder_name}")
    return folder_name


# ===================== 2. 手动实现卷积操作 =====================
def manual_convolve(img, kernel):
    """手动实现二维卷积"""
    img_h, img_w = img.shape
    kernel_k = kernel.shape[0]
    pad = kernel_k // 2

    padded_img = np.pad(img, pad_width=pad, mode='constant', constant_values=0)
    conv_img = np.zeros_like(img, dtype=np.float32)

    for i in range(img_h):
        for j in range(img_w):
            roi = padded_img[i:i + kernel_k, j:j + kernel_k]
            conv_img[i, j] = np.sum(roi * kernel)

    conv_img = np.clip(conv_img, 0, 255).astype(np.uint8)
    return conv_img


# ===================== 3. Sobel算子滤波 =====================
def sobel_filter(img_gray):
    """手动实现Sobel算子滤波（x和y方向）"""
    sobel_x_kernel = np.array([[1, 0, -1],
                               [2, 0, -2],
                               [1, 0, -1]], dtype=np.float32)
    sobel_y_kernel = np.array([[1, 2, 1],
                               [0, 0, 0],
                               [-1, -2, -1]], dtype=np.float32)

    sobel_x = manual_convolve(img_gray, sobel_x_kernel)
    sobel_y = manual_convolve(img_gray, sobel_y_kernel)

    sobel_combined = np.sqrt(sobel_x.astype(np.float32) ** 2 + sobel_y.astype(np.float32) ** 2)
    sobel_combined = np.clip(sobel_combined, 0, 255).astype(np.uint8)
    return sobel_x, sobel_y, sobel_combined


# ===================== 4. 指定卷积核滤波 =====================
def custom_kernel_filter(img_gray):
    """手动实现实验指定的卷积核滤波"""
    custom_kernel = np.array([[1, 0, -1],
                              [2, 0, -2],
                              [1, 0, -1]], dtype=np.float32)
    return manual_convolve(img_gray, custom_kernel)


# ===================== 5. 手动计算颜色直方图 =====================
def compute_color_histogram(img_bgr):
    """手动计算RGB三通道的颜色直方图"""
    b, g, r = cv2.split(img_bgr)
    h, w = b.shape
    total_pixels = h * w

    hist_r = np.zeros(256, dtype=np.int32)
    hist_g = np.zeros(256, dtype=np.int32)
    hist_b = np.zeros(256, dtype=np.int32)

    for i in range(h):
        for j in range(w):
            hist_r[r[i, j]] += 1
            hist_g[g[i, j]] += 1
            hist_b[b[i, j]] += 1

    hist_r = hist_r / total_pixels
    hist_g = hist_g / total_pixels
    hist_b = hist_b / total_pixels
    return hist_r, hist_g, hist_b


# ===================== 6. 手动提取纹理特征 =====================
def extract_texture_features(img_gray, distance=1, angle=0):
    """手动计算灰度共生矩阵，提取纹理特征"""
    # 灰度级压缩
    img_gray = (img_gray // 16).astype(np.uint8)
    max_gray = 16
    # 初始化灰度共生矩阵
    glcm = np.zeros((max_gray, max_gray), dtype=np.int32)
    h, w = img_gray.shape

    if angle == 0:
        for i in range(h):
            for j in range(w - distance):
                g1 = img_gray[i, j]
                g2 = img_gray[i, j + distance]
                glcm[g1, g2] += 1
    elif angle == 90:
        for i in range(h - distance):
            for j in range(w):
                g1 = img_gray[i, j]
                g2 = img_gray[i + distance, j]
                glcm[g1, g2] += 1

    glcm = glcm / np.sum(glcm)

    contrast = 0.0
    energy = 0.0
    entropy = 0.0
    correlation = 0.0
    # 计算GLCM的均值和方差
    mean_i = np.sum(np.arange(max_gray) * np.sum(glcm, axis=1))
    mean_j = np.sum(np.arange(max_gray) * np.sum(glcm, axis=0))
    var_i = np.sum(((np.arange(max_gray) - mean_i) ** 2) * np.sum(glcm, axis=1))
    var_j = np.sum(((np.arange(max_gray) - mean_j) ** 2) * np.sum(glcm, axis=0))
    # 遍历GLCM，计算四个纹理特征
    for i in range(max_gray):
        for j in range(max_gray):
            p = glcm[i, j]
            if p == 0:
                continue
            contrast += (i - j) ** 2 * p
            energy += p ** 2
            entropy += -p * np.log2(p)
            correlation += ((i - mean_i) * (j - mean_j) * p) / np.sqrt(var_i * var_j)

    features = {
        "contrast": contrast,
        "energy": energy,
        "entropy": entropy,
        "correlation": correlation
    }
    return features


# ===================== 7. 主函数：实验全流程 =====================
def image_filtering_experiment(img_path, folder_name="实验一结果"):
    # 初始化实验文件夹
    exp_folder = init_experiment_folder(folder_name)

    # 1. 读取图像
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise ValueError("无法读取图像，请检查路径！")
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # 2. Sobel算子滤波 + 保存结果
    sobel_x, sobel_y, sobel_combined = sobel_filter(img_gray)
    cv2.imwrite(os.path.join(exp_folder, "Sobel_X滤波结果.jpg"), sobel_x)

    cv2.imwrite(os.path.join(exp_folder, "Sobel_Y滤波结果.jpg"), sobel_y)
    cv2.imwrite(os.path.join(exp_folder, "Sobel合并边缘图.jpg"), sobel_combined)
    # 3. 指定卷积核滤波 + 保存结果
    custom_filtered = custom_kernel_filter(img_gray)
    cv2.imwrite(os.path.join(exp_folder, "指定卷积核滤波结果.jpg"), custom_filtered)

    # 4. 颜色直方图 + 保存可视化图
    hist_r, hist_g, hist_b = compute_color_histogram(img_bgr)
    plt.figure(figsize=(8, 5))
    plt.plot(hist_r, color="red", label="Red")
    plt.plot(hist_g, color="green", label="Green")
    plt.plot(hist_b, color="blue", label="Blue")
    plt.title("RGB颜色直方图")
    plt.xlabel("像素值")
    plt.ylabel("归一化频率")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(exp_folder, "颜色直方图.jpg"), dpi=150, bbox_inches='tight')
    plt.close()

    # 5. 纹理特征 + 保存到npy文件
    texture_features = extract_texture_features(img_gray)
    np.save(os.path.join(exp_folder, "纹理特征.npy"), texture_features)
    # 额外保存特征为txt（方便实验报告复制）
    with open(os.path.join(exp_folder, "纹理特征.txt"), "w", encoding="utf-8") as f:
        f.write("=== 实验一纹理特征 ===\n")
        for key, value in texture_features.items():
            f.write(f"{key}：{value:.6f}\n")

    plt.figure(figsize=(18, 8))
    # 原始图像
    plt.subplot(2, 3, 1)
    plt.imshow(img_rgb)
    plt.title("原始图像")
    plt.axis("off")
    # Sobel-X
    plt.subplot(2, 3, 2)
    plt.imshow(sobel_x, cmap="gray")
    plt.title("Sobel X滤波")
    plt.axis("off")
    # Sobel-Y
    plt.subplot(2, 3, 3)
    plt.imshow(sobel_y, cmap="gray")
    plt.title("Sobel Y滤波")
    plt.axis("off")
    # 指定卷积核
    plt.subplot(2, 3, 6)
    plt.imshow(custom_filtered, cmap="gray")
    plt.title("指定卷积核滤波")
    plt.axis("off")
    # 颜色直方图
    plt.subplot(2, 3, 4)
    plt.plot(hist_r, color="red", label="Red")
    plt.plot(hist_g, color="green", label="Green")
    plt.plot(hist_b, color="blue", label="Blue")
    plt.title("RGB颜色直方图")
    plt.xlabel("像素值")
    plt.ylabel("归一化频率")
    plt.legend()
    # Sobel合并边缘
    plt.subplot(2, 3, 5)
    plt.imshow(sobel_combined, cmap="gray")
    plt.title("Sobel合并边缘图")
    plt.axis("off")
    # 保存总可视化图
    plt.tight_layout()
    plt.savefig(os.path.join(exp_folder, "实验一总结果图.jpg"), dpi=150, bbox_inches='tight')
    plt.close()

    # 打印结果汇总
    print("\n=== 实验一结果汇总 ===")
    print(f"所有结果已保存到文件夹：{exp_folder}")
    print("生成的文件列表：")
    file_list = os.listdir(exp_folder)
    for i, file in enumerate(file_list, 1):
        print(f"{i}. {file}")
    print("\n纹理特征：")
    for key, value in texture_features.items():
        print(f"{key}：{value:.6f}")


# ===================== 8. 调用实验 =====================
if __name__ == "__main__":
    # 替换为你的图像路径
    IMAGE_PATH = "exp1.png"
    # 实验结果文件夹名称
    EXP_FOLDER = "实验一结果"

    # 执行实验
    image_filtering_experiment(IMAGE_PATH, EXP_FOLDER)
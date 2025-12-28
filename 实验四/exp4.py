import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import os

# -------------------------- 解决OMP冲突：先设置环境变量 --------------------------
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 关键：允许重复加载OpenMP库
os.environ["OMP_NUM_THREADS"] = "1"  # 限制线程数，减少冲突


# -------------------------- 1. 加载模型--------------------------
def load_bicycle_model(model_path):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # 替换分类头（和训练时一致）
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)  # 2类：背景+单车

    # 加载训练好的权重
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=True)

    # 加载权重到模型
    model.load_state_dict(checkpoint['model_state_dict'])

    # 设置为评估模式
    model.eval()
    return model


# -------------------------- 2. 图片预处理 --------------------------
def preprocess_image(img_path):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    img = Image.open(img_path).convert('RGB')
    img_tensor = transform(img)
    return img, img_tensor


# -------------------------- 3. 检测并可视化结果 --------------------------
def detect_bicycle(model, img_path, threshold=0.5):
    # 预处理图片
    img_pil, img_tensor = preprocess_image(img_path)

    # 推理（无梯度计算）
    with torch.no_grad():
        prediction = model([img_tensor])[0]  # 模型输出是列表，取第一个元素

    # 过滤置信度>threshold的检测框
    boxes = prediction['boxes'][prediction['scores'] > threshold].numpy()
    scores = prediction['scores'][prediction['scores'] > threshold].numpy()

    # 绘制检测框
    draw = ImageDraw.Draw(img_pil)
    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = box
        # 画框（红色，宽度2）
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=2)
        # 写置信度
        draw.text((x1, y1 - 15), f"bicycle {score:.2f}", fill=(255, 0, 0))

    # 显示结果
    plt.figure(figsize=(10, 8))
    plt.imshow(img_pil)
    plt.axis('off')
    plt.show()
    img_pil.save("exp4_detection_result.jpg")
    # 输出单车位置（返回所有检测框的坐标）
    print("=" * 50)
    print("检测到的共享单车位置（x1, y1, x2, y2）：")
    if len(boxes) == 0:
        print("未检测到共享单车！")
    else:
        for i, box in enumerate(boxes, 1):
            print(f"单车{i}：{box.tolist()}")
    print("=" * 50)


# -------------------------- 主函数 --------------------------
if __name__ == "__main__":
    # 模型路径（替换为你的模型文件路径）
    MODEL_PATH = "bicycle_detection_model.pth"
    # 待检测图片路径
    IMG_PATH = "exp4.jpg"

    print("正在加载模型...")
    model = load_bicycle_model(MODEL_PATH)
    # 检测并输出结果
    print("正在检测图片...")
    detect_bicycle(model, IMG_PATH)
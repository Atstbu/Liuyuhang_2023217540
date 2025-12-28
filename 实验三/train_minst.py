import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image
import os
import numpy as np

# ===================== 1. 配置基础参数 =====================
# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 数据集路径（替换为你的实际路径）
TRAIN_DATA_PATH = r"D:\HuaweiMoveData\Users\lyuhang\Desktop\机器视觉\minist_dataset\training"
TEST_DATA_PATH = r"D:\HuaweiMoveData\Users\lyuhang\Desktop\机器视觉\minist_dataset\our_testing_without_labels"

# 训练参数
BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 0.001

# ===================== 2. 数据预处理 =====================
# 定义预处理流程
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # 转为单通道灰度图
    transforms.Resize((28, 28)),  # 调整为MNIST的28x28尺寸
    transforms.ToTensor(),  # 转为Tensor，范围[0,1]
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST官方均值/方差，提升训练效果
])

# ===================== 3. 加载训练集（按文件夹标签） =====================
# ImageFolder会自动将子文件夹名（0-9）作为标签，加载对应图片
train_dataset = ImageFolder(
    root=TRAIN_DATA_PATH,
    transform=transform
)
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,  # 打乱训练集，提升泛化能力
    num_workers=0  # Windows系统建议设为0，避免多线程报错
)

# 验证训练集标签映射（确认文件夹0对应标签0，以此类推）
print("训练集标签映射：", train_dataset.class_to_idx)


# ===================== 4. 定义无标签测试集加载器 =====================
class UnlabeledMNISTDataset(Dataset):
    """加载无标签的测试集图片"""

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [os.path.join(root_dir, f) for f in os.listdir(root_dir)
                            if f.endswith(('.png'))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('L')
        if self.transform:
            img = self.transform(img)
        # 返回图片和文件名
        return img, os.path.basename(img_path)


# 加载无标签测试集
test_dataset = UnlabeledMNISTDataset(
    root_dir=TEST_DATA_PATH,
    transform=transform
)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ===================== 5. 定义CNN模型 =====================
class MNIST_CNN(nn.Module):
    def __init__(self):
        super(MNIST_CNN, self).__init__()
        # 卷积层（提取图像特征）
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # 输入1通道，输出32通道
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 池化后尺寸14x14

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 池化后尺寸7x7

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        # 全连接层（分类）
        self.fc_layers = nn.Sequential(
            nn.Flatten(),  # 展平为一维向量
            nn.Linear(64 * 7 * 7, 64),  # 7x7x64 → 64维特征
            nn.ReLU(),
            nn.Linear(64, 10)  # 64 → 10（数字0-9）
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


# ===================== 6. 初始化模型/损失函数/优化器 =====================
model = MNIST_CNN().to(device)
criterion = nn.CrossEntropyLoss()  # 分类任务核心损失函数
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


# ===================== 7. 训练函数 =====================
def train_model(model, train_loader, criterion, optimizer, epochs):
    model.train()  # 切换为训练模式
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            # 数据移到设备上（GPU/CPU）
            images, labels = images.to(device), labels.to(device)

            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 反向传播+优化
            optimizer.zero_grad()  # 清空梯度
            loss.backward()  # 计算梯度
            optimizer.step()  # 更新参数

            # 统计训练集准确率
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # 每100个批次打印一次进度
            if (batch_idx + 1) % 100 == 0:
                print(
                    f'Epoch [{epoch + 1}/{epochs}], Batch [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')

        # 打印每轮训练结果
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        print(f'Epoch [{epoch + 1}/{epochs}] 完成 | 平均损失: {epoch_loss:.4f} | 训练集准确率: {epoch_acc:.2f}%')


# ===================== 8. 测试函数 =====================
def test_unlabeled_data(model, test_loader):
    model.eval()  # 切换为评估模式
    predictions = {}  # 存储{文件名: 预测数字}

    with torch.no_grad():  # 禁用梯度计算，提升速度
        for images, filenames in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            # 保存每个文件的预测结果
            for filename, pred in zip(filenames, predicted.cpu().numpy()):
                predictions[filename] = int(pred)

    # 打印测试集预测结果
    print("\n测试集预测结果（文件名: 预测数字）：")
    for filename, pred in predictions.items():
        print(f"{filename} -> {pred}")
    return predictions


# ===================== 9. 执行训练+测试 =====================
if __name__ == '__main__':
    # 训练模型
    print("开始训练模型...")
    train_model(model, train_loader, criterion, optimizer, EPOCHS)

    # 保存训练好的模型
    torch.save(model.state_dict(), 'mnist_model.pth')
    print("\n模型已保存为 mnist_model.pth")

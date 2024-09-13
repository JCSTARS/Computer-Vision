import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision.models as models

import logging
from datetime import datetime
# 获取当前时间并格式化为字符串，例如 '2023-09-12_18-30-00'
current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
# 配置 logging
log_filename = f'training_log_{current_time}.txt'
logging.basicConfig(filename=log_filename, level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# 数据预处理，将图像大小调整为 224x224，并进行归一化处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet 的输入大小为 224x224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet 预训练模型的标准化
])

# 加载 CIFAR-10 数据集
train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)

# 数据加载器
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 加载预训练的 VGG16 模型
model = models.efficientnet_b0(pretrained=True)

# 替换 EfficientNet 的分类头，使其适应 CIFAR-10 的 10 个类别
num_ftrs = model.classifier[1].in_features  # EfficientNet 的最后一层是 model.classifier[1]
model.classifier[1] = nn.Linear(num_ftrs, 10)

# 将模型移动到 GPU（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 评估函数
def evaluate(model, data_loader):
    model.eval()  # 设置模型为评估模式
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

max_test_acc = 0
# 训练过程
num_epochs = 50
for epoch in range(num_epochs):
    model.train()  # 设置模型为训练模式
    running_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    # 每个 epoch 结束后评估
    train_accuracy = evaluate(model, train_loader)
    test_accuracy = evaluate(model, test_loader)
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/i:.4f}, Train Accuracy: {train_accuracy*100:.2f}%, Test Accuracy: {test_accuracy*100:.2f}%")
    logging.info(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/i:.4f}, Train Accuracy: {train_accuracy*100:.2f}%, Test Accuracy: {test_accuracy*100:.2f}%")

    if test_accuracy > max_test_acc and test_accuracy > 0.9:
        max_test_acc = test_accuracy
        # 保存最佳模型
        torch.save(model.state_dict(), 'best_model_vit_b_16_cifar10.pth')

import torch
import torchvision
import torchvision.transforms as transforms
from resnet50_cifar10 import *
import logging
from datetime import datetime
# 获取当前时间并格式化为字符串，例如 '2023-09-12_18-30-00'
current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
# 配置 logging
log_filename = f'training_log_{current_time}.txt'
logging.basicConfig(filename=log_filename, level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

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
    
# 判断是否有GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_epochs = 20  # 10轮
batch_size = 128  # 128步长
learning_rate = 0.01  # 学习率0.01

# 图像预处理
transform = transforms.Compose([
    transforms.Pad(4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32),
    transforms.ToTensor()])

# CIFAR-10 数据集下载
train_dataset = torchvision.datasets.CIFAR10(root='data/',
                                             train=True,
                                             transform=transform,
                                             download=True)

test_dataset = torchvision.datasets.CIFAR10(root='data/',
                                            train=False,
                                            transform=transforms.ToTensor())

# 数据载入
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


# -50 3-4-6-3 总计(3+4+6+3)*3=48 个conv层 加上开头的两个Conv 一共50层
model = ResNet(ResidualBlock, [3, 4, 6, 3]).to(device)

# 损失函数
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 训练数据集
total_step = len(train_loader)
curr_lr = learning_rate

max_test_acc = 0

for epoch in range(num_epochs):
    model.train()
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_acc = evaluate(model, train_loader)
    test_acc = evaluate(model, test_loader)

    print("Epoch [{}/{}],{:.4f},{:.4f},Loss: {:.4f}"
                  .format(epoch + 1, num_epochs, train_acc, test_acc, loss.item()))
    logging.info("Epoch [%d/%d], Loss: %.4f, Train_acc: %.2f, Test_acc: %.2f", 
                 epoch + 1, num_epochs, loss.item(), train_acc*100, test_acc*100)
    
    if test_acc > max_test_acc:
        max_test_acc = test_acc
        # 保存最佳模型
        torch.save(model.state_dict(), 'best_model_resnet50_cifar10.pth')
    
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from matplotlib import pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 定义卷积神经网络模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 第一个卷积层：输入通道数为3（RGB图像），输出通道数为16，卷积核大小为3x3
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)  # 批归一化
        # 第二个卷积层
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        # 最大池化层，池化核大小为2x2
        self.pool = nn.MaxPool2d(2, 2)
        # 第三个卷积层
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(32)
        self.conv5 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        # 全连接层，输入大小为64 * 16 * 16，输出大小为5 (因为有 5 个分类类别)
        self.fc1 = nn.Linear(64 * 16 * 16, 256)
        self.fc2 = nn.Linear(256, 5)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))  # 第一层卷积、批归一化、ReLU激活函数
        x = F.relu(self.bn2(self.conv2(x)))  # 第二层卷积、批归一化、ReLU激活函数
        x = self.pool(x)                     # 最大池化层
        x = F.relu(self.bn4(self.conv4(x)))  # 第三层卷积、批归一化、ReLU激活函数
        x = F.relu(self.bn5(self.conv5(x)))  # 第四层卷积、批归一化、ReLU激活函数
        x = self.pool(x)                     # 第二个最大池化层
        x = x.view(x.size(0), -1)            # 展平操作
        x = F.relu(self.fc1(x))              # 第一个全连接层、ReLU激活函数
        x = self.fc2(x)                      # 第二个全连接层
        return x


# 创建卷积神经网络模型
model = CNN()
# 定义损失函数为交叉熵损失函数
loss_fn = nn.CrossEntropyLoss()
# 定义优化器为Adam优化器，学习率为0.001，权重衰减为0.0001
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

train_losses = []      # 用于存储训练过程中训练样本损失值
train_accuracies = []  # 用于存储训练过程中训练样本准确率值
test_losses = []       # 用于存储训练过程中测试样本损失值
test_accuracies = []   # 用于存储训练过程中测试样本准确率值


def evaluate_model(cnn_model, data_loader):
    correct = 0  # 初始化正确预测数量
    total = 0    # 初始化样本总数
    loss = 0     # 初始化损失值
    with torch.no_grad():  # 关闭梯度计算
        for inputs, labels in data_loader:                 # 遍历数据集
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = cnn_model(inputs)                    # 前向传播
            loss += loss_fn(outputs, labels).item()        # 累加损失值
            _, predicted = torch.max(outputs, 1)           # 获取预测结果中的最大值及其索引
            total += labels.size(0)                        # 更新样本总数
            correct += (predicted == labels).sum().item()  # 统计正确预测数量

    return loss / len(data_loader), 100 * correct / total  # 返回平均测试损失值和测试准确率


# 训练函数
def train(num_epoch, train_loader, test_loader):
    model.to(device)

    for epoch in range(num_epoch):
        running_loss = 0.0   # 记录累计损失值
        # 遍历训练数据集
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()            # 梯度清零
            outputs = model(inputs)          # 前向传播
            loss = loss_fn(outputs, labels)  # 计算损失值
            loss.backward()                  # 反向传播计算梯度
            optimizer.step()                 # 更新模型参数
            running_loss += loss.item()      # 累加损失值

        # 在训练集上测试模型
        train_loss, train_accuracy = evaluate_model(model, train_loader)
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        # 在测试集上测试模型
        test_loss, test_accuracy = evaluate_model(model, test_loader)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

        # 打印训练和测试结果
        print(f'Epoch [{epoch + 1}/{num_epoch}], '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, '
              f'Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.2f}%')


def main():
    # 处理训练数据和测试数据
    train_transform = transforms.Compose([  # 组合多个图像变换操作
        transforms.Resize((64, 64)),        # 调整图像大小为64x64
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.RandomRotation(15),      # 随机旋转角度在[-15, 15]之间
        transforms.ToTensor(),              # 将图像转换为张量
    ])

    test_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    # 创建训练数据集、训练数据加载器、测试数据集、测试数据加载器
    train_dataset = ImageFolder(root='./train', transform=train_transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
    test_dataset = ImageFolder(root='./test', transform=test_transform)
    test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)

    # 训练模型
    train(num_epoch=50, train_loader=train_loader, test_loader=test_loader)

    # 绘制训练过程的损失函数值曲线和准确率曲线
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.plot(test_losses, label='Test Loss', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy', color='blue')
    plt.plot(test_accuracies, label='Test Accuracy', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Test Accuracy')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()

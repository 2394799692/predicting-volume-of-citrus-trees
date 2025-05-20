import torch
from torch.utils.data import DataLoader, Dataset
from torch import nn, optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np
import random
import os
from PIL import Image
from torchvision import transforms
import copy


# 设置随机种子
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


# 固定的随机种子值
seed = 6000
set_seed(seed)

# 读取数据
file_path = 'D:/data/336.csv'
data = pd.read_csv(file_path, encoding='ISO-8859-1')

# 提取输入和输出
X = data[['x', 'y', 'z']].values
y = data[['tiji', 'biaomianji']].values

# 标准化数据
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# 固定训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=seed)

# 转换为 PyTorch 的 Tensor 并确保类型为 Float
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# 定义图像转换
image_transforms = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])


# 自定义数据集类
class MultimodalDataset(Dataset):
    def __init__(self, text_data, image_dir, targets, transform=None):
        self.text_data = text_data
        self.image_dir = image_dir
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.text_data)

    def __getitem__(self, idx):
        text = self.text_data[idx].astype(np.float32)  # 确保文本数据为 Float 类型
        image1_path = os.path.join(self.image_dir, f"{idx + 1}-c.png")
        image2_path = os.path.join(self.image_dir, f"{idx + 1}-z.png")

        image1 = Image.open(image1_path)
        image2 = Image.open(image2_path)

        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)

        target = self.targets[idx].astype(np.float32)  # 确保目标数据为 Float 类型
        return text, image1, image2, target


# 创建数据集和数据加载器
image_dir = 'D:/data/image'
train_dataset = MultimodalDataset(X_train, image_dir, y_train, transform=image_transforms)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

test_dataset = MultimodalDataset(X_test, image_dir, y_test, transform=image_transforms)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# 定义多模态神经网络
class MultimodalNN(nn.Module):
    def __init__(self):
        super(MultimodalNN, self).__init__()
        self.text_fc1 = nn.Linear(3, 128)
        self.text_fc2 = nn.Linear(128, 64)

        self.image_conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.image_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.image_conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.image_fc1 = nn.Linear(32 * 32 * 32, 128)

        self.fc1 = nn.Linear(64 + 128 + 128, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)
        self.dropout = nn.Dropout(0.2)

    def forward(self, text_input, image1_input, image2_input):
        # 文本数据处理
        text_input = text_input.float()
        text_features = torch.relu(self.text_fc1(text_input))
        text_features = torch.relu(self.text_fc2(text_features))

        # 图像数据处理
        image1_features = self.image_pool(torch.relu(self.image_conv1(image1_input)))
        image1_features = self.image_pool(torch.relu(self.image_conv2(image1_features)))
        image1_features = image1_features.view(image1_features.size(0), -1)
        image1_features = torch.relu(self.image_fc1(image1_features))

        image2_features = self.image_pool(torch.relu(self.image_conv1(image2_input)))
        image2_features = self.image_pool(torch.relu(self.image_conv2(image2_features)))
        image2_features = image2_features.view(image2_features.size(0), -1)
        image2_features = torch.relu(self.image_fc1(image2_features))

        # 合并特征
        combined = torch.cat((text_features, image1_features, image2_features), dim=1)
        combined = torch.relu(self.fc1(combined))
        combined = self.dropout(combined)
        combined = torch.relu(self.fc2(combined))
        output = self.fc3(combined)
        return output


model = MultimodalNN().cuda()

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001)

# 使用学习率调度器
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)

# 训练模型
num_epochs = 100
early_stopping_patience = 20
best_model_wts = copy.deepcopy(model.state_dict())
best_loss = float('inf')
early_stopping_counter = 0

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for text_inputs, image1_inputs, image2_inputs, targets in train_loader:
        text_inputs, image1_inputs, image2_inputs, targets = text_inputs.cuda(), image1_inputs.cuda(), image2_inputs.cuda(), targets.cuda()

        # 前向传播
        outputs = model(text_inputs, image1_inputs, image2_inputs)
        loss = criterion(outputs, targets)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)
    scheduler.step(epoch_loss)

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}')

    # Early stopping
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        best_model_wts = copy.deepcopy(model.state_dict())
        early_stopping_counter = 0
    else:
        early_stopping_counter += 1

    if early_stopping_counter >= early_stopping_patience:
        print("Early stopping")
        break

# 加载最佳模型权重
model.load_state_dict(best_model_wts)
# 保存最佳模型
torch.save(model.state_dict(), 'best_model1.pth')
# 预测
model.eval()
predictions = []
with torch.no_grad():
    for text_inputs, image1_inputs, image2_inputs, _ in test_loader:
        text_inputs, image1_inputs, image2_inputs = text_inputs.cuda(), image1_inputs.cuda(), image2_inputs.cuda()
        outputs = model(text_inputs, image1_inputs, image2_inputs)
        predictions.append(outputs.cpu().numpy())

predictions = np.vstack(predictions)

# 将预测结果反标准化
predictions = scaler_y.inverse_transform(predictions)
y_test_original = scaler_y.inverse_transform(y_test)

# 计算评估指标
r2_1 = r2_score(y_test_original[:, 0], predictions[:, 0])
r2_2 = r2_score(y_test_original[:, 1], predictions[:, 1])

mse_1 = mean_squared_error(y_test_original[:, 0], predictions[:, 0])
mse_2 = mean_squared_error(y_test_original[:, 1], predictions[:, 1])

rmse_1 = np.sqrt(mse_1)
rmse_2 = np.sqrt(mse_2)

mae_1 = mean_absolute_error(y_test_original[:, 0], predictions[:, 0])
mae_2 = mean_absolute_error(y_test_original[:, 1], predictions[:, 1])

# MAPE计算
mape_1 = np.mean(np.abs((y_test_original[:, 0] - predictions[:, 0]) / y_test_original[:, 0])) * 100
mape_2 = np.mean(np.abs((y_test_original[:, 1] - predictions[:, 1]) / y_test_original[:, 1])) * 100



print(f'R² score for the first output: {r2_1:.4f}')
print(f'R² score for the second output: {r2_2:.4f}')

print(f'MSE for the first output: {mse_1:.4f}')
print(f'MSE for the second output: {mse_2:.4f}')

print(f'RMSE for the first output: {rmse_1:.4f}')
print(f'RMSE for the second output: {rmse_2:.4f}')

print(f'MAE for the first output: {mae_1:.4f}')
print(f'MAE for the second output: {mae_2:.4f}')

print(f'MAPE for the first output: {mape_1:.4f}%')
print(f'MAPE for the first output: {mape_1:.4f}%')
print(f'MAPE for the second output: {mape_2:.4f}%')

# 输出预测结果
print(predictions)


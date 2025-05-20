import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import nn, optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np
import random
import copy
from torchvision import models, transforms
from PIL import Image
import os

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

# 转换为 PyTorch 的 Tensor
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# 加载图像
def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    image = preprocess(image)
    return image

# 创建图像特征提取器（使用预训练的ResNet）
class ResNetFeatureExtractor(nn.Module):
    def __init__(self):
        super(ResNetFeatureExtractor, self).__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])

    def forward(self, x):
        with torch.no_grad():
            x = self.feature_extractor(x)
        return x.view(x.size(0), -1)

image_feature_extractor = ResNetFeatureExtractor().cuda()

# 创建 DataLoader
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, text_data, targets, image_dir, transform=None):
        self.text_data = text_data
        self.targets = targets
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.text_data)

    def __getitem__(self, idx):
        text_features = self.text_data[idx]
        target = self.targets[idx]
        image1_path = os.path.join(self.image_dir, f'{idx+1}-c.png')
        image2_path = os.path.join(self.image_dir, f'{idx+1}-z.png')

        image1 = load_image(image1_path)
        image2 = load_image(image2_path)

        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)

        return text_features, target, image1, image2

train_dataset = CustomDataset(X_train_tensor, y_train_tensor, 'D:/data/image')
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 定义更深的神经网络模型
class DeeperNN(nn.Module):
    def __init__(self):
        super(DeeperNN, self).__init__()
        self.text_fc1 = nn.Linear(3, 128)
        self.text_fc2 = nn.Linear(128, 64)

        self.image_fc1 = nn.Linear(2048, 512)
        self.image_fc2 = nn.Linear(512, 64)

        self.fc1 = nn.Linear(192, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)
        self.dropout = nn.Dropout(0.2)

    def forward(self, text_input, image1, image2):
        text_features = torch.relu(self.text_fc1(text_input))
        text_features = torch.relu(self.text_fc2(text_features))

        image1_features = torch.relu(self.image_fc1(image1))
        image1_features = torch.relu(self.image_fc2(image1_features))

        image2_features = torch.relu(self.image_fc1(image2))
        image2_features = torch.relu(self.image_fc2(image2_features))

        combined = torch.cat((text_features, image1_features, image2_features), dim=1)
        combined = torch.relu(self.fc1(combined))
        combined = self.dropout(combined)
        combined = torch.relu(self.fc2(combined))
        output = self.fc3(combined)
        return output

model = DeeperNN().cuda()

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
    for text_inputs, targets, image1_inputs, image2_inputs in train_loader:
        text_inputs, targets = text_inputs.cuda(), targets.cuda()
        image1_inputs, image2_inputs = image1_inputs.cuda(), image2_inputs.cuda()

        # 提取图像特征
        image1_features = image_feature_extractor(image1_inputs)
        image2_features = image_feature_extractor(image2_inputs)

        # 前向传播
        outputs = model(text_inputs, image1_features, image2_features)
        loss = criterion(outputs, targets)  # 修正：应该是 targets 而不是 text_inputs

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

# 保存模型
torch.save(model.state_dict(), 'best_model3.pth')

# 创建测试集的 DataLoader
test_dataset = CustomDataset(X_test_tensor, y_test_tensor, 'D:/data/image')
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 预测
model.eval()
predictions = []
with torch.no_grad():
    for text_inputs, targets, image1_inputs, image2_inputs in test_loader:
        text_inputs, image1_inputs, image2_inputs = text_inputs.cuda(), image1_inputs.cuda(), image2_inputs.cuda()

        # 提取图像特征
        image1_features = image_feature_extractor(image1_inputs)
        image2_features = image_feature_extractor(image2_inputs)

        # 前向传播
        outputs = model(text_inputs, image1_features, image2_features)
        predictions.append(outputs.cpu().numpy())

# 将预测结果反标准化
predictions = np.vstack(predictions)
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
print(f'MAPE for the second output: {mape_2:.4f}%')

# 输出预测结果
print(predictions)
import joblib

# 保存标准化器
joblib.dump(scaler_X, 'scaler_X2.pkl')
joblib.dump(scaler_y, 'scaler_y2.pkl')
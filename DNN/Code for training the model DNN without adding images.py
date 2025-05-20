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

# 创建 DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 定义更深的神经网络模型
class DeeperNN(nn.Module):
    def __init__(self):
        super(DeeperNN, self).__init__()
        self.fc1 = nn.Linear(3, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64, 2)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn3(self.fc3(x)))
        x = self.fc4(x)
        return x

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
    for inputs, targets in train_loader:
        inputs, targets = inputs.cuda(), targets.cuda()

        # 前向传播
        outputs = model(inputs)
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
torch.save(model.state_dict(), 'best_model0.pth')
# 预测
model.eval()
with torch.no_grad():
    X_test_tensor = X_test_tensor.cuda()
    predictions = model(X_test_tensor)
    predictions = predictions.cpu().numpy()

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
print(f'MAPE for the second output: {mape_2:.4f}%')

# 创建 DataFrame 并保存为 Excel 文件
predictions_df = pd.DataFrame(predictions, columns=['Predicted_Volume', 'Predicted_Surface_Area'])
y_test_df = pd.DataFrame(y_test_original, columns=['True_Volume', 'True_Surface_Area'])
# 合并真实值和预测值
results_df = pd.concat([y_test_df, predictions_df], axis=1)

# 保存到 Excel 文件
results_df.to_excel('0.xlsx', index=False)
import joblib

# 保存标准化器
joblib.dump(scaler_X, 'scaler_X.pkl')
joblib.dump(scaler_y, 'scaler_y.pkl')
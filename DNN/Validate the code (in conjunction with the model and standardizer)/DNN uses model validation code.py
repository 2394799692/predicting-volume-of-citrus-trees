import torch
from torch import nn
import pandas as pd
import joblib

# 定义更深的神经网络模型（与训练时相同）
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

# 加载保存的模型权重
model = DeeperNN().cuda()
model.load_state_dict(torch.load('best_model0.pth'))
model.eval()

# 加载标准化器
scaler_X = joblib.load('scaler_X.pkl')
scaler_y = joblib.load('scaler_y.pkl')

# 读取并预处理新的数据
new_data_file_path = 'D:/data/zyp.csv'
new_data = pd.read_csv(new_data_file_path, encoding='ISO-8859-1')

# 提取输入
X_new = new_data[['x', 'y', 'z']].values

# 使用之前的标准化器进行数据标准化
X_new_scaled = scaler_X.transform(X_new)

# 转换为 PyTorch 的 Tensor
X_new_tensor = torch.tensor(X_new_scaled, dtype=torch.float32).cuda()

# 进行预测
with torch.no_grad():
    predictions_new = model(X_new_tensor)
    predictions_new = predictions_new.cpu().numpy()

# 将预测结果反标准化
predictions_new = scaler_y.inverse_transform(predictions_new)

# 创建 DataFrame 并保存为 Excel 文件
predictions_new_df = pd.DataFrame(predictions_new, columns=['Predicted_Volume', 'Predicted_Surface_Area'])
predictions_new_df.to_excel('zyp-new_predictions.xlsx', index=False)

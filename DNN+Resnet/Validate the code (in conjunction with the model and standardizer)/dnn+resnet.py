
import joblib

import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import nn, optim
import pandas as pd

import numpy as np

from torchvision import models, transforms
from PIL import Image
import os
# 定义更深的神经网络模型（与训练时相同）
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

# 加载保存的模型权重
model = DeeperNN().cuda()
model.load_state_dict(torch.load('best_model3.pth'))
model.eval()

# 加载标准化器
scaler_X = joblib.load('scaler_X2.pkl')
scaler_y = joblib.load('scaler_y2.pkl')

# 读取并预处理新的数据
new_data_file_path = 'D:/data/zyp.csv'
new_data = pd.read_csv(new_data_file_path, encoding='ISO-8859-1')

# 提取输入
X_new = new_data[['x', 'y', 'z']].values

# 使用之前的标准化器进行数据标准化
X_new_scaled = scaler_X.transform(X_new)

# 转换为 PyTorch 的 Tensor
X_new_tensor = torch.tensor(X_new_scaled, dtype=torch.float32).cuda()

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

# 创建自定义数据集类
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, text_data, image_dir, transform=None):
        self.text_data = text_data
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.text_data)

    def __getitem__(self, idx):
        text_features = self.text_data[idx]
        # image1_path = os.path.join(self.image_dir, f'{idx+1}-c.png')
        # image2_path = os.path.join(self.image_dir, f'{idx+1}-z.png')
        # 修改为 .jpg
        image1_path = os.path.join(self.image_dir, f'{idx + 1}-c.jpg')
        image2_path = os.path.join(self.image_dir, f'{idx + 1}-z.jpg')

        image1 = load_image(image1_path)
        image2 = load_image(image2_path)

        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)

        return text_features, image1, image2

# 创建测试集的 DataLoader
test_dataset = CustomDataset(X_new_tensor, 'D:/data/yanzheng/资源圃/image')
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 预测
model.eval()
predictions = []
with torch.no_grad():
    for text_inputs, image1_inputs, image2_inputs in test_loader:
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

# 创建 DataFrame 并保存为 Excel 文件
predictions_new_df = pd.DataFrame(predictions, columns=['Predicted_Volume', 'Predicted_Surface_Area'])
predictions_new_df.to_excel('zyp-predictions.xlsx', index=False)

# 输出预测结果
print(predictions)

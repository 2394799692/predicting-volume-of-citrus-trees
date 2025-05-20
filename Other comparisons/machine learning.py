import torch
import numpy as np
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb
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

# 定义模型
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=seed),
    'Gradient Boosting': GradientBoostingRegressor(random_state=seed),
    'SVR': SVR(kernel='rbf'),
    'KNN': KNeighborsRegressor(n_neighbors=5),
    'XGBoost': xgb.XGBRegressor(objective='reg:squarederror', random_state=seed)
}

# 训练和评估模型
results = {}
predictions_dict = {}

for name, model in models.items():
    if name == 'SVR':
        # SVR 需要单独处理每个目标变量
        svr_model_1 = SVR(kernel='rbf')
        svr_model_2 = SVR(kernel='rbf')
        svr_model_1.fit(X_train, y_train[:, 0])
        svr_model_2.fit(X_train, y_train[:, 1])
        predictions_1 = svr_model_1.predict(X_test)
        predictions_2 = svr_model_2.predict(X_test)
        predictions = np.column_stack((predictions_1, predictions_2))
    else:
        # 对于其他模型，分别训练每个目标变量
        model_1 = copy.deepcopy(model)
        model_2 = copy.deepcopy(model)
        model_1.fit(X_train, y_train[:, 0])
        model_2.fit(X_train, y_train[:, 1])
        predictions_1 = model_1.predict(X_test)
        predictions_2 = model_2.predict(X_test)
        predictions = np.column_stack((predictions_1, predictions_2))

    # 反标准化
    predictions = scaler_y.inverse_transform(predictions)
    y_test_original = scaler_y.inverse_transform(y_test)

    # 保存真实值和预测值
    predictions_dict[name] = pd.DataFrame({
        'True Tiji': y_test_original[:, 0],
        'Predicted Tiji': predictions[:, 0],
        'True Biaomianji': y_test_original[:, 1],
        'Predicted Biaomianji': predictions[:, 1]
    })

    # 计算评估指标
    r2_1 = r2_score(y_test_original[:, 0], predictions[:, 0])
    r2_2 = r2_score(y_test_original[:, 1], predictions[:, 1])
    mse_1 = mean_squared_error(y_test_original[:, 0], predictions[:, 0])
    mse_2 = mean_squared_error(y_test_original[:, 1], predictions[:, 1])
    rmse_1 = np.sqrt(mse_1)
    rmse_2 = np.sqrt(mse_2)
    mae_1 = mean_absolute_error(y_test_original[:, 0], predictions[:, 0])
    mae_2 = mean_absolute_error(y_test_original[:, 1], predictions[:, 1])

    results[name] = {
        'R2 Score (tiji)': r2_1,
        'R2 Score (biaomianji)': r2_2,
        'MSE (tiji)': mse_1,
        'MSE (biaomianji)': mse_2,
        'RMSE (tiji)': rmse_1,
        'RMSE (biaomianji)': rmse_2,
        'MAE (tiji)': mae_1,
        'MAE (biaomianji)': mae_2
    }

# 保存结果到多个Excel文件
for name, df in predictions_dict.items():
    df.to_excel(f'{name}_predictions.xlsx', index=False)

import pprint
pprint.pprint(results)

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch
from torch import nn
from torch.optim import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import wandb

import time
import csv
import ast

import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--lr", type=float, default=0.01,
                        help="Learning rate")

parser.add_argument("--name", type=str, default="CNN",
                        help="model name")

args = parser.parse_args()

# 初始化wanda
wandb.init(
    # set the wandb project where this run will be logged
    project="0714",
    name = args.name,
    # track hyperparameters and run metadata
    config={
    "model": "LSTM",
    "Learning Rate": args.lr
    }
)

# 读取train.csv文件

# 将CSV文件导入为DataFrame
df = pd.read_csv('/Users/cglin/Desktop/0715比赛/dataset/train/train.csv')

# 计算第4列和第5列的平均值
mean_column4 = df['日产液量'].mean()
mean_column5 = df['日产气量'].mean()

# 将第4列和第5列的NaN值替换为各自列的平均值
df['日产液量'].fillna(mean_column4, inplace=True)
df['日产气量'].fillna(mean_column5, inplace=True)

# 将第二列和第三列的字符串形式的列表转换为实际的列表
df['位移'] = df['位移'].apply(ast.literal_eval)
df['载荷'] = df['载荷'].apply(ast.literal_eval)

# 平铺第二列
column2_df = pd.DataFrame(df['位移'].to_list())
column2_df.columns = [f'位移_{i}' for i in column2_df.columns]
df = pd.concat([df.drop('位移', axis=1), column2_df], axis=1)

# 平铺第三列
column3_df = pd.DataFrame(df['载荷'].to_list())
column3_df.columns = [f'载荷_{i}' for i in column3_df.columns]
df = pd.concat([df.drop('载荷', axis=1), column3_df], axis=1)

df = df.iloc[:, 1:]
lables = df['类别'] - 1
print(lables.describe())
df = df.drop('类别', axis=1)

X = df.values
print(df.columns)
y = lables.values

# 分割数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=123)
# X_train, y_train = X, y

# 读取test.csv文件

# 将CSV文件导入为DataFrame
file_path_test = "/Users/cglin/Desktop/0715比赛/dataset/test/test.csv"
df_test = pd.read_csv(file_path_test)

# 计算第4列和第5列的平均值
mean_column4_test = df_test['日产液量'].mean()
mean_column5_test = df_test['日产气量'].mean()

# 将第4列和第5列的NaN值替换为各自列的平均值
df_test['日产液量'].fillna(mean_column4_test, inplace=True)
df_test['日产气量'].fillna(mean_column5_test, inplace=True)

# 将第二列和第三列的字符串形式的列表转换为实际的列表
df_test['位移'] = df_test['位移'].apply(ast.literal_eval)
df_test['载荷'] = df_test['载荷'].apply(ast.literal_eval)

# 平铺第二列
column2_df_test = pd.DataFrame(df_test['位移'].to_list())
column2_df_test.columns = [f'位移_{i}' for i in column2_df_test.columns]
df_test = pd.concat([df_test.drop('位移', axis=1), column2_df_test], axis=1)

# 平铺第三列
column3_df_test = pd.DataFrame(df_test['载荷'].to_list())
column3_df_test.columns = [f'载荷_{i}' for i in column3_df_test.columns]
df_test = pd.concat([df_test.drop('载荷', axis=1), column3_df_test], axis=1)

df_test = df_test.iloc[:, 1:]
# print(df.columns)

X_Test = df_test.values

# 创建一个MinMaxScaler对象
scaler = MinMaxScaler()

# 使用训练集和测试集数据来拟合 scaler，并对它们进行转换
X_1 = np.concatenate((X_train, X_test), axis=0)
X_all = np.concatenate((X_Test, X_1), axis=0)

scaler.fit(X_all)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# 定义 LSTM 模型
class LSTMModel(nn.Module):
    def __init__(self):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size=1, hidden_size=50, num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=1, hidden_size=50, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(102, 50)
        self.fc2 = nn.Linear(50, 9)

    def forward(self, x):
        x3 = x[:, 0:2]
        x1 = x[:, 2:122, None]
        x2 = x[:, 122:, None]
        
        x1 = torch.from_numpy(x1)
        x1 = x1.float()
        x2 = torch.from_numpy(x2)
        x2 = x2.float()
        x3 = torch.from_numpy(x3)
        x3 = x3.float() 
        
        out1, _ = self.lstm1(x1)
        out2, _ = self.lstm2(x2)
        out = torch.cat((out1[:, -1, :], out2[:, -1, :], x3), dim=-1)
        out = torch.relu(self.fc1(out))
        out = self.fc2(out)
        return out

# 初始化模型、优化器和损失函数
model = LSTMModel()
optimizer = Adam(model.parameters(), lr = args.lr)
criterion = nn.CrossEntropyLoss()

def l1_regularizer(model):
    l1_reg = None
    for W in model.parameters():
        if l1_reg is None:
            l1_reg = W.norm(1)
        else:
            l1_reg = l1_reg + W.norm(1)
    return l1_reg

# 训练模型
y_train = torch.from_numpy(y_train)
epoch_num = 200
for epoch in range(epoch_num):  # 这里只是示例，实际训练的 epoch 数量可以调整
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    
    Train_loss =  loss.item()
    # Add L1 regularization
    loss += 0.0001 * l1_regularizer(model)
    
    
    loss.backward()
    optimizer.step()

    # 每训练1个epoch打印一次损失值
    if epoch % 1 == 0:
        print('Train Epoch: {}\tLoss: {:.6f}'.format(epoch, loss.item()))
        print('Train Epoch: {}\tTrain_Loss: {:.6f}'.format(epoch, Train_loss))
        wandb.log({"loss":  loss.item()})
        

# 测试模型
model.eval()
with torch.no_grad():
    outputs = model(X_test)
predicted = torch.argmax(outputs, dim=1).cpu().numpy()

y_test = torch.from_numpy(y_test)

precision = precision_score(y_test.cpu().numpy(), predicted, average='weighted')
recall = recall_score(y_test.cpu().numpy(), predicted, average='weighted')
f1 = f1_score(y_test.cpu().numpy(), predicted, average='weighted')
accuracy = accuracy_score(y_test.cpu().numpy(), predicted)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

wandb.log({"Accuracy":  accuracy})
wandb.log({"Precision":  precision})
wandb.log({"Recall":  recall})
wandb.log({"F1 Score":  f1})

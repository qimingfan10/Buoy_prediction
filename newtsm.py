import numpy as np
import pandas as pd
import os
import re
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 强制使用 CPU (PyTorch 默认 CPU，无需显式设置)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import concurrent.futures
import time  # 导入 time 模块

# 自定义 TSMixer Block (PyTorch 版本)
class MixerLayer(nn.Module):
    def __init__(self, num_features, num_tokens, token_mix_dim, channel_mix_dim, dropout_rate=0.1):
        super(MixerLayer, self).__init__()
        self.num_features = num_features
        self.num_tokens = num_tokens
        self.token_mix_dim = token_mix_dim
        self.channel_mix_dim = channel_mix_dim
        self.dropout_rate = dropout_rate

        self.token_mixing_mlp = nn.Sequential(
            nn.Linear(num_tokens, token_mix_dim), # 注意输入维度是 num_tokens
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(token_mix_dim, num_tokens) # 输出维度是 num_tokens
        )

        self.channel_mixing_mlp = nn.Sequential(
            nn.Linear(num_features, channel_mix_dim), # 注意输入维度是 num_features
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(channel_mix_dim, num_features) # 输出维度是 num_features
        )

        self.ln_token = nn.LayerNorm(num_features) # LayerNorm 在 feature 维度
        self.ln_channel = nn.LayerNorm(num_features) # LayerNorm 在 feature 维度

    def forward(self, x):
        # 1. Token-Mixing (Time-Mixing)
        # 输入形状: (batch_size, sequence_length, num_features)
        # Permute 变为: (batch_size, num_features, sequence_length)
        x_token = x.permute(0, 2, 1) # PyTorch 中 permute 使用维度索引
        x_token = self.token_mixing_mlp(x_token) # (batch_size, num_features, sequence_length)
        x_token = x_token.permute(0, 2, 1) # 恢复形状: (batch_size, sequence_length, num_features)
        x_token = self.ln_token(x_token + x) # 残差连接和 Layer Normalization

        # 2. Channel-Mixing (Feature-Mixing)
        x_channel = self.channel_mixing_mlp(x_token) # (batch_size, sequence_length, num_features)
        x_channel = self.ln_channel(x_channel + x_token) # 残差连接和 Layer Normalization

        return x_channel

def extract_profile_features(df_profile):
    """特征提取函数 (保持不变)"""
    lon = df_profile['lon'].iloc[0]
    lat = df_profile['lat'].iloc[0]
    while lon == 1000 or lat == 1000:
        df_profile = df_profile.iloc[1:]
        if df_profile.empty:
            return None
        lon = df_profile['lon'].iloc[0]
        lat = df_profile['lat'].iloc[0]
    temperature = df_profile['temperature'].values[:245]
    salinity = df_profile['salinity'].values[:245]
    temperature[temperature == 1000] = 0
    salinity[salinity == 1000] = 0
    if len(temperature) < 245:
        temperature = np.pad(temperature, (0, 245 - len(temperature)), 'constant', constant_values=0)
    if len(salinity) < 245:
        salinity = np.pad(salinity, (0, 245 - len(salinity)), 'constant', constant_values=0)
    feature_vector = np.concatenate(([lon], [lat], temperature, salinity))
    if np.count_nonzero(feature_vector == 0) > 256:
        return None
    return feature_vector

def get_profile_label_from_filename(filename):
    """标签函数 (保持不变)"""
    label_part = filename.split('_')[-1]
    label = label_part.split('.')[0]
    print(f'Filename: {filename}, Extracted Label: {label}')
    if label == '1':
        return 1
    else:
        return 0

def natural_sort_key(s):
    """自然排序函数 (保持不变)"""
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

def load_single_file(file_path):
    """加载单个文件的函数"""
    df = pd.read_csv(file_path)
    df['filename'] = os.path.basename(file_path)
    return df

def load_data_from_folder(folder_path, n=1500):
    """加载文件夹数据函数 (使用并行处理)"""
    all_data = []
    profile_filenames = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    profile_filenames.sort(key=natural_sort_key)
    if n is not None:
        profile_filenames = profile_filenames[:n]

    # 使用 ThreadPoolExecutor 并行加载文件
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(load_single_file, os.path.join(folder_path, filename)): filename for filename in profile_filenames}
        for future in concurrent.futures.as_completed(futures):
            try:
                df = future.result()
                all_data.append(df)
            except Exception as e:
                print(f"加载文件时出错: {futures[future]} - {e}")

    return pd.concat(all_data, ignore_index=True)

def load_data_from_csv(folder_path, sequence_length):
    """数据加载函数"""
    all_profile_data = load_data_from_folder(folder_path)
    sequences = []
    labels = []
    profile_filenames = sorted(list(set(all_profile_data['filename'])), key=natural_sort_key)
    
    # 计算需要多少组，每组至少包含 sequence_length 个连续剖面
    group_size = sequence_length
    num_groups = len(profile_filenames) // group_size
    
    for group_idx in range(num_groups):
        start_idx = group_idx * group_size
        end_idx = start_idx + group_size
        group_filenames = profile_filenames[start_idx:end_idx]
        
        # 处理这一组内的序列
        if len(group_filenames) >= sequence_length:
            sequence_features = []
            current_labels = []
            for filename in group_filenames:
                profile_df = all_profile_data[all_profile_data['filename'] == filename].copy()
                feature_vector = extract_profile_features(profile_df)
                if feature_vector is not None:
                    sequence_features.append(feature_vector)
                    current_label = get_profile_label_from_filename(filename)
                    current_labels.append(current_label)
                else:
                    sequence_features = []
                    break
            
            if len(sequence_features) == sequence_length and len(current_labels) == sequence_length:
                if len(set(current_labels)) == 1:
                    sequences.append(sequence_features)
                    labels.append(current_labels[0])
                
    return np.array(sequences), np.array(labels)

def split_data(sequences, labels, train_size=0.6, val_size=0.2, test_size=0.2, random_state=42):
    """三向划分：训练集60%，验证集20%，测试集20%"""
    # 首先划分出测试集
    train_val_sequences, test_sequences, train_val_labels, test_labels = train_test_split(
        sequences, 
        labels, 
        test_size=test_size, 
        random_state=random_state,
        stratify=labels
    )
    
    # 然后将剩下的数据划分为训练集和验证集
    # 验证集比例需要重新计算：0.2 / (1 - 0.2) = 0.25
    train_sequences, val_sequences, train_labels, val_labels = train_test_split(
        train_val_sequences,
        train_val_labels,
        test_size=val_size/(train_size + val_size),
        random_state=random_state,
        stratify=train_val_labels
    )
    
    return train_sequences, val_sequences, test_sequences, train_labels, val_labels, test_labels

# 首先定义路径和序列长度
folder_path = '/root/0'
sequence_length = 4

# 然后加载数据
sequences, labels = load_data_from_csv(folder_path, sequence_length)

# 进行数据集划分
train_sequences, val_sequences, test_sequences, train_labels, val_labels, test_labels = split_data(
    sequences, labels, train_size=0.6, val_size=0.2, test_size=0.2, random_state=42
)

# 检查数据加载
if sequences.size == 0 or labels.size == 0:
    print("错误: 未能成功加载任何数据，请检查数据文件和路径.")
    exit()

print(f"加载数据完成，序列数量: {len(sequences)}, 标签数量: {len(labels)}")
print(f"序列数据形状: {sequences.shape}, 标签数据形状: {labels.shape}")

# 打印各个数据集的形状
print(f"训练集序列形状: {train_sequences.shape}, 训练集标签形状: {train_labels.shape}")
print(f"验证集序列形状: {val_sequences.shape}, 验证集标签形状: {val_labels.shape}")
print(f"测试集序列形状: {test_sequences.shape}, 测试集标签形状: {test_labels.shape}")

# 数据集类 (PyTorch 版本)
class ProfileDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = torch.tensor(sequences, dtype=torch.float32) # 转换为 PyTorch Tensor
        self.labels = torch.tensor(labels, dtype=torch.float32) # 转换为 PyTorch Tensor

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

# 创建数据集和数据加载器
train_dataset = ProfileDataset(train_sequences, train_labels)
val_dataset = ProfileDataset(val_sequences, val_labels)
test_dataset = ProfileDataset(test_sequences, test_labels)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 2. 构建 TSMixer 模型 (PyTorch 版本)
class TSMixerModel(nn.Module):
    def __init__(self, sequence_length, num_features):
        super(TSMixerModel, self).__init__()
        self.input_layer = nn.Linear(num_features, num_features) # 可选的输入层
        self.mixer_layer1 = MixerLayer(num_features=num_features, num_tokens=sequence_length, token_mix_dim=256, channel_mix_dim=2048)
        self.mixer_layer2 = MixerLayer(num_features=num_features, num_tokens=sequence_length, token_mix_dim=256, channel_mix_dim=2048) # 可以添加更多层
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1) #  PyTorch 的 Global Average Pooling 1D
        self.fc_output = nn.Linear(num_features, 1) # 输出层

    def forward(self, x):
        # x 形状: (batch_size, sequence_length, num_features)
        x = self.input_layer(x) # 可选的输入层
        x = self.mixer_layer1(x)
        x = self.mixer_layer2(x)
        # Global Average Pooling 需要输入形状为 (batch_size, num_features, sequence_length) 或 (batch_size, num_features, *)
        # 因此需要 Permute 或 transpose 维度
        x = x.permute(0, 2, 1) # 将 sequence_length 放到最后一个维度 -> (batch_size, num_features, sequence_length)
        x = self.global_avg_pool(x) # -> (batch_size, num_features, 1)
        x = x.squeeze(dim=-1) # 移除最后一个维度，变为 (batch_size, num_features)
        x = self.fc_output(x) # -> (batch_size, 1)
        return torch.sigmoid(x) # 输出 sigmoid 概率

# 初始化模型
num_features = 492
model = TSMixerModel(sequence_length, num_features)

# 3. 定义损失函数和优化器 (PyTorch 版本)
criterion = nn.BCELoss() # 二分类交叉熵损失函数
optimizer = torch.optim.Adam(model.parameters()) # Adam 优化器

# 4. 模型训练 (PyTorch 版本)
epochs = 30
device = torch.device("cuda") # 使用 CPU 训练 (如果 GPU 可用，可以改为 "cuda")
model.to(device) # 将模型移动到指定设备

best_loss = float('inf')  # 初始化最佳损失为无穷大

start_time = time.time()  # 记录开始时间

for epoch in range(epochs):
    model.train() # 设置为训练模式
    train_loss = 0.0
    correct_predictions = 0  # 记录正确预测的数量
    total_predictions = 0  # 记录总预测数量

    for sequences_batch, labels_batch in train_loader:
        sequences_batch, labels_batch = sequences_batch.to(device), labels_batch.to(device) # 数据移动到设备
        optimizer.zero_grad() # 梯度清零
        outputs = model(sequences_batch) # 前向传播
        loss = criterion(outputs, labels_batch.unsqueeze(1)) # 计算损失, labels_batch 需要调整形状为 (batch_size, 1)
        loss.backward() # 反向传播
        optimizer.step() # 更新权重
        train_loss += loss.item()

        # 计算准确率
        predicted_labels_batch = (outputs > 0.5).int().flatten()  # 概率转为标签
        correct_predictions += (predicted_labels_batch == labels_batch.int()).sum().item()  # 统计正确预测
        total_predictions += labels_batch.size(0)  # 更新总预测数量

    avg_train_loss = train_loss / len(train_loader)
    train_accuracy = correct_predictions / total_predictions  # 计算训练准确率
    print(f"Epoch [{epoch+1}/{epochs}], 平均训练损失: {avg_train_loss:.4f}, 训练准确率: {train_accuracy:.4f}")

    # 验证步骤
    model.eval()  # 设置为评估模式
    val_loss = 0.0
    val_correct_predictions = 0  # 记录验证集正确预测的数量
    val_total_predictions = 0  # 记录验证集总预测数量

    with torch.no_grad():
        for sequences_batch, labels_batch in val_loader:
            sequences_batch, labels_batch = sequences_batch.to(device), labels_batch.to(device)
            outputs = model(sequences_batch)
            loss = criterion(outputs, labels_batch.unsqueeze(1))
            val_loss += loss.item()

            # 计算验证集准确率
            predicted_labels_batch = (outputs > 0.5).int().flatten()  # 概率转为标签
            val_correct_predictions += (predicted_labels_batch == labels_batch.int()).sum().item()  # 统计正确预测
            val_total_predictions += labels_batch.size(0)  # 更新总预测数量

    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = val_correct_predictions / val_total_predictions  # 计算验证准确率
    print(f"验证集损失: {avg_val_loss:.4f}, 验证准确率: {val_accuracy:.4f}")

    # 保存最佳权重
    if avg_val_loss < best_loss:
        best_loss = avg_val_loss
        torch.save(model.state_dict(), 'best_model_weights.pth')  # 保存模型权重
        print("保存最佳模型权重.")

end_time = time.time()  # 记录结束时间
total_time = end_time - start_time  # 计算总用时
print(f"总训练时间: {total_time:.2f} 秒")  # 打印总用时

# 5. 模型评估与预测 (PyTorch 版本)
model.eval() # 设置为评估模式
test_loss = 0.0
all_predicted_labels = []
all_true_labels = []

with torch.no_grad(): # 评估阶段不需要计算梯度
    for sequences_batch, labels_batch in test_loader:
        sequences_batch, labels_batch = sequences_batch.to(device), labels_batch.to(device)
        outputs = model(sequences_batch)
        loss = criterion(outputs, labels_batch.unsqueeze(1))
        test_loss += loss.item()
        predicted_labels_batch = (outputs > 0.5).int().flatten() # 概率转为标签
        all_predicted_labels.extend(predicted_labels_batch.cpu().numpy()) # 收集预测标签
        all_true_labels.extend(labels_batch.cpu().numpy()) # 收集真实标签

avg_test_loss = test_loss / len(test_loader)
test_accuracy = accuracy_score(all_true_labels, all_predicted_labels)
print(f"测试集损失: {avg_test_loss:.4f}, 测试集准确率: {test_accuracy:.4f}")

print("\n分类报告:")
print(classification_report(all_true_labels, all_predicted_labels))

# (可选) 打印部分预测结果 (前 10 个)
# print("\n部分预测结果 (前 10 个):")
# for i in range(min(10, len(all_true_labels))):
#     print(f"预测概率: {outputs[i][0]:.4f}, 预测标签: {all_predicted_labels[i]}, 真实标签: {all_true_labels[i]}") # 注意: 这里需要修改为 PyTorch 的预测概率获取方式 (如果需要打印概率)
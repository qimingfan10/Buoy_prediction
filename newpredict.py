import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

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
            nn.Linear(num_tokens, token_mix_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(token_mix_dim, num_tokens)
        )

        self.channel_mixing_mlp = nn.Sequential(
            nn.Linear(num_features, channel_mix_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(channel_mix_dim, num_features)
        )

        self.ln_token = nn.LayerNorm(num_features)
        self.ln_channel = nn.LayerNorm(num_features)

    def forward(self, x):
        x_token = x.permute(0, 2, 1)
        x_token = self.token_mixing_mlp(x_token)
        x_token = x_token.permute(0, 2, 1)
        x_token = self.ln_token(x_token + x)

        x_channel = self.channel_mixing_mlp(x_token)
        x_channel = self.ln_channel(x_channel + x_token)

        return x_channel

class TSMixerModel(nn.Module):
    def __init__(self, sequence_length, num_features):
        super(TSMixerModel, self).__init__()
        self.input_layer = nn.Linear(num_features, num_features)
        self.mixer_layer1 = MixerLayer(num_features=num_features, num_tokens=sequence_length, token_mix_dim=256, channel_mix_dim=2048)
        self.mixer_layer2 = MixerLayer(num_features=num_features, num_tokens=sequence_length, token_mix_dim=256, channel_mix_dim=2048)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc_output = nn.Linear(num_features, 1)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.mixer_layer1(x)
        x = self.mixer_layer2(x)
        x = x.permute(0, 2, 1)
        x = self.global_avg_pool(x)
        x = x.squeeze(dim=-1)
        x = self.fc_output(x)
        return torch.sigmoid(x)

class ProfileDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]

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

def load_data_from_csv(folder_path, sequence_length):
    """加载文件夹数据函数 (使用并行处理)"""
    all_data = []
    profile_filenames = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    for filename in profile_filenames:
        print(filename)
        file_path = os.path.join(folder_path, filename)
        df = pd.read_csv(file_path)
        feature_vector = extract_profile_features(df)
        if feature_vector is not None:
            all_data.append(feature_vector)

    # 确保返回的数组是三维的
    if len(all_data) < sequence_length:
        raise ValueError("数据不足以形成一个完整的序列。")
    
    # 将数据转换为三维数组
    sequences = []
    for i in range(len(all_data) - sequence_length + 1):
        sequences.append(all_data[i:i + sequence_length])
    
    return np.array(sequences)

# 加载模型
def load_model(sequence_length, num_features, model_path):
    model = TSMixerModel(sequence_length, num_features)
    model.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))
    model.eval()
    return model

# 预测函数
def predict(model, data_loader):
    all_predicted_labels = []
    with torch.no_grad():
        for sequences_batch in data_loader:
            outputs = model(sequences_batch)
            predicted_labels_batch = (outputs > 0.5).int().flatten()
            all_predicted_labels.extend(predicted_labels_batch.cpu().numpy())
    return all_predicted_labels

if __name__ == "__main__":
    folder_path = './test'  # 数据文件夹路径
    sequence_length = 32  # 根据之前训练的模型设置
    num_features = 492  # 根据你的模型设置
    model_path = 'best_model_n2743_seq32.pth'  # 权重文件路径

    # 加载数据
    sequences = load_data_from_csv(folder_path, sequence_length)
    dataset = ProfileDataset(sequences)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=False)

    # 加载模型
    model = load_model(sequence_length, num_features, model_path)

    # 进行预测
    predicted_labels = predict(model, data_loader)

    # 打印预测结果
    print("预测标签:", predicted_labels)
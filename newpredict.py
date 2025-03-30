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

def load_data_from_csv(folder_path, sequence_length, num_features):
    """加载文件夹数据并返回文件名和特征"""
    all_data = []
    filenames = []  # 存储文件名
    profile_filenames = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    for filename in profile_filenames:
        file_path = os.path.join(folder_path, filename)
        df = pd.read_csv(file_path)
        feature_vector = extract_profile_features(df)
        if feature_vector is not None:
            all_data.append(feature_vector)
            filenames.append(filename)  # 添加文件名

    if len(all_data) == 0:
        print("警告：没有找到有效的数据文件或所有文件都返回了 None。")
        return np.array([]), []

    all_data = np.array(all_data, dtype=np.float32)

    if len(all_data) < sequence_length:
        print(f"警告：数据不足以形成一个完整的序列 ({sequence_length})。正在填充...")
        num_missing = sequence_length - len(all_data)
        padding = np.zeros((num_missing, num_features), dtype=np.float32)
        all_data = np.concatenate((all_data, padding), axis=0)
        # 对于填充的数据，我们不应该有对应的文件名，所以我们添加 None 或空字符串
        filenames.extend([''] * num_missing) #或者使用 filenames.extend([None] * num_missing)

    sequences = []
    sequence_filenames = [] #存储序列对应的文件名列表
    for i in range(len(all_data) - sequence_length + 1):
        sequences.append(all_data[i:i + sequence_length])
        #关键：取序列的第一个文件作为序列的文件名
        sequence_filenames.append(filenames[i])

    return np.array(sequences, dtype=np.float32), sequence_filenames

# 加载模型
def load_model(sequence_length, num_features, model_path):
    model = TSMixerModel(sequence_length, num_features)
    model.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))
    model.eval()
    return model

def preprocess_input(data, sequence_length=32, num_features=492):
    """预处理输入数据"""
    if isinstance(data, list):
        data = np.array(data, dtype=np.float32)
    if not isinstance(data, np.ndarray):
        raise TypeError("输入数据必须是 NumPy 数组或列表。")
    if data.dtype != np.float32:
        data = data.astype(np.float32)

    if data.ndim != 3:
        raise ValueError(f"输入数据必须是三维数组 (batch_size, sequence_length, num_features)，但实际是 {data.ndim} 维。")
    if data.shape[1] != sequence_length:
        raise ValueError(f"输入数据的第二个维度 (序列长度) 必须是 {sequence_length}，但实际是 {data.shape[1]}")
    if data.shape[2] != num_features:
        raise ValueError(f"输入数据的第三个维度（特征维度）必须是 {num_features}，但实际是 {data.shape[2]}")
    return data

# 预测函数
def predict(model, data_loader):
    all_predicted_probabilities = [] # 修改：存储预测概率
    with torch.no_grad():
        for sequences_batch in data_loader:
            outputs = model(sequences_batch)
            all_predicted_probabilities.extend(outputs.cpu().numpy()) # 修改：存储概率
    return np.array(all_predicted_probabilities) # 修改：返回概率数组

if __name__ == "__main__":
    folder_path = './test'  # 数据文件夹路径
    sequence_length = 16  # 根据之前训练的模型设置
    num_features = 492  # 根据你的模型设置
    model_path = 'best_seq16_fold0_testacc0.7667.pth'  # 权重文件路径

    # 加载数据
    sequences, sequence_filenames = load_data_from_csv(folder_path, sequence_length, num_features)
    dataset = ProfileDataset(sequences)
    data_loader = DataLoader(dataset, batch_size=16, shuffle=False)

    # 加载模型
    model = load_model(sequence_length, num_features, model_path)

    # 进行预测
    predictions = predict(model, data_loader) # 修改：接收概率

    print("Predictions (probabilities):", predictions)
    threshold = 0.5
    binary_predictions = (predictions > threshold).astype(int)
    #print("Binary Predictions:", binary_predictions)
    print("预测标签数量:", len(binary_predictions.flatten()))

    # 打印每个预测对应的文件名
    print("\n预测结果及对应文件名:")
    for i, prediction in enumerate(binary_predictions.flatten()):
        # 注意：sequence_filenames 的长度可能与 predictions 的长度不同，需要仔细处理
        # predictions 的长度是滑动窗口的数量，而 sequence_filenames 的长度也是滑动窗口的数量，
        # 因为我们在 load_data_from_csv 中为每个序列都记录了起始文件名。
        # 因此，可以直接使用相同的索引 i。
        if i < len(sequence_filenames):
            print(f"预测: {prediction}, 文件名: {sequence_filenames[i]}")
        else:
            print(f"预测: {prediction}, 文件名: (文件名信息缺失)")
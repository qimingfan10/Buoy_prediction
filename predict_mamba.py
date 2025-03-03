import torch
import pandas as pd
import numpy as np
from pathlib import Path
from mamba import TemporalModel
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import time

start_time = time.time()
class PredictionDataset(Dataset):
    def __init__(self, file_paths, scaler_path=None, max_depth=100):
        self.samples = []
        self.lengths = []
        self.file_names = []
        
        # 读取所有CSV文件
        print("正在加载CSV文件...")
        for csv_file in tqdm(file_paths):
            df = pd.read_csv(csv_file)
            length = len(df)
            if length > max_depth:
                df = df.iloc[:max_depth]
                length = max_depth
            self.lengths.append(length)
            self.samples.append(df.values)
            self.file_names.append(Path(csv_file).name)
        
        # 标准化处理
        print("正在进行数据标准化...")
        if scaler_path and Path(scaler_path).exists():
            # 如果提供了训练好的scaler，直接加载使用
            import joblib
            self.scaler = joblib.load(scaler_path)
        else:
            # 否则使用当前数据拟合一个新的scaler
            all_data = np.vstack(self.samples)
            self.scaler = StandardScaler().fit(all_data)
            
        self.samples = [self.scaler.transform(s) for s in tqdm(self.samples)]

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.samples[idx]), self.lengths[idx], self.file_names[idx]

def collate_fn(batch):
    """自定义批处理函数"""
    inputs, lengths, file_names = zip(*batch)
    max_len = max(lengths)
    
    padded_inputs = torch.zeros(len(inputs), max_len, inputs[0].shape[1])
    for i, (input, length) in enumerate(zip(inputs, lengths)):
        padded_inputs[i, :length] = input
    
    lengths = torch.tensor(lengths)
    return padded_inputs, lengths, file_names

def predict_files(model, data_loader, device):
    """使用训练好的模型进行预测"""
    model.eval()
    predictions = []
    probabilities = []
    file_names = []
    
    print("正在进行预测...")
    with torch.no_grad():
        for inputs, lengths, batch_file_names in tqdm(data_loader, desc="预测进度"):
            inputs = inputs.to(device)
            lengths = lengths.to(device)
            
            outputs = model(inputs, lengths).squeeze()
            
            # 处理单个样本的情况
            if outputs.ndim == 0:
                probs = torch.sigmoid(outputs).item()
                preds = int(probs > 0.5)
                predictions.append(preds)
                probabilities.append(probs)
            else:
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).long()
                predictions.extend(preds.cpu().tolist())
                probabilities.extend(probs.cpu().tolist())
                
            file_names.extend(batch_file_names)
    
    return predictions, probabilities, file_names

def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 获取要预测的CSV文件路径
    input_folder = 'data/predict'  # 存放待预测CSV文件的文件夹
    if not Path(input_folder).exists():
        raise FileNotFoundError(f"找不到文件夹: {input_folder}")
    
    csv_files = list(Path(input_folder).glob("*.csv"))
    if not csv_files:
        raise ValueError(f"在 {input_folder} 中没有找到CSV文件")
    
    # 创建数据集和加载器
    print("准备数据...")
    dataset = PredictionDataset(
        csv_files,
        scaler_path='model/scaler.joblib'  # 如果有保存的scaler就使用它
    )
    
    data_loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=False  # 确保处理最后一个不完整的批次
    )
    
    # 加载模型
    print("加载模型...")
    model = TemporalModel().to(device)
    model.load_state_dict(torch.load('1200new_mamba_model.pth'))
    
    # 进行预测
    predictions, probabilities, file_names = predict_files(model, data_loader, device)
    
    # 将预测结果转换为描述性标签
    label_mapping = {0: "不在漩涡中心", 1: "在漩涡中心"}
    descriptive_predictions = [label_mapping[pred] for pred in predictions]

    # 保存结果
    results_df = pd.DataFrame({
        'file_name': file_names,
        'prediction': descriptive_predictions,  # 使用描述性标签
        'probability': probabilities
    })
    
    # 输出结果
    print("\n预测结果:")
    print("文件名 | 预测类别")
    print("-" * 40)
    for _, row in results_df.iterrows():
        print(f"{row['file_name']:<20} | {row['prediction']:^8}")

    # 修改统计信息的计算方式
    predictions_array = np.array(predictions)  # 转换为numpy数组
    print("\n预测统计:")
    print(f"不在漩涡中心的数量: {np.sum(predictions_array == 0)}")
    print(f"在漩涡中心的数量: {np.sum(predictions_array == 1)}")
    
    # 添加百分比统计
    total = len(predictions)
    print(f"\n类别分布:")
    print(f"不在漩涡中心的占比: {np.sum(predictions_array == 0) / total * 100:.2f}%")
    print(f"在漩涡中心的占比: {np.sum(predictions_array == 1) / total * 100:.2f}%")
    
    # 添加概率分布统计
    # probabilities_array = np.array(probabilities)
    # print(f"\n预测概率统计:")
    # print(f"最小概率: {probabilities_array.min():.4f}")
    # print(f"最大概率: {probabilities_array.max():.4f}")
    # print(f"平均概率: {probabilities_array.mean():.4f}")
    # print(f"中位数概率: {np.median(probabilities_array):.4f}")

if __name__ == "__main__":
    main()
    end_time = time.time()    # 记录结束时间
    run_time = end_time - start_time  # 计算运行时间

    print(f"代码运行时间：{run_time} 秒")
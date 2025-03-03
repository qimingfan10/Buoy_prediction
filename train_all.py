import torch
import torch.nn as nn
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import time
import json
from datetime import datetime
import openlocationcode as olc  # 添加 Plus Code 库
from sklearn.metrics import mean_squared_error, r2_score  # 导入必要的库

# 导入所有模型
from mamba import TemporalModel
from lstm import LSTMModel
from gru import GRUModel
from rnn import SimpleRNN
from transf import TransformerCls
from tcn import TCNClassifier
from tsm import TSMixer
from dli import DLinearModel
from itrans import iTransformerClassifier

class BuoyDataset(Dataset):
    def __init__(self, folder_0, folder_1, max_depth=100):
        self.samples = []
        self.lengths = []
        
        print("正在加载数据...")
        
        # 读取文件夹0（类别0）
        files_0 = list(Path(folder_0).glob("*.csv"))
        for csv_file in tqdm(files_0, desc="加载类别0文件"):
            df = pd.read_csv(csv_file)
            length = len(df)
            if length > max_depth:
                df = df.iloc[:max_depth]
                length = max_depth
            
            # 直接使用原始数据
            processed_data = df.values  # 直接使用原始数据
            
            self.lengths.append(length)
            self.samples.append((processed_data, 0))
            
        # 读取文件夹1（类别1）
        files_1 = list(Path(folder_1).glob("*.csv"))
        for csv_file in tqdm(files_1, desc="加载类别1文件"):
            df = pd.read_csv(csv_file)
            length = len(df)
            if length > max_depth:
                df = df.iloc[:max_depth]
                length = max_depth
            
            # 直接使用原始数据
            processed_data = df.values  # 直接使用原始数据
            
            self.lengths.append(length)
            self.samples.append((processed_data, 1))
        
        print(f"已加载类别0文件: {len(files_0)}个")
        print(f"已加载类别1文件: {len(files_1)}个")
        
        # 标准化处理
        print("正在进行数据标准化...")
        all_data = np.vstack([s[0] for s in self.samples])
        self.scaler = StandardScaler().fit(all_data)
        self.samples = [(self.scaler.transform(s[0]), s[1]) for s in tqdm(self.samples)]

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        data, label = self.samples[idx]
        length = self.lengths[idx]
        return torch.FloatTensor(data), torch.tensor(label), length

def collate_fn(batch):
    data, labels, lengths = zip(*batch)
    max_len = max(lengths)
    padded_data = []
    for d, l in zip(data, lengths):
        if l < max_len:
            pad = torch.zeros((max_len - l, d.size(1)))
            d = torch.cat([d, pad], dim=0)
        padded_data.append(d)
    return torch.stack(padded_data), torch.tensor(labels), torch.tensor(lengths)

class ModelTrainer:
    def __init__(self, model_name, model, train_loader, val_loader, device, log_dir):
        self.model_name = model_name
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 训练记录
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_acc': [],
            'best_acc': 0,
            'epochs': [],
            'learning_rates': [],
            'training_time': [],
            'mse': [],  # 添加 mse 记录
            'r2': []    # 添加 r2 记录
        }
        
        # 创建结果文件
        self.result_file = self.log_dir / f'{self.model_name}_results.txt'
        with open(self.result_file, 'w') as f:
            f.write("Epoch\tTrain Loss\tVal Loss\tVal Acc\tMSE\tR²\n")  # 写入表头

    def save_history(self):
        # 保存训练历史到文件
        with open(self.result_file, 'a') as f:
            f.write(f"{self.history['epochs'][-1]}\t{self.history['train_loss'][-1]:.4f}\t{self.history['val_loss'][-1]:.4f}\t{self.history['val_acc'][-1]:.4f}\t{self.history['mse'][-1]:.4f}\t{self.history['r2'][-1]:.4f}\n")

    def train(self, epochs=50):
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
        
        start_time = time.time()
        best_acc = 0
        patience = 10
        no_improve = 0
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            # 训练阶段
            self.model.train()
            train_loss = 0
            train_pbar = tqdm(self.train_loader, desc=f'{self.model_name} Epoch {epoch+1}/{epochs} [Train]')
            
            for inputs, labels, lengths in train_pbar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                lengths = lengths.to(self.device)
                
                outputs = self.model(inputs, lengths).squeeze()
                loss = criterion(outputs, labels.float())
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item()
                train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            train_loss /= len(self.train_loader)
            
            # 验证阶段
            self.model.eval()
            val_loss = 0
            correct = 0
            val_pbar = tqdm(self.val_loader, desc=f'{self.model_name} Epoch {epoch+1}/{epochs} [Val]')
            
            all_outputs = []  # 用于存储所有输出
            all_labels = []   # 用于存储所有标签
            
            with torch.no_grad():
                for inputs, labels, lengths in val_pbar:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    lengths = lengths.to(self.device)
                    
                    outputs = self.model(inputs, lengths).squeeze()
                    loss = criterion(outputs, labels.float())
                    val_loss += loss.item()
                    
                    preds = (torch.sigmoid(outputs) > 0.5).long()
                    correct += (preds == labels).sum().item()
                    
                    val_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                    
                    all_outputs.extend(outputs.cpu().numpy())  # 收集输出
                    all_labels.extend(labels.cpu().numpy())     # 收集标签
            
            # 打印调试信息
            print(f"all_outputs shape: {np.array(all_outputs).shape}, all_labels shape: {np.array(all_labels).shape}")
            
            val_loss /= len(self.val_loader)
            val_acc = correct / len(self.val_loader.dataset)
            
            # 计算 MSE 和 R²
            all_outputs_sigmoid = 1 / (1 + np.exp(-np.array(all_outputs)))  # 将 logits 转换为概率
            mse = mean_squared_error(all_labels, all_outputs_sigmoid)
            r2 = r2_score(all_labels, all_outputs_sigmoid)
            
            # 记录 MSE 和 R²
            self.history['mse'].append(mse)
            self.history['r2'].append(r2)
            
            print(f"MSE: {mse:.4f}, R²: {r2:.4f}")
            
            # 更新学习率
            scheduler.step(val_acc)
            
            # 记录训练信息
            epoch_time = time.time() - epoch_start
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['epochs'].append(epoch + 1)
            self.history['learning_rates'].append(optimizer.param_groups[0]['lr'])
            self.history['training_time'].append(epoch_time)
            
            print(f"\n{self.model_name} Epoch {epoch+1}/{epochs} Summary:")
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            print(f"Val Acc: {val_acc:.4f}")
            print(f"MSE: {mse:.4f}, R²: {r2:.4f}")  # 打印 MSE 和 R²
            
            # 将结果写入文件
            with open(self.result_file, 'a') as f:
                f.write(f"{epoch+1}\t{train_loss:.4f}\t{val_loss:.4f}\t{val_acc:.4f}\t{mse:.4f}\t{r2:.4f}\n")
            
            # 保存最佳模型
            if val_acc > best_acc:
                best_acc = val_acc
                self.history['best_acc'] = best_acc
                no_improve = 0
                torch.save(self.model.state_dict(), self.log_dir / f'best_{self.model_name}_model.pth')
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"{self.model_name} Early stopping triggered")
                    break
            
            # 保存训练记录
            self.save_history()
        
        total_time = time.time() - start_time
        print(f"\n{self.model_name} 总训练时间: {total_time:.2f}秒")
        return self.history

def main():
    # 设置设备
    device = torch.device('cpu')  # 强制使用 CPU 进行调试
    print(f"使用设备: {device}")
    
    # 创建数据加载器
    train_dataset = BuoyDataset('data/train/folder_0', 'data/train/folder_1')
    val_dataset = BuoyDataset('data/val/folder_0', 'data/val/folder_1')
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=32,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # 创建日志目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = Path(f'training_logs_{timestamp}')
    
    # 定义所有模型
    models = {
        'Mamba': TemporalModel(),
        'LSTM': LSTMModel(),
        'GRU': GRUModel(),
        'RNN': SimpleRNN(),
        'Transformer': TransformerCls(),
        'TCN': TCNClassifier(),
        'TSMixer': TSMixer(),
        'DLinear': DLinearModel(),
        'iTransformer': iTransformerClassifier()
    }
    
    # 训练所有模型
    all_histories = {}
    for name, model in models.items():
        print(f"\n开始训练 {name} 模型...")
        model = model.to(device)
        trainer = ModelTrainer(name, model, train_loader, val_loader, device, log_dir)
        history = trainer.train()
        all_histories[name] = history
    
    # 保存总体训练记录
    with open(log_dir / 'all_models_history.json', 'w') as f:
        json.dump(all_histories, f, indent=4)
    
    # 打印最终结果比较
    print("\n所有模型训练完成！最终结果比较：")
    print("-" * 60)
    print(f"{'模型名称':<15} {'最佳验证准确率':>15} {'总训练时间':>15}")
    print("-" * 60)
    for name, history in all_histories.items():
        total_time = sum(history['training_time'])
        print(f"{name:<15} {history['best_acc']:>15.4f} {total_time:>15.2f}s")

if __name__ == "__main__":
    main() 
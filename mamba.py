import torch
import torch.nn as nn
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report
from torch.optim.lr_scheduler import ReduceLROnPlateau
from mamba_ssm.modules.mamba_simple import Mamba
from dataclasses import dataclass
from tqdm import tqdm
#help(Mamba)
class BuoyDataset(Dataset):
    def __init__(self, folder_0, folder_1, max_depth=100):
        self.samples = []
        self.lengths = []  # 存储序列长度
        
        # 检查文件夹是否存在
        if not Path(folder_0).exists():
            raise FileNotFoundError(f"文件夹不存在: {folder_0}")
        if not Path(folder_1).exists():
            raise FileNotFoundError(f"文件夹不存在: {folder_1}")
            
        # 读取文件夹0（类别0）
        files_0 = list(Path(folder_0).glob("*.csv"))
        if not files_0:
            raise ValueError(f"在 {folder_0} 中没有找到CSV文件")
        
        print("正在加载类别0文件...")
        for csv_file in tqdm(files_0):
            df = pd.read_csv(csv_file)
            length = len(df)
            if length > max_depth:
                df = df.iloc[:max_depth]
                length = max_depth
            self.lengths.append(length)
            self.samples.append((df.values, 0))
            
        # 读取文件夹1（类别1）
        files_1 = list(Path(folder_1).glob("*.csv"))
        if not files_1:
            raise ValueError(f"在 {folder_1} 中没有找到CSV文件")
            
        print("正在加载类别1文件...")    
        for csv_file in tqdm(files_1):
            df = pd.read_csv(csv_file)
            length = len(df)
            if length > max_depth:
                df = df.iloc[:max_depth]
                length = max_depth
            self.lengths.append(length)
            self.samples.append((df.values, 1))
        
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

@dataclass
class ModelArgs:
    d_model: int
    n_layer: int
    vocab_size: int
    
    def __post_init__(self):
        self.d_state = 16  # Mamba 状态维度
        self.d_conv = 4    # 卷积核大小
        self.expand = 2    # 扩展因子
        self.dt_rank = None
        self.d_inner = int(self.expand * self.d_model)
        self.bias = True

def collate_fn(batch):
    """
    自定义批处理函数，处理变长序列
    """
    # 解压批次数据
    inputs, labels, lengths = zip(*batch)
    
    # 获取最大长度
    max_len = max(lengths)
    
    # 填充序列到最大长度
    padded_inputs = torch.zeros(len(inputs), max_len, inputs[0].shape[1])
    for i, (input, length) in enumerate(zip(inputs, lengths)):
        padded_inputs[i, :length] = input
    
    # 转换为张量
    labels = torch.tensor(labels)
    lengths = torch.tensor(lengths)
    
    return padded_inputs, labels, lengths

class TemporalModel(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=128, num_layers=2):
        super().__init__()
        
        # 输入投影层
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Mamba层
        self.layers = nn.ModuleList([
            Mamba(
                d_model=hidden_dim,
                d_state=16,  # 默认值
                d_conv=4,    # 默认值
                expand=2     # 默认值
            ) for _ in range(num_layers)
        ])
        
        # 输出层
        self.output_layer = nn.Linear(hidden_dim, 1)
        
    def forward(self, x, lengths):
        # x shape: [batch_size, seq_len, input_dim]
        
        # 投影到模型维度
        x = self.input_proj(x)
        
        # 依次通过每个Mamba层
        for layer in self.layers:
            x = layer(x)
        
        # 使用序列最后一个有效位置的输出
        batch_size = x.size(0)
        x = x[torch.arange(batch_size), lengths - 1]
        
        # 输出层
        x = self.output_layer(x)
        
        return x

def train_model(model, train_loader, val_loader, device, epochs=50):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )
    
    best_acc = 0
    patience = 10
    no_improve = 0
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
        
        for inputs, labels, lengths in train_pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            lengths = lengths.to(device)
            
            outputs = model(inputs, lengths).squeeze()
            loss = criterion(outputs, labels.float())
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
        train_loss /= len(train_loader)
        
        # 验证阶段
        model.eval()
        val_loss = 0
        correct = 0
        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [Val]')
        
        with torch.no_grad():
            for inputs, labels, lengths in val_pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                lengths = lengths.to(device)
                
                outputs = model(inputs, lengths).squeeze()
                loss = criterion(outputs, labels.float())
                val_loss += loss.item()
                
                preds = (torch.sigmoid(outputs) > 0.5).long()
                correct += (preds == labels).sum().item()
                
                val_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
        val_loss /= len(val_loader)
        val_acc = correct / len(val_loader.dataset)
        
        # 更新学习率
        scheduler.step(val_acc)
        
        print(f"\nEpoch {epoch+1}/{epochs} Summary:")
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"Val Acc: {val_acc:.4f}")
        
        # 早停
        if val_acc > best_acc:
            best_acc = val_acc
            no_improve = 0
            torch.save(model.state_dict(), 'best_mamba_model.pth')
        else:
            no_improve += 1
            if no_improve >= patience:
                print("Early stopping triggered")
                break

def evaluate(model, test_loader, device):
    model.eval()
    y_true, y_pred = [], []
    test_loss = 0
    criterion = nn.BCEWithLogitsLoss()
    
    test_pbar = tqdm(test_loader, desc='Evaluating')
    with torch.no_grad():
        for inputs, labels, lengths in test_pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            lengths = lengths.to(device)
            
            outputs = model(inputs, lengths).squeeze()
            loss = criterion(outputs, labels.float())
            test_loss += loss.item()
            
            preds = (torch.sigmoid(outputs) > 0.5).long()
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())
            
            test_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    test_loss /= len(test_loader)
    print(f"\nTest Loss: {test_loss:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

# 使用示例
if __name__ == "__main__":
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
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
    
    # 创建模型
    model = TemporalModel().to(device)
    
    # 训练模型
    train_model(model, train_loader, val_loader, device)
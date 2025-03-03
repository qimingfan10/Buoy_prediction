import torch
import torch.nn as nn
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report
import math
from torch.optim.lr_scheduler import ReduceLROnPlateau

class BuoyDataset(Dataset):
    def __init__(self, folder_0, folder_1, max_depth=100):
        self.samples = []
        self.lengths = []  # 存储序列长度
        
        # 读取文件夹0（类别0）
        for csv_file in Path(folder_0).glob("*.csv"):
            df = pd.read_csv(csv_file)
            length = len(df)
            if length > max_depth:
                df = df.iloc[:max_depth]
                length = max_depth
            self.lengths.append(length)
            self.samples.append((df.values, 0))
            
        # 读取文件夹1（类别1）
        for csv_file in Path(folder_1).glob("*.csv"):
            df = pd.read_csv(csv_file)
            length = len(df)
            if length > max_depth:
                df = df.iloc[:max_depth]
                length = max_depth
            self.lengths.append(length)
            self.samples.append((df.values, 1))
        
        # 标准化处理
        all_data = np.vstack([s[0] for s in self.samples])
        self.scaler = StandardScaler().fit(all_data)
        self.samples = [(self.scaler.transform(s[0]), s[1]) for s in self.samples]

    def __len__(self): 
        return len(self.samples)
    
    def __getitem__(self, idx):
        data, label = self.samples[idx]
        length = self.lengths[idx]
        return torch.FloatTensor(data), torch.tensor(label), length

def collate_fn(batch):
    # 自定义批处理函数
    data, labels, lengths = zip(*batch)
    max_len = max(lengths)
    # 填充到最大长度
    padded_data = []
    for d, l in zip(data, lengths):
        if l < max_len:
            pad = torch.zeros((max_len - l, d.size(1)))
            d = torch.cat([d, pad], dim=0)
        padded_data.append(d)
    return torch.stack(padded_data), torch.tensor(labels), torch.tensor(lengths)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TransformerCls(nn.Module):
    def __init__(self, input_size=5, d_model=64, nhead=4, num_layers=3, dropout=0.1):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(input_size, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1)
        )
        
    def forward(self, x, lengths=None):
        # x: [batch_size, seq_len, input_size]
        x = self.embed(x)  # [batch_size, seq_len, d_model]
        x = self.pos_encoder(x)
        
        if lengths is not None:
            # 创建注意力掩码
            mask = torch.arange(x.size(1), device=x.device)[None, :] >= lengths[:, None]
            x = self.transformer_encoder(x, src_key_padding_mask=mask)
            
            # 使用真实长度进行池化
            mask = ~mask
            x = (x * mask.float().unsqueeze(-1)).sum(dim=1) / lengths.float().unsqueeze(-1)
        else:
            x = self.transformer_encoder(x)
            x = x.mean(dim=1)  # 全局平均池化
            
        return self.classifier(x)

def train_model(model, train_loader, val_loader, device, epochs=50):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
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
        for inputs, labels, lengths in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            lengths = lengths.to(device)
            
            outputs = model(inputs, lengths).squeeze()
            loss = criterion(outputs, labels.float())
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        
        # 验证阶段
        model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for inputs, labels, lengths in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                lengths = lengths.to(device)
                
                outputs = model(inputs, lengths).squeeze()
                val_loss += criterion(outputs, labels.float()).item()
                
                preds = (torch.sigmoid(outputs) > 0.5).long()
                correct += (preds == labels).sum().item()
            
        val_loss /= len(val_loader)
        val_acc = correct / len(val_loader.dataset)
        
        # 更新学习率
        scheduler.step(val_acc)
        
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"Val Acc: {val_acc:.4f}")
        
        # 早停
        if val_acc > best_acc:
            best_acc = val_acc
            no_improve = 0
            torch.save(model.state_dict(), 'best_transformer_model.pth')
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
    
    with torch.no_grad():
        for inputs, labels, lengths in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            lengths = lengths.to(device)
            
            outputs = model(inputs, lengths).squeeze()
            test_loss += criterion(outputs, labels.float()).item()
            
            preds = (torch.sigmoid(outputs) > 0.5).long()
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())
    
    test_loss /= len(test_loader)
    print(f"\nTest Loss: {test_loss:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

# 使用示例
if __name__ == "__main__":
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 设置数据路径
    data_root = Path("data")  # 使用Path对象
    
    # 创建必要的目录结构
    for split in ['train', 'val']:
        for label in ['folder_0', 'folder_1']:
            folder = data_root / split / label
            folder.mkdir(parents=True, exist_ok=True)
            
            # 创建一些示例数据（如果目录为空）
            if not list(folder.glob('*.csv')):
                print(f"在 {folder} 中创建示例数据...")
                # 创建示例数据
                for i in range(5):  # 每个文件夹创建5个样本
                    # 创建随机时间序列数据
                    data = np.random.randn(50, 5)  # 50个时间步，5个特征
                    df = pd.DataFrame(data, columns=['feature_'+str(j) for j in range(5)])
                    df.to_csv(folder / f'sample_{i}.csv', index=False)
    
    # 设置具体路径
    train_folder_0 = data_root / "train/folder_0"
    train_folder_1 = data_root / "train/folder_1"
    val_folder_0 = data_root / "val/folder_0"
    val_folder_1 = data_root / "val/folder_1"
    
    print("数据目录结构已创建")
    
    # 创建数据加载器
    train_dataset = BuoyDataset(train_folder_0, train_folder_1)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=32, 
        shuffle=True,
        collate_fn=collate_fn
    )
    
    val_dataset = BuoyDataset(val_folder_0, val_folder_1)
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # 创建模型
    model = TransformerCls().to(device)
    
    # 训练模型
    train_model(model, train_loader, val_loader, device)
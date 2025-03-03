import torch
import torch.nn as nn
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report
from torch.optim.lr_scheduler import ReduceLROnPlateau

class BuoyDataset(Dataset):
    def __init__(self, folder_0, folder_1, max_depth=100):
        self.samples = []
        self.lengths = []  # 存储序列长度
        
        # 检查路径是否存在
        folder_0 = Path(folder_0)
        folder_1 = Path(folder_1)
        
        if not folder_0.exists():
            raise ValueError(f"路径不存在: {folder_0}")
        if not folder_1.exists():
            raise ValueError(f"路径不存在: {folder_1}")
            
        # 读取文件夹0（类别0）
        files_0 = list(folder_0.glob("*.csv"))
        if not files_0:
            raise ValueError(f"在 {folder_0} 中没有找到CSV文件")
            
        for csv_file in files_0:
            df = pd.read_csv(csv_file)
            length = len(df)
            if length > max_depth:
                df = df.iloc[:max_depth]
                length = max_depth
            self.lengths.append(length)
            self.samples.append((df.values, 0))
            
        # 读取文件夹1（类别1）
        files_1 = list(folder_1.glob("*.csv"))
        if not files_1:
            raise ValueError(f"在 {folder_1} 中没有找到CSV文件")
            
        for csv_file in files_1:
            df = pd.read_csv(csv_file)
            length = len(df)
            if length > max_depth:
                df = df.iloc[:max_depth]
                length = max_depth
            self.lengths.append(length)
            self.samples.append((df.values, 1))
        
        print(f"加载了 {len(files_0)} 个类别0样本和 {len(files_1)} 个类别1样本")
        
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

class TSMixer(nn.Module):
    def __init__(self, feat_dim=5, hidden_dim=64, num_layers=2, dropout=0.2):
        super().__init__()
        
        # 时间混合器：对每个特征通道单独处理时间维度
        self.temporal_mixers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feat_dim, hidden_dim),  # 先混合特征维度
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, feat_dim)
            ) for _ in range(num_layers)
        ])
        
        # 特征混合器：在每个时间步上混合特征
        self.feature_mixers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(feat_dim),
                nn.Linear(feat_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, feat_dim)
            ) for _ in range(num_layers)
        ])
        
        # 全局池化
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        # 批归一化
        self.batch_norm = nn.BatchNorm1d(feat_dim)
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x, lengths=None):
        # x: [batch_size, seq_len, feat_dim]
        batch_size, seq_len, feat_dim = x.shape
        
        # 应用长度掩码
        if lengths is not None:
            mask = torch.arange(seq_len, device=x.device)[None, :] < lengths[:, None]
            mask = mask.float().unsqueeze(-1)
            x = x * mask
        
        # 多层混合
        for temp_mixer, feat_mixer in zip(self.temporal_mixers, self.feature_mixers):
            # 时间维度混合：在每个时间步上混合特征
            temp_out = temp_mixer(x)
            x = x + temp_out
            
            # 特征维度混合
            feat_out = feat_mixer(x)
            x = x + feat_out
            
            # 重新应用掩码
            if lengths is not None:
                x = x * mask
        
        # 全局池化
        x = x.transpose(1, 2)  # [batch_size, feat_dim, seq_len]
        x = self.pool(x).squeeze(-1)  # [batch_size, feat_dim]
        
        # 批归一化
        x = self.batch_norm(x)
        
        # 分类
        return self.classifier(x)

def train_model(model, train_loader, val_loader, device, epochs=50):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
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
            torch.save(model.state_dict(), 'best_tsmixer_model.pth')
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
    model = TSMixer().to(device)
    
    # 训练模型
    train_model(model, train_loader, val_loader, device)
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

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=kernel_size//2)

    def forward(self, x):
        # x: [Batch, Input_dim, Length]
        x = x.transpose(1, 2)  # 调整维度顺序以适应1D卷积
        x = self.avg(x)
        x = x.transpose(1, 2)  # 恢复维度顺序
        return x

class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class DLinearModel(nn.Module):
    """
    Decomposition-Linear
    """
    def __init__(self, input_dim=5, seq_len=100, hidden_dim=64, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim
        
        # 序列分解
        self.decomposition = series_decomp(kernel_size=25)
        
        # 季节性分量和趋势分量的处理
        self.seasonal_layer = nn.Sequential(
            # 修改输入维度以匹配实际数据
            nn.Linear(input_dim, hidden_dim),  # 每个时间步单独处理
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.trend_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  # 每个时间步单独处理
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 最终分类层
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # 合并两个分支的特征
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, lengths=None):
        batch_size = x.size(0)
        
        # 序列分解
        seasonal_part, trend_part = self.decomposition(x)
        
        # 处理季节性分量
        seasonal_output = self.seasonal_layer(seasonal_part)  # [batch, seq_len, hidden_dim]
        seasonal_output = seasonal_output.mean(dim=1)  # 全局平均池化 [batch, hidden_dim]
        
        # 处理趋势分量
        trend_output = self.trend_layer(trend_part)  # [batch, seq_len, hidden_dim]
        trend_output = trend_output.mean(dim=1)  # 全局平均池化 [batch, hidden_dim]
        
        # 合并特征
        combined = torch.cat([seasonal_output, trend_output], dim=1)
        
        # 分类预测
        output = self.classifier(combined)
        return output

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
            
            outputs = model(inputs).squeeze()
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
                
                outputs = model(inputs).squeeze()
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
            torch.save(model.state_dict(), 'best_dlinear_model.pth')
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
            
            outputs = model(inputs).squeeze()
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
    model = DLinearModel().to(device)
    
    # 训练模型
    train_model(model, train_loader, val_loader, device)
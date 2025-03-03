import torch
import torch.nn as nn
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from sklearn.metrics import classification_report

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

    def process_coordinates(self, df):
        # 归一化经纬度
        df['lat'] = (df['lat'] + 90) / 180  # 将纬度归一化到 [0, 1]
        df['lon'] = (df['lon'] + 180) / 360  # 将经度归一化到 [0, 1]
        
        # 选择需要的特征，假设我们需要保留原始数据中的其他特征
        # df = df[['lat', 'lon', 'feature1', 'feature2', ..., 'feature11']]
        
        # 返回归一化后的经纬度和其他特征
        return df.values  # 返回所有特征

def collate_fn(batch):
    # 自定义批处理函数
    data, labels, lengths = zip(*batch)
    data = pad_sequence([torch.FloatTensor(d) for d in data], batch_first=True)  # 确保数据是 FloatTensor
    return data, torch.tensor(labels), torch.tensor(lengths)

class LSTMModel(nn.Module):
    def __init__(self, input_size=5, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True,
            dropout=dropout
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)  # 输出节点数设置为 1
        )
    
    def forward(self, x, lengths):
        # x shape: [batch_size, seq_len, input_dim]
        
        # 需要将输入转换为 (seq_len, batch_size, input_dim)
        x = x.permute(1, 0, 2)  # 转换为 (seq_len, batch_size, input_dim)
        
        # 将 lengths 移动到 CPU
        lengths = lengths.cpu()
        
        packed_x = pack_padded_sequence(x, lengths, enforce_sorted=False)
        packed_out, _ = self.lstm(packed_x)
        
        # 处理输出
        out, _ = pad_packed_sequence(packed_out)
        
        # 取最后一个时间步的输出
        last_out = out[-1]  # 取最后一个时间步的输出，形状为 (batch_size, hidden_size)
        
        # 通过分类器
        return self.classifier(last_out)  # 返回形状为 (batch_size, 1)

def train_model(model, train_loader, val_loader, device, epochs=50):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
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
            
            outputs = model(inputs, lengths)  # outputs 形状为 (batch_size, 1)
            
            loss = criterion(outputs.squeeze(), labels.float())  # 使用 squeeze() 将形状变为 (batch_size,)
            
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
            torch.save(model.state_dict(), 'best_model.pth')
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

# 多分类指标参考 

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
    model = LSTMModel().to(device)
    
    # 训练模型
    train_model(model, train_loader, val_loader, device) 

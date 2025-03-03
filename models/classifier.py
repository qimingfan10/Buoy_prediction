import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes, device):
        super(GraphClassifier, self).__init__()
        
        # 增加网络深度和复杂度
        self.layer1 = nn.Linear(in_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.layer3 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.output = nn.Linear(hidden_dim, n_classes)
        
        # 添加批归一化
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim * 2)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        
        # 添加dropout
        self.dropout = nn.Dropout(0.3)
        
        self.device = device
        
    def forward(self, x):
        # 前向传播
        x = F.relu(self.bn1(self.layer1(x)))
        x = self.dropout(x)
        
        x = F.relu(self.bn2(self.layer2(x)))
        x = self.dropout(x)
        
        x = F.relu(self.bn3(self.layer3(x)))
        x = self.dropout(x)
        
        return self.output(x)

    def cv_train(self, dataset, batchSize, num_epochs, lr, kFold, savePath, device):
        # 使用更好的优化器
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=0.01)
        
        # 使用学习率调度器
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5, verbose=True
        )
        
        # 使用 focal loss 来处理可能的类别不平衡问题
        criterion = FocalLoss(alpha=0.25, gamma=2)
        
        f1_scores = []
        precision_scores = []
        recall_scores = []
        
        for fold in range(kFold):
            # ... 训练代码 ...
            
            # 在每个epoch结束后更新学习率
            scheduler.step(val_f1)  # 假设有验证集的F1分数
            
            # 计算指标
            f1 = self.calculate_f1(y_true, y_pred)
            precision = self.calculate_precision(y_true, y_pred)
            recall = self.calculate_recall(y_true, y_pred)
            
            print(f"Fold {fold + 1}: F1 = {f1:.4f}, Precision = {precision:.4f}, Recall = {recall:.4f}")
            
            f1_scores.append(f1)
            precision_scores.append(precision)
            recall_scores.append(recall)

        # 计算平均值
        avg_f1 = sum(f1_scores) / kFold
        avg_precision = sum(precision_scores) / kFold
        avg_recall = sum(recall_scores) / kFold
        
        print(f"Average F1: {avg_f1:.4f}")
        print(f"Average Precision: {avg_precision:.4f}")
        print(f"Average Recall: {avg_recall:.4f}")

# Focal Loss 实现
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean() 
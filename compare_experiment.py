import numpy as np
import pandas as pd
import os
import time
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from newtsm import (load_data_from_folder, extract_profile_features, 
                   ProfileDataset, TSMixerModel)
from torch.utils.data import DataLoader
import re
from collections import defaultdict
import concurrent.futures
import multiprocessing
from tqdm import tqdm
import random
import numpy as np
import torch

# 设置随机种子
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
# 设置matplotlib中文字体
try:
    # 尝试设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    print("已设置中文字体支持")
except:
    print("警告: 无法设置中文字体，图表中的中文可能无法正确显示")

# 设置实验参数
# n_values = [5487, 4115, 2743, 1371, 548]
n_values = [ 5487 ]
seq_length_values = [2, 4, 8, 16, 32, 64]
folder_path = '/root/0'

# 创建结果存储结构
results = {
    'n': [],
    'sequence_length': [],
    'train_accuracy': [],
    'val_accuracy': [],
    'test_accuracy': [],
    'training_time': [],
    'data_loading_time': []
}

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 获取所有文件名
def get_sorted_files(folder_path, n_limit=None):
    """获取排序后的文件列表"""
    all_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.csv')], 
                      key=natural_sort_key)
    if n_limit and n_limit < len(all_files):
        all_files = all_files[:n_limit]
    return all_files

# 从文件名中提取标签，不打印文件名
def get_label_without_print(filename):
    """从文件名中提取标签，不打印文件名"""
    label_part = filename.split('_')[-1]
    label = label_part.split('.')[0]
    if label == '1':
        return 1
    else:
        return 0

# 添加自然排序函数的定义
def natural_sort_key(s):
    """自然排序函数"""
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

# 修改数据加载和处理函数
def load_and_process_data_in_chunks(folder_path, n, chunk_size=100):
    """分块加载和处理数据，使用更小的chunk_size"""
    features_dict = {}
    all_files = get_sorted_files(folder_path, n)
    total_chunks = (len(all_files) + chunk_size - 1) // chunk_size
    
    for i in range(0, len(all_files), chunk_size):
        chunk_files = all_files[i:i + chunk_size]
        print(f"\n处理第 {i//chunk_size + 1}/{total_chunks} 块，共 {len(chunk_files)} 个文件")
        
        for file in tqdm(chunk_files, desc="处理文件"):
            file_path = os.path.join(folder_path, file)
            try:
                # 一次只读取一个文件
                df = pd.read_csv(file_path)
                feature_vector = extract_profile_features(df)
                if feature_vector is not None:
                    label = get_label_without_print(file)
                    features_dict[file] = (feature_vector, label)
                
                # 立即清理内存
                del df
                
            except Exception as e:
                print(f"无法处理文件 {file}: {str(e)}")
                continue
            
        # 每处理完一个块就清理一次内存
        torch.cuda.empty_cache()
        
    return features_dict

# 主程序部分
print("使用设备:", device)

# 使用更小的chunk_size进行分块加载和处理
print("开始加载和处理数据...")
data_load_start = time.time()
features_dict = load_and_process_data_in_chunks(folder_path, n=max(n_values), chunk_size=100)
data_load_time = time.time() - data_load_start
print(f"数据处理完成，用时: {data_load_time:.2f}秒")
print(f"有效特征文件数: {len(features_dict)}")

# 获取所有文件名并排序
all_filenames = sorted(list(features_dict.keys()), key=natural_sort_key)

# 清理GPU内存
torch.cuda.empty_cache()

def prepare_sequences_fast(n_limit, sequence_length):
    """快速准备序列数据，避免数据泄露"""
    start_time = time.time()
    
    # 限制文件数量
    filenames = all_filenames[:n_limit] if n_limit < len(all_filenames) else all_filenames
    
    sequences = []
    labels = []
    
    # 计算组数
    group_size = sequence_length
    num_groups = len(filenames) // group_size
    
    print(f"准备序列数据 (n={n_limit}, sequence_length={sequence_length})...")
    # 按组处理数据，避免数据泄露
    for group_idx in tqdm(range(num_groups), desc="序列生成进度"):
        start_idx = group_idx * group_size
        end_idx = start_idx + group_size
        group_filenames = filenames[start_idx:end_idx]
        
        sequence_features = []
        current_labels = []
        valid_sequence = True
        
        # 处理当前组的所有文件
        for filename in group_filenames:
            if filename in features_dict:
                feature_vector, label = features_dict[filename]
                sequence_features.append(feature_vector)
                current_labels.append(label)
            else:
                valid_sequence = False
                break
        
        # 只有当序列完整且标签一致时才保存
        if valid_sequence and len(sequence_features) == sequence_length:
            if len(set(current_labels)) == 1:  # 确保序列中所有标签一致
                sequences.append(sequence_features)
                labels.append(current_labels[0])
    
    prep_time = time.time() - start_time
    return np.array(sequences), np.array(labels), prep_time

# 修改模型初始化和训练部分
def init_weights(m):
    """初始化模型权重"""
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

def safe_train_test_split(*arrays, **kwargs):
    """安全的训练集划分函数，处理样本量少的情况"""
    try:
        return train_test_split(*arrays, **kwargs)
    except ValueError as e:
        if "The least populated class in y has only 1 member" in str(e):
            # 如果样本太少，取消分层抽样
            print("警告：样本量太少，取消分层抽样...")
            kwargs.pop('stratify', None)
            return train_test_split(*arrays, **kwargs)
        else:
            raise e

# 在实验主循环中修改这些部分
for n in n_values:
    for seq_length in seq_length_values:
        print(f"\n开始实验: n={n}, sequence_length={seq_length}")
        
        # 清理GPU内存
        torch.cuda.empty_cache()
        
        # 为当前n和sequence_length准备数据
        sequences, labels, data_prep_time = prepare_sequences_fast(n, seq_length)
        
        # 检查数据准备
        if sequences.size == 0 or labels.size == 0:
            print(f"错误: n={n}, sequence_length={seq_length} 未能成功准备任何数据，跳过此组合")
            continue
            
        print(f"数据准备完成，用时: {data_prep_time:.2f}秒")
        print(f"序列数量: {len(sequences)}, 标签数量: {len(labels)}")
        print(f"序列数据形状: {sequences.shape}, 标签数据形状: {labels.shape}")
        
        # 添加训练开始时间记录
        train_start_time = time.time()
        
        # 修改数据集划分部分
        try:
            # 首先尝试进行分层抽样的划分
            train_val_sequences, test_sequences, train_val_labels, test_labels = safe_train_test_split(
                sequences, labels, 
                test_size=0.2, 
                random_state=42, 
                stratify=labels
            )
            
            # 然后划分训练集和验证集
            train_sequences, val_sequences, train_labels, val_labels = safe_train_test_split(
                train_val_sequences, train_val_labels, 
                test_size=0.25,  # 这样最终会得到 60%训练，20%验证，20%测试
                random_state=42, 
                stratify=train_val_labels
            )
        except Exception as e:
            print(f"错误: 数据集划分失败 - {str(e)}")
            print(f"跳过当前实验组合: n={n}, sequence_length={seq_length}")
            continue
            
        # 检查划分后的数据集大小
        if len(train_sequences) == 0 or len(val_sequences) == 0 or len(test_sequences) == 0:
            print(f"错误: 划分后的数据集为空，跳过当前实验组合: n={n}, sequence_length={seq_length}")
            continue
            
        print(f"数据集划分完成:")
        print(f"训练集大小: {len(train_sequences)}")
        print(f"验证集大小: {len(val_sequences)}")
        print(f"测试集大小: {len(test_sequences)}")
        
        # 创建 PyTorch Dataset
        train_dataset = ProfileDataset(train_sequences, train_labels)
        val_dataset = ProfileDataset(val_sequences, val_labels)
        test_dataset = ProfileDataset(test_sequences, test_labels)
        
        # 创建 DataLoader
        batch_size = 16
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,pin_memory=True)
        
        # 修改模型初始化
        num_features = sequences.shape[2]
        model = TSMixerModel(seq_length, num_features)
        model.apply(init_weights)  # 应用权重初始化
        model.to(device)
        
        # 修改损失函数和优化器
        criterion = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)  # 添加L2正则化
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
        
        # 添加早停
        early_stopping_patience = 5
        early_stopping_counter = 0
        best_val_loss = float('inf')
        
        # 修改训练循环
        epochs = 50  # 增加训练轮数
        for epoch in range(epochs):
            # 训练阶段
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            # 添加 dropout
            model.train()  # 确保dropout层处于训练模式
            
            for sequences_batch, labels_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [训练]"):
                sequences_batch = sequences_batch.to(device)
                labels_batch = labels_batch.to(device)
                
                # 添加数据增强
                if random.random() < 0.2:  # 20%的概率进行数据增强
                    noise = torch.randn_like(sequences_batch) * 0.01
                    sequences_batch = sequences_batch + noise
                
                optimizer.zero_grad()
                outputs = model(sequences_batch)
                loss = criterion(outputs, labels_batch.unsqueeze(1))
                
                # 添加L1正则化
                l1_lambda = 0.01
                l1_norm = sum(p.abs().sum() for p in model.parameters())
                loss = loss + l1_lambda * l1_norm
                
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                train_loss += loss.item()
                predicted = (outputs > 0.5).int().flatten()
                train_correct += (predicted == labels_batch.int()).sum().item()
                train_total += labels_batch.size(0)
            
            train_accuracy = train_correct / train_total
            avg_train_loss = train_loss / len(train_loader)
            
            # 验证阶段
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for sequences_batch, labels_batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [验证]"):
                    sequences_batch = sequences_batch.to(device)
                    labels_batch = labels_batch.to(device)
                    outputs = model(sequences_batch)
                    loss = criterion(outputs, labels_batch.unsqueeze(1))
                    
                    val_loss += loss.item()
                    predicted = (outputs > 0.5).int().flatten()
                    val_correct += (predicted == labels_batch.int()).sum().item()
                    val_total += labels_batch.size(0)
            
            val_accuracy = val_correct / val_total
            avg_val_loss = val_loss / len(val_loader)
            
            # 学习率调整
            scheduler.step(avg_val_loss)
            
            # 早停检查
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                early_stopping_counter = 0
                # 保存最佳模型
                torch.save(model.state_dict(), f'best_model_n{n}_seq{seq_length}.pth')
            else:
                early_stopping_counter += 1
                
            if early_stopping_counter >= early_stopping_patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
            
            print(f"Epoch [{epoch+1}/{epochs}], "
                  f"训练损失: {avg_train_loss:.4f}, 训练准确率: {train_accuracy:.4f}, "
                  f"验证损失: {avg_val_loss:.4f}, 验证准确率: {val_accuracy:.4f}")
        
        training_time = time.time() - train_start_time
        print(f"训练完成，用时: {training_time:.2f}秒")
        
        # 加载最佳模型进行测试
        model.load_state_dict(torch.load(f'best_model_n{n}_seq{seq_length}.pth'))
        model.eval()
        
        test_correct = 0
        test_total = 0
        
        # 添加测试进度条
        with torch.no_grad():
            test_pbar = tqdm(test_loader, desc="测试评估")
            for sequences_batch, labels_batch in test_pbar:
                sequences_batch, labels_batch = sequences_batch.to(device), labels_batch.to(device)
                outputs = model(sequences_batch)
                predicted = (outputs > 0.5).int().flatten()
                test_correct += (predicted == labels_batch.int()).sum().item()
                test_total += labels_batch.size(0)
                
                # 更新进度条信息
                test_pbar.set_postfix({'acc': f"{test_correct/test_total:.4f}"})
        
        test_accuracy = test_correct / test_total
        print(f"测试准确率: {test_accuracy:.4f}")
        
        # 存储结果
        results['n'].append(n)
        results['sequence_length'].append(seq_length)
        results['train_accuracy'].append(train_accuracy)
        results['val_accuracy'].append(val_accuracy)
        results['test_accuracy'].append(test_accuracy)
        results['training_time'].append(training_time)
        results['data_loading_time'].append(data_prep_time)
        
        # 保存当前结果（以防中途出错）
        pd.DataFrame(results).to_csv('3experiment_results.csv', index=False)
        print(f"已保存当前实验结果到 experiment_results.csv")
        
        # 每次实验结束后清理
        del train_loader, val_loader, test_loader
        del train_dataset, val_dataset, test_dataset
        del model
        torch.cuda.empty_cache()

# 创建最终结果表格
results_df = pd.DataFrame(results)
results_df.to_csv('3final_experiment_results.csv', index=False)
print("所有实验完成，最终结果已保存到 final_experiment_results.csv")

# 使用英文标签绘图，避免中文显示问题
plt.figure(figsize=(15, 10))

# 测试准确率对比图
plt.subplot(2, 2, 1)
for n in n_values:
    n_results = results_df[results_df['n'] == n]
    plt.plot(n_results['sequence_length'], n_results['test_accuracy'], marker='o', label=f'n={n}')
plt.xlabel('Sequence Length')
plt.ylabel('Test Accuracy')
plt.title('Test Accuracy for Different Data Sizes and Sequence Lengths')
plt.legend()
plt.grid(True)

# 训练时间对比图
plt.subplot(2, 2, 2)
for n in n_values:
    n_results = results_df[results_df['n'] == n]
    plt.plot(n_results['sequence_length'], n_results['training_time'], marker='o', label=f'n={n}')
plt.xlabel('Sequence Length')
plt.ylabel('Training Time (s)')
plt.title('Training Time for Different Data Sizes and Sequence Lengths')
plt.legend()
plt.grid(True)

# 数据加载时间对比图
plt.subplot(2, 2, 3)
for n in n_values:
    n_results = results_df[results_df['n'] == n]
    plt.plot(n_results['sequence_length'], n_results['data_loading_time'], marker='o', label=f'n={n}')
plt.xlabel('Sequence Length')
plt.ylabel('Data Preparation Time (s)')
plt.title('Data Preparation Time for Different Data Sizes and Sequence Lengths')
plt.legend()
plt.grid(True)

# 验证准确率对比图
plt.subplot(2, 2, 4)
for n in n_values:
    n_results = results_df[results_df['n'] == n]
    plt.plot(n_results['sequence_length'], n_results['val_accuracy'], marker='o', label=f'n={n}')
plt.xlabel('Sequence Length')
plt.ylabel('Validation Accuracy')
plt.title('Validation Accuracy for Different Data Sizes and Sequence Lengths')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('3experiment_results.png')
print("结果可视化图表已保存到 experiment_results.png")

# 创建热力图
plt.figure(figsize=(10, 8))
pivot_table = results_df.pivot_table(values='test_accuracy', index='n', columns='sequence_length')
sns.heatmap(pivot_table, annot=True, cmap='YlGnBu', fmt='.4f')
plt.title('Test Accuracy Heatmap for Different Data Sizes and Sequence Lengths')
plt.ylabel('Data Size (n)')
plt.xlabel('Sequence Length')
plt.tight_layout()
plt.savefig('3accuracy_heatmap.png')
print("准确率热力图已保存到 accuracy_heatmap.png") 
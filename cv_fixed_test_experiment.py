import numpy as np
import pandas as pd
import os
import time
import torch
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
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
import shutil

# 设置随机种子
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# 设置matplotlib中文字体
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    print("已设置中文字体支持")
except:
    print("警告: 无法设置中文字体，图表中的中文可能无法正确显示")

# 设置实验参数
n_values = [5487]  # 简化实验，只用最大数据量
seq_length_values = [2, 4, 8, 16, 32, 64]
folder_path = '/root/0'
num_folds = 5  # 设置交叉验证折数

# 创建结果存储结构
results = {
    'n': [],
    'sequence_length': [],
    'fold': [],
    'train_accuracy': [],
    'val_accuracy': [],
    'test_accuracy': [],
    'training_time': [],
    'class_distribution': []
}

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 复用原来的函数（get_sorted_files, get_label_without_print, natural_sort_key）
def get_sorted_files(folder_path, n_limit=None):
    all_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.csv')], 
                     key=natural_sort_key)
    if n_limit and n_limit < len(all_files):
        all_files = all_files[:n_limit]
    return all_files

def get_label_without_print(filename):
    label_part = filename.split('_')[-1]
    label = label_part.split('.')[0]
    if label == '1':
        return 1
    else:
        return 0

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

# 数据加载函数
def load_and_process_data_in_chunks(folder_path, n, chunk_size=100):
    features_dict = {}
    all_files = get_sorted_files(folder_path, n)
    total_chunks = (len(all_files) + chunk_size - 1) // chunk_size
    
    for i in range(0, len(all_files), chunk_size):
        chunk_files = all_files[i:i + chunk_size]
        print(f"\n处理第 {i//chunk_size + 1}/{total_chunks} 块，共 {len(chunk_files)} 个文件")
        
        for file in tqdm(chunk_files, desc="处理文件"):
            file_path = os.path.join(folder_path, file)
            try:
                df = pd.read_csv(file_path)
                feature_vector = extract_modified_profile_features(df)
                if feature_vector is not None:
                    label = get_label_without_print(file)
                    features_dict[file] = (feature_vector, label)
                del df
            except Exception as e:
                print(f"无法处理文件 {file}: {str(e)}")
        torch.cuda.empty_cache()
        
    return features_dict

# 新增函数：修改后的特征提取函数
def extract_modified_profile_features(df):
    """
    提取修改后的特征向量：1经度，1纬度，191盐度（从第5行到第195行），191温度（从第5行到195行）
    """
    try:
        # 提取经纬度（第一行）
        longitude = df.iloc[0, 0]
        latitude = df.iloc[0, 1]
        
        # 提取盐度和温度（第5-195行）
        salinity = df.iloc[4:195, 2].values
        temperature = df.iloc[4:195, 3].values
        
        # 检查数据长度
        if len(salinity) != 191 or len(temperature) != 191:
            print(f"警告：数据长度不正确 - 盐度：{len(salinity)}, 温度：{len(temperature)}")
            return None
        
        # 检查是否有NaN值
        if (np.isnan(longitude) or np.isnan(latitude) or 
            np.isnan(salinity).any() or np.isnan(temperature).any()):
            return None
        
        # 组合特征
        feature_vector = np.concatenate([[longitude, latitude], salinity, temperature])
        
        # 验证特征向量长度
        expected_length = 2 + 191 + 191  # 经度纬度 + 盐度 + 温度
        if len(feature_vector) != expected_length:
            print(f"警告：特征向量长度不正确 - 预期：{expected_length}，实际：{len(feature_vector)}")
            return None
            
        return feature_vector
        
    except Exception as e:
        print(f"特征提取错误: {str(e)}")
        return None

# 准备序列数据
def prepare_sequences_fast(n_limit, sequence_length, all_filenames, features_dict):
    start_time = time.time()
    
    filenames = all_filenames[:n_limit] if n_limit < len(all_filenames) else all_filenames
    sequences = []
    labels = []
    filenames_used = []
    
    group_size = sequence_length
    num_groups = len(filenames) // group_size
    
    print(f"准备序列数据 (n={n_limit}, sequence_length={sequence_length})...")
    
    # 首先验证特征向量维度
    first_feature = None
    for filename in features_dict:
        feature_vector, _ = features_dict[filename]
        if first_feature is None:
            first_feature = feature_vector
            feature_dim = len(feature_vector)
            print(f"特征向量维度: {feature_dim}")
        else:
            if len(feature_vector) != feature_dim:
                print(f"警告：特征向量维度不一致 - 预期：{feature_dim}，实际：{len(feature_vector)}")
                return None, None, None, None, None
    
    for group_idx in tqdm(range(num_groups), desc="序列生成进度"):
        start_idx = group_idx * group_size
        end_idx = start_idx + group_size
        group_filenames = filenames[start_idx:end_idx]
        
        sequence_features = []
        current_labels = []
        valid_sequence = True
        
        for filename in group_filenames:
            if filename in features_dict:
                feature_vector, label = features_dict[filename]
                if len(feature_vector) != feature_dim:
                    valid_sequence = False
                    break
                sequence_features.append(feature_vector)
                current_labels.append(label)
            else:
                valid_sequence = False
                break
        
        if valid_sequence and len(sequence_features) == sequence_length:
            if len(set(current_labels)) == 1:  # 确保序列标签一致
                sequences.append(np.array(sequence_features))  # 确保每个序列都是numpy数组
                labels.append(current_labels[0])
                filenames_used.append(group_filenames)
    
    if not sequences:
        print("警告：没有有效的序列生成")
        return None, None, None, None, None
    
    # 转换为numpy数组前再次验证维度
    sequences_array = np.array(sequences)
    print(f"序列数组形状: {sequences_array.shape}")
    
    prep_time = time.time() - start_time
    class_distribution = "正样本比例: {:.2f}% ({}/{})".format(
        100 * np.mean(labels), sum(labels), len(labels))
    print(f"类别分布: {class_distribution}")
    
    return sequences_array, np.array(labels), filenames_used, prep_time, class_distribution

# 修改 safe_train_test_split 函数
def safe_train_test_split(*arrays, **kwargs):
    try:
        # 获取标签数组的索引（通常是第二个参数）
        label_idx = 1
        labels = arrays[label_idx]
        
        # 检查是否有足够的正样本和负样本
        positive_indices = np.where(labels == 1)[0]
        negative_indices = np.where(labels == 0)[0]
        
        if len(positive_indices) == 0 or len(negative_indices) == 0:
            print("错误：数据集中缺少正样本或负样本")
            return None
        
        # 计算测试集中需要的样本数量
        test_size = kwargs.get('test_size', 0.2)
        if isinstance(test_size, float):
            n_test = int(len(labels) * test_size)
        else:
            n_test = test_size
            
        # 确保测试集至少包含一个正样本和一个负样本
        min_samples_per_class = max(1, n_test // 4)  # 每个类别至少占25%
        
        if len(positive_indices) < min_samples_per_class or len(negative_indices) < min_samples_per_class:
            print("警告：某个类别的样本数量过少，无法保证测试集的类别平衡")
            return train_test_split(*arrays, **kwargs)
        
        # 分别从正样本和负样本中抽取测试集
        test_pos_indices = np.random.choice(positive_indices, min_samples_per_class, replace=False)
        test_neg_indices = np.random.choice(negative_indices, min_samples_per_class, replace=False)
        
        # 合并测试集索引
        test_indices = np.concatenate([test_pos_indices, test_neg_indices])
        np.random.shuffle(test_indices)
        
        # 剩余样本作为训练集
        train_indices = np.array([i for i in range(len(labels)) if i not in test_indices])
        
        # 根据索引分割所有数组
        result = []
        for array in arrays:
            result.extend([array[train_indices], array[test_indices]])
            
        return tuple(result)
        
    except Exception as e:
        print(f"警告：测试集划分出现异常 - {str(e)}")
        print("尝试使用常规划分方法...")
        return train_test_split(*arrays, **kwargs)

# 初始化模型权重
def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

# 主实验流程
print("使用设备:", device)

# 加载和处理数据
print("开始加载和处理数据...")
data_load_start = time.time()
features_dict = load_and_process_data_in_chunks(folder_path, n=max(n_values), chunk_size=100)
data_load_time = time.time() - data_load_start
print(f"数据处理完成，用时: {data_load_time:.2f}秒")
print(f"有效特征文件数: {len(features_dict)}")

# 获取文件名并排序
all_filenames = sorted(list(features_dict.keys()), key=natural_sort_key)
torch.cuda.empty_cache()

# 实验循环
for n in n_values:
    for seq_length in seq_length_values:
        print(f"\n开始实验: n={n}, sequence_length={seq_length}")
        torch.cuda.empty_cache()
        
        # 准备序列数据
        sequences, labels, filenames_used, data_prep_time, class_distribution = prepare_sequences_fast(
            n, seq_length, all_filenames, features_dict)
        
        if sequences is None or labels is None:
            print(f"错误: 未能准备任何数据，跳过此组合")
            continue
            
        print(f"数据准备完成，用时: {data_prep_time:.2f}秒")
        print(f"序列数量: {sequences.shape}, 标签数量: {labels.shape}")
        
        # 首先划分出固定的测试集（20%）
        try:
            split_result = safe_train_test_split(
                sequences, labels, filenames_used, 
                test_size=0.2, 
                random_state=42
            )
            
            if split_result is None:
                print(f"错误：无法创建包含正负样本的测试集，跳过此组合")
                continue
            
            train_val_sequences, test_sequences, train_val_labels, test_labels, train_val_filenames, test_filenames = split_result
            
            # 检查测试集的类别分布
            test_pos_ratio = np.mean(test_labels)
            print(f"测试集正样本比例: {test_pos_ratio:.2%} ({sum(test_labels)}/{len(test_labels)})")
            
            if test_pos_ratio == 0 or test_pos_ratio == 1:
                print("错误：测试集中只包含一个类别的样本，跳过此组合")
                continue
            
        except Exception as e:
            print(f"错误: 测试集划分失败 - {str(e)}")
            continue
            
        # 分析测试集类别分布
        test_class_dist = "测试集正样本比例: {:.2f}% ({}/{})".format(
            100 * np.mean(test_labels), sum(test_labels), len(test_labels))
        print(test_class_dist)
        
        # 准备K折交叉验证
        kfold = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
        fold_results = []
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(train_val_sequences, train_val_labels)):
            print(f"\n执行交叉验证 - 折 {fold+1}/{num_folds}")
            
            # 获取当前折的训练集和验证集
            fold_train_sequences = train_val_sequences[train_idx]
            fold_train_labels = train_val_labels[train_idx]
            fold_val_sequences = train_val_sequences[val_idx]
            fold_val_labels = train_val_labels[val_idx]
            
            # 分析当前折的类别分布
            fold_train_dist = "训练集正样本比例: {:.2f}% ({}/{})".format(
                100 * np.mean(fold_train_labels), sum(fold_train_labels), len(fold_train_labels))
            fold_val_dist = "验证集正样本比例: {:.2f}% ({}/{})".format(
                100 * np.mean(fold_val_labels), sum(fold_val_labels), len(fold_val_labels))
            print(fold_train_dist)
            print(fold_val_dist)
            
            # 创建数据集和数据加载器
            train_dataset = ProfileDataset(fold_train_sequences, fold_train_labels)
            val_dataset = ProfileDataset(fold_val_sequences, fold_val_labels)
            test_dataset = ProfileDataset(test_sequences, test_labels)
            
            batch_size = 16
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
            
            # 初始化模型
            train_start_time = time.time()
            num_features = sequences.shape[2]
            model = TSMixerModel(seq_length, num_features)
            model.apply(init_weights)
            model.to(device)
            
            # 定义损失函数和优化器
            criterion = torch.nn.BCELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
            
            # 早停设置
            early_stopping_patience = 5
            early_stopping_counter = 0
            best_val_loss = float('inf')
            
            # 训练循环
            epochs = 50
            for epoch in range(epochs):
                # 训练阶段
                model.train()
                train_loss = 0.0
                train_correct = 0
                train_total = 0
                
                for sequences_batch, labels_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [训练]"):
                    sequences_batch = sequences_batch.to(device)
                    labels_batch = labels_batch.to(device)
                    
                    # 数据增强
                    if random.random() < 0.2:
                        noise = torch.randn_like(sequences_batch) * 0.01
                        sequences_batch = sequences_batch + noise
                    
                    optimizer.zero_grad()
                    outputs = model(sequences_batch)
                    
                    # 计算BCE损失
                    bce_loss = criterion(outputs, labels_batch.unsqueeze(1))
                    
                    # 添加L1正则化，但使用更小的权重
                    l1_lambda = 0.0001  # 降低L1正则化权重
                    l1_norm = sum(p.abs().sum() for p in model.parameters())
                    total_loss = bce_loss + l1_lambda * l1_norm
                    
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    
                    # 记录BCE损失而不是总损失
                    train_loss += bce_loss.item()
                    predicted = (outputs > 0.5).int().flatten()
                    train_correct += (predicted == labels_batch.int()).sum().item()
                    train_total += labels_batch.size(0)
                
                train_accuracy = train_correct / train_total
                avg_train_loss = train_loss / len(train_loader)  # 使用BCE损失计算平均值
                
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
                    torch.save(model.state_dict(), f'best_model_n{n}_seq{seq_length}_fold{fold}.pth')
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
            model.load_state_dict(torch.load(f'best_model_n{n}_seq{seq_length}_fold{fold}.pth'))
            model.eval()
            
            # 在验证集上评估
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for sequences_batch, labels_batch in val_loader:
                    sequences_batch, labels_batch = sequences_batch.to(device), labels_batch.to(device)
                    outputs = model(sequences_batch)
                    predicted = (outputs > 0.5).int().flatten()
                    val_correct += (predicted == labels_batch.int()).sum().item()
                    val_total += labels_batch.size(0)
            
            final_val_accuracy = val_correct / val_total
            
            # 在测试集上评估
            test_correct = 0
            test_total = 0
            test_preds = []
            test_true = []
            
            with torch.no_grad():
                for sequences_batch, labels_batch in test_loader:
                    sequences_batch, labels_batch = sequences_batch.to(device), labels_batch.to(device)
                    outputs = model(sequences_batch)
                    predicted = (outputs > 0.5).int().flatten()
                    test_correct += (predicted == labels_batch.int()).sum().item()
                    test_total += labels_batch.size(0)
                    
                    test_preds.extend(predicted.cpu().numpy())
                    test_true.extend(labels_batch.cpu().numpy())
            
            test_accuracy = test_correct / test_total
            
            # 输出混淆矩阵
            conf_matrix = confusion_matrix(test_true, test_preds)
            print("\n测试集混淆矩阵:")
            print(conf_matrix)
            
            # 存储结果
            fold_results.append({
                'fold': fold,
                'train_accuracy': train_accuracy,
                'val_accuracy': final_val_accuracy,
                'test_accuracy': test_accuracy,
                'training_time': training_time,
                'conf_matrix': conf_matrix
            })
            
            # 清理
            del train_loader, val_loader, test_loader
            del train_dataset, val_dataset, test_dataset
            del model
            torch.cuda.empty_cache()
        
        # 计算平均结果
        avg_train_acc = np.mean([r['train_accuracy'] for r in fold_results])
        avg_val_acc = np.mean([r['val_accuracy'] for r in fold_results])
        avg_test_acc = np.mean([r['test_accuracy'] for r in fold_results])
        avg_time = np.mean([r['training_time'] for r in fold_results])
        
        print(f"\n{num_folds}折交叉验证平均结果:")
        print(f"平均训练准确率: {avg_train_acc:.4f}")
        print(f"平均验证准确率: {avg_val_acc:.4f}")
        print(f"平均测试准确率: {avg_test_acc:.4f}")
        print(f"平均训练时间: {avg_time:.2f}秒")
        
        # 存储结果
        for r in fold_results:
            results['n'].append(n)
            results['sequence_length'].append(seq_length)
            results['fold'].append(r['fold'])
            results['train_accuracy'].append(r['train_accuracy'])
            results['val_accuracy'].append(r['val_accuracy'])
            results['test_accuracy'].append(r['test_accuracy'])
            results['training_time'].append(r['training_time'])
            results['class_distribution'].append(class_distribution)
        
        # 保存当前结果
        pd.DataFrame(results).to_csv('cv_fixed_test_results.csv', index=False)
        print(f"已保存当前实验结果到 cv_fixed_test_results.csv")

# 创建最终结果表格
results_df = pd.DataFrame(results)
results_df.to_csv('cv_fixed_test_final_results.csv', index=False)

# 计算每个序列长度的平均性能
avg_results = results_df.groupby(['n', 'sequence_length']).agg({
    'train_accuracy': 'mean',
    'val_accuracy': 'mean', 
    'test_accuracy': 'mean',
    'training_time': 'mean'
}).reset_index()

print("所有实验完成，最终结果已保存")

# 绘制结果图表
plt.figure(figsize=(15, 10))

# 测试准确率折线图
plt.subplot(2, 2, 1)
for n in n_values:
    n_results = avg_results[avg_results['n'] == n]
    plt.plot(n_results['sequence_length'], n_results['test_accuracy'], marker='o', linewidth=2)
    # 在每个点添加实际值
    for i, point in n_results.iterrows():
        plt.text(point['sequence_length'], point['test_accuracy'], f"{point['test_accuracy']:.4f}", 
                 fontsize=8, ha='center')
plt.xlabel('Sequence Length')
plt.ylabel('Test Accuracy')
plt.title('Mean Test Accuracy (5-fold CV)')
plt.grid(True)

# 验证准确率折线图
plt.subplot(2, 2, 2)
for n in n_values:
    n_results = avg_results[avg_results['n'] == n]
    plt.plot(n_results['sequence_length'], n_results['val_accuracy'], marker='o', linewidth=2)
    # 在每个点添加实际值
    for i, point in n_results.iterrows():
        plt.text(point['sequence_length'], point['val_accuracy'], f"{point['val_accuracy']:.4f}", 
                 fontsize=8, ha='center')
plt.xlabel('Sequence Length')
plt.ylabel('Validation Accuracy')
plt.title('Mean Validation Accuracy (5-fold CV)')
plt.grid(True)

# 测试与验证准确率对比
plt.subplot(2, 2, 3)
bar_width = 0.3
for n in n_values:
    n_results = avg_results[avg_results['n'] == n]
    x = np.arange(len(n_results))
    plt.bar(x - bar_width/2, n_results['test_accuracy'], bar_width, label='Test')
    plt.bar(x + bar_width/2, n_results['val_accuracy'], bar_width, label='Validation')
    plt.xticks(x, n_results['sequence_length'])
plt.xlabel('Sequence Length')
plt.ylabel('Accuracy')
plt.title('Test vs Validation Accuracy')
plt.legend()
plt.grid(True, axis='y')

# 训练时间
plt.subplot(2, 2, 4)
for n in n_values:
    n_results = avg_results[avg_results['n'] == n]
    plt.plot(n_results['sequence_length'], n_results['training_time'], marker='o', linewidth=2)
plt.xlabel('Sequence Length')
plt.ylabel('Training Time (s)')
plt.title('Mean Training Time')
plt.grid(True)

plt.tight_layout()
plt.savefig('cv_fixed_test_results.png')
print("结果可视化图表已保存到 cv_fixed_test_results.png")

# 在所有实验完成后，找出并保存最佳模型
print("\n正在识别和保存最佳模型...")

# 找出测试集准确率最高的组合
best_row = results_df.loc[results_df['test_accuracy'].idxmax()]
best_n = int(best_row['n'])
best_seq_length = int(best_row['sequence_length'])
best_fold = int(best_row['fold'])
best_test_acc = best_row['test_accuracy']
best_val_acc = best_row['val_accuracy']

# 源文件和目标文件路径
source_model_path = f'best_model_n{best_n}_seq{best_seq_length}_fold{best_fold}.pth'
best_model_path = f'BEST_MODEL_seq{best_seq_length}_fold{best_fold}_testacc{best_test_acc:.4f}.pth'

# 复制最佳模型权重文件
shutil.copy(source_model_path, best_model_path)

print(f"\n最佳模型信息:")
print(f"序列长度: {best_seq_length}")
print(f"交叉验证折: {best_fold}")
print(f"测试准确率: {best_test_acc:.4f}")
print(f"验证准确率: {best_val_acc:.4f}")
print(f"模型已保存至: {best_model_path}")

# 找出每个序列长度下的最佳模型
print("\n各序列长度下的最佳模型:")
for seq_length in seq_length_values:
    seq_results = results_df[results_df['sequence_length'] == seq_length]
    if not seq_results.empty:
        seq_best_row = seq_results.loc[seq_results['test_accuracy'].idxmax()]
        seq_best_fold = int(seq_best_row['fold'])
        seq_best_test_acc = seq_best_row['test_accuracy']
        
        # 为每个序列长度保存最佳模型
        seq_source_path = f'best_model_n{best_n}_seq{seq_length}_fold{seq_best_fold}.pth'
        seq_best_path = f'best_seq{seq_length}_fold{seq_best_fold}_testacc{seq_best_test_acc:.4f}.pth'
        shutil.copy(seq_source_path, seq_best_path)
        
        print(f"序列长度 {seq_length}: 最佳折={seq_best_fold}, 测试准确率={seq_best_test_acc:.4f}, 保存为 {seq_best_path}")

# 保存模型性能摘要表
summary_table = avg_results.sort_values('test_accuracy', ascending=False)
summary_table.to_csv('model_performance_summary.csv', index=False)
print("\n模型性能摘要已保存至 model_performance_summary.csv") 
import os
import numpy as np
import pandas as pd
import onnxruntime as ort
from tqdm import tqdm

# 特征提取函数
def extract_modified_profile_features(df):
    """提取修改后的特征向量：经度，纬度，盐度，温度"""
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

# 加载数据并创建序列
def load_and_create_sequences(folder_path, sequence_length=16):
    """加载数据并创建序列"""
    print("加载数据...")
    
    # 加载原始数据
    features_dict = {}
    all_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.csv')])
    
    for file in tqdm(all_files, desc="处理数据文件"):
        try:
            df = pd.read_csv(os.path.join(folder_path, file))
            feature_vector = extract_modified_profile_features(df)
            if feature_vector is not None:
                features_dict[file] = feature_vector
        except Exception as e:
            print(f"处理文件 {file} 时出错: {str(e)}")
            continue
    
    # 准备序列数据
    sequences = []
    sequence_files = []
    
    # 按照文件名排序
    sorted_files = sorted(features_dict.keys())
    
    # 创建所有可能的序列
    for i in range(0, len(sorted_files) - sequence_length + 1):
        seq_files = sorted_files[i:i + sequence_length]
        seq_features = [features_dict[f] for f in seq_files]
        sequences.append(np.array(seq_features))
        sequence_files.append(seq_files)
    
    return np.array(sequences, dtype=np.float32), sequence_files

def main():
    # 设置参数
    model_path = './model_quantized.onnx'
    data_folder = './lianghua2'
    sequence_length = 16
    
    # 加载ONNX模型
    print(f"加载量化模型: {model_path}")
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    
    # 加载数据
    sequences, sequence_files = load_and_create_sequences(data_folder, sequence_length)
    
    if len(sequences) == 0:
        print("错误: 未能创建任何序列，请检查数据")
        return
    
    print(f"创建了 {len(sequences)} 个序列")
    
    # 进行预测
    print("进行预测...")
    results = []
    
    for i, seq in enumerate(tqdm(sequences, desc="预测")):
        # 添加批次维度并确保是float32类型
        input_data = np.expand_dims(seq, axis=0).astype(np.float32)
        
        # 预测
        output = session.run(None, {input_name: input_data})[0]
        probability = output[0][0]
        prediction = 1 if probability > 0.5 else 0
        
        # 保存结果
        results.append({
            'sequence_index': i,
            'files': sequence_files[i],
            'probability': probability,
            'prediction': prediction
        })
    
    # 输出结果
    print("\n预测结果:")
    for result in results:
        # 获取当前序列的最后一个文件名
        last_file = result['files'][-1]
        
        # 推断下一个文件名
        # 假设文件名格式为 "itp102_stationXXX_Y.csv"
        parts = last_file.split('_')
        if len(parts) >= 2:
            station_part = parts[1]  # 获取"stationXXX"部分
            if station_part.startswith('station'):
                try:
                    # 提取站点编号
                    station_num = int(station_part[7:])
                    # 生成下一个站点的文件名
                    next_station_num = station_num + 1
                    next_file = f"{parts[0]}_station{next_station_num}_0.csv"
                    
                    print(f"\n序列 {result['sequence_index']}:")
                    print(f"预测: {next_file} {'正例' if result['prediction'] == 1 else '负例'} (概率: {result['probability']:.6f})")
                    print(f"基于文件: {result['files']}")
                except ValueError:
                    # 如果无法解析站点编号，则使用原始输出
                    print(f"\n序列 {result['sequence_index']}:")
                    print(f"预测: {'正例' if result['prediction'] == 1 else '负例'} (概率: {result['probability']:.6f})")
                    print(f"文件: {result['files']}")
            else:
                # 如果不是预期的格式，则使用原始输出
                print(f"\n序列 {result['sequence_index']}:")
                print(f"预测: {'正例' if result['prediction'] == 1 else '负例'} (概率: {result['probability']:.6f})")
                print(f"文件: {result['files']}")
        else:
            # 如果不是预期的格式，则使用原始输出
            print(f"\n序列 {result['sequence_index']}:")
            print(f"预测: {'正例' if result['prediction'] == 1 else '负例'} (概率: {result['probability']:.6f})")
            print(f"文件: {result['files']}")
    
    # 统计结果
    positive_count = sum(1 for r in results if r['prediction'] == 1)
    negative_count = sum(1 for r in results if r['prediction'] == 0)
    
    print(f"\n统计信息:")
    print(f"总序列数: {len(results)}")
    print(f"正例数量: {positive_count} ({positive_count/len(results)*100:.2f}%)")
    print(f"负例数量: {negative_count} ({negative_count/len(results)*100:.2f}%)")

if __name__ == "__main__":
    main() 
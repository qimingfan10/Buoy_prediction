from pathlib import Path
import random
import shutil
from tqdm import tqdm

def balance_folder_pair(folder_0, folder_1):
    """平衡两个文件夹中的文件数量"""
    # 获取两个文件夹中的所有CSV文件
    files_0 = list(folder_0.glob("*.csv"))
    files_1 = list(folder_1.glob("*.csv"))
    
    print(f"\n处理文件夹对: {folder_0.parent.name}")
    print(f"原始数量 - 类别0: {len(files_0)}, 类别1: {len(files_1)}")
    
    # 如果类别0的文件更多，随机删除多余的文件
    if len(files_0) > len(files_1):
        files_to_remove = random.sample(files_0, len(files_0) - len(files_1))
        print(f"需要从类别0中删除 {len(files_to_remove)} 个文件")
        
        # 创建backup文件夹
        backup_dir = folder_0.parent.parent / "backup" / folder_0.parent.name / folder_0.name
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        # 移动文件到backup文件夹
        print("正在移动文件到备份文件夹...")
        for file in tqdm(files_to_remove):
            shutil.move(str(file), str(backup_dir / file.name))
        
        # 重新计算数量
        files_0 = list(folder_0.glob("*.csv"))
        print(f"平衡后数量 - 类别0: {len(files_0)}, 类别1: {len(files_1)}")
    else:
        print("文件夹已经平衡，无需操作")

def main():
    # 设置数据根目录
    data_root = Path("data")
    
    # 检查数据目录是否存在
    if not data_root.exists():
        raise FileNotFoundError("找不到data文件夹")
    
    # 处理训练集
    train_folder_0 = data_root / "train/folder_0"
    train_folder_1 = data_root / "train/folder_1"
    
    if not train_folder_0.exists() or not train_folder_1.exists():
        raise FileNotFoundError("找不到训练数据文件夹")
    
    balance_folder_pair(train_folder_0, train_folder_1)
    
    # 处理验证集
    val_folder_0 = data_root / "val/folder_0"
    val_folder_1 = data_root / "val/folder_1"
    
    if not val_folder_0.exists() or not val_folder_1.exists():
        raise FileNotFoundError("找不到验证数据文件夹")
    
    balance_folder_pair(val_folder_0, val_folder_1)
    
    print("\n数据集平衡完成！")
    print("原始文件已备份到 data/backup 文件夹中")

if __name__ == "__main__":
    main() 
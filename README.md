# 浮标预测

这是一个包含模型预测相关代码和文件的仓库，提供了PyTorch和ONNX两种格式的模型预测示例。

## 仓库结构

```
.
├── newpredict_best.py              # PyTorch版本预测函数
├── BEST_MODEL_seq16_fold2_testacc0.8462.pth # PyTorch模型权重文件
├── requirement.txt                 # 项目依赖库列表
├── cv_fixed_test_experiment.py     # 预训练文件，使用五折交叉验证
└── model_quantized/                # 量化ONNX模型及相关文件
    ├── model_quantized.onnx        # 量化后的ONNX模型文件
    ├── predict_quantized.py        # 量化后的预测文件
    └── ...                         
```

## 环境设置

1.  克隆本仓库到本地：
    ```bash
    git clone https://github.com/qimingfan10/Buoy_prediction.git
    cd Buoy_prediction
    ```
2.  安装所需的依赖库。请确保你已经安装了 Python。
    ```bash
    pip install -r requirement.txt
    ```
    这会安装 `torch`, `pandas`, `onnxruntime` 等依赖。

## 使用说明

本项目提供了使用PyTorch和ONNX格式进行模型预测的示例。

### 1. PyTorch 模型预测

使用 `newpredict_best.py` 脚本和 `BEST_MODEL_seq16_fold2_testacc0.8462.pth` 模型文件进行预测。该脚本会加载模型和示例测试数据集。

运行命令：

```bash
python newpredict_best.py
```

**注意:**
* 请确保 `BEST_MODEL_seq16_fold2_testacc0.8462.pth` 文件位于与 `newpredict_best.py` 相同的目录下。

### 2. ONNX 模型预测 (量化版本)

`model_quantized` 文件夹下包含了量化后的ONNX模型 `model_quantized.onnx` 以及其它用于预训练的相关文件。

运行 `predict_quantized.py` 用于 ONNX 预测：

```bash
cd model_quantized
python predict_quantized.py
```

**注意:**
* 你需要使用 ONNX Runtime 或其他支持 ONNX 的推理引擎来加载和运行 `model_quantized.onnx`。

## 文件说明

* `newpredict_best.py`: 基于 PyTorch 加载 `.pth` 模型进行预测的主脚本。
* `BEST_MODEL_seq16_fold2_testacc0.8462.pth`: 训练好的 PyTorch 模型权重文件。
* `requirement.txt`: 列出了运行项目所需的 Python 库及其版本。
* `model_quantized/`: 存放 ONNX 量化模型及相关文件的目录。
* `model_quantized/model_quantized.onnx`: 经过量化处理的 ONNX 模型文件，通常体积更小，推理速度更快。

---

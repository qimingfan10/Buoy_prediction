[English](#en) | [中文](#zh-cn)

---

<a name="en"></a>

# Buoy Prediction

This repository contains code and files related to model prediction for buoys, providing examples for both PyTorch and ONNX model formats.

## Repository Structure


.
├── newpredict_best.py # PyTorch version prediction function
├── BEST_MODEL_seq16_fold2_testacc0.8462.pth # PyTorch model weights file
├── requirement.txt # List of project dependency libraries
├── cv_fixed_test_experiment.py # Pre-training file using 5-fold cross-validation
└── model_quantized/ # Directory for quantized ONNX model and related files
├── model_quantized.onnx # Quantized ONNX model file
├── predict_quantized.py # Prediction script for the quantized model
└── ... # Other related files

## Environment Setup

1.  Clone this repository to your local machine:
    ```bash
    git clone https://github.com/qimingfan10/Buoy_prediction.git
    cd Buoy_prediction
    ```
2.  Install the required dependencies. Make sure you have Python installed.
    ```bash
    pip install -r requirement.txt
    ```
    This will install `torch`, `pandas`, `onnxruntime`, and other dependencies.

## Usage Instructions

This project provides examples for model prediction using both PyTorch and ONNX formats.

### 1. PyTorch Model Prediction

Use the `newpredict_best.py` script and the `BEST_MODEL_seq16_fold2_testacc0.8462.pth` model file for prediction. The script loads the model and an example test dataset.

Run the command:

```bash
python newpredict_best.py
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
IGNORE_WHEN_COPYING_END

Note:

Ensure the BEST_MODEL_seq16_fold2_testacc0.8462.pth file is located in the same directory as newpredict_best.py.

2. ONNX Model Prediction (Quantized Version)

The model_quantized folder contains the quantized ONNX model model_quantized.onnx and other related files for pre-training.

Run predict_quantized.py for ONNX prediction:

cd model_quantized
python predict_quantized.py
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END

Note:

You need to use ONNX Runtime or another inference engine that supports ONNX to load and run model_quantized.onnx.

File Descriptions

newpredict_best.py: The main script for prediction using the PyTorch .pth model.

BEST_MODEL_seq16_fold2_testacc0.8462.pth: The trained PyTorch model weights file.

requirement.txt: Lists the Python libraries and their versions required to run the project.

cv_fixed_test_experiment.py: Script used for pre-training the model, employing 5-fold cross-validation.

model_quantized/: Directory containing the ONNX quantized model and related files.

model_quantized/model_quantized.onnx: The ONNX model file that has undergone quantization, typically resulting in smaller size and faster inference.

model_quantized/predict_quantized.py: Prediction script specifically for the quantized ONNX model.

<a name="zh-cn"></a>

浮标预测

这是一个包含模型预测相关代码和文件的仓库，提供了PyTorch和ONNX两种格式的模型预测示例。

仓库结构
.
├── newpredict_best.py               # PyTorch版本预测函数
├── BEST_MODEL_seq16_fold2_testacc0.8462.pth # PyTorch模型权重文件
├── requirement.txt                  # 项目依赖库列表
├── cv_fixed_test_experiment.py      # 预训练文件，使用五折交叉验证
└── model_quantized/                 # 量化ONNX模型及相关文件
    ├── model_quantized.onnx         # 量化后的ONNX模型文件
    ├── predict_quantized.py         # 量化后的预测文件
    └── ...                          # 其他相关文件
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
IGNORE_WHEN_COPYING_END
环境设置

克隆本仓库到本地：

git clone https://github.com/qimingfan10/Buoy_prediction.git
cd Buoy_prediction
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END

安装所需的依赖库。请确保你已经安装了 Python。

pip install -r requirement.txt
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END

这会安装 torch, pandas, onnxruntime 等依赖。

使用说明

本项目提供了使用PyTorch和ONNX格式进行模型预测的示例。

1. PyTorch 模型预测

使用 newpredict_best.py 脚本和 BEST_MODEL_seq16_fold2_testacc0.8462.pth 模型文件进行预测。该脚本会加载模型和示例测试数据集。

运行命令：

python newpredict_best.py
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END

注意:

请确保 BEST_MODEL_seq16_fold2_testacc0.8462.pth 文件位于与 newpredict_best.py 相同的目录下。

2. ONNX 模型预测 (量化版本)

model_quantized 文件夹下包含了量化后的ONNX模型 model_quantized.onnx 以及其它用于预训练的相关文件。

运行 predict_quantized.py 用于 ONNX 预测：

cd model_quantized
python predict_quantized.py
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END

注意:

你需要使用 ONNX Runtime 或其他支持 ONNX 的推理引擎来加载和运行 model_quantized.onnx。

文件说明

newpredict_best.py: 基于 PyTorch 加载 .pth 模型进行预测的主脚本。

BEST_MODEL_seq16_fold2_testacc0.8462.pth: 训练好的 PyTorch 模型权重文件。

requirement.txt: 列出了运行项目所需的 Python 库及其版本。

cv_fixed_test_experiment.py: 用于预训练模型的文件，采用了五折交叉验证方法。

model_quantized/: 存放 ONNX 量化模型及相关文件的目录。

model_quantized/model_quantized.onnx: 经过量化处理的 ONNX 模型文件，通常体积更小，推理速度更快。

model_quantized/predict_quantized.py: 针对量化 ONNX 模型的预测脚本。

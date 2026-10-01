# 番茄病害智能诊断系统

基于深度学习的番茄叶片病害识别与诊断系统。上传一张番茄叶片图片后，系统会给出前三名病害（或健康）预测、置信度，以及对应的防治建议。

数据来源于 PlantVillage 番茄叶片子集，采用 ImageNet 预训练的 **ResNet50** 迁移学习，对 **10 类** 叶片状态进行分类。验证集准确率约 **97.6%**（20 个 epoch）。

## 功能特点

- 叶片图像分类：细菌性斑点病、早疫病、晚疫病、叶霉病、壳针孢叶斑病、靶斑病、花叶病毒病、黄化曲叶病毒病、二斑叶螨，以及健康叶片
- Web 诊断界面：Gradio 上传图片即可诊断，并展示 Top-3 概率条
- 防治建议：针对主预测类别给出用药、环境调控与农事操作建议（仅供参考）
- 模型评估可视化：混淆矩阵、ROC / PR 曲线、训练曲线、Grad-CAM、样本预测对比

## 技术栈

| 模块 | 说明 |
|------|------|
| 模型 | ResNet50（ImageNet 预训练，冻结骨干 + 全局平均池化 + Dropout + Softmax） |
| 输入 | RGB 图像，尺寸 224×224，ResNet50 预处理 |
| 框架 | TensorFlow / Keras |
| 界面 | Gradio 3.x |
| 数据 | PlantVillage 番茄叶片图像（`data/color_tom`） |

## 识别类别

| 标签 | 中文名称 |
|------|----------|
| Tomato___Bacterial_spot | 番茄细菌性斑点病 |
| Tomato___Early_blight | 番茄早疫病 |
| Tomato___Late_blight | 番茄晚疫病 |
| Tomato___Leaf_Mold | 番茄叶霉病 |
| Tomato___Septoria_leaf_spot | 番茄壳针孢叶斑病 |
| Tomato___Target_Spot | 番茄靶斑病 |
| Tomato___Tomato_mosaic_virus | 番茄花叶病毒病 |
| Tomato___Tomato_Yellow_Leaf_Curl_Virus | 番茄黄化曲叶病毒病 |
| Tomato___Spider_mites Two-spotted_spider_mite | 番茄二斑叶螨 |
| Tomato___healthy | 健康 |

类别中文映射见 `pest_detection/models/trained_model/label_mapping.json`。

## 项目结构

```
tmwk/
├── pest_detection/                 # 诊断系统主目录
│   ├── gradio_app/
│   │   ├── app.py                  # Gradio 诊断应用（启动入口）
│   │   └── examples/               # 界面示例图
│   ├── models/
│   │   ├── trained_model/
│   │   │   ├── best_model.keras    # 推理用 Keras 模型
│   │   │   ├── best_model.h5
│   │   │   └── label_mapping.json
│   │   └── exported_model/         # SavedModel 导出
│   ├── scripts/
│   │   ├── data_processing.py      # 数据加载、类别名修正、采样与增强
│   │   └── train_model.py          # ResNet50 模型构建与训练骨架
│   ├── data/raw_data/              # 原始叶片图像（按类别分子目录）
│   └── requirements.txt
├── data/color_tom/                 # 训练/评估用番茄叶片数据集
├── Model_file/                     # 训练导出模型与相关 notebook
│   ├── leaf_disease_resnet50.h5
│   ├── best_model.h5
│   ├── exported_model/
│   └── leaf-disease.ipynb
├── visualization/                  # 评估与可视化脚本
│   ├── plot_training_curves.py
│   ├── confusion_matrix.py
│   ├── confusion_matrix_layor.py
│   ├── precision_recall_Auc_F1score_Curve2.py
│   └── PredictCompare.py
├── data.ipynb                      # 数据清洗与类别统计
├── mlExp.ipynb                     # 方案设计说明
├── Grad_cm.py                      # Grad-CAM 可视化
├── plot_training_curves.py         # 训练/验证准确率与损失曲线
├── train.tfrecord / test.tfrecord  # TFRecord 数据（体积较大）
└── README.md
```

根目录还有若干评估脚本与结果图（混淆矩阵、ROC、PR、训练曲线等），与 `visualization/` 中脚本对应。

## 环境要求

- Python 3.10～3.12（当前仓库按 3.12 + TensorFlow 2.18 配置依赖）
- Windows / Linux 均可
- 建议使用虚拟环境；推理需要已训练好的模型权重

依赖以 `pest_detection/requirements.txt` 为准。`numpy==1.21.6` 仅支持 Python 3.10 及以下，3.12 环境请使用文件中已更新的 `numpy>=1.26` 约束。

可视化脚本可能还需 `opencv-python`、`scikit-learn`。部分评估代码在 TensorFlow 2.19 下运行过。

## 启动诊断系统

### 1. 安装依赖

```bash
cd pest_detection
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
# source .venv/bin/activate

pip install -r requirements.txt
有可能会遇到下载 TensorFlow 时网络超时的问题，可以使用以下命令
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
安装时间可能会有些长，请耐心等待
```

### 2. 确认模型文件

应用按脚本所在位置解析相对路径，无需改盘符。请确认以下文件存在：

- `pest_detection/models/trained_model/best_model.keras`
- `pest_detection/models/trained_model/label_mapping.json`

### 3. 启动 Web 界面

```bash
python gradio_app/app.py
```

默认监听 `localhost:7860`。浏览器打开：

```
http://localhost:7860
```

上传叶片图片，或点击界面中的示例图，即可查看诊断结果与防治建议。

如需局域网访问，将 `app.py` 末尾的 `server_name="localhost"` 改为 `"0.0.0.0"`。

## 数据与训练

1. 将番茄叶片图像按类别放入 `data/color_tom/` 或 `pest_detection/data/raw_data/`（一类一个子目录）。
2. 数据清洗与统计可参考根目录 `data.ipynb`。
3. 预处理脚本：`pest_detection/scripts/data_processing.py`（类别名修正、欠采样、数据增强）。
4. 训练脚本：`pest_detection/scripts/train_model.py`（ResNet50，batch size 32，最多 50 epoch，按验证准确率保存 `best_model.keras`）。

`train_model.py` 中的 `train_dataset` / `val_dataset` 需按实际 TFRecord 或 `image_dataset_from_directory` 路径补全后再运行。

训练过程可视化：

```bash
python plot_training_curves.py
# 或
python visualization/plot_training_curves.py
```

## 模型评估

`visualization/` 及根目录脚本按项目根目录解析相对路径，加载 SavedModel（`Model_file/exported_model`）并在 `data/color_tom` 上评估。

| 脚本 | 作用 |
|------|------|
| `visualization/confusion_matrix.py` | 混淆矩阵 |
| `visualization/precision_recall_Auc_F1score_Curve2.py` | PR / ROC / F1 |
| `visualization/PredictCompare.py` | 随机样本预测对比 |
| `Grad_cm.py` | Grad-CAM 关注区域 |

## 注意事项

- 防治建议仅供学习与演示，实际用药请遵循当地植保规范。
- `train.tfrecord`、`test.tfrecord` 以及 `data/` 下图片体积较大，克隆或拷贝仓库时注意磁盘空间。
- 本仓库为课程/实验项目，`pest_detection/scripts/train_model.py` 为训练骨架，完整训练流程以 notebook 与已保存权重为主。

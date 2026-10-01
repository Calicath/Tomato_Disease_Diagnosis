import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_recall_curve, roc_curve, auc, f1_score
from sklearn.preprocessing import label_binarize

# 固定所有随机种子
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)
os.environ['TF_DETERMINISTIC_OPS'] = '1'

# 设置中文字体
plt.rcParams["font.family"] = "Microsoft YaHei"
plt.rcParams['axes.unicode_minus'] = False

# 参数
IMG_SIZE = (224, 224)
NUM_CLASSES = 10
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(PROJECT_ROOT, "data", "color_tom")
model_path = os.path.join(PROJECT_ROOT, "Model_file", "exported_model")
BATCH_SIZE = 32  # 每批处理的样本数

# 获取类别
class_names = sorted(os.listdir(data_dir))
label_encoder = LabelEncoder()
label_encoder.fit(class_names)

# 加载 SavedModel
model = tf.saved_model.load(model_path)
predict_fn = model.signatures["serving_default"]
input_key = list(predict_fn.structured_input_signature[1].keys())[0]

# 分批处理所有样本
all_outputs = []
all_true_indices = []

for class_name in class_names:
    class_path = os.path.join(data_dir, class_name)
    files = os.listdir(class_path)
    class_label = label_encoder.transform([class_name])[0]
    
    # 分批处理当前类别的所有文件
    for i in range(0, len(files), BATCH_SIZE):
        batch_files = files[i:i+BATCH_SIZE]
        batch_images = []
        
        # 加载当前批次的图像
        for file in batch_files:
            path = os.path.join(class_path, file)
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, IMG_SIZE)
            img = img.astype(np.float32)
            batch_images.append(img)
        
        # 转换为 tensor 并推理
        if batch_images:
            input_tensor = tf.convert_to_tensor(np.array(batch_images), dtype=tf.float32)
            outputs = predict_fn(**{input_key: input_tensor})
            output_tensor = list(outputs.values())[0].numpy()
            
            # 收集结果
            all_outputs.append(output_tensor)
            all_true_indices.extend([class_label] * len(batch_files))

# 合并所有批次的结果
output_tensor = np.vstack(all_outputs)
true_indices = np.array(all_true_indices)

# Binarize the true labels for multi-class metrics
true_binary = label_binarize(true_indices, classes=np.arange(NUM_CLASSES))

# 后续代码保持不变...
# Calculate precision, recall, F1-score for each class
precision = dict()
recall = dict()
f1_scores = []
for i in range(NUM_CLASSES):
    precision[i], recall[i], _ = precision_recall_curve(true_binary[:, i], output_tensor[:, i])
    f1_scores.append(f1_score(true_binary[:, i], output_tensor[:, i] > 0.5))

# Calculate AUC for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(NUM_CLASSES):
    fpr[i], tpr[i], _ = roc_curve(true_binary[:, i], output_tensor[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot Precision-Recall curve
plt.figure(figsize=(10, 8))
for i in range(NUM_CLASSES):
    plt.plot(recall[i], precision[i], lw=2, label=f'Class {class_names[i]} (F1 = {f1_scores[i]:.2f})')
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend(loc="best")
plt.savefig("precision_recall_curve.png")
plt.show()

# Plot ROC curve
plt.figure(figsize=(10, 8))
for i in range(NUM_CLASSES):
    plt.plot(fpr[i], tpr[i], lw=2, label=f'Class {class_names[i]} (AUC = {roc_auc[i]:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend(loc="best")
plt.savefig("roc_curve.png")
plt.show()
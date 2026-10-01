import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
import random
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# 固定所有随机种子
random.seed(42)          # Python随机模块
np.random.seed(42)       # NumPy
tf.random.set_seed(42)   # TensorFlow
os.environ['TF_DETERMINISTIC_OPS'] = '1'  # 强制TensorFlow使用确定性操作

# 设置中文字体
plt.rcParams["font.family"] = "Microsoft YaHei"
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 参数
IMG_SIZE = (224, 224)
NUM_CLASSES = 10
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(PROJECT_ROOT, "data", "color_tom")
model_path = os.path.join(PROJECT_ROOT, "Model_file", "exported_model")

# 获取类别
class_names = sorted(os.listdir(data_dir))
label_encoder = LabelEncoder()
label_encoder.fit(class_names)

# 加载 SavedModel
model = tf.saved_model.load(model_path)
predict_fn = model.signatures["serving_default"]
input_key = list(predict_fn.structured_input_signature[1].keys())[0]

# 准备数据（不要除以255）
images = []
true_labels = []
num_per_class = 10

for class_name in class_names:
    class_path = os.path.join(data_dir, class_name)
    files = os.listdir(class_path)
    selected_files = random.sample(files, min(num_per_class, len(files)))
    
    for file in selected_files:
        path = os.path.join(class_path, file)
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, IMG_SIZE)
        img = img.astype(np.float32)  # 保持原值（0~255）
        images.append(img)
        true_labels.append(class_name)

# 转换为 tensor
input_tensor = tf.convert_to_tensor(np.array(images), dtype=tf.float32)

# 推理
outputs = predict_fn(**{input_key: input_tensor})
output_tensor = list(outputs.values())[0].numpy()  # softmax 概率
pred_indices = np.argmax(output_tensor, axis=1)
true_indices = label_encoder.transform(true_labels)

# 绘制混淆矩阵
cm = confusion_matrix(true_indices, pred_indices)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)

plt.figure(figsize=(30, 28))  # 调整图像大小
disp.plot(cmap="Blues", xticks_rotation=45)

plt.title("Confusion Matrix（混淆矩阵）", fontsize=18)
plt.xticks(fontsize=6)
plt.yticks(fontsize=6)
plt.subplots_adjust(bottom=0.5)  # 底部留白50%，根据需要调整大小
plt.setp(plt.gca().get_xticklabels(), ha="right", rotation=70)

plt.tight_layout()
plt.savefig("confusion_matrix_h5_improved.png")
plt.show()

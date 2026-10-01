import os
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
#本代码每次从数据集中随机选择8张图片进行预测，并绘制预测结果

# 设置中文字体
plt.rcParams["font.family"] = "Microsoft YaHei"
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 模型路径
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SAVED_MODEL_PATH = os.path.join(PROJECT_ROOT, "Model_file", "exported_model")

# 加载模型
print(f"正在加载 SavedModel: {SAVED_MODEL_PATH}")
model = tf.keras.Sequential([
    tf.keras.layers.TFSMLayer(SAVED_MODEL_PATH, call_endpoint='serving_default')
])

# 类别名
class_names = ['Bacterial_spot', 'Early_blight', 'Healthy', 'Late_blight','Leaf_Mold', 'Septoria_leaf_spot', 'Spider_mites', 'Target_Spot', 'Mosaic_virus' , 'Yellow_Leaf_Curl_Virus']

# 测试集目录
test_data_dir = os.path.join(PROJECT_ROOT, "data", "color_tom")

# 加载测试集
test_dataset = tf.keras.utils.image_dataset_from_directory(
    test_data_dir,
    labels='inferred',
    label_mode='int',
    image_size=(224, 224),
    batch_size=32,
    shuffle=False
)

# 读取所有图片和标签（tf.data.Dataset -> numpy）
images, labels = [], []
for batch in test_dataset:
    x, y = batch
    images.extend(x.numpy())
    labels.extend(y.numpy())
images = np.array(images)
labels = np.array(labels)

# 随机选8张样本
num_samples = 8
indices = random.sample(range(len(images)), num_samples)
selected_images = images[indices]
true_labels = labels[indices]

# 模型推理（TFSMLayer返回字典）
raw_output = model(selected_images)
print("模型输出键：", raw_output.keys())
predictions = raw_output['dense_1'].numpy()

# 取Top-3预测
top3_indices = np.argsort(predictions, axis=1)[:, -3:][:, ::-1]

# 绘制结果
plt.figure(figsize=(20, 10))
for i in range(num_samples):
    ax = plt.subplot(2, 4, i + 1)
    img = selected_images[i]
    true_label = class_names[true_labels[i]]
    top3 = top3_indices[i]
    top3_labels = [class_names[j] for j in top3]
    top3_probs = predictions[i][top3]
    
    # 格式化预测概率为百分比，保留2位小数
    prob_percent = [f"{prob*100:.2f}%" for prob in top3_probs]
    
    # 创建标题和预测信息文本
    title = f"真实: {true_label}\n预测: {top3_labels[0]} ({prob_percent[0]})"
    
    # 创建Top-3预测详细信息
    top3_info = "Top-3预测:\n"
    for j in range(3):
        top3_info += f"{j+1}. {top3_labels[j]}: {prob_percent[j]}\n"

    ax.imshow(img.astype(np.uint8))
    ax.set_title(title, fontsize=12)
    ax.set_xlabel(top3_info, fontsize=10)
    ax.axis("off")

plt.tight_layout()
plt.savefig("sample_predictions_top3.png", dpi=300)
plt.show()
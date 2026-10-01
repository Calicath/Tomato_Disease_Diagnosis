import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import random
import os
import cv2

# 模型路径
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SAVED_MODEL_PATH = os.path.join(PROJECT_ROOT, "Model_file", "exported_model")

print(f"正在加载 SavedModel: {SAVED_MODEL_PATH}")
model = tf.keras.Sequential([
    tf.keras.layers.TFSMLayer(SAVED_MODEL_PATH, call_endpoint='serving_default')
])

# 类别名称（手动设置）
class_names = ['Bacterial_spot', 'Early_blight', 'Healthy', 'Late_blight',
'Leaf_Mold', 'Septoria_leaf_spot', 'Spider_mites', 'Target_Spot',
'Mosaic_virus','Yellow_Leaf_Curl_Virus']

# 加载测试集
test_dir = os.path.join(PROJECT_ROOT, "data", "color_tom")
img_size = (224, 224)
test_dataset = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=img_size,
    batch_size=32,
    shuffle=False
)

# 提取测试集图像和标签
batch_size = 32  # 保持和dataset一致
misclassified_images = []
misclassified_true_labels = []
misclassified_preds = []

for x_batch, y_batch in test_dataset:
    # 获取当前batch的预测
    raw_output = model(x_batch, training=False)
    if isinstance(raw_output, dict):
        batch_preds = list(raw_output.values())[0].numpy()
    else:
        batch_preds = raw_output.numpy()
    batch_pred_labels = np.argmax(batch_preds, axis=1)
    
    # 记录分类错误的样本
    for i in range(len(y_batch)):
        if batch_pred_labels[i] != y_batch[i]:
            misclassified_images.append(x_batch[i].numpy())
            misclassified_true_labels.append(y_batch[i].numpy())
            misclassified_preds.append(batch_pred_labels[i])
    
    # 如果已收集足够样本，提前停止
    if len(misclassified_images) >= 6:
        break

# 随机选择6个错误样本（如果超过6个）
if len(misclassified_images) > 6:
    indices = random.sample(range(len(misclassified_images)), 6)
    selected_images = np.array([misclassified_images[i] for i in indices])
    true_labels = np.array([misclassified_true_labels[i] for i in indices])
    wrong_preds = np.array([misclassified_preds[i] for i in indices])
else:
    selected_images = np.array(misclassified_images)
    true_labels = np.array(misclassified_true_labels)
    wrong_preds = np.array(misclassified_preds)


# 获取预测值
raw_output = model(misclassified_images, training=False)
if isinstance(raw_output, dict):
    predictions = list(raw_output.values())[0].numpy()
else:
    predictions = raw_output.numpy()
predicted_labels = np.argmax(predictions, axis=1)

# 选取分类错误的样本
misclassified_indices = np.where(predicted_labels != misclassified_true_labels)[0]
sample_indices = random.sample(list(misclassified_indices), min(6, len(misclassified_indices)))
selected_images = misclassified_images[sample_indices]
true_labels = misclassified_true_labels[sample_indices]
wrong_preds = predicted_labels[sample_indices]

# Grad-CAM 函数
def make_gradcam_heatmap(img_array, model, class_index, last_conv_layer_name="resnet50/conv5_block3_out"):
    grad_model = tf.keras.models.Model(
        [model.input],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# 可视化
plt.figure(figsize=(18, 9))
for i in range(len(sample_indices)):
    img = selected_images[i]
    img_array = np.expand_dims(img, axis=0)
    true_label = class_names[true_labels[i]]
    pred_label = class_names[wrong_preds[i]]

    # 生成 heatmap
    heatmap = make_gradcam_heatmap(img_array, model, wrong_preds[i])

    # 叠加 heatmap
    heatmap_resized = cv2.resize(heatmap, (img_size[1], img_size[0]))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img.astype(np.uint8), 0.6, heatmap_colored, 0.4, 0)

    # 绘图
    ax = plt.subplot(2, 3, i + 1)
    ax.imshow(superimposed_img)
    ax.set_title(f"真实: {true_label}\n预测: {pred_label}", fontsize=12)
    ax.axis("off")

plt.tight_layout()
plt.savefig("gradcam_misclassified.png")
plt.show()

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from tensorflow.keras.layers import BatchNormalization  # type: ignore

# 设置中文字体
plt.rcParams["font.family"] = "Microsoft YaHei"
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def plot_confusion_matrix(y_true, y_pred, classes, normalize=True, title='混淆矩阵', cmap=plt.cm.Blues):
    """
    绘制混淆矩阵
    
    参数:
    y_true: 真实标签
    y_pred: 预测标签
    classes: 类别名称列表
    normalize: 是否将混淆矩阵归一化
    title: 图表标题
    cmap: 颜色映射
    """
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    
    # 归一化
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f%%'
        cm_title = f'{title} (百分比)'
    else:
        fmt = 'd'
        cm_title = f'{title} (数量)'
    
    # 创建图形和轴
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    
    # 添加颜色条
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('比例', rotation=-90, va="bottom", fontsize=12)
    
    # 设置刻度和标签
    ax.set(xticks=np.arange(cm.shape[1]),yticks=np.arange(cm.shape[0]),xticklabels=classes, yticklabels=classes,title=cm_title,ylabel='真实类别',xlabel='预测类别')
    
    # 旋转x轴标签
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",rotation_mode="anchor", fontsize=12)
    plt.setp(ax.get_yticklabels(), fontsize=12)
    
    # 在每个单元格中添加数值
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if normalize:
                value = format(cm[i, j] * 100, '.2f')
                text = f"{value}%"
            else:
                text = format(cm[i, j], 'd')
            
            ax.text(j, i, text,
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=14)
    
    # 添加网格线
    ax.grid(False)
    fig.tight_layout()
    return fig, ax

def load_test_data(data_dir, target_size=(224, 224), batch_size=32):
    """加载测试数据"""
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255
    )
    
    test_generator = test_datagen.flow_from_directory(
        data_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False  # 确保标签顺序正确
    )
    
    return test_generator

def main():
    # 路径配置
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    PEST_DIR = os.path.join(CURRENT_DIR, "pest_detection")

    SAVED_MODEL_PATH = os.path.join(CURRENT_DIR, "Model_file", "exported_model")
    LABEL_MAP_PATH = os.path.join(PEST_DIR, "models", "trained_model", "label_mapping.json")
    TEST_DATA_PATH = os.path.join(CURRENT_DIR, "data", "color_tom")

    # 打印调试信息
    print(f"SavedModel 路径: {SAVED_MODEL_PATH}")
    print(f"文件是否存在: {os.path.exists(SAVED_MODEL_PATH)}")

    # 加载模型
    print(f"正在加载 SavedModel: {SAVED_MODEL_PATH}")
    model = tf.keras.Sequential([
        tf.keras.layers.TFSMLayer(SAVED_MODEL_PATH, call_endpoint='serving_default')
    ])

    # 加载标签映射
    print(f"加载标签映射: {LABEL_MAP_PATH}")
    with open(LABEL_MAP_PATH) as f:
        label_mapping = json.load(f)

    index_to_class = {int(k): v for k, v in label_mapping.items()}
    class_names = [index_to_class[i] for i in sorted(index_to_class.keys())]

    # 加载测试数据
    print(f"加载测试数据: {TEST_DATA_PATH}")
    test_generator = load_test_data(TEST_DATA_PATH)

    true_labels = test_generator.classes

    # 获取模型预测
    print("正在进行预测...")
    predictions = model.predict(test_generator)
    
    # 查看模型输出键名
    print("模型输出 keys:", predictions.keys())

    # 动态获取输出层名称并提取 logits
    output_layer_name = next(iter(predictions.keys()))
    logits = predictions[output_layer_name]

    # 打印 logits 示例验证是否正常
    print("logits shape:", logits.shape)
    print("前5个预测 logits:", logits[:5])

    # 如果输出是 logits，转换为概率
    if logits.max() > 1:  # 简单判断是否是 logits
        probabilities = tf.nn.softmax(logits, axis=-1).numpy()
    else:
        probabilities = logits

    # 获取预测类别
    predicted_labels = np.argmax(probabilities, axis=1)

    # 打印前几个预测概率用于调试
    print("前5个预测概率:", probabilities[:5])

    # 获取测试集类别顺序作为 class_names
    class_names = sorted(test_generator.class_indices.keys())

    # 转换为类别预测
    predicted_labels = np.argmax(logits, axis=1)

    # 绘制混淆矩阵
    print("生成混淆矩阵...")
    plot_confusion_matrix(test_generator.classes, predicted_labels, class_names)

    # 保存图像
    output_path = os.path.join(CURRENT_DIR, "confusion_matrix.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"混淆矩阵已保存至: {output_path}")

    # 显示图表
    plt.show()

if __name__ == "__main__":
    main()
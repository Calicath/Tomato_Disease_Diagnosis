import os
import tensorflow as tf

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SAVED_MODEL_PATH = os.path.join(PROJECT_ROOT, "Model_file", "exported_model")

# 用 tf.keras.models.load_model 直接加载模型（不是用 TFSMLayer）
real_model = tf.keras.models.load_model(os.path.join(PROJECT_ROOT, "Model_file", "leaf_disease_resnet50.h5"))

# 打印所有层的名称和类型
for i, layer in enumerate(real_model.layers):
    print(f"{i}: {layer.name} ({layer.__class__.__name__})")

# 自动找出最后一个卷积层
conv_layers = [layer for layer in real_model.layers if isinstance(layer, tf.keras.layers.Conv2D)]
if conv_layers:
    last_conv_layer = conv_layers[-1]
    print("最后一个卷积层名：", last_conv_layer.name)
else:
    print("未找到卷积层。")

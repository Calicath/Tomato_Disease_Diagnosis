import tensorflow as tf
import os

# 设置正确的 SavedModel 路径
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SAVED_MODEL_PATH = os.path.join(PROJECT_ROOT, "Model_file", "exported_model")

print(f"正在尝试加载 SavedModel（路径: {SAVED_MODEL_PATH}）")

# 检查文件是否存在
pb_file = os.path.join(SAVED_MODEL_PATH, 'saved_model.pb')
variables_dir = os.path.join(SAVED_MODEL_PATH, 'variables')

print(f"saved_model.pb 存在？{os.path.exists(pb_file)}")
print(f"variables/ 文件夹存在？{os.path.exists(variables_dir)}")

try:
    model = tf.keras.Sequential([
        tf.keras.layers.TFSMLayer(SAVED_MODEL_PATH, call_endpoint='serving_default')
    ])
    print("✅ 模型加载成功！")

    # 打印签名信息（用于调试）
    loaded = tf.saved_model.load(SAVED_MODEL_PATH)
    print("🔎 模型签名信息:")
    print(loaded.signatures)

except Exception as e:
    print("❌ 加载失败:", str(e))
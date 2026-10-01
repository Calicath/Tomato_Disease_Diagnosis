import os
import cv2
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 修正数据集名称映射
CLASS_NAME_CORRECTIONS = {
    "Tomato_Bacterial_SPOTAL": "Tomato_Bacterial_spot",
    "Tomato_Early_light": "Tomato_Early_blight",
    "Tomato_Late_ blight": "Tomato_Late_blight",
    "Tomato_Spider_mites Two-spottedSpider_mite": "Tomato_Spider_mites_two_spotted",
    "Tomato_ Tomato_mosaic virus": "Tomato_mosaic_virus",
    "Tomato_ Tomato_Yellow_Leaf_Curl_Virus": "Tomato_Yellow_Leaf_Curl_Virus"
}


def load_dataset(base_dir):
    images = []
    labels = []

    for orig_name in os.listdir(base_dir):
        # 修正类名
        class_name = CLASS_NAME_CORRECTIONS.get(orig_name, orig_name.replace("_", " ").title().replace(" ", "_"))
        class_dir = os.path.join(base_dir, orig_name)

        if os.path.isdir(class_dir):
            for filename in os.listdir(class_dir):
                img_path = os.path.join(class_dir, filename)
                try:
                    img = cv2.imread(img_path)
                    if img is not None:
                        img = cv2.resize(img, (224, 224))
                        images.append(img)
                        labels.append(class_name)
                except Exception as e:
                    print(f"Error loading {img_path}: {str(e)}")

    return np.array(images), np.array(labels)


def main():
    # 配置参数
    DATASET_DIR = "data/raw_data"
    TARGET_SIZE = (224, 224)

    # 加载并修正数据
    images, labels = load_dataset(DATASET_DIR)

    # 创建标签编码器
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)

    # 平衡采样
    rus = RandomUnderSampler(sampling_strategy={k: 1000 for k in np.unique(labels_encoded)})
    X_res, y_res = rus.fit_resample(images.reshape(len(images), -1), labels_encoded)

    # 数据增强
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True
    )

    # 生成增强数据
    augmented_images = []
    for img_array in X_res.reshape(-1, *TARGET_SIZE, 3):
        aug_iter = datagen.flow(img_array[np.newaxis, ...], batch_size=1)
        for _ in range(2):
            augmented_images.append(next(aug_iter)[0])

    # 生成标签映射
    label_mapping = {str(i): cls for i, cls in enumerate(le.classes_)}
    with open("models/trained_model/label_mapping.json", "w") as f:
        json.dump(label_mapping, f, indent=2)


if __name__ == "__main__":
    main()
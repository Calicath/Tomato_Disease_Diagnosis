import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam


# 构建模型
def build_model(num_classes):
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


# 训练配置
BATCH_SIZE = 32
EPOCHS = 50

if __name__ == "__main__":
    # 这里需要根据实际数据路径调整
    train_dataset = ...
    val_dataset = ...

    # 创建模型
    model = build_model(num_classes=10)

    # 训练回调
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            'models/trained_model/best_model.keras',
            save_best_only=True,
            monitor='val_accuracy'
        )
    ]

    # 开始训练
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        callbacks=callbacks
    )
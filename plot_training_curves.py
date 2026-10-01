# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

epochs = range(1, 21)

train_loss = [
    0.2610, 0.2336, 0.2057, 0.1836, 0.1683, 0.1539, 0.1405, 0.1351,
    0.1308, 0.1152, 0.1165, 0.1062, 0.1041, 0.1038, 0.0988, 0.1010,
    0.1109, 0.0891, 0.0867, 0.0864
]

train_acc = [
    0.9100, 0.9193, 0.9295, 0.9347, 0.9417, 0.9439, 0.9499, 0.9514,
    0.9533, 0.9588, 0.9582, 0.9625, 0.9628, 0.9624, 0.9654, 0.9641,
    0.9615, 0.9685, 0.9694, 0.9685
]

val_loss = [
    0.1503, 0.1353, 0.1505, 0.1089, 0.1102, 0.0976, 0.1070, 0.0893,
    0.0842, 0.0879, 0.0899, 0.0934, 0.0762, 0.0871, 0.0805, 0.0810,
    0.0721, 0.0872, 0.0814, 0.0667
]

val_acc = [
    0.9490, 0.9527, 0.9447, 0.9603, 0.9627, 0.9640, 0.9607, 0.9688,
    0.9707, 0.9690, 0.9693, 0.9663, 0.9750, 0.9717, 0.9738, 0.9718,
    0.9722, 0.9710, 0.9712, 0.9760
]

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(epochs, train_acc, 'b-o', label='训练准确率', linewidth=2, markersize=6)
plt.plot(epochs, val_acc, 'r-s', label='验证准确率', linewidth=2, markersize=6)
plt.title('训练和验证准确率', fontsize=14)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.xticks(epochs)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(fontsize=12)

plt.subplot(1, 2, 2)
plt.plot(epochs, train_loss, 'b-o', label='训练损失', linewidth=2, markersize=6)
plt.plot(epochs, val_loss, 'r-s', label='验证损失', linewidth=2, markersize=6)
plt.title('训练和验证损失', fontsize=14)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.xticks(epochs)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(fontsize=12)

plt.tight_layout()

plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')

plt.show()
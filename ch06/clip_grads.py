# coding: utf-8
import numpy as np

# dW1, dW2 模拟两组梯度，每个元素随机在 0~10
# grads 是一个梯度列表（通常对应模型的所有参数）
# max_norm 是梯度的最大允许范数（阈值）
dW1 = np.random.rand(3, 3) * 10
dW2 = np.random.rand(3, 3) * 10
grads = [dW1, dW2]
max_norm = 5.0


def clip_grads(grads, max_norm):
    total_norm = 0
    for grad in grads:
        total_norm += np.sum(grad**2)
    total_norm = np.sqrt(total_norm)

    # rate：缩放系数 如果当前范数比 max_norm 大（rate < 1），就整体按比例缩小
    rate = max_norm / (total_norm + 1e-6)
    if rate < 1:
        for grad in grads:
            grad *= rate


print("before:", dW1.flatten())
clip_grads(grads, max_norm)
print("after:", dW1.flatten())

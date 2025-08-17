# coding: utf-8
import sys

sys.path.append("..")  # 为了引入父目录的文件而进行的设定
import numpy as np
from common.optimizer import SGD
from dataset import spiral
import matplotlib.pyplot as plt
from two_layer_net import TwoLayerNet


# 设定超参数
max_epoch = 300  # 总共训练300轮
batch_size = 30  # 每次使用30个样本训练（小批量）
hidden_size = 10  # 隐藏层神经元个数
learning_rate = 1.0  # 学习率

# x.shape = (300, 2)：每个样本是一个二维点
# t.shape = (300, 3)：目标标签为 one-hot 编码（3类）
x, t = spiral.load_data()  # 导入 spiral 数据集，是一个 2D 的三分类非线性数据集。

# TwoLayerNet：
# 输入层：2个神经元
# 隐藏层：10个神经元
# 输出层：3个神经元（表示3类）
model = TwoLayerNet(input_size=2, hidden_size=hidden_size, output_size=3)

optimizer = SGD(lr=learning_rate)  # SGD：使用随机梯度下降优化网络参数


# 学习用的变量
data_size = len(x)  # 总共300个样本
max_iters = data_size // batch_size  # 每个epoch中执行的迭代次数
total_loss = 0
loss_count = 0
loss_list = []  # 用于记录每次输出的平均loss

# * 开始训练
for epoch in range(max_epoch):
    # 打乱数据，每个 epoch 开始都打乱数据，避免模型记住样本顺序。
    idx = np.random.permutation(data_size)
    x = x[idx]
    t = t[idx]

    for iters in range(max_iters):
        # 从打乱后的数据中，依次取出小批量进行训练
        batch_x = x[iters * batch_size : (iters + 1) * batch_size]
        batch_t = t[iters * batch_size : (iters + 1) * batch_size]

        # 计算梯度，更新参数
        loss = model.forward(batch_x, batch_t)
        model.backward()
        optimizer.update(model.params, model.grads)

        total_loss += loss
        loss_count += 1

        # 定期输出学习过程，每10次 iteration 输出一次平均损失并记录
        if (iters + 1) % 10 == 0:
            avg_loss = total_loss / loss_count
            print(
                "| epoch %d |  iter %d / %d | loss %.2f"
                % (epoch + 1, iters + 1, max_iters, avg_loss)
            )
            loss_list.append(avg_loss)
            total_loss, loss_count = 0, 0


# 绘制学习结果
plt.plot(np.arange(len(loss_list)), loss_list, label="train")
plt.xlabel("iterations (x10)")
plt.ylabel("loss")
plt.show()

# 绘制决策边界
h = 0.001
x_min, x_max = x[:, 0].min() - 0.1, x[:, 0].max() + 0.1
y_min, y_max = x[:, 1].min() - 0.1, x[:, 1].max() + 0.1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
X = np.c_[xx.ravel(), yy.ravel()]
score = model.predict(X)
predict_cls = np.argmax(score, axis=1)
Z = predict_cls.reshape(xx.shape)
plt.contourf(xx, yy, Z)
plt.axis("off")

# 绘制数据点
x, t = spiral.load_data()
N = 100
CLS_NUM = 3
markers = ["o", "x", "^"]
for i in range(CLS_NUM):
    plt.scatter(
        x[i * N : (i + 1) * N, 0], x[i * N : (i + 1) * N, 1], s=40, marker=markers[i]
    )
plt.show()

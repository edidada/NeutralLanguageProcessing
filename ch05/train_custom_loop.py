# coding: utf-8
import sys

sys.path.append(".")
import matplotlib.pyplot as plt
import numpy as np
from common.optimizer import SGD
from dataset import ptb
from simple_rnnlm import SimpleRnnlm


# * 设定超参数
# batch_size：一次训练多少条句子片段
# hidden_size：RNN 隐藏状态维度
# wordvec_size：词向量维度
# time_size：Truncated BPTT 的时间步数（每次反向传播只展开这么多时间步）
# lr：学习率
# max_epoch：训练多少轮
batch_size = 10
wordvec_size = 100
hidden_size = 100
time_size = 5  # Truncated BPTT的时间跨度大小
lr = 0.1
max_epoch = 100

# * 读入训练数据（缩小了数据集）
# 从 PTB 数据集中读入训练集（整数 ID 序列）
# corpus_size = 1000：这里只用前 1000 个词，减少训练时间
# vocab_size：词表大小 = 最大单词 ID + 1
corpus, word_to_id, id_to_word = ptb.load_data("train")
corpus_size = 1000
corpus = corpus[:corpus_size]
vocab_size = int(max(corpus) + 1)

xs = corpus[:-1]  # 输入
ts = corpus[1:]  # 输出（监督标签）
data_size = len(xs)
print("corpus size: %d, vocabulary size: %d" % (corpus_size, vocab_size))

# * 学习用的参数
max_iters = data_size // (batch_size * time_size)  # 一个 epoch 里有多少个 mini-batch
time_idx = 0
total_loss = 0
loss_count = 0
ppl_list = []

# 生成模型
model = SimpleRnnlm(vocab_size, wordvec_size, hidden_size)
optimizer = SGD(lr)

# 计算读入mini-batch的各笔样本数据的开始位置
jump = (corpus_size - 1) // batch_size  # 每个样本在语料中的起始间隔
offsets = [i * jump for i in range(batch_size)]  # 存储每个 batch 里样本的起始位置

for epoch in range(max_epoch):
    for iter in range(max_iters):
        # 获取mini-batch
        # batch_x[i, t]：第 i 个样本，第 t 个时间步的输入单词 ID
        # batch_t[i, t]：对应的标签
        # time_idx 每次加 1，确保下一批数据向前滑动
        batch_x = np.empty((batch_size, time_size), dtype="i")
        batch_t = np.empty((batch_size, time_size), dtype="i")
        for t in range(time_size):
            for i, offset in enumerate(offsets):
                batch_x[i, t] = xs[(offset + time_idx) % data_size]
                batch_t[i, t] = ts[(offset + time_idx) % data_size]
            time_idx += 1

        # 前向 + 反向 + 更新   计算梯度，更新参数
        loss = model.forward(batch_x, batch_t)
        model.backward()
        optimizer.update(model.params, model.grads)
        total_loss += loss
        loss_count += 1

    # 困惑度（Perplexity）：语言模型评估指标  各个epoch的困惑度评价
    ppl = np.exp(total_loss / loss_count)
    print("| epoch: %d | perplexity: %.4f" % (epoch + 1, ppl))
    ppl_list.append(float(ppl))
    total_loss, loss_count = 0, 0

# 绘制图形
x = np.arange(len(ppl_list))
plt.plot(x, ppl_list, label="train")
plt.xlabel("epochs")
plt.ylabel("perplexity")
plt.show()

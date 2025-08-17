# coding: utf-8
import sys

sys.path.append("..")
import numpy as np
from common.time_layers import *


class SimpleRnnlm:

    # vocab_size (V)：词表大小
    # wordvec_size (D)：词向量维度
    # hidden_size (H)：RNN 隐藏状态维度
    # rn 是 np.random.randn 的简写
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V, D, H = vocab_size, wordvec_size, hidden_size
        rn = np.random.randn

        # * 初始化权重
        # 词向量矩阵 V x D，每一行对应一个词的向量 除以 100 是为了让初始化值比较小（防止初期梯度爆炸）
        embed_W = (rn(V, D) / 100).astype("f")
        # 输入到隐藏层的权重，D x H 除以 √D 是 Xavier 初始化方式之一
        rnn_Wx = (rn(D, H) / np.sqrt(D)).astype("f")
        # 隐藏状态到隐藏状态的权重（RNN 循环部分），H x H
        rnn_Wh = (rn(H, H) / np.sqrt(H)).astype("f")
        # RNN 偏置
        rnn_b = np.zeros(H).astype("f")
        # 从隐藏状态到词表大小的全连接层权重、偏置 输出维度是 V，因为要预测下一个词
        affine_W = (rn(H, V) / np.sqrt(H)).astype("f")
        affine_b = np.zeros(V).astype("f")

        # * 生成层
        # imeEmbedding: 输入是单词 ID（整数），输出是对应的词向量序列。
        # TimeRNN: 接收词向量序列，按时间步运行 RNN（stateful=True 表示状态在不同批次之间可以保留）。
        # TimeAffine: 把 RNN 输出的隐藏状态映射到词表空间，用于预测下一个词。
        self.layers = [
            TimeEmbedding(embed_W),
            TimeRNN(rnn_Wx, rnn_Wh, rnn_b, stateful=True),
            TimeAffine(affine_W, affine_b),
        ]
        self.loss_layer = TimeSoftmaxWithLoss()
        self.rnn_layer = self.layers[1]

        # 将所有的权重和梯度整理到列表中
        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

    # 单词ID序列 xs → TimeEmbedding → TimeRNN → TimeAffine → TimeSoftmaxWithLoss → loss
    # xs：形状 (batch_size, time_steps)     ts：目标单词ID（标签）
    def forward(self, xs, ts):
        for layer in self.layers:
            xs = layer.forward(xs)
        loss = self.loss_layer.forward(xs, ts)
        return loss

    # loss_layer → TimeAffine → TimeRNN → TimeEmbedding
    # dout=1 表示从损失开始反向传播  反向遍历 self.layers，逐步更新梯度
    def backward(self, dout=1):
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    # 当 stateful=True 时，RNN 会记住上一次批次的隐藏状态
    # 调用这个方法可以清空状态（例如处理新句子时）
    def reset_state(self):
        self.rnn_layer.reset_state()

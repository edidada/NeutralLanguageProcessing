# coding: utf-8
import sys

sys.path.append("..")
import numpy as np
from common.layers import MatMul, SoftmaxWithLoss


class SimpleCBOW:
    def __init__(self, vocab_size, hidden_size):
        # vocab_size：词汇表大小（每个词都被映射为一个 one-hot 向量）
        # hidden_size：词向量的维度（即中间的隐含层维度）
        V, H = vocab_size, hidden_size

        # 初始化权重
        # W_in：输入层的权重矩阵（词向量矩阵），形状 [V, H]
        # W_out：输出层的权重矩阵，形状 [H, V]
        W_in = 0.01 * np.random.randn(V, H).astype("float32")
        W_out = 0.01 * np.random.randn(H, V).astype("float32")

        # 构建计算图的各层
        self.in_layer0 = MatMul(W_in)
        self.in_layer1 = MatMul(W_in)
        self.out_layer = MatMul(W_out)
        self.loss_layer = SoftmaxWithLoss()

        # 将所有的权重和梯度整理到列表中
        layers = [self.in_layer0, self.in_layer1, self.out_layer]
        self.params, self.grads = [], []
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads

        # 将单词的分布式表示设置为成员变量
        self.word_vecs = W_in

    def forward(self, contexts, target):
        h0 = self.in_layer0.forward(contexts[:, 0])
        h1 = self.in_layer1.forward(contexts[:, 1])
        h = (h0 + h1) * 0.5
        score = self.out_layer.forward(h)
        loss = self.loss_layer.forward(score, target)
        return loss

    def backward(self, dout=1):
        ds = self.loss_layer.backward(dout)
        da = self.out_layer.backward(ds)
        da *= 0.5
        self.in_layer1.backward(da)
        self.in_layer0.backward(da)
        return None

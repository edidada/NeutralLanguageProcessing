# coding: utf-8
from common.np import *  # import numpy as np (or import cupy as np)
from common.layers import *
from common.functions import softmax, sigmoid


class RNN:
    # Wx: 输入到隐藏状态的权重矩阵，形状为 (输入维度, 隐藏层维度)
    # Wh: 隐藏状态到隐藏状态的权重矩阵，形状为 (隐藏层维度, 隐藏层维度)
    # b: 偏置项，形状为 (隐藏层维度,)
    # self.params: 储存参数，供外部优化器使用
    # self.grads: 对应参数的梯度（在 backward 中计算）
    # self.cache: 缓存前向传播中中间变量，用于反向传播
    def __init__(self, Wx, Wh, b):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.cache = None

    # x: 当前时刻的输入，形状为 (batch_size, input_dim)
    # h_prev: 上一时刻的隐藏状态，形状为 (batch_size, hidden_dim)
    def forward(self, x, h_prev):
        Wx, Wh, b = self.params
        t = np.dot(h_prev, Wh) + np.dot(x, Wx) + b
        h_next = np.tanh(t)

        # 缓存 (x, h_prev, h_next) 以备反向传播使用
        self.cache = (x, h_prev, h_next)
        return h_next

    # dh_next: 来自下一时间步隐藏状态的梯度，形状为 (batch_size, hidden_dim)
    def backward(self, dh_next):
        Wx, Wh, b = self.params
        x, h_prev, h_next = self.cache

        # h_next = tanh(t)，其中 t = x @ Wx + h_prev @ Wh + b
        # 反向传播到 tanh： tanh 的导数是 1 − tanh(t) ^ 2
        # 结合链式法则乘以上层梯度 dh_next
        dt = dh_next * (1 - h_next**2)

        # 偏置 b 是加在每个样本上的向量。所以要在 batch 维度上对 dt 求和
        db = np.sum(dt, axis=0)  # 对偏置求和
        dWh = np.dot(h_prev.T, dt)  # 用于更新 Wh
        dh_prev = np.dot(dt, Wh.T)  #  向前传递的梯度，用于上一个时间步
        dWx = np.dot(x.T, dt)  #  用于更新 Wx
        dx = np.dot(dt, Wx.T)  # 输入 x 的梯度（供嵌套网络或嵌入层使用）

        # 存储梯度
        self.grads[0][...] = dWx
        self.grads[1][...] = dWh
        self.grads[2][...] = db

        return dx, dh_prev


class TimeRNN:
    # Wx：输入权重矩阵（D → H），形状 (D, H)
    # Wh：隐藏层权重矩阵（H → H），形状 (H, H)
    # b：偏置项，形状 (H,)
    # params：用于存储参数
    # grads：用于存储反向传播得到的参数梯度
    # h：保存当前的隐藏状态
    # stateful：是否“有记忆”，即前一个 batch 的 hidden state 是否在下一个 batch 中继续用（处理长序列时分段处理）
    def __init__(self, Wx, Wh, b, stateful=False):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.layers = None

        self.h, self.dh = None, None
        self.stateful = stateful

    # 输入 xs：形状为 (N, T, D)，N 是 batch size，T 是时间步数，D 是输入维度
    def forward(self, xs):
        Wx, Wh, b = self.params
        N, T, D = xs.shape
        D, H = Wx.shape

        self.layers = []
        hs = np.empty((N, T, H), dtype="f")  # 初始化输出序列 hs：形状为 (N, T, H)

        # 如果 stateful=False，则每次调用 forward 都会重置隐藏状态 h 为零向量
        # 如果 True，则保留上一次 forward 的隐藏状态用于本次
        if not self.stateful or self.h is None:
            self.h = np.zeros((N, H), dtype="f")

        # 对每个时间步 t：实例化一个 RNN 单元
        # 使用上一个时间步的隐藏状态 self.h 和当前输入 xs[:, t, :] 做前向传播
        # 得到当前的隐藏状态，保存到 hs 中
        # 每个时间步的 RNN 层被保存到 self.layers 中，以便反向传播用
        for t in range(T):
            layer = RNN(*self.params)
            self.h = layer.forward(xs[:, t, :], self.h)
            hs[:, t, :] = self.h
            self.layers.append(layer)

        return hs

    # dhs 是对所有时间步输出 hs 的损失梯度，形状为 (N, T, H)
    def backward(self, dhs):
        Wx, Wh, b = self.params
        N, T, H = dhs.shape
        D, H = Wx.shape

        dxs = np.empty((N, T, D), dtype="f")
        dh = 0
        grads = [0, 0, 0]
        for t in reversed(range(T)):
            layer = self.layers[t]
            dx, dh = layer.backward(dhs[:, t, :] + dh)
            dxs[:, t, :] = dx

            for i, grad in enumerate(layer.grads):
                grads[i] += grad

        for i, grad in enumerate(grads):
            self.grads[i][...] = grad
        self.dh = dh

        return dxs

    def set_state(self, h):
        self.h = h

    def reset_state(self):
        self.h = None


class LSTM:
    def __init__(self, Wx, Wh, b):
        """

        Parameters
        ----------
        Wx: 输入`x`用的权重参数（整合了4个权重）
        Wh: 隐藏状态`h`用的权重参数（整合了4个权重）
        b: 偏置（整合了4个偏置）
        """
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.cache = None

    def forward(self, x, h_prev, c_prev):
        Wx, Wh, b = self.params
        N, H = h_prev.shape
        # 将 4 个门的计算合并成一次矩阵运算，提高效率 Wx 和 Wh 的列数是 4H，因为包含了 4 组权重
        A = np.dot(x, Wx) + np.dot(h_prev, Wh) + b

        f = sigmoid(A[:, :H])  # 遗忘门  控制丢弃多少旧信息
        g = np.tanh(A[:, H : 2 * H])  # 候选状态  新信息的候选值
        i = sigmoid(A[:, 2 * H : 3 * H])  # 输入门  控制写入多少新信息
        o = sigmoid(A[:, 3 * H :])  # 输出门  控制输出多少细胞状态信息

        f = sigmoid(f)
        g = np.tanh(g)
        i = sigmoid(i)
        o = sigmoid(o)

        # 遗忘门决定保留多少旧的 c_prev
        # 输入门 & 候选状态共同决定写入多少新信息
        # 输出门控制最终暴露多少状态给 h_next
        c_next = f * c_prev + g * i  # 更新细胞状态
        h_next = o * np.tanh(c_next)  # 计算隐藏状

        self.cache = (x, h_prev, c_prev, i, f, g, o, c_next)
        return h_next, c_next

    def backward(self, dh_next, dc_next):
        Wx, Wh, b = self.params
        x, h_prev, c_prev, i, f, g, o, c_next = self.cache

        tanh_c_next = np.tanh(c_next)

        # dc_next：来自未来时刻细胞状态的梯度
        # (dh_next * o) * (1 - tanh^2)：来自未来隐藏状态的梯度反传到 c_next
        ds = dc_next + (dh_next * o) * (1 - tanh_c_next**2)

        # 细胞状态的梯度通过遗忘门传给上一个时间步
        dc_prev = ds * f

        # 分别求出门的梯度（还没经过激活函数的梯度）
        di = ds * g
        df = ds * c_prev
        do = dh_next * tanh_c_next
        dg = ds * i

        # 反向通过门的激活函数
        di *= i * (1 - i)  # sigmoid' = i*(1-i)
        df *= f * (1 - f)
        do *= o * (1 - o)
        dg *= 1 - g**2  # tanh' = 1 - g^2

        # 再把 4 个门的梯度合并回一个矩阵，方便矩阵乘法求权重梯度
        dA = np.hstack((df, dg, di, do))

        # 对权重和偏置求梯度
        dWh = np.dot(h_prev.T, dA)
        dWx = np.dot(x.T, dA)
        db = dA.sum(axis=0)

        # 反传到上一层输入和上一时刻隐藏状态
        self.grads[0][...] = dWx
        self.grads[1][...] = dWh
        self.grads[2][...] = db

        dx = np.dot(dA, Wx.T)
        dh_prev = np.dot(dA, Wh.T)

        return dx, dh_prev, dc_prev


class TimeLSTM:
    # Wx: 输入到各门的权重 (D × 4H)
    # Wh: 隐藏状态到各门的权重 (H × 4H)
    # b: 偏置 (4H)
    # stateful: 控制是否跨批次保留隐藏状态 (h) 和细胞状态 (c)。
    def __init__(self, Wx, Wh, b, stateful=False):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.layers = None

        self.h, self.c = None, None
        self.dh = None
        self.stateful = stateful

    def forward(self, xs):
        Wx, Wh, b = self.params
        N, T, D = xs.shape
        H = Wh.shape[0]

        self.layers = []
        hs = np.empty((N, T, H), dtype="f")

        if not self.stateful or self.h is None:
            self.h = np.zeros((N, H), dtype="f")
        if not self.stateful or self.c is None:
            self.c = np.zeros((N, H), dtype="f")

        for t in range(T):
            layer = LSTM(*self.params)  # 一个时间步的 LSTM 单元
            self.h, self.c = layer.forward(
                xs[:, t, :],  # 第 t 个时间步的输入
                self.h,  # 上一时间步的隐藏状态
                self.c,  # 上一时间步的细胞状态
            )
            hs[:, t, :] = self.h
            self.layers.append(layer)  # 存起来，反向传播要用

        return hs

    def backward(self, dhs):
        Wx, Wh, b = self.params
        N, T, H = dhs.shape
        D = Wx.shape[0]

        dxs = np.empty((N, T, D), dtype="f")
        dh, dc = 0, 0

        grads = [0, 0, 0]
        for t in reversed(range(T)):
            layer = self.layers[t]
            # dx: 输入的梯度
            # dh: 隐藏状态的梯度（传回前一时间步）
            # dc: 细胞状态的梯度（传回前一时间步）
            dx, dh, dc = layer.backward(dhs[:, t, :] + dh, dc)
            dxs[:, t, :] = dx
            for i, grad in enumerate(layer.grads):
                grads[i] += grad

        for i, grad in enumerate(grads):
            self.grads[i][...] = grad
        self.dh = dh
        return dxs

    def set_state(self, h, c=None):
        self.h, self.c = h, c

    def reset_state(self):
        self.h, self.c = None, None


# 把一个批次的序列中每个单词 ID 转换成对应的词向量（embedding）
class TimeEmbedding:
    # W：词向量矩阵，形状 (V, D)
    # V = 词汇表大小（vocabulary size）
    # D = 词向量维度（embedding dim）
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.layers = None
        self.W = W

    def forward(self, xs):
        N, T = xs.shape
        V, D = self.W.shape

        out = np.empty((N, T, D), dtype="f")
        self.layers = []

        for t in range(T):
            layer = Embedding(self.W)
            out[:, t, :] = layer.forward(xs[:, t])
            self.layers.append(layer)

        return out

    def backward(self, dout):
        N, T, D = dout.shape

        grad = 0
        for t in range(T):
            layer = self.layers[t]
            layer.backward(dout[:, t, :])
            grad += layer.grads[0]

        self.grads[0][...] = grad
        return None


class TimeAffine:
    def __init__(self, W, b):
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.x = None

    def forward(self, x):
        N, T, D = x.shape
        W, b = self.params

        rx = x.reshape(N * T, -1)
        out = np.dot(rx, W) + b
        self.x = x
        return out.reshape(N, T, -1)

    def backward(self, dout):
        x = self.x
        N, T, D = x.shape
        W, b = self.params

        dout = dout.reshape(N * T, -1)
        rx = x.reshape(N * T, -1)

        db = np.sum(dout, axis=0)
        dW = np.dot(rx.T, dout)
        dx = np.dot(dout, W.T)
        dx = dx.reshape(*x.shape)

        self.grads[0][...] = dW
        self.grads[1][...] = db

        return dx


class TimeSoftmaxWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.cache = None
        self.ignore_label = -1

    def forward(self, xs, ts):
        N, T, V = xs.shape

        if ts.ndim == 3:  # 在监督标签为one-hot向量的情况下
            ts = ts.argmax(axis=2)

        mask = ts != self.ignore_label

        # 按批次大小和时序大小进行整理（reshape）
        xs = xs.reshape(N * T, V)
        ts = ts.reshape(N * T)
        mask = mask.reshape(N * T)

        ys = softmax(xs)
        ls = np.log(ys[np.arange(N * T), ts])
        ls *= mask  # 与ignore_label相应的数据将损失设为0
        loss = -np.sum(ls)
        loss /= mask.sum()

        self.cache = (ts, ys, mask, (N, T, V))
        return loss

    def backward(self, dout=1):
        ts, ys, mask, (N, T, V) = self.cache

        dx = ys
        dx[np.arange(N * T), ts] -= 1
        dx *= dout
        dx /= mask.sum()
        dx *= mask[:, np.newaxis]  # 与ignore_label相应的数据将梯度设为0

        dx = dx.reshape((N, T, V))

        return dx


class TimeDropout:
    def __init__(self, dropout_ratio=0.5):
        self.params, self.grads = [], []
        self.dropout_ratio = dropout_ratio
        self.mask = None
        self.train_flg = True

    def forward(self, xs):
        if self.train_flg:
            flg = np.random.rand(*xs.shape) > self.dropout_ratio
            scale = 1 / (1.0 - self.dropout_ratio)
            self.mask = flg.astype(np.float32) * scale

            return xs * self.mask
        else:
            return xs

    def backward(self, dout):
        return dout * self.mask


class TimeBiLSTM:
    def __init__(self, Wx1, Wh1, b1, Wx2, Wh2, b2, stateful=False):
        self.forward_lstm = TimeLSTM(Wx1, Wh1, b1, stateful)
        self.backward_lstm = TimeLSTM(Wx2, Wh2, b2, stateful)
        self.params = self.forward_lstm.params + self.backward_lstm.params
        self.grads = self.forward_lstm.grads + self.backward_lstm.grads

    def forward(self, xs):
        o1 = self.forward_lstm.forward(xs)
        o2 = self.backward_lstm.forward(xs[:, ::-1])
        o2 = o2[:, ::-1]

        out = np.concatenate((o1, o2), axis=2)
        return out

    def backward(self, dhs):
        H = dhs.shape[2] // 2
        do1 = dhs[:, :, :H]
        do2 = dhs[:, :, H:]

        dxs1 = self.forward_lstm.backward(do1)
        do2 = do2[:, ::-1]
        dxs2 = self.backward_lstm.backward(do2)
        dxs2 = dxs2[:, ::-1]
        dxs = dxs1 + dxs2
        return dxs


# ====================================================================== #
# 如下所示的层是本书中没有说明的层的实现
# 或者为了使代码容易理解而牺牲了处理速度的层的实现。
#
# TimeSigmoidWithLoss: 用于时序数据的sigmoid损失层
# GRU: GRU层
# TimeGRU: 用于时序数据的GRU层
# BiTimeLSTM: 双向LSTM层
# Simple_TimeSoftmaxWithLoss：简单的TimeSoftmaxWithLoss层的实现
# Simple_TimeAffine: 简单的TimeAffine层的实现
# ====================================================================== #


class TimeSigmoidWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.xs_shape = None
        self.layers = None

    def forward(self, xs, ts):
        N, T = xs.shape
        self.xs_shape = xs.shape

        self.layers = []
        loss = 0

        for t in range(T):
            layer = SigmoidWithLoss()
            loss += layer.forward(xs[:, t], ts[:, t])
            self.layers.append(layer)

        return loss / T

    def backward(self, dout=1):
        N, T = self.xs_shape
        dxs = np.empty(self.xs_shape, dtype="f")

        dout *= 1 / T
        for t in range(T):
            layer = self.layers[t]
            dxs[:, t] = layer.backward(dout)

        return dxs


class GRU:
    def __init__(self, Wx, Wh, b):
        """

        Parameters
        ----------
        Wx: 用于输入`x`的权重参数（整合了3个权重）
        Wh: 用于隐藏状态`h` 的权重参数（整合了3个权重）
        b: 偏置（整合了3个偏置）
        """
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.cache = None

    def forward(self, x, h_prev):
        Wx, Wh, b = self.params
        H = Wh.shape[0]
        Wxz, Wxr, Wxh = Wx[:, :H], Wx[:, H : 2 * H], Wx[:, 2 * H :]
        Whz, Whr, Whh = Wh[:, :H], Wh[:, H : 2 * H], Wh[:, 2 * H :]
        bz, br, bh = b[:H], b[H : 2 * H], b[2 * H :]

        z = sigmoid(np.dot(x, Wxz) + np.dot(h_prev, Whz) + bz)
        r = sigmoid(np.dot(x, Wxr) + np.dot(h_prev, Whr) + br)
        h_hat = np.tanh(np.dot(x, Wxh) + np.dot(r * h_prev, Whh) + bh)
        h_next = (1 - z) * h_prev + z * h_hat

        self.cache = (x, h_prev, z, r, h_hat)

        return h_next

    def backward(self, dh_next):
        Wx, Wh, b = self.params
        H = Wh.shape[0]
        Wxz, Wxr, Wxh = Wx[:, :H], Wx[:, H : 2 * H], Wx[:, 2 * H :]
        Whz, Whr, Whh = Wh[:, :H], Wh[:, H : 2 * H], Wh[:, 2 * H :]
        x, h_prev, z, r, h_hat = self.cache

        dh_hat = dh_next * z
        dh_prev = dh_next * (1 - z)

        # tanh
        dt = dh_hat * (1 - h_hat**2)
        dbh = np.sum(dt, axis=0)
        dWhh = np.dot((r * h_prev).T, dt)
        dhr = np.dot(dt, Whh.T)
        dWxh = np.dot(x.T, dt)
        dx = np.dot(dt, Wxh.T)
        dh_prev += r * dhr

        # update gate(z)
        dz = dh_next * h_hat - dh_next * h_prev
        dt = dz * z * (1 - z)
        dbz = np.sum(dt, axis=0)
        dWhz = np.dot(h_prev.T, dt)
        dh_prev += np.dot(dt, Whz.T)
        dWxz = np.dot(x.T, dt)
        dx += np.dot(dt, Wxz.T)

        # rest gate(r)
        dr = dhr * h_prev
        dt = dr * r * (1 - r)
        dbr = np.sum(dt, axis=0)
        dWhr = np.dot(h_prev.T, dt)
        dh_prev += np.dot(dt, Whr.T)
        dWxr = np.dot(x.T, dt)
        dx += np.dot(dt, Wxr.T)

        self.dWx = np.hstack((dWxz, dWxr, dWxh))
        self.dWh = np.hstack((dWhz, dWhr, dWhh))
        self.db = np.hstack((dbz, dbr, dbh))

        self.grads[0][...] = self.dWx
        self.grads[1][...] = self.dWh
        self.grads[2][...] = self.db

        return dx, dh_prev


class TimeGRU:
    def __init__(self, Wx, Wh, b, stateful=False):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.layers = None
        self.h, self.dh = None, None
        self.stateful = stateful

    def forward(self, xs):
        Wx, Wh, b = self.params
        N, T, D = xs.shape
        H = Wh.shape[0]
        self.layers = []
        hs = np.empty((N, T, H), dtype="f")

        if not self.stateful or self.h is None:
            self.h = np.zeros((N, H), dtype="f")

        for t in range(T):
            layer = GRU(*self.params)
            self.h = layer.forward(xs[:, t, :], self.h)
            hs[:, t, :] = self.h
            self.layers.append(layer)
        return hs

    def backward(self, dhs):
        Wx, Wh, b = self.params
        N, T, H = dhs.shape
        D = Wx.shape[0]

        dxs = np.empty((N, T, D), dtype="f")

        dh = 0
        grads = [0, 0, 0]
        for t in reversed(range(T)):
            layer = self.layers[t]
            dx, dh = layer.backward(dhs[:, t, :] + dh)
            dxs[:, t, :] = dx

            for i, grad in enumerate(layer.grads):
                grads[i] += grad

        for i, grad in enumerate(grads):
            self.grads[i][...] = grad

        self.dh = dh
        return dxs

    def set_state(self, h):
        self.h = h

    def reset_state(self):
        self.h = None


class Simple_TimeSoftmaxWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.cache = None

    def forward(self, xs, ts):
        N, T, V = xs.shape
        layers = []
        loss = 0

        for t in range(T):
            layer = SoftmaxWithLoss()
            loss += layer.forward(xs[:, t, :], ts[:, t])
            layers.append(layer)
        loss /= T

        self.cache = (layers, xs)
        return loss

    def backward(self, dout=1):
        layers, xs = self.cache
        N, T, V = xs.shape
        dxs = np.empty(xs.shape, dtype="f")

        dout *= 1 / T
        for t in range(T):
            layer = layers[t]
            dxs[:, t, :] = layer.backward(dout)

        return dxs


class Simple_TimeAffine:
    def __init__(self, W, b):
        self.W, self.b = W, b
        self.dW, self.db = None, None
        self.layers = None

    def forward(self, xs):
        N, T, D = xs.shape
        D, M = self.W.shape

        self.layers = []
        out = np.empty((N, T, M), dtype="f")
        for t in range(T):
            layer = Affine(self.W, self.b)
            out[:, t, :] = layer.forward(xs[:, t, :])
            self.layers.append(layer)

        return out

    def backward(self, dout):
        N, T, M = dout.shape
        D, M = self.W.shape

        dxs = np.empty((N, T, D), dtype="f")
        self.dW, self.db = 0, 0
        for t in range(T):
            layer = self.layers[t]
            dxs[:, t, :] = layer.backward(dout[:, t, :])

            self.dW += layer.dW
            self.db += layer.db

        return dxs

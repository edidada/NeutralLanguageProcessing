# coding: utf-8
import sys

sys.path.append("..")
import numpy
import time
import matplotlib.pyplot as plt
from common.np import *  # import numpy as np
from common.util import clip_grads


class Trainer:
    def __init__(self, model, optimizer):
        self.model = model  # 模型对象，必须具备 forward() 和 backward() 方法
        self.optimizer = optimizer  # 优化器对象，必须有 update() 方法
        self.loss_list = []  # 存储训练过程中的损失值
        self.eval_interval = None  # 每隔多少次迭代进行一次评估
        self.current_epoch = 0  # 当前训练的轮数

    def fit(self, x, t, max_epoch=10, batch_size=32, max_grad=None, eval_interval=20):
        # | `x`             | 输入数据（如训练样本）
        # | `t`             | 标签数据（如目标词的ID）
        # | `max_epoch`     | 最大训练轮数（epoch）
        # | `batch_size`    | 每次训练的样本数量
        # | `max_grad`      | 梯度裁剪阈值（防止梯度爆炸）
        # | `eval_interval` | 每隔多少个迭代输出一次训练信息

        # * 数据准备与打乱
        data_size = len(x)  # 数据总量
        max_iters = data_size // batch_size  # 每轮(epoch)中的迭代次数
        self.eval_interval = eval_interval
        model, optimizer = self.model, self.optimizer
        total_loss = 0
        loss_count = 0

        start_time = time.time()
        for epoch in range(max_epoch):
            # 对训练数据进行随机打乱，提高训练鲁棒性
            idx = numpy.random.permutation(numpy.arange(data_size))
            x = x[idx]
            t = t[idx]

            for iters in range(max_iters):
                batch_x = x[iters * batch_size : (iters + 1) * batch_size]
                batch_t = t[iters * batch_size : (iters + 1) * batch_size]

                # 计算梯度，更新参数
                loss = model.forward(batch_x, batch_t)
                model.backward()

                # 去重权重 + 梯度裁剪 将共享的权重整合为1个
                params, grads = remove_duplicate(model.params, model.grads)

                # 梯度裁剪：避免梯度爆炸，将所有梯度的范数限制在 max_grad 范围内。
                if max_grad is not None:
                    clip_grads(grads, max_grad)

                # 使用优化器（SGD、Adam 等）根据当前的梯度来更新参数
                optimizer.update(params, grads)
                total_loss += loss
                loss_count += 1

                # 评价  每隔 eval_interval 次迭代就输出一次日志
                if (eval_interval is not None) and (iters % eval_interval) == 0:
                    avg_loss = total_loss / loss_count
                    elapsed_time = time.time() - start_time
                    print(
                        "| epoch %d |  iter %d / %d | time %d[s] | loss %.2f"
                        % (
                            self.current_epoch + 1,
                            iters + 1,
                            max_iters,
                            elapsed_time,
                            avg_loss,
                        )
                    )
                    self.loss_list.append(float(avg_loss))
                    total_loss, loss_count = 0, 0

            self.current_epoch += 1

    def plot(self, ylim=None):
        x = numpy.arange(len(self.loss_list))
        if ylim is not None:
            plt.ylim(*ylim)
        plt.plot(x, self.loss_list, label="train")
        plt.xlabel("iterations (x" + str(self.eval_interval) + ")")
        plt.ylabel("loss")
        plt.show()


class RnnlmTrainer:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.time_idx = None
        self.ppl_list = None
        self.eval_interval = None
        self.current_epoch = 0

    def get_batch(self, x, t, batch_size, time_size):
        batch_x = np.empty((batch_size, time_size), dtype="i")
        batch_t = np.empty((batch_size, time_size), dtype="i")

        data_size = len(x)
        jump = data_size // batch_size
        offsets = [
            i * jump for i in range(batch_size)
        ]  # mini-batch的各笔样本数据的开始位置

        for time in range(time_size):
            for i, offset in enumerate(offsets):
                batch_x[i, time] = x[(offset + self.time_idx) % data_size]
                batch_t[i, time] = t[(offset + self.time_idx) % data_size]
            self.time_idx += 1
        return batch_x, batch_t

    def fit(
        self,
        xs,
        ts,
        max_epoch=10,
        batch_size=20,
        time_size=35,
        max_grad=None,
        eval_interval=20,
    ):
        data_size = len(xs)
        max_iters = data_size // (batch_size * time_size)
        self.time_idx = 0
        self.ppl_list = []
        self.eval_interval = eval_interval
        model, optimizer = self.model, self.optimizer
        total_loss = 0
        loss_count = 0

        start_time = time.time()
        for epoch in range(max_epoch):
            for iters in range(max_iters):
                batch_x, batch_t = self.get_batch(xs, ts, batch_size, time_size)

                # 计算梯度，更新参数
                loss = model.forward(batch_x, batch_t)
                model.backward()
                params, grads = remove_duplicate(
                    model.params, model.grads
                )  # 将共享的权重整合为1个
                if max_grad is not None:
                    clip_grads(grads, max_grad)
                optimizer.update(params, grads)
                total_loss += loss
                loss_count += 1

                # 评价困惑度
                if (eval_interval is not None) and (iters % eval_interval) == 0:
                    ppl = np.exp(total_loss / loss_count)
                    elapsed_time = time.time() - start_time
                    print(
                        "| epoch %d |  iter %d / %d | time %d[s] | perplexity %.2f"
                        % (
                            self.current_epoch + 1,
                            iters + 1,
                            max_iters,
                            elapsed_time,
                            ppl,
                        )
                    )
                    self.ppl_list.append(float(ppl))
                    total_loss, loss_count = 0, 0

            self.current_epoch += 1

    def plot(self, ylim=None):
        x = numpy.arange(len(self.ppl_list))
        if ylim is not None:
            plt.ylim(*ylim)
        plt.plot(x, self.ppl_list, label="train")
        plt.xlabel("iterations (x" + str(self.eval_interval) + ")")
        plt.ylabel("perplexity")
        plt.show()


# todo 为了减少参数量或增强语言建模能力，常会多个模块共用同一权重，但训练时每次前向、
# todo 反向传播都会生成独立的参数引用和梯度，需要合并处理，否则训练会出错或不收敛。
def remove_duplicate(params, grads):
    """
    将参数列表中重复的权重整合为1个，
    加上与该权重对应的梯度
    """
    params, grads = params[:], grads[:]  # copy list

    while True:
        find_flg = False
        L = len(params)

        for i in range(0, L - 1):
            for j in range(i + 1, L):
                # 在共享权重的情况下
                if params[i] is params[j]:
                    grads[i] += grads[j]  # 加上梯度
                    find_flg = True
                    params.pop(j)
                    grads.pop(j)
                # 在作为转置矩阵共享权重的情况下（weight tying）
                elif (
                    params[i].ndim == 2
                    and params[j].ndim == 2
                    and params[i].T.shape == params[j].shape
                    and np.all(params[i].T == params[j])
                ):
                    grads[i] += grads[j].T
                    find_flg = True
                    params.pop(j)
                    grads.pop(j)

                if find_flg:
                    break
            if find_flg:
                break

        if not find_flg:
            break

    return params, grads

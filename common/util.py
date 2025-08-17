# coding: utf-8
import sys

sys.path.append("..")
import os
from common.np import *


def preprocess(text):
    text = text.lower()  # 全部转为小写，统一处理大小写
    text = text.replace(".", " .")  # 给句号（.）加空格，使其成为独立的词
    words = text.split(" ")  # 按空格分隔文本，得到一个词列表

    # * 构建词典
    word_to_id = {}  # word_to_id：将每个单词映射为唯一的数字 ID
    id_to_word = {}  # id_to_word：反向映射，用于从 ID 找回原单词
    for word in words:
        if word not in word_to_id:
            new_id = len(word_to_id)
            word_to_id[word] = new_id
            id_to_word[new_id] = word

    # 生成语料库（corpus） 把整个文本用 ID 替换，生成整数序列（用于模型输入）
    corpus = np.array([word_to_id[w] for w in words])

    return corpus, word_to_id, id_to_word


def cos_similarity(x, y, eps=1e-8):
    """计算余弦相似度

    :param x: 向量
    :param y: 向量
    :param eps: 用于防止“除数为0”的微小值
    :return:
    """
    nx = x / (np.sqrt(np.sum(x**2)) + eps)
    ny = y / (np.sqrt(np.sum(y**2)) + eps)
    return np.dot(nx, ny)


def most_similar(query, word_to_id, id_to_word, word_matrix, top=5):
    """相似单词的查找

    :param query: 查询词
    :param word_to_id: 从单词到单词ID的字典
    :param id_to_word: 从单词ID到单词的字典
    :param word_matrix: 汇总了单词向量的矩阵，假定保存了与各行对应的单词向量
    :param top: 显示到前几位
    """

    # 如果查询词不在字典中，直接报错并退出
    if query not in word_to_id:
        print("%s is not found" % query)
        return

    # 查到查询词的 ID，然后从 word_matrix 中取出对应的词向量
    print("\n[query] " + query)
    query_id = word_to_id[query]
    query_vec = word_matrix[query_id]

    # 计算与所有词的相似度
    vocab_size = len(id_to_word)
    similarity = np.zeros(vocab_size)
    for i in range(vocab_size):
        similarity[i] = cos_similarity(word_matrix[i], query_vec)

    # 输出最相似的前 top 个单词
    # (-1 * similarity).argsort()：相当于按相似度从高到低排序
    count = 0
    for i in (-1 * similarity).argsort():
        if id_to_word[i] == query:
            continue
        print(" %s: %s" % (id_to_word[i], similarity[i]))

        count += 1
        if count >= top:
            return


def convert_one_hot(corpus, vocab_size):
    """转换为one-hot表示

    :param corpus: 单词ID列表（一维或二维的NumPy数组）
    :param vocab_size: 词汇个数
    :return: one-hot表示（二维或三维的NumPy数组）
    """
    N = corpus.shape[0]

    if corpus.ndim == 1:
        one_hot = np.zeros((N, vocab_size), dtype=np.int32)
        for idx, word_id in enumerate(corpus):
            one_hot[idx, word_id] = 1

    elif corpus.ndim == 2:
        C = corpus.shape[1]
        one_hot = np.zeros((N, C, vocab_size), dtype=np.int32)
        for idx_0, word_ids in enumerate(corpus):
            for idx_1, word_id in enumerate(word_ids):
                one_hot[idx_0, idx_1, word_id] = 1

    return one_hot


def create_co_matrix(corpus, vocab_size, window_size=1):
    """生成共现矩阵
    :param corpus:      语料库（单词ID列表）
    :param vocab_size:  词汇表大小，即共有多少个唯一词（比如 len(word_to_id)）
    :param window_size: 窗口大小，默认值为1，表示只考虑左右各1个词作为上下文
    :return:            共现矩阵
    """

    # 初始化共现矩阵
    corpus_size = len(corpus)
    co_matrix = np.zeros((vocab_size, vocab_size), dtype=np.int32)

    # 外层循环遍历语料中每一个词的下标和对应的词ID
    # 内层循环根据 window_size， 查找当前词左边第 i 个、右边第 i 个词的 ID
    for idx, word_id in enumerate(corpus):
        for i in range(1, window_size + 1):
            left_idx = idx - i
            right_idx = idx + i

            if left_idx >= 0:
                left_word_id = corpus[left_idx]
                co_matrix[word_id, left_word_id] += 1

            if right_idx < corpus_size:
                right_word_id = corpus[right_idx]
                co_matrix[word_id, right_word_id] += 1

    return co_matrix


def ppmi(C, verbose=False, eps=1e-8):
    """从共现矩阵 C 计算正的点互信息（PPMI）矩阵，消除词频干扰。

    :param C: 共现矩阵
    :param verbose: 是否输出进展情况
    :return:
    """

    M = np.zeros_like(C, dtype=np.float32)  # 和 C 同形状的矩阵，用来保存最终的 PPMI 值
    N = np.sum(C)  # 所有共现频次的总和，也就是共现矩阵中所有元素之和
    S = np.sum(C, axis=0)  # S 是一个向量，记录了每个单词作为上下文出现的频率
    total = C.shape[0] * C.shape[1]  # 共现矩阵中所有元素的数量
    cnt = 0  # 进度条

    # 遍历矩阵中每一个位置 [i, j]，计算对应位置的 PMI 值
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            pmi = np.log2(C[i, j] * N / (S[j] * S[i]) + eps)
            M[i, j] = max(0, pmi)

            if verbose:
                cnt += 1
                if cnt % (total // 100 + 1) == 0:
                    print("%.1f%% done" % (100 * cnt / total))
    return M


def create_contexts_target(corpus, window_size=1):
    """生成上下文和目标词

    :param corpus: 语料库（单词ID列表）
    :param window_size: 窗口大小（当窗口大小为1时，左右各1个单词为上下文）
    :return:
    """
    target = corpus[window_size:-window_size]
    contexts = []

    for idx in range(window_size, len(corpus) - window_size):
        cs = []
        for t in range(-window_size, window_size + 1):
            if t == 0:
                continue
            cs.append(corpus[idx + t])
        contexts.append(cs)

    return np.array(contexts), np.array(target)


def to_cpu(x):
    import numpy

    if type(x) == numpy.ndarray:
        return x
    return np.asnumpy(x)


def to_gpu(x):
    import cupy

    if type(x) == cupy.ndarray:
        return x
    return cupy.asarray(x)


def clip_grads(grads, max_norm):
    total_norm = 0
    for grad in grads:
        total_norm += np.sum(grad**2)
    total_norm = np.sqrt(total_norm)

    rate = max_norm / (total_norm + 1e-6)
    if rate < 1:
        for grad in grads:
            grad *= rate


def eval_perplexity(model, corpus, batch_size=10, time_size=35):
    print("evaluating perplexity ...")
    corpus_size = len(corpus)
    total_loss, loss_cnt = 0, 0
    max_iters = (corpus_size - 1) // (batch_size * time_size)
    jump = (corpus_size - 1) // batch_size

    for iters in range(max_iters):
        xs = np.zeros((batch_size, time_size), dtype=np.int32)
        ts = np.zeros((batch_size, time_size), dtype=np.int32)
        time_offset = iters * time_size
        offsets = [time_offset + (i * jump) for i in range(batch_size)]
        for t in range(time_size):
            for i, offset in enumerate(offsets):
                xs[i, t] = corpus[(offset + t) % corpus_size]
                ts[i, t] = corpus[(offset + t + 1) % corpus_size]

        try:
            loss = model.forward(xs, ts, train_flg=False)
        except TypeError:
            loss = model.forward(xs, ts)
        total_loss += loss

        sys.stdout.write("\r%d / %d" % (iters, max_iters))
        sys.stdout.flush()

    print("")
    ppl = np.exp(total_loss / max_iters)
    return ppl


def eval_seq2seq(model, question, correct, id_to_char, verbos=False, is_reverse=False):
    correct = correct.flatten()
    # 开头的分隔符
    start_id = correct[0]
    correct = correct[1:]
    guess = model.generate(question, start_id, len(correct))

    # 转换为字符串
    question = "".join([id_to_char[int(c)] for c in question.flatten()])
    correct = "".join([id_to_char[int(c)] for c in correct])
    guess = "".join([id_to_char[int(c)] for c in guess])

    if verbos:
        if is_reverse:
            question = question[::-1]

        colors = {"ok": "\033[92m", "fail": "\033[91m", "close": "\033[0m"}
        print("Q", question)
        print("T", correct)

        is_windows = os.name == "nt"

        if correct == guess:
            mark = colors["ok"] + "☑" + colors["close"]
            if is_windows:
                mark = "O"
            print(mark + " " + guess)
        else:
            mark = colors["fail"] + "☒" + colors["close"]
            if is_windows:
                mark = "X"
            print(mark + " " + guess)
        print("---")

    return 1 if guess == correct else 0


def analogy(a, b, c, word_to_id, id_to_word, word_matrix, top=5, answer=None):
    for word in (a, b, c):
        if word not in word_to_id:
            print("%s is not found" % word)
            return

    print("\n[analogy] " + a + ":" + b + " = " + c + ":?")
    a_vec, b_vec, c_vec = (
        word_matrix[word_to_id[a]],
        word_matrix[word_to_id[b]],
        word_matrix[word_to_id[c]],
    )
    query_vec = b_vec - a_vec + c_vec
    query_vec = normalize(query_vec)

    similarity = np.dot(word_matrix, query_vec)

    if answer is not None:
        print(
            "==>"
            + answer
            + ":"
            + str(np.dot(word_matrix[word_to_id[answer]], query_vec))
        )

    count = 0
    for i in (-1 * similarity).argsort():
        if np.isnan(similarity[i]):
            continue
        if id_to_word[i] in (a, b, c):
            continue
        print(" {0}: {1}".format(id_to_word[i], similarity[i]))

        count += 1
        if count >= top:
            return


def normalize(x):
    if x.ndim == 2:
        s = np.sqrt((x * x).sum(1))
        x /= s.reshape((s.shape[0], 1))
    elif x.ndim == 1:
        s = np.sqrt((x * x).sum())
        x /= s
    return x

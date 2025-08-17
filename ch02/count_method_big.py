# coding: utf-8
import sys

sys.path.append(".")
import numpy as np
from common.util import most_similar, create_co_matrix, ppmi
from dataset import ptb


window_size = 2  # 共现窗口大小，左右各2个单词
wordvec_size = 100  # SVD后词向量的维度

# corpus: 单词ID组成的列表。
# word_to_id: 单词 → ID 的映射。
# id_to_word: ID → 单词的映射。
# vocab_size: 词汇总数。
corpus, word_to_id, id_to_word = ptb.load_data("train")
vocab_size = len(word_to_id)

# C: 大小为 (vocab_size, vocab_size) 的共现矩阵。每个位置 C[i][j] 表示词 i 和 j 在窗口中一起出现的次数。
print("counting  co-occurrence ...")
C = create_co_matrix(corpus, vocab_size, window_size)

# 将共现矩阵转换为 PPMI（Positive Pointwise Mutual Information）矩阵
print("calculating PPMI ...")
W = ppmi(C, verbose=True)

print("calculating SVD ...")
try:
    # truncated SVD (fast!)
    from sklearn.utils.extmath import randomized_svd

    U, S, V = randomized_svd(W, n_components=wordvec_size, n_iter=5, random_state=None)
    print("truncated SVD (fast!)\n")
except ImportError:
    # SVD (slow)
    U, S, V = np.linalg.svd(W)
    print("SVD (slow)\n")

# 使用 U 的前 100 个主成分（向量维度）作为词向量
word_vecs = U[:, :wordvec_size]

# most_similar() 通过计算 余弦相似度，找到最相似的 top=5 个词。
querys = ["you", "year", "car", "toyota"]
for query in querys:
    most_similar(query, word_to_id, id_to_word, word_vecs, top=5)

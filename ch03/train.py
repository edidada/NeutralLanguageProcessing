# coding: utf-8
import sys

sys.path.append(".")  # 为了引入父目录的文件而进行的设定
from common.trainer import Trainer
from common.optimizer import Adam
from simple_cbow import SimpleCBOW
from common.util import preprocess, create_contexts_target, convert_one_hot


window_size = 1  # 上下文窗口大小
hidden_size = 5  # 词向量维度
batch_size = 3  # 批量大小
max_epoch = 1000  # 训练轮数

# 文本处理
text = "You say goodbye and I say hello."
corpus, word_to_id, id_to_word = preprocess(text)

# 构造训练数据
vocab_size = len(word_to_id)
contexts, target = create_contexts_target(corpus, window_size)
target = convert_one_hot(target, vocab_size)
contexts = convert_one_hot(contexts, vocab_size)

# 模型训练
# vocab_size：词汇表大小（每个词都被映射为一个 one-hot 向量）
# hidden_size：词向量的维度（即中间的隐含层维度）
model = SimpleCBOW(vocab_size, hidden_size)
optimizer = Adam()
trainer = Trainer(model, optimizer)

trainer.fit(contexts, target, max_epoch, batch_size)
trainer.plot()

# 词向量输出
# 从训练好的模型中取出词向量（就是输入矩阵 W_in），然后打印出每个单词对应的向量
word_vecs = model.word_vecs
for word_id, word in id_to_word.items():
    print(word, word_vecs[word_id])

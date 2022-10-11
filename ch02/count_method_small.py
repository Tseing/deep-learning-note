import sys
sys.path.append("..")
import numpy as np
import matplotlib.pyplot as plt
from common.util import preprocess, creat_co_matrix, ppmi


text = "You say goodbye and I say hello."
corpus, word_to_id, id_to_word = preprocess(text)
vocab_size = len(id_to_word)
C = creat_co_matrix(corpus, vocab_size, window_size=1)
W = ppmi(C)

# SVD 奇异值分解
U, S, V = np.linalg.svd(W)

# 共现矩阵 [0 1 0 0 0 0 0]
# print(C[0])

# PPMI 矩阵 [0.        1.8073549 0.        0.        0.        0.        0.       ]
# print(W[0])

# SVD [-1.1102230e-16  3.4094876e-01 -1.2051624e-01 -4.1633363e-16 -1.1102230e-16 -9.3232495e-01 -2.4257469e-17]
# print(U[0])

# 将词向量降为二维向量
for word, word_id in word_to_id.items():
    plt.annotate(word, (U[word_id, 0], U[word_id, 1]))

plt.scatter(U[:, 0], U[:, 1], alpha=0.5)
plt.show()

import sys
sys.path.append("..")
import numpy as np
from common.util import preprocess, creat_co_matrix, cos_similarity, ppmi

text = "You say goodbye and I say hello."
corpus, word_to_id, id_to_word = preprocess(text)
vocab_size = len(word_to_id)
C = creat_co_matrix(corpus, vocab_size)
W = ppmi(C)

np.set_printoptions(precision=3)  # 3 位有效数字
print("covariance matrix")
print(C)
print("-" * 50)
print("PPMI")
print(W)

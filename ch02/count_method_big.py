import sys
sys.path.append("..")
import numpy as np
from common.util import most_similarity, creat_co_matrix, ppmi
from dataset import ptb

window_size = 2
wordvec_size = 100

corpus, word_to_id, id_to_word = ptb.load_data("train")
vocab_size = len(word_to_id)
print("counting co-occurrence ...")
C = creat_co_matrix(corpus, vocab_size, window_size)
print("calculating PPMI ...")
W = ppmi(C, verbose=True)

print("calculating SVD ...")
U, S, V = np.linalg.svd(W)

word_vecs = U[:, :wordvec_size]

querys = ["you", "year", "car", "toyota"]
for query in querys:
    most_similarity(query, word_to_id, id_to_word, word_vecs, top=5)

import numpy as np


def clip_grads(grads, max_norm):
    """
    梯度裁剪

    :param grads:
    :param max_norm:
    :return:
    """
    total_norm = 0
    for grad in grads:
        total_norm += np.sum(grad ** 2)
    total_norm = np.sqrt(total_norm)

    rate = max_norm / (total_norm + 1e-6)
    if rate < 1:
        for grad in grads:
            grad *= rate


def preprocess(text):
    """
    预处理文本，将文本按词划分

    :param text: (str) 文本
    :returns: corpus (list) 使用单词 id 表示的文本，
              word_to_id (dict) 可以使用单词查询其 id 的字典，
              id_to_word (dict) 可以使用 id 查询单词的字典
    """
    text = text.lower()
    text = text.replace(".", " .")  # 在句号前后插入空格用于分词
    words = text.split(" ")

    # key_to_value
    word_to_id = {}
    id_to_word = {}
    for word in words:
        if word not in word_to_id:
            new_id = len(word_to_id)
            word_to_id[word] = new_id
            id_to_word[new_id] = word

    corpus = np.array([word_to_id[w] for w in words])  # 所有 word id 的列表

    return corpus, word_to_id, id_to_word


def creat_co_matrix(corpus, vocal_size, window_size=1):
    """
    :param corpus: (str) 句子
    :param vocal_size: (int) 词数量，共现矩阵方阵的行与列
    :param window_size: (int) 窗口值，计算窗口值范围内邻近词的共现矩阵
    :return: (2d nparray) 共现矩阵
    """
    corpus_size = len(corpus)
    co_matrix = np.zeros((vocal_size, vocal_size), dtype=np.int32)

    for idx, word_id in enumerate(corpus):  # enumerate 将列表索引与内容转为字典
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


def cos_similarity(x, y, eps=1e-8):
    """
    计算两向量的余弦相似度

    :param x: (ndarray) 词向量
    :param y: (ndarray) 词向量
    :param eps: (float) 防止溢出的微小量
    :return: (float) 余弦相似度
    """
    nx = x / (np.sqrt(np.sum(x ** 2)) + eps)
    ny = y / (np.sqrt(np.sum(y ** 2)) + eps)

    return np.dot(nx, ny)


def most_similarity(query, word_to_id, id_to_word, word_matrix, top=5):
    """
    输出与搜索的单词最相似的几个词量向

    :param query: (str) 搜索的单词
    :param word_to_id: (dict) {word:id}
    :param id_to_word: (dict) {id:word}
    :param word_matrix: (2d nparray) 句子的共现矩阵
    :param top: (int) 输出单词数
    :return: None
    """
    if query not in word_to_id:
        print("%s is not found" % query)
        return None

    print("\n[query] " + query)
    query_id = word_to_id[query]
    query_vec = word_matrix[query_id]

    vocab_size = len(id_to_word)
    similarity = np.zeros(vocab_size)
    for i in range(vocab_size):
        similarity[i] = cos_similarity(word_matrix[i], query_vec)

    count = 0
    for i in (-1 * similarity).argsort():  # argsort 正序排列，加负号即为倒序
        if id_to_word[i] == query:
            continue
        print(" %s: %s" % (id_to_word[i], similarity[i]))

        count += 1
        if count >= top:
            return None


def ppmi(C, verbose=False, eps=1e-8):
    """
    计算正的点交互信息

    :param C: (ndarray) 共现矩阵
    :param verbose: (bool) 是否输出运行结果
    :param eps: 防止溢出的微小值
    :return: (ndarray) 正的点交互信息矩阵
    """
    M = np.zeros_like(C, dtype=np.float32)
    N = np.sum(C)  # 约为总单词出现次数
    S = np.sum(C, axis=0)  # 约为各个单词的出现次数
    total = C.shape[0] * C.shape[1]
    cnt = 0

    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            pmi = np.log2(C[i, j] * N / (S[j] * S[i]) + eps)
            M[i, j] = max(0, pmi)

            if verbose:  # 是否输出运行状态
                cnt += 1
                if cnt % (total // 100 + 1) == 0:
                    print("%.1f%% done" % (100 * cnt / total))

    return M


def create_contexts_target(corpus, window_size=1):
    """
    针对文本中的每个单词（target）生成左右相邻的上下文单词信息

    :param corpus: (list) 预处理后文本
    :param window_size: (int) 窗口值，左或右方向上相邻单词个数
    :returns: contexts: (list) 每个单词的上下文单词信息，
              target: (list) 目标单词
    """
    target = corpus[window_size:-window_size]
    contexts = []

    for idx in range(window_size, len(corpus) - window_size):
        cs = []
        for t in range(-window_size, window_size + 1):
            if t == 0:  # corpus[idx] = target
                continue
            cs.append(corpus[idx + t])
        contexts.append(cs)

    return np.array(contexts), np.array(target)


def convert_one_hot(corpus, vocab_size):
    """
    将文本序列转换为 one-hot 编码
    :param corpus: (list) 文本序列
    :param vocab_size: (int) 文本中所具有的不同单词数量
    :return: (ndarray) 包含每个单词 one-hot 向量的矩阵
    """
    N = corpus.shape[0]
    one_hot = None

    # 文本序列为一维列表
    if corpus.ndim == 1:
        one_hot = np.zeros((N, vocab_size), dtype=np.int32)
        for idx, word_id in enumerate(corpus):
            one_hot[idx, word_id] = 1

    # 文本序列为二维列表 (?)
    elif corpus.ndim == 2:
        C = corpus.shape[1]
        one_hot = np.zeros((N, C, vocab_size), dtype=np.int32)
        for idx_0, word_ids in enumerate(corpus):
            for idx_1, word_id in enumerate(word_ids):
                one_hot[idx_0, idx_1, word_id] = 1

    return one_hot

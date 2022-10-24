import sys
sys.path.append("..")
import numpy as np
from common.layers import MatMul, SoftmaxWithLoss


class SimeSkipGram:
    def __init__(self, vocab_size, hidden_size):
        V, H = vocab_size, hidden_size

        W_in = 0.01 * np.random.randn(V, H).astype("f")
        W_out = 0.01 * np.random.randn(H, V).astype("f")

        self.in_layer = MatMul(W_in)
        self.out_layer = MatMul(W_out)
        self.loss_layer0 = SoftmaxWithLoss()
        self.loss_layer1 = SoftmaxWithLoss()

        layers = [self.in_layer, self.out_layer]
        self.params, self.grams = [], []
        for layer in layers:
            self.params += layer.params
            self.grams += layer.grads

    def forward(self, center, contexts):
        h = self.in_layer.forward(center)   # 缺少左右邻接词的中心词
        score = self.out_layer.forward(h)
        loss0 = self.loss_layer0.forward(score, contexts[:, 0])
        loss1 = self.loss_layer1.forward(score, contexts[:, 1])
        loss = loss0 + loss1
        return loss

    def backward(self, dout0=1, dout1=1):
        ds0 = self.loss_layer0.backward(dout0)
        ds1 = self.loss_layer1.backward(dout1)
        da = ds0 + ds1
        dh = self.out_layer.backward(da)
        self.in_layer.backward(dh)

        return None

# 手撕实现一个完整的MLP网络，包含前向传播和反向传播


import numpy as np
from typing import List


class MLP:
    def __init__(self, input_dim: int,
                 hidden_layers: List[int],
                 output_dim: int):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_laysers = hidden_layers

        self.weight = []
        self.bias = []
        layers = [input_dim] + hidden_layers + [output_dim]

        for i in range(len(layers)-1):
            self.weight.append(np.zeros(layers[i],layers[i+1]))
            self.bias.append(0)
            # 初始化权重矩阵和偏置


    def weight_init(self):
        # 实现权重偏置高斯分布初始化 N(0,1)
        for i in range(len(self.weight)):
            self.weight[i] = 
            self.bias[i] = np.random()

    def forward(self):

        pass

    def bac

        
        
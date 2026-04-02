"""
softmax函数是将模型直接的logit输出转化为和为1的概率分布
x_i为模型输出 例如为多分类模型的输出 那么再套一层softmax即为
softmax(x_i) = \frac{e^{x_i}}{\sum_{j=1}^{n} e^{x_j}}
"""

from math import exp
import numpy as np

def softmax(x: np.ndarray):
    # 输入为二维ndarray数组 （batch， x_i）
    x_shape = x.shape()
    denom 

    return exp(x)


def main():
    return

if __name__ == "__main__":
    main()

